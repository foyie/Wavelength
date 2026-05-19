"""
Thompson Sampling Contextual Bandit
------------------------------------
Each (user, song) arm has a Beta distribution over its reward probability:
    p ~ Beta(alpha, beta)

On every recommendation:
  - Sample p_i ~ Beta(alpha_i, beta_i) for each candidate arm
  - Multiply by a contextual prior from audio-feature similarity
  - Return top-N by sampled score

On every feedback event (via Kafka consumer):
  - alpha += reward   (successes)
  - beta  += (1 - clipped_reward)   (failures)

This gives the key exploration property: songs with few observations have
high posterior variance and get sampled more often, naturally balancing
explore/exploit without any explicit epsilon parameter.

Key design decisions:
  - Beta params are stored in Postgres (BanditState table) for persistence
  - Redis caches the full (alpha, beta) vectors per user for sub-10ms lookup
  - Candidate pool is capped at top 500 songs to bound scoring latency
  - Context multiplier uses cosine similarity so audio features guide cold-start
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# How much the audio-feature context modulates the bandit score.
# 0.0 = pure bandit (ignores context), 1.0 = context dominates.
CONTEXT_WEIGHT = 0.35
TEMPO_NORM = 200.0


class ThompsonBandit:
    """
    Stateless scorer — state lives in Postgres/Redis, not in this object.
    Accepts pre-loaded alpha/beta arrays and returns ranked indices.
    """

    def sample_scores(
        self,
        alphas: np.ndarray,           # shape (n_songs,)
        betas: np.ndarray,            # shape (n_songs,)
        context_scores: np.ndarray,   # shape (n_songs,) in [0, 1]
        n_samples: int = 1,
    ) -> np.ndarray:
        """
        Draw one Thompson sample per arm and blend with context.
        Returns final score array, shape (n_songs,).
        """
        # Draw from Beta posteriors
        bandit_samples = np.random.beta(alphas, betas)    # (n_songs,)

        # Blend: (1 - w) * bandit + w * context
        w = CONTEXT_WEIGHT
        final = (1.0 - w) * bandit_samples + w * context_scores

        return final

    def rank(
        self,
        alphas: np.ndarray,
        betas: np.ndarray,
        context_scores: np.ndarray,
        top_k: int,
    ) -> list[int]:
        """Return indices of top_k songs by sampled score."""
        scores = self.sample_scores(alphas, betas, context_scores)
        # argpartition is O(n) vs O(n log n) for argsort — faster for large n
        if top_k >= len(scores):
            return list(np.argsort(scores)[::-1])
        part = np.argpartition(scores, -top_k)[-top_k:]
        top_sorted = part[np.argsort(scores[part])[::-1]]
        return top_sorted.tolist()

    def update(
        self,
        alpha: float,
        beta: float,
        reward: float,
    ) -> tuple[float, float]:
        """
        Bayesian update for a single arm.
        reward should be in [0, 1] — clamp before calling.
        Returns updated (alpha, beta).
        """
        clipped = max(0.0, min(1.0, reward))
        new_alpha = alpha + clipped
        new_beta = beta + (1.0 - clipped)
        return new_alpha, new_beta

    def explain(
        self,
        alpha: float,
        beta: float,
        context_score: float,
        song_name: str,
    ) -> str:
        """Generate a human-readable explanation for why this song was ranked here."""
        n_obs = int(alpha + beta - 2)  # subtract the 1,1 prior
        mean = alpha / (alpha + beta)

        if n_obs < 3:
            return f"New to your taste — exploring for you · {int(context_score * 100)}% audio match"
        elif mean > 0.75:
            return f"Strong match based on {n_obs} signals · {int(context_score * 100)}% audio similarity"
        elif context_score > 0.75:
            return f"High audio feature match with songs you love"
        else:
            return f"Recommended based on {n_obs} listening signals"


def build_context_scores(
    songs: list,          # list of Song ORM objects
    user,                 # User ORM object or None
) -> np.ndarray:
    """
    Compute cosine similarity between each song's feature vector
    and the user's preference centroid. Falls back to popularity if no user.
    Returns array in [0, 1], shape (n_songs,).
    """
    if user is None:
        return np.array([(s.popularity or 0) / 100.0 for s in songs])

    user_vec = np.array([
        user.avg_danceability or 0.5,
        user.avg_energy or 0.5,
        user.avg_valence or 0.5,
        (user.avg_tempo or 120.0) / TEMPO_NORM,
        user.avg_acousticness or 0.5,
    ])

    context = np.zeros(len(songs))
    for i, song in enumerate(songs):
        song_vec = np.array([
            song.danceability or 0.5,
            song.energy or 0.5,
            song.valence or 0.5,
            (song.tempo or 120.0) / TEMPO_NORM,
            song.acousticness or 0.5,
        ])
        dot = np.dot(song_vec, user_vec)
        norm = np.linalg.norm(song_vec) * np.linalg.norm(user_vec)
        cos = dot / norm if norm > 0 else 0.0
        context[i] = (cos + 1.0) / 2.0  # shift from [-1,1] to [0,1]

    return context


# Module-level singleton — stateless, safe to share across async tasks
bandit = ThompsonBandit()
