"""
Phase 1 Recommender — Popularity Baseline
-----------------------------------------
Scores songs using a weighted combination of:
  - Spotify popularity  (60%)
  - User preference similarity  (40%)

Phase 2 replaces score() with Thompson Sampling posterior sampling.
The interface (recommend()) stays the same so the router doesn't change.
"""
import logging
import time
from typing import Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, not_

from app.models import Song, User, PlaylistItem
from app.services.feature_store import feature_store
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FEATURE_KEYS = ["danceability", "energy", "valence", "tempo", "acousticness"]
TEMPO_NORM = 200.0  # rough max BPM for normalisation


def normalise_song(song: Song) -> np.ndarray:
    """Return a 5-element feature vector in [0, 1]."""
    return np.array([
        song.danceability or 0.5,
        song.energy or 0.5,
        song.valence or 0.5,
        (song.tempo or 120.0) / TEMPO_NORM,
        song.acousticness or 0.5,
    ])


def normalise_user(user: User) -> np.ndarray:
    return np.array([
        user.avg_danceability or 0.5,
        user.avg_energy or 0.5,
        user.avg_valence or 0.5,
        (user.avg_tempo or 120.0) / TEMPO_NORM,
        user.avg_acousticness or 0.5,
    ])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class PopularityRecommender:
    """
    Baseline recommender for Phase 1.
    Replace `score()` in Phase 2 with bandit posterior sampling.
    """

    def score(self, song: Song, user: Optional[User]) -> float:
        """
        Combined score: popularity + user similarity.
        Returns a value in [0, 1].
        """
        popularity_score = (song.popularity or 0) / 100.0

        if user is None:
            return popularity_score

        song_vec = normalise_song(song)
        user_vec = normalise_user(user)
        similarity = (cosine_similarity(song_vec, user_vec) + 1.0) / 2.0  # shift to [0,1]

        w = settings.popularity_weight
        return w * popularity_score + (1 - w) * similarity

    def explain(self, song: Song, user: Optional[User]) -> str:
        """Simple text explanation. Phase 4 replaces with SHAP."""
        if user is None:
            return f"Popular track with {song.popularity}/100 popularity score"

        if song.energy and user.avg_energy:
            if abs(song.energy - user.avg_energy) < 0.15:
                return f"Matches your energy preference · {song.popularity}/100 popularity"

        if song.danceability and user.avg_danceability:
            if abs(song.danceability - user.avg_danceability) < 0.15:
                return f"High danceability match · {song.popularity}/100 popularity"

        return f"Trending track · {song.popularity}/100 popularity"

    async def recommend(
        self,
        user_id: str,
        db: AsyncSession,
        limit: int = 20,
        exclude_playlist: bool = True,
    ) -> tuple[list[dict], float]:
        """
        Returns (ranked_songs, latency_ms).
        Each dict: song fields + score + rank + explanation.
        """
        t0 = time.monotonic()

        # Load user (may be None for new users)
        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()

        # Songs already in playlist
        excluded_ids: set[str] = set()
        if exclude_playlist:
            pl_result = await db.execute(
                select(PlaylistItem.song_id).where(PlaylistItem.user_id == user_id)
            )
            excluded_ids = {row[0] for row in pl_result.fetchall()}

        # Fetch candidate songs (top 500 by popularity for fast scoring)
        q = select(Song).order_by(Song.popularity.desc()).limit(500)
        if excluded_ids:
            q = q.where(not_(Song.id.in_(excluded_ids)))
        songs_result = await db.execute(q)
        songs = songs_result.scalars().all()

        # Score every candidate
        scored = [(song, self.score(song, user)) for song in songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:limit]

        results = []
        for rank, (song, score) in enumerate(top, start=1):
            results.append({
                "id": song.id,
                "name": song.name,
                "artist": song.artist,
                "album": song.album,
                "album_art_url": song.album_art_url,
                "preview_url": song.preview_url,
                "duration_ms": song.duration_ms,
                "popularity": song.popularity,
                "explicit": song.explicit,
                "danceability": song.danceability,
                "energy": song.energy,
                "valence": song.valence,
                "tempo": song.tempo,
                "acousticness": song.acousticness,
                "genres": song.genres or [],
                "score": round(score, 4),
                "rank": rank,
                "explanation": self.explain(song, user),
            })

        latency_ms = (time.monotonic() - t0) * 1000
        return results, latency_ms


# Singleton — Phase 2 will replace this with a bandit-backed recommender
recommender = PopularityRecommender()
