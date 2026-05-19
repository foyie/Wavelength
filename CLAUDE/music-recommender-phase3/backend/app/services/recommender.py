"""
Phase 2 Recommender — Thompson Sampling Bandit
-----------------------------------------------
Drop-in replacement for PopularityRecommender.
Same recommend() signature — routers unchanged.

Algorithm:
  1. Fetch top-500 candidate songs (popularity filter for bounded latency)
  2. Load (alpha, beta) for each (user, song) arm from Redis / Postgres
  3. Compute context scores from audio-feature cosine similarity
  4. Draw one Thompson sample per arm: p_i ~ Beta(alpha_i, beta_i)
  5. Final score = (1 - w) * p_i + w * context_i
  6. Return top-N by final score

Cold-start:
  - New user  → default prior (1, 1), context drives ranking
  - New song  → default prior (1, 1), popularity pushes it into candidates
"""
import logging
import time
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, not_

from app.models import Song, User, PlaylistItem
from app.services.bandit.thompson import bandit, build_context_scores
from app.services.bandit.state_manager import bandit_state_manager

logger = logging.getLogger(__name__)

CANDIDATE_POOL = 500


class BanditRecommender:

    async def recommend(
        self,
        user_id: str,
        db: AsyncSession,
        limit: int = 20,
        exclude_playlist: bool = True,
    ) -> tuple[list[dict], float]:
        t0 = time.monotonic()

        user_result = await db.execute(select(User).where(User.id == user_id))
        user = user_result.scalar_one_or_none()

        excluded_ids: set[str] = set()
        if exclude_playlist:
            pl_result = await db.execute(
                select(PlaylistItem.song_id).where(PlaylistItem.user_id == user_id)
            )
            excluded_ids = {row[0] for row in pl_result.fetchall()}

        q = select(Song).order_by(Song.popularity.desc()).limit(CANDIDATE_POOL)
        if excluded_ids:
            q = q.where(not_(Song.id.in_(excluded_ids)))
        songs_result = await db.execute(q)
        songs: list[Song] = list(songs_result.scalars().all())

        if not songs:
            return [], (time.monotonic() - t0) * 1000

        song_ids = [s.id for s in songs]

        alphas_dict, betas_dict = await bandit_state_manager.load_user_state(
            user_id, song_ids, db
        )

        alphas = np.array([alphas_dict[sid] for sid in song_ids])
        betas  = np.array([betas_dict[sid]  for sid in song_ids])
        context = build_context_scores(songs, user)

        top_indices = bandit.rank(alphas, betas, context, top_k=limit)
        final_scores = bandit.sample_scores(alphas, betas, context)

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            song = songs[idx]
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
                "score": round(float(final_scores[idx]), 4),
                "rank": rank,
                "explanation": bandit.explain(
                    float(alphas[idx]), float(betas[idx]),
                    float(context[idx]), song.name
                ),
            })

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"Bandit recommend: user={user_id} n={len(results)} latency={latency_ms:.1f}ms")
        return results, latency_ms


recommender = BanditRecommender()
