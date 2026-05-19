"""
Phase 4 Recommender — Thompson Sampling + SHAP Explanations
------------------------------------------------------------
Extends Phase 2/3 recommender with:
  1. Per-song SHAP feature attributions (explain_mode=True)
  2. Experiment arm awareness (routes users into A/B arms)
  3. Richer explanation text replacing the bandit.explain() heuristic

Same recommend() signature — routers unchanged.
"""
import logging
import time
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, not_

from app.models import Song, User, PlaylistItem
from app.services.bandit.thompson import bandit, build_context_scores
from app.services.bandit.state_manager import bandit_state_manager
from app.services.explainability.shap_explainer import explainer

logger = logging.getLogger(__name__)

CANDIDATE_POOL = 500


def _user_feature_dict(user: User | None) -> dict:
    if user is None:
        return {
            "avg_danceability": 0.5, "avg_energy": 0.5, "avg_valence": 0.5,
            "avg_tempo": 120.0, "avg_acousticness": 0.5,
        }
    return {
        "avg_danceability": user.avg_danceability,
        "avg_energy": user.avg_energy,
        "avg_valence": user.avg_valence,
        "avg_tempo": user.avg_tempo,
        "avg_acousticness": user.avg_acousticness,
    }


class BanditRecommender:

    async def recommend(
        self,
        user_id: str,
        db: AsyncSession,
        limit: int = 20,
        exclude_playlist: bool = True,
        explain: bool = False,           # Phase 4: include SHAP breakdown
        experiment_id: str | None = None, # Phase 4: A/B experiment context
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

        alphas  = np.array([alphas_dict[sid] for sid in song_ids])
        betas   = np.array([betas_dict[sid]  for sid in song_ids])
        context = build_context_scores(songs, user)

        top_indices  = bandit.rank(alphas, betas, context, top_k=limit)
        final_scores = bandit.sample_scores(alphas, betas, context)

        # Phase 4: batch SHAP explanations
        user_feats = _user_feature_dict(user)
        shap_results = None
        if explain:
            top_songs  = [songs[i] for i in top_indices]
            top_alphas = [float(alphas[i]) for i in top_indices]
            top_betas  = [float(betas[i])  for i in top_indices]
            top_scores = [float(final_scores[i]) for i in top_indices]
            song_feat_dicts = [
                {
                    "danceability": s.danceability, "energy": s.energy,
                    "valence": s.valence, "tempo": s.tempo,
                    "acousticness": s.acousticness,
                }
                for s in top_songs
            ]
            shap_results = explainer.explain_batch(
                song_feat_dicts, user_feats, top_alphas, top_betas, top_scores
            )

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            song  = songs[idx]
            a_i   = float(alphas[idx])
            b_i   = float(betas[idx])
            s_i   = float(final_scores[idx])
            shap  = shap_results[rank - 1] if shap_results else None

            explanation_text = (
                shap.explanation_text if shap
                else bandit.explain(a_i, b_i, float(context[idx]), song.name)
            )

            row = {
                "id":           song.id,
                "name":         song.name,
                "artist":       song.artist,
                "album":        song.album,
                "album_art_url":song.album_art_url,
                "preview_url":  song.preview_url,
                "duration_ms":  song.duration_ms,
                "popularity":   song.popularity,
                "explicit":     song.explicit,
                "danceability": song.danceability,
                "energy":       song.energy,
                "valence":      song.valence,
                "tempo":        song.tempo,
                "acousticness": song.acousticness,
                "genres":       song.genres or [],
                "score":        round(s_i, 4),
                "rank":         rank,
                "explanation":  explanation_text,
            }

            if shap:
                row["shap"] = {
                    "feature_contributions": shap.feature_contributions,
                    "dominant_feature":      shap.dominant_feature,
                    "score_breakdown":       shap.score_breakdown,
                }

            results.append(row)

        latency_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            f"Bandit recommend: user={user_id} n={len(results)} "
            f"explain={explain} latency={latency_ms:.1f}ms"
        )
        return results, latency_ms


recommender = BanditRecommender()
