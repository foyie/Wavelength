"""
Explainability Router — Phase 4
---------------------------------
On-demand SHAP feature attribution for recommendations.

GET  /api/explain/{user_id}/{song_id}      — explain why this song was recommended
POST /api/explain/{user_id}/batch          — explain a batch of songs at once
GET  /api/explain/{user_id}/profile        — explain the user's taste profile
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional

from app.db import get_db
from app.models import Song, User, BanditState
from app.services.explainability.shap_explainer import explainer
from app.services.bandit.state_manager import bandit_state_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/explain", tags=["explainability"])

FEATURE_LABELS = {
    "danceability": "Danceability",
    "energy": "Energy",
    "valence": "Mood/Positivity",
    "tempo_norm": "Tempo",
    "acousticness": "Acoustic Feel",
    "bandit_prior": "Learned Preference",
}


class BatchExplainRequest(BaseModel):
    song_ids: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/{user_id}/{song_id}")
async def explain_recommendation(
    user_id: str,
    song_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Explain why a specific song was (or would be) recommended to a user.

    Returns SHAP feature contributions, dominant feature, and a
    human-readable explanation sentence.
    """
    # Load song
    song_result = await db.execute(select(Song).where(Song.id == song_id))
    song = song_result.scalar_one_or_none()
    if not song:
        raise HTTPException(status_code=404, detail=f"Song '{song_id}' not found")

    # Load user
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()

    # Load bandit arm
    alpha, beta = await bandit_state_manager.get_single_arm(user_id, song_id, db)
    n_obs = max(0, int(alpha + beta - 2))

    # Build feature dicts
    song_features = {
        "danceability": song.danceability,
        "energy": song.energy,
        "valence": song.valence,
        "tempo": song.tempo,
        "acousticness": song.acousticness,
    }

    user_features = {
        "avg_danceability": user.avg_danceability if user else 0.5,
        "avg_energy": user.avg_energy if user else 0.5,
        "avg_valence": user.avg_valence if user else 0.5,
        "avg_tempo": user.avg_tempo if user else 120.0,
        "avg_acousticness": user.avg_acousticness if user else 0.5,
    }

    # Score approximation (mean of Beta distribution)
    from app.services.bandit.thompson import CONTEXT_W
    from app.services.bandit.thompson import build_context_scores
    import numpy as np

    # Approximate score for explanation purposes
    bandit_mean = alpha / (alpha + beta)
    user_obj_dummy = type("U", (), user_features)()
    context = build_context_scores([song], user_obj_dummy if user else None)
    approx_score = (1 - CONTEXT_W) * bandit_mean + CONTEXT_W * float(context[0])

    result = explainer.explain(
        song_features=song_features,
        user_features=user_features,
        alpha=alpha,
        beta=beta,
        final_score=approx_score,
    )

    return {
        "user_id": user_id,
        "song": {
            "id": song.id,
            "name": song.name,
            "artist": song.artist,
        },
        "explanation_text": result.explanation_text,
        "dominant_feature": result.dominant_feature,
        "dominant_feature_label": FEATURE_LABELS.get(result.dominant_feature, result.dominant_feature),
        "feature_contributions": {
            FEATURE_LABELS.get(k, k): round(v, 4)
            for k, v in result.feature_contributions.items()
        },
        "score_breakdown": result.score_breakdown,
        "bandit_info": {
            "alpha": round(alpha, 3),
            "beta": round(beta, 3),
            "mean_reward": round(bandit_mean, 4),
            "n_observations": n_obs,
            "uncertainty": round(
                ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5, 4
            ),
        },
    }


@router.post("/{user_id}/batch")
async def explain_batch(
    user_id: str,
    payload: BatchExplainRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Explain a batch of song recommendations in one call.
    Used by the frontend to show feature bars on the recommendation list.
    Limited to 50 songs per request.
    """
    if len(payload.song_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 songs per batch request")

    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()

    user_features = {
        "avg_danceability": user.avg_danceability if user else 0.5,
        "avg_energy": user.avg_energy if user else 0.5,
        "avg_valence": user.avg_valence if user else 0.5,
        "avg_tempo": user.avg_tempo if user else 120.0,
        "avg_acousticness": user.avg_acousticness if user else 0.5,
    }

    results = []
    for song_id in payload.song_ids:
        song_result = await db.execute(select(Song).where(Song.id == song_id))
        song = song_result.scalar_one_or_none()
        if not song:
            continue

        alpha, beta = await bandit_state_manager.get_single_arm(user_id, song_id, db)
        bandit_mean = alpha / (alpha + beta)

        song_features = {
            "danceability": song.danceability,
            "energy": song.energy,
            "valence": song.valence,
            "tempo": song.tempo,
            "acousticness": song.acousticness,
        }

        shap = explainer.explain(
            song_features=song_features,
            user_features=user_features,
            alpha=alpha,
            beta=beta,
            final_score=bandit_mean,
        )

        results.append({
            "song_id": song_id,
            "explanation_text": shap.explanation_text,
            "dominant_feature": shap.dominant_feature,
            "feature_contributions": {
                k: round(v, 4) for k, v in shap.feature_contributions.items()
            },
            "score_breakdown": shap.score_breakdown,
        })

    return {
        "user_id": user_id,
        "total": len(results),
        "explanations": results,
    }


@router.get("/{user_id}/profile")
async def explain_user_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Explain the user's current taste profile — what their preference
    centroid looks like and how it compares to the population average.
    """
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    # Population average (rough estimate — could be a DB aggregate)
    POP_AVG = {
        "danceability": 0.58,
        "energy": 0.62,
        "valence": 0.48,
        "tempo": 120.0,
        "acousticness": 0.25,
    }

    profile = {
        "user_id": user_id,
        "total_interactions": user.total_interactions,
        "preference_vector": {
            "danceability": round(user.avg_danceability or 0.5, 3),
            "energy": round(user.avg_energy or 0.5, 3),
            "valence": round(user.avg_valence or 0.5, 3),
            "tempo": round(user.avg_tempo or 120.0, 1),
            "acousticness": round(user.avg_acousticness or 0.5, 3),
        },
        "vs_population_avg": {
            "danceability": round((user.avg_danceability or 0.5) - POP_AVG["danceability"], 3),
            "energy": round((user.avg_energy or 0.5) - POP_AVG["energy"], 3),
            "valence": round((user.avg_valence or 0.5) - POP_AVG["valence"], 3),
            "tempo": round((user.avg_tempo or 120.0) - POP_AVG["tempo"], 1),
            "acousticness": round((user.avg_acousticness or 0.5) - POP_AVG["acousticness"], 3),
        },
        "taste_summary": _summarise_taste(user),
    }

    return profile


def _summarise_taste(user: User) -> str:
    """Generate a short plain-text taste summary from the user's EMA vector."""
    traits = []

    if (user.avg_energy or 0.5) > 0.7:
        traits.append("high-energy")
    elif (user.avg_energy or 0.5) < 0.4:
        traits.append("mellow")

    if (user.avg_danceability or 0.5) > 0.7:
        traits.append("danceable")

    if (user.avg_valence or 0.5) > 0.65:
        traits.append("upbeat")
    elif (user.avg_valence or 0.5) < 0.35:
        traits.append("introspective")

    if (user.avg_acousticness or 0.5) > 0.6:
        traits.append("acoustic")
    elif (user.avg_acousticness or 0.5) < 0.2:
        traits.append("electronic/produced")

    if not traits:
        return "Balanced taste across features — still learning your preferences"

    return f"You tend to prefer {', '.join(traits[:-1])} and {traits[-1]} music" \
        if len(traits) > 1 else f"You tend to prefer {traits[0]} music"
