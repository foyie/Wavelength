import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import RecommendationResponse, RecommendedSong
from app.services.recommender import recommender
from app.services.feature_store import feature_store
from app.services.experiments.registry import experiment_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recommend", tags=["recommendations"])


@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    exclude_playlist: bool = Query(default=True),
    bypass_cache: bool = Query(default=False),
    explain: bool = Query(default=False, description="Include SHAP feature attributions"),
    experiment_id: Optional[str] = Query(default=None, description="A/B experiment to enrol in"),
    db: AsyncSession = Depends(get_db),
):
    """
    Phase 4: Thompson Sampling + optional SHAP explanations + A/B experiment routing.

    - explain=true adds per-song feature_contributions (SHAP) to each result
    - experiment_id enrols the user in the named A/B experiment
    - bypass_cache=true forces a fresh bandit sample
    """
    # Determine algorithm label (may include experiment arm)
    algorithm = "thompson_sampling"
    if experiment_id:
        arm = experiment_registry.assign_arm(user_id, experiment_id)
        if arm:
            algorithm = f"thompson_sampling:{experiment_id}:{arm}"

    if not bypass_cache and not explain:
        cache_key = f"{user_id}:limit={limit}"
        cached = await feature_store.get_cached_recommendations(cache_key)
        if cached:
            return RecommendationResponse(
                user_id=user_id,
                songs=[RecommendedSong(**s) for s in cached],
                algorithm=f"{algorithm}_cached",
                latency_ms=0.5,
            )

    songs_data, latency_ms = await recommender.recommend(
        user_id=user_id,
        db=db,
        limit=limit,
        exclude_playlist=exclude_playlist,
        explain=explain,
        experiment_id=experiment_id,
    )

    if not explain:
        cache_key = f"{user_id}:limit={limit}"
        await feature_store.set_cached_recommendations(cache_key, songs_data, ttl=300)

    logger.info(
        f"Recommendations: user={user_id} n={len(songs_data)} "
        f"explain={explain} latency={latency_ms:.1f}ms"
    )
    return RecommendationResponse(
        user_id=user_id,
        songs=[RecommendedSong(**s) for s in songs_data],
        algorithm=algorithm,
        latency_ms=round(latency_ms, 2),
    )
