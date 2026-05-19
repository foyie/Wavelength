import logging
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import RecommendationResponse, RecommendedSong
from app.services.recommender import recommender
from app.services.feature_store import feature_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/recommend", tags=["recommendations"])


@router.get("/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    exclude_playlist: bool = Query(default=True),
    bypass_cache: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
):
    """
    Phase 2: Thompson Sampling bandit recommendations.
    Set bypass_cache=true to force a fresh bandit sample.
    """
    if not bypass_cache:
        cache_key = f"{user_id}:limit={limit}"
        cached = await feature_store.get_cached_recommendations(cache_key)
        if cached:
            return RecommendationResponse(
                user_id=user_id,
                songs=[RecommendedSong(**s) for s in cached],
                algorithm="thompson_sampling_cached",
                latency_ms=0.5,
            )

    songs_data, latency_ms = await recommender.recommend(
        user_id=user_id, db=db, limit=limit, exclude_playlist=exclude_playlist,
    )

    cache_key = f"{user_id}:limit={limit}"
    await feature_store.set_cached_recommendations(cache_key, songs_data, ttl=300)

    logger.info(f"Recommendations: user={user_id} n={len(songs_data)} latency={latency_ms:.1f}ms")
    return RecommendationResponse(
        user_id=user_id,
        songs=[RecommendedSong(**s) for s in songs_data],
        algorithm="thompson_sampling",
        latency_ms=round(latency_ms, 2),
    )
