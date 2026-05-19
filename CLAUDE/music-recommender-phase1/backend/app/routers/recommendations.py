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
    db: AsyncSession = Depends(get_db),
):
    """
    Return ranked song recommendations for a user.
    Phase 1: Popularity + preference similarity baseline.
    Phase 2: Thompson Sampling bandit (drop-in replacement).
    """
    # Check recommendation cache (5 min TTL)
    cache_key = f"{user_id}:limit={limit}"
    cached = await feature_store.get_cached_recommendations(cache_key)
    if cached:
        logger.debug(f"Cache hit for user {user_id}")
        songs = [RecommendedSong(**s) for s in cached]
        return RecommendationResponse(
            user_id=user_id,
            songs=songs,
            algorithm="popularity_cached",
            latency_ms=0.5,
        )

    songs_data, latency_ms = await recommender.recommend(
        user_id=user_id,
        db=db,
        limit=limit,
        exclude_playlist=exclude_playlist,
    )

    # Cache for 5 minutes
    await feature_store.set_cached_recommendations(cache_key, songs_data, ttl=300)

    songs = [RecommendedSong(**s) for s in songs_data]
    logger.info(f"Recommendations served for {user_id} in {latency_ms:.1f}ms")

    return RecommendationResponse(
        user_id=user_id,
        songs=songs,
        algorithm="popularity_baseline",
        latency_ms=round(latency_ms, 2),
    )
