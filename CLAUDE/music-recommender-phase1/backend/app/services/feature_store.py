"""
Feature store backed by Redis.
Fast path: Redis (5-10ms).  Fallback: compute from PostgreSQL + populate cache.
"""
import json
import logging
from typing import Optional
import redis.asyncio as redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeatureStore:
    def __init__(self):
        self._client: Optional[redis.Redis] = None

    def _get_client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return self._client

    # ── Song features ────────────────────────────────────────────────────────

    async def get_song_features(self, song_id: str) -> Optional[dict]:
        client = self._get_client()
        raw = await client.get(f"song:{song_id}:features")
        if raw:
            return json.loads(raw)
        return None

    async def set_song_features(self, song_id: str, features: dict):
        client = self._get_client()
        await client.setex(
            f"song:{song_id}:features",
            settings.redis_feature_ttl,
            json.dumps(features),
        )

    async def set_song_features_bulk(self, songs: list[dict]):
        """Pipeline bulk insert for initial data load."""
        client = self._get_client()
        async with client.pipeline() as pipe:
            for song in songs:
                features = {
                    "id": song["id"],
                    "popularity": song.get("popularity", 0),
                    "danceability": song.get("danceability"),
                    "energy": song.get("energy"),
                    "valence": song.get("valence"),
                    "tempo": song.get("tempo"),
                    "acousticness": song.get("acousticness"),
                }
                pipe.setex(
                    f"song:{song['id']}:features",
                    settings.redis_feature_ttl,
                    json.dumps(features),
                )
            await pipe.execute()
        logger.info(f"Bulk-loaded features for {len(songs)} songs")

    # ── User features ─────────────────────────────────────────────────────────

    async def get_user_features(self, user_id: str) -> Optional[dict]:
        client = self._get_client()
        raw = await client.get(f"user:{user_id}:features")
        if raw:
            return json.loads(raw)
        return None

    async def set_user_features(self, user_id: str, features: dict):
        client = self._get_client()
        await client.setex(
            f"user:{user_id}:features",
            settings.redis_feature_ttl,
            json.dumps(features),
        )

    async def invalidate_user_features(self, user_id: str):
        """Call after feedback to force recompute on next request."""
        client = self._get_client()
        await client.delete(f"user:{user_id}:features")

    # ── Recommendation cache ──────────────────────────────────────────────────

    async def get_cached_recommendations(self, user_id: str) -> Optional[list]:
        client = self._get_client()
        raw = await client.get(f"user:{user_id}:recs")
        if raw:
            return json.loads(raw)
        return None

    async def set_cached_recommendations(self, user_id: str, recs: list, ttl: int = 300):
        """Cache recommendations for 5 minutes by default."""
        client = self._get_client()
        await client.setex(
            f"user:{user_id}:recs",
            ttl,
            json.dumps(recs),
        )

    async def invalidate_recommendations(self, user_id: str):
        client = self._get_client()
        await client.delete(f"user:{user_id}:recs")

    # ── Health ────────────────────────────────────────────────────────────────

    async def ping(self) -> str:
        try:
            client = self._get_client()
            await client.ping()
            return "healthy"
        except Exception as e:
            return f"unhealthy: {e}"

    async def close(self):
        if self._client:
            await self._client.aclose()


# Singleton
feature_store = FeatureStore()
