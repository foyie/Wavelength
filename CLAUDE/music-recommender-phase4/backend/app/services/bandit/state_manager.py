"""
BanditStateManager
------------------
Single source of truth for per-user, per-song Beta distribution parameters.

Read path  (hot):   Redis hash  →  O(1) HGETALL per user
Write path (warm):  Kafka consumer updates Redis first, Postgres async
Fallback:           If Redis miss, load from Postgres and repopulate cache

Redis key schema:
  bandit:{user_id}:alphas   → hash of {song_id: float}
  bandit:{user_id}:betas    → hash of {song_id: float}
  bandit:{user_id}:updated  → timestamp string

Default priors: alpha=1, beta=1  (uniform Beta — maximum uncertainty)
"""
import json
import logging
from datetime import datetime
from typing import Optional

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models import BanditState
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

BANDIT_TTL = 7 * 24 * 3600   # 7 days — evict inactive users
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0


class BanditStateManager:

    def __init__(self):
        self._redis: Optional[redis.Redis] = None

    def _r(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return self._redis

    # ── Read ────────────────────────────────────────────────────────────────

    async def load_user_state(
        self,
        user_id: str,
        song_ids: list[str],
        db: AsyncSession,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Returns (alphas_dict, betas_dict) for all song_ids.
        Missing arms get the default prior (1, 1).
        """
        alphas, betas = await self._load_from_redis(user_id, song_ids)
        missing = [sid for sid in song_ids if sid not in alphas]

        if missing:
            db_alphas, db_betas = await self._load_from_postgres(user_id, missing, db)
            alphas.update(db_alphas)
            betas.update(db_betas)

            # Backfill Redis from Postgres
            if db_alphas:
                await self._write_to_redis(user_id, db_alphas, db_betas, extend_ttl=False)

        # Any still-missing arms → default prior
        for sid in song_ids:
            alphas.setdefault(sid, DEFAULT_ALPHA)
            betas.setdefault(sid, DEFAULT_BETA)

        return alphas, betas

    async def _load_from_redis(
        self,
        user_id: str,
        song_ids: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        r = self._r()
        alpha_key = f"bandit:{user_id}:alphas"
        beta_key = f"bandit:{user_id}:betas"

        try:
            pipe = r.pipeline()
            pipe.hmget(alpha_key, song_ids)
            pipe.hmget(beta_key, song_ids)
            results = await pipe.execute()

            alphas = {}
            betas = {}
            for sid, a, b in zip(song_ids, results[0], results[1]):
                if a is not None:
                    alphas[sid] = float(a)
                if b is not None:
                    betas[sid] = float(b)
            return alphas, betas
        except Exception as e:
            logger.warning(f"Redis read failed for {user_id}: {e}")
            return {}, {}

    async def _load_from_postgres(
        self,
        user_id: str,
        song_ids: list[str],
        db: AsyncSession,
    ) -> tuple[dict[str, float], dict[str, float]]:
        result = await db.execute(
            select(BanditState).where(
                BanditState.user_id == user_id,
                BanditState.song_id.in_(song_ids),
            )
        )
        rows = result.scalars().all()
        alphas = {r.song_id: r.alpha for r in rows}
        betas = {r.song_id: r.beta_param for r in rows}
        return alphas, betas

    # ── Write ────────────────────────────────────────────────────────────────

    async def update_arm(
        self,
        user_id: str,
        song_id: str,
        new_alpha: float,
        new_beta: float,
        db: AsyncSession,
    ):
        """Update a single arm in both Redis and Postgres."""
        await self._write_to_redis(
            user_id,
            {song_id: new_alpha},
            {song_id: new_beta},
            extend_ttl=True,
        )
        await self._upsert_postgres(user_id, song_id, new_alpha, new_beta, db)

    async def _write_to_redis(
        self,
        user_id: str,
        alphas: dict[str, float],
        betas: dict[str, float],
        extend_ttl: bool = True,
    ):
        r = self._r()
        alpha_key = f"bandit:{user_id}:alphas"
        beta_key = f"bandit:{user_id}:betas"
        try:
            pipe = r.pipeline()
            pipe.hset(alpha_key, mapping={k: str(v) for k, v in alphas.items()})
            pipe.hset(beta_key, mapping={k: str(v) for k, v in betas.items()})
            if extend_ttl:
                pipe.expire(alpha_key, BANDIT_TTL)
                pipe.expire(beta_key, BANDIT_TTL)
            await pipe.execute()
        except Exception as e:
            logger.warning(f"Redis write failed for {user_id}: {e}")

    async def _upsert_postgres(
        self,
        user_id: str,
        song_id: str,
        alpha: float,
        beta: float,
        db: AsyncSession,
    ):
        stmt = pg_insert(BanditState).values(
            user_id=user_id,
            song_id=song_id,
            alpha=alpha,
            beta_param=beta,
            updated_at=datetime.utcnow(),
        ).on_conflict_do_update(
            index_elements=["user_id", "song_id"],
            set_={"alpha": alpha, "beta_param": beta, "updated_at": datetime.utcnow()},
        )
        await db.execute(stmt)
        await db.commit()

    async def get_single_arm(
        self,
        user_id: str,
        song_id: str,
        db: AsyncSession,
    ) -> tuple[float, float]:
        """Convenience: fetch (alpha, beta) for one arm."""
        alphas, betas = await self.load_user_state(user_id, [song_id], db)
        return alphas[song_id], betas[song_id]

    async def close(self):
        if self._redis:
            await self._redis.aclose()


# Singleton
bandit_state_manager = BanditStateManager()
