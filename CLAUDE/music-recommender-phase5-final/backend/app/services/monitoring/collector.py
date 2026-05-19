"""
Background Metrics Collector
------------------------------
An asyncio background task that periodically refreshes Prometheus gauges
that can't be updated in the hot path (e.g. DB aggregates, pool stats).

Runs every COLLECTION_INTERVAL_SECONDS.
Started and stopped in main.py lifespan alongside the Kafka consumer.

Metrics updated here:
  - Active users (24h window)        — DB aggregate, too expensive per-request
  - Catalog quality gauges           — songs total / with features
  - Bandit exploration rate          — fraction of arms with < 5 observations
  - Bandit mean reward               — rolling mean across sampled users
  - DB pool size                     — from SQLAlchemy engine stats
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, func, and_

from app.db import AsyncSessionLocal, engine
from app.models import User, Song, BanditState, Interaction
from app.services.monitoring.metrics import (
    ACTIVE_USERS_24H,
    SONGS_IN_CATALOG,
    SONGS_WITH_FEATURES,
    BANDIT_MEAN_REWARD,
    BANDIT_EXPLORATION_RATE,
    DB_POOL_SIZE,
)
from app.services.monitoring.data_quality import update_catalog_quality_metrics

logger = logging.getLogger(__name__)

COLLECTION_INTERVAL_SECONDS = 60   # update gauges every minute


class MetricsCollector:

    def __init__(self, interval: int = COLLECTION_INTERVAL_SECONDS):
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info(f"MetricsCollector started (interval={self.interval}s)")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MetricsCollector stopped")

    async def _collect_loop(self):
        # Run once immediately on startup, then on interval
        await self._collect_all()
        while self._running:
            await asyncio.sleep(self.interval)
            if self._running:
                try:
                    await self._collect_all()
                except Exception as e:
                    logger.error(f"MetricsCollector error: {e}", exc_info=True)

    async def _collect_all(self):
        async with AsyncSessionLocal() as db:
            await asyncio.gather(
                self._collect_active_users(db),
                self._collect_catalog_quality(db),
                self._collect_bandit_health(db),
                return_exceptions=True,
            )
        self._collect_db_pool()

    async def _collect_active_users(self, db):
        cutoff = datetime.utcnow() - timedelta(hours=24)
        result = await db.execute(
            select(func.count(User.id)).where(User.last_active >= cutoff)
        )
        count = result.scalar() or 0
        ACTIVE_USERS_24H.set(count)
        logger.debug(f"Active users (24h): {count}")

    async def _collect_catalog_quality(self, db):
        await update_catalog_quality_metrics(db)

    async def _collect_bandit_health(self, db):
        """
        Sample bandit arm statistics:
          - Exploration rate: fraction of arms with < 5 observations
          - Mean reward: mean(alpha / (alpha + beta)) across all arms
        """
        result = await db.execute(
            select(BanditState.alpha, BanditState.beta_param).limit(5000)
        )
        rows = result.fetchall()

        if not rows:
            return

        total = len(rows)
        unexplored = sum(1 for a, b in rows if (a + b - 2) < 5)
        exploration_rate = unexplored / total if total > 0 else 0.0
        mean_reward = sum(a / (a + b) for a, b in rows) / total

        BANDIT_EXPLORATION_RATE.set(exploration_rate)
        BANDIT_MEAN_REWARD.set(mean_reward)
        logger.debug(
            f"Bandit health: n_arms={total} exploration={exploration_rate:.3f} "
            f"mean_reward={mean_reward:.3f}"
        )

    def _collect_db_pool(self):
        try:
            pool = engine.pool
            DB_POOL_SIZE.set(pool.size())
        except Exception:
            pass   # pool stats not always available


# Singleton
metrics_collector = MetricsCollector()
