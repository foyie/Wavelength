"""
Kafka Consumer — Feedback → Bandit Update Loop
-----------------------------------------------
Runs as a background asyncio task started in main.py lifespan.

For each FeedbackEvent consumed:
  1. Load current (alpha, beta) for the (user, song) arm
  2. Apply Thompson Sampling Bayesian update
  3. Persist updated (alpha, beta) to Redis + Postgres
  4. Invalidate the recommendation cache for this user

This is the heart of Phase 2 — every user action drives an immediate
update to the bandit posterior, so the next recommendation call already
reflects what the user just told us.

Error handling:
  - Exceptions per-message are caught and logged; consumer never crashes
  - Failed messages are logged with full context for debugging
  - Consumer resumes from committed offset on restart (at-least-once delivery)
"""
import asyncio
import logging
from typing import Optional

from app.services.kafka.producer import FeedbackEvent, FEEDBACK_TOPIC
from app.services.bandit.thompson import bandit
from app.services.bandit.state_manager import bandit_state_manager
from app.services.feature_store import feature_store
from app.db import AsyncSessionLocal
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeedbackConsumer:
    """
    Async Kafka consumer that runs in a background task.
    Gracefully skips if aiokafka is not installed or Kafka is unreachable.
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "bandit-updater"):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the consumer loop as a background task."""
        try:
            from aiokafka import AIOKafkaConsumer
            self._consumer = AIOKafkaConsumer(
                FEEDBACK_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
                value_deserializer=lambda v: v.decode("utf-8"),
            )
            await self._consumer.start()
            self._running = True
            self._task = asyncio.create_task(self._consume_loop())
            logger.info(f"Kafka consumer started: group={self.group_id} topic={FEEDBACK_TOPIC}")
        except ImportError:
            logger.warning("aiokafka not installed — Kafka consumer disabled")
        except Exception as e:
            logger.warning(f"Kafka consumer failed to start ({e}) — bandit updates via direct path")

    async def _consume_loop(self):
        """Main consume loop — processes one message at a time."""
        while self._running:
            try:
                async for msg in self._consumer:
                    if not self._running:
                        break
                    try:
                        event = FeedbackEvent.from_json(msg.value)
                        await self._process_event(event)
                    except Exception as e:
                        logger.error(
                            f"Error processing feedback message "
                            f"partition={msg.partition} offset={msg.offset}: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                if self._running:
                    logger.error(f"Consumer loop error: {e}", exc_info=True)
                    await asyncio.sleep(2)  # backoff before retry

    async def _process_event(self, event: FeedbackEvent):
        """
        Core update logic:
          1. Load current bandit state for this (user, song) arm
          2. Apply Thompson update: alpha += reward, beta += (1 - reward)
          3. Persist updated state
          4. Invalidate recommendation cache
        """
        async with AsyncSessionLocal() as db:
            current_alpha, current_beta = await bandit_state_manager.get_single_arm(
                event.user_id, event.song_id, db
            )

            new_alpha, new_beta = bandit.update(
                current_alpha, current_beta, event.reward
            )

            await bandit_state_manager.update_arm(
                event.user_id, event.song_id, new_alpha, new_beta, db
            )

        # Invalidate recommendation cache so next call is fresh
        await feature_store.invalidate_recommendations(event.user_id)

        logger.debug(
            f"Bandit updated: user={event.user_id} song={event.song_id} "
            f"action={event.action} reward={event.reward:.2f} "
            f"alpha={new_alpha:.2f} beta={new_beta:.2f}"
        )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception:
                pass


# Singleton — started/stopped in main.py lifespan
feedback_consumer = FeedbackConsumer()
