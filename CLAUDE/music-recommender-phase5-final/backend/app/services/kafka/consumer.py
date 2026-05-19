"""
Kafka Consumer — Phase 3 update
---------------------------------
Adds to Phase 2:
  1. FeedbackValidator  — rejects malformed events before bandit update
  2. DriftPipeline      — runs ADWIN + KS detectors on every event
  3. Prometheus counters for Kafka processing metrics
"""
import asyncio
import logging
import time
from typing import Optional

from app.services.kafka.producer import FeedbackEvent, FEEDBACK_TOPIC
from app.services.bandit.thompson import bandit
from app.services.bandit.state_manager import bandit_state_manager
from app.services.feature_store import feature_store
from app.services.monitoring.data_quality import feedback_validator
from app.services.monitoring.drift_pipeline import drift_pipeline
from app.services.monitoring.metrics import (
    FEEDBACK_EVENTS,
    REWARD_DISTRIBUTION,
    FEEDBACK_LATENCY,
    KAFKA_PUBLISH_SUCCESS,
    BANDIT_ARM_UPDATES,
)
from app.db import AsyncSessionLocal
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FeedbackConsumer:

    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "bandit-updater"):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
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
            logger.warning(f"Kafka consumer failed to start ({e}) — using direct path")

    async def _consume_loop(self):
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
                            f"Error processing message partition={msg.partition} "
                            f"offset={msg.offset}: {e}",
                            exc_info=True,
                        )
            except Exception as e:
                if self._running:
                    logger.error(f"Consumer loop error: {e}", exc_info=True)
                    await asyncio.sleep(2)

    async def _process_event(self, event: FeedbackEvent):
        t0 = time.monotonic()

        # ── 1. Validate ───────────────────────────────────────────────────
        vr = feedback_validator.validate(
            event.user_id, event.song_id, event.action, event.reward
        )
        if not vr.valid:
            logger.warning(
                f"Feedback validation failed: user={event.user_id} "
                f"song={event.song_id} errors={vr.errors}"
            )
            return

        # ── 2. Bandit update ──────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            current_alpha, current_beta = await bandit_state_manager.get_single_arm(
                event.user_id, event.song_id, db
            )
            new_alpha, new_beta = bandit.update(current_alpha, current_beta, event.reward)
            await bandit_state_manager.update_arm(
                event.user_id, event.song_id, new_alpha, new_beta, db
            )

        # ── 3. Prometheus metrics ─────────────────────────────────────────
        FEEDBACK_EVENTS.labels(action=event.action).inc()
        REWARD_DISTRIBUTION.observe(event.reward)
        BANDIT_ARM_UPDATES.inc()

        # ── 4. Drift detection ────────────────────────────────────────────
        context = event.context or {}
        drift_events = await drift_pipeline.process_feedback_event(
            reward=event.reward,
            action=event.action,
            context=context,
        )

        # On drift: invalidate ALL users' rec caches (force fresh bandit samples)
        if drift_events:
            critical = [e for e in drift_events if e.severity == "critical"]
            if critical:
                logger.warning(
                    f"Critical drift detected ({len(critical)} signals) — "
                    f"invalidating recommendation caches"
                )
                # Targeted invalidation: just this user for now
                # Phase 4 adds broadcast invalidation via Redis pub/sub
                await feature_store.invalidate_recommendations(event.user_id)
            else:
                await feature_store.invalidate_recommendations(event.user_id)
        else:
            await feature_store.invalidate_recommendations(event.user_id)

        # ── 5. Latency tracking ───────────────────────────────────────────
        latency_ms = (time.monotonic() - t0) * 1000
        FEEDBACK_LATENCY.observe(latency_ms)

        logger.debug(
            f"Event processed: user={event.user_id} song={event.song_id} "
            f"action={event.action} reward={event.reward:.2f} "
            f"alpha={new_alpha:.2f} beta={new_beta:.2f} "
            f"latency={latency_ms:.1f}ms drift={len(drift_events)}"
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


feedback_consumer = FeedbackConsumer()
