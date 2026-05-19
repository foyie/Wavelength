"""
Kafka Producer
--------------
Publishes feedback events to the `feedback` topic.
The consumer (feedback_consumer.py) processes them to update bandit state.

Why Kafka instead of direct DB writes?
  - Decouples the API response path from bandit update latency
  - Enables replay for retraining, auditing, and drift detection (Phase 3)
  - Provides a durable event log for the full interaction history

Topic: music.feedback
Key:   user_id  (ensures all events for a user land on the same partition)
Value: JSON-serialized FeedbackEvent

Falls back gracefully if Kafka is unavailable — events are still written
directly to Postgres via the legacy feedback.py path.
"""
import json
import logging
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

FEEDBACK_TOPIC = "music.feedback"


@dataclass
class FeedbackEvent:
    user_id: str
    song_id: str
    action: str
    reward: float
    timestamp: str
    play_duration_ms: Optional[int] = None
    context: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "FeedbackEvent":
        return cls(**json.loads(data))


class FeedbackProducer:
    """
    Async Kafka producer with graceful degradation.
    If aiokafka is not installed or Kafka is unreachable,
    publish() returns False and the caller falls back to direct DB writes.
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self._producer = None
        self._available = False
        self._init_attempted = False

    async def start(self):
        if self._init_attempted:
            return
        self._init_attempted = True
        try:
            from aiokafka import AIOKafkaProducer
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
                acks="all",               # wait for all replicas
                compression_type="gzip",
                linger_ms=5,              # micro-batch for throughput
            )
            await self._producer.start()
            self._available = True
            logger.info(f"Kafka producer connected: {self.bootstrap_servers}")
        except ImportError:
            logger.warning("aiokafka not installed — Kafka publishing disabled")
        except Exception as e:
            logger.warning(f"Kafka unavailable ({e}) — falling back to direct writes")

    async def publish(self, event: FeedbackEvent) -> bool:
        """Returns True if published to Kafka, False if fallback needed."""
        if not self._available or self._producer is None:
            return False
        try:
            await self._producer.send(
                FEEDBACK_TOPIC,
                key=event.user_id,
                value=event.to_json(),
            )
            return True
        except Exception as e:
            logger.error(f"Kafka publish failed: {e}")
            return False

    async def stop(self):
        if self._producer and self._available:
            await self._producer.stop()


# Singleton — started in main.py lifespan
feedback_producer = FeedbackProducer()
