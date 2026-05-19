"""
Experiment Registry
--------------------
In-memory store of all SPRT experiments with Redis persistence.
Experiments survive restarts via Redis serialisation.

Design:
  - Experiments are stored as JSON in Redis (key: exp:{experiment_id})
  - User→arm assignment is deterministic (hash of user_id + experiment_id)
    so the same user always gets the same arm across requests
  - Registry is loaded from Redis on startup
  - Every mutation (new obs, status change) syncs back to Redis

Traffic allocation:
  Users are split into control/treatment by:
    arm = hash(user_id + experiment_id) % 100 < traffic_split_pct
  Default split: 50/50.

Experiment lifecycle:
  RUNNING → CONTROL_WINS | TREATMENT_WINS | INCONCLUSIVE

Concurrency: asyncio single-threaded, no locks needed.
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Optional
import redis.asyncio as redis

from app.services.experiments.sprt import (
    SPRTExperiment, ExperimentArm, ExperimentStatus
)
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

EXP_KEY_PREFIX = "experiment:"
EXP_INDEX_KEY  = "experiments:index"
EXP_TTL        = 90 * 24 * 3600   # 90 days


class ExperimentRegistry:
    """
    Manages creation, assignment, observation recording, and
    persistence of all A/B experiments.
    """

    def __init__(self):
        self._experiments: dict[str, SPRTExperiment] = {}
        self._redis: Optional[redis.Redis] = None

    def _r(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=5,
            )
        return self._redis

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def load_from_redis(self):
        """Call on startup to restore experiments that survived a restart."""
        r = self._r()
        try:
            exp_ids = await r.smembers(EXP_INDEX_KEY)
            for exp_id in exp_ids:
                data = await r.get(f"{EXP_KEY_PREFIX}{exp_id}")
                if data:
                    exp = self._deserialise(json.loads(data))
                    self._experiments[exp_id] = exp
            logger.info(f"Loaded {len(self._experiments)} experiments from Redis")
        except Exception as e:
            logger.warning(f"Could not load experiments from Redis: {e}")

    async def _persist(self, exp: SPRTExperiment):
        """Write experiment state to Redis."""
        r = self._r()
        try:
            payload = json.dumps(self._serialise(exp))
            await r.setex(f"{EXP_KEY_PREFIX}{exp.id}", EXP_TTL, payload)
            await r.sadd(EXP_INDEX_KEY, exp.id)
        except Exception as e:
            logger.warning(f"Could not persist experiment {exp.id}: {e}")

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def create(
        self,
        experiment_id: str,
        name: str,
        description: str,
        control_name: str,
        treatment_name: str,
        alpha: float = 0.05,
        beta: float = 0.20,
        delta: float = 0.02,
        max_samples_per_arm: int = 10_000,
        traffic_split_pct: int = 50,
    ) -> SPRTExperiment:
        if experiment_id in self._experiments:
            raise ValueError(f"Experiment '{experiment_id}' already exists")

        exp = SPRTExperiment(
            id=experiment_id,
            name=name,
            description=description,
            control_arm=ExperimentArm(name=control_name),
            treatment_arm=ExperimentArm(name=treatment_name),
            alpha=alpha,
            beta=beta,
            delta=delta,
            max_samples_per_arm=max_samples_per_arm,
        )
        # Store traffic split as an attribute (not in dataclass to keep it simple)
        exp._traffic_split_pct = traffic_split_pct  # type: ignore

        self._experiments[experiment_id] = exp
        await self._persist(exp)
        logger.info(f"Experiment created: {experiment_id} ({name})")
        return exp

    def get(self, experiment_id: str) -> Optional[SPRTExperiment]:
        return self._experiments.get(experiment_id)

    def list_all(self) -> list[SPRTExperiment]:
        return list(self._experiments.values())

    def list_running(self) -> list[SPRTExperiment]:
        return [e for e in self._experiments.values()
                if e.status == ExperimentStatus.RUNNING]

    async def delete(self, experiment_id: str):
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
        r = self._r()
        try:
            await r.delete(f"{EXP_KEY_PREFIX}{experiment_id}")
            await r.srem(EXP_INDEX_KEY, experiment_id)
        except Exception as e:
            logger.warning(f"Could not delete experiment {experiment_id} from Redis: {e}")

    # ── Assignment & observation ──────────────────────────────────────────────

    def assign_arm(self, user_id: str, experiment_id: str) -> Optional[str]:
        """
        Deterministically assign a user to an arm.
        Returns arm name or None if experiment not found / not running.
        """
        exp = self._experiments.get(experiment_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        split = getattr(exp, "_traffic_split_pct", 50)
        h = int(hashlib.md5(f"{user_id}:{experiment_id}".encode()).hexdigest(), 16)
        bucket = h % 100
        if bucket < split:
            return exp.treatment_arm.name
        else:
            return exp.control_arm.name

    async def record_observation(
        self,
        experiment_id: str,
        user_id: str,
        reward: float,
    ) -> Optional[ExperimentStatus]:
        """
        Record a reward observation for the correct arm of an experiment.
        Returns updated status, or None if experiment not found.
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        arm_name = self.assign_arm(user_id, experiment_id)
        if not arm_name:
            return None

        prev_status = exp.status
        new_status = exp.add_observation(arm_name, reward)

        # Persist on every N observations or on status change
        if exp.n_updates % 50 == 0 or new_status != prev_status:
            await self._persist(exp)

        return new_status

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _serialise(self, exp: SPRTExperiment) -> dict:
        return {
            "id": exp.id,
            "name": exp.name,
            "description": exp.description,
            "alpha": exp.alpha,
            "beta": exp.beta,
            "delta": exp.delta,
            "max_samples_per_arm": exp.max_samples_per_arm,
            "log_lambda": exp.log_lambda,
            "status": exp.status.value,
            "winner": exp.winner,
            "created_at": exp.created_at,
            "decided_at": exp.decided_at,
            "n_updates": exp.n_updates,
            "traffic_split_pct": getattr(exp, "_traffic_split_pct", 50),
            "control": {
                "name": exp.control_arm.name,
                "n_observations": exp.control_arm.n_observations,
                "total_reward": exp.control_arm.total_reward,
            },
            "treatment": {
                "name": exp.treatment_arm.name,
                "n_observations": exp.treatment_arm.n_observations,
                "total_reward": exp.treatment_arm.total_reward,
            },
        }

    def _deserialise(self, d: dict) -> SPRTExperiment:
        control = ExperimentArm(
            name=d["control"]["name"],
            n_observations=d["control"]["n_observations"],
            total_reward=d["control"]["total_reward"],
        )
        treatment = ExperimentArm(
            name=d["treatment"]["name"],
            n_observations=d["treatment"]["n_observations"],
            total_reward=d["treatment"]["total_reward"],
        )
        exp = SPRTExperiment(
            id=d["id"],
            name=d["name"],
            description=d["description"],
            control_arm=control,
            treatment_arm=treatment,
            alpha=d["alpha"],
            beta=d["beta"],
            delta=d["delta"],
            max_samples_per_arm=d["max_samples_per_arm"],
        )
        exp.log_lambda = d["log_lambda"]
        exp.status = ExperimentStatus(d["status"])
        exp.winner = d.get("winner")
        exp.created_at = d["created_at"]
        exp.decided_at = d.get("decided_at")
        exp.n_updates = d.get("n_updates", 0)
        exp._traffic_split_pct = d.get("traffic_split_pct", 50)  # type: ignore
        return exp

    async def close(self):
        if self._redis:
            await self._redis.aclose()


# Singleton
experiment_registry = ExperimentRegistry()
