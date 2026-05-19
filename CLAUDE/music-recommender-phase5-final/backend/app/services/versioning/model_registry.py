"""
Model Version Registry
-----------------------
Tracks snapshots of the bandit's learned state with full lineage:
who trained it, when, what data it saw, what metrics it achieved.

We do not use MLflow's heavy model serialisation for the bandit itself
(the state lives in Postgres/Redis as (alpha, beta) pairs — not a file).
Instead we use MLflow as an experiment tracker and version registry to:
  1. Record a snapshot of aggregate bandit statistics at a point in time
  2. Tag the snapshot with the drift events that prompted it
  3. Track performance metrics (mean reward, CTR, exploration rate)
  4. Register named versions: "bandit-v1", "bandit-v2", etc.
  5. Allow rollback by restoring (alpha, beta) from a snapshot

Why version a bandit?
  - Audit trail: know exactly what state the model was in when a drift occurred
  - Rollback: if a bandit reset causes regressions, restore the prior version
  - Comparison: compare mean reward before/after a context_weight change
  - Compliance: document model lineage for regulated use cases

Architecture:
  - MLflow tracking server runs as a sidecar (see docker-compose)
  - Snapshots are stored as JSON artifacts in MLflow's artifact store
  - Redis caches the latest registered version name for fast lookup
  - The registry is async-safe (no MLflow sync SDK in hot path)
"""
import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow not installed — model versioning disabled. Run: pip install mlflow")


@dataclass
class BanditSnapshot:
    """Captures the aggregate state of the bandit at a point in time."""
    version_name: str
    created_at: str
    trigger: str                  # "manual" | "drift_detected" | "scheduled"
    drift_signal: Optional[str]   # e.g. "adwin_reward" | "ks_feature:energy"
    n_users: int
    n_arms_total: int
    mean_alpha: float
    mean_beta: float
    mean_reward: float            # mean(alpha / (alpha + beta)) across all arms
    exploration_rate: float       # fraction of arms with < 5 observations
    total_interactions: int
    context_weight: float         # bandit_context_weight at time of snapshot
    candidate_pool: int
    notes: str = ""


class ModelVersionRegistry:
    """
    Async wrapper around MLflow for bandit state versioning.
    All MLflow calls run in a thread pool to avoid blocking the event loop.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "music-bandit"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._experiment_id: Optional[str] = None
        self._available = False

    async def initialise(self):
        """Set up MLflow experiment. Called once on startup."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available — versioning disabled")
            return
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._setup_mlflow
            )
            self._available = True
            logger.info(f"MLflow versioning enabled: {self.tracking_uri}/{self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed ({e}) — versioning disabled")

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        client = mlflow.tracking.MlflowClient()
        exps = client.search_experiments(filter_string=f"name='{self.experiment_name}'")
        if exps:
            self._experiment_id = exps[0].experiment_id

    async def save_snapshot(self, snapshot: BanditSnapshot) -> Optional[str]:
        """
        Save a bandit snapshot as an MLflow run.
        Returns the run_id or None if MLflow is unavailable.
        """
        if not self._available:
            logger.info(f"Snapshot skipped (MLflow unavailable): {snapshot.version_name}")
            return None

        try:
            run_id = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._save_run(snapshot)
            )
            logger.info(f"Snapshot saved: {snapshot.version_name} run_id={run_id}")
            return run_id
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def _save_run(self, snapshot: BanditSnapshot) -> str:
        with mlflow.start_run(run_name=snapshot.version_name) as run:
            # Log metrics (queryable in MLflow UI)
            mlflow.log_metrics({
                "mean_reward":       snapshot.mean_reward,
                "exploration_rate":  snapshot.exploration_rate,
                "mean_alpha":        snapshot.mean_alpha,
                "mean_beta":         snapshot.mean_beta,
                "n_arms_total":      float(snapshot.n_arms_total),
                "n_users":           float(snapshot.n_users),
                "total_interactions": float(snapshot.total_interactions),
            })

            # Log parameters (indexed, filterable)
            mlflow.log_params({
                "trigger":        snapshot.trigger,
                "drift_signal":   snapshot.drift_signal or "none",
                "context_weight": snapshot.context_weight,
                "candidate_pool": snapshot.candidate_pool,
            })

            # Log tags
            mlflow.set_tags({
                "version_name": snapshot.version_name,
                "created_at":   snapshot.created_at,
                "notes":        snapshot.notes,
            })

            # Log full snapshot as JSON artifact
            snapshot_json = json.dumps(asdict(snapshot), indent=2)
            mlflow.log_text(snapshot_json, "snapshot.json")

            return run.info.run_id

    async def list_snapshots(self, max_results: int = 20) -> list[dict]:
        """List recent snapshots ordered by creation time."""
        if not self._available:
            return []
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._list_runs(max_results)
            )
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []

    def _list_runs(self, max_results: int) -> list[dict]:
        client = mlflow.tracking.MlflowClient()
        if not self._experiment_id:
            return []
        runs = client.search_runs(
            experiment_ids=[self._experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return [
            {
                "run_id": r.info.run_id,
                "version_name": r.data.tags.get("version_name", r.info.run_name),
                "created_at": r.data.tags.get("created_at", ""),
                "trigger": r.data.params.get("trigger", ""),
                "mean_reward": r.data.metrics.get("mean_reward", 0),
                "exploration_rate": r.data.metrics.get("exploration_rate", 0),
                "n_arms_total": int(r.data.metrics.get("n_arms_total", 0)),
                "total_interactions": int(r.data.metrics.get("total_interactions", 0)),
                "status": r.info.status,
            }
            for r in runs
        ]

    async def get_snapshot(self, run_id: str) -> Optional[dict]:
        """Fetch full snapshot artifact by run_id."""
        if not self._available:
            return None
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._get_run_artifact(run_id)
            )
        except Exception as e:
            logger.error(f"Failed to fetch snapshot {run_id}: {e}")
            return None

    def _get_run_artifact(self, run_id: str) -> Optional[dict]:
        client = mlflow.tracking.MlflowClient()
        try:
            artifact_path = client.download_artifacts(run_id, "snapshot.json")
            with open(artifact_path) as f:
                return json.load(f)
        except Exception:
            return None

    @property
    def is_available(self) -> bool:
        return self._available


# Singleton — initialised in main.py lifespan
model_registry = ModelVersionRegistry()
