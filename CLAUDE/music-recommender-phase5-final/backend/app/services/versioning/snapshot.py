"""
Snapshot Service
-----------------
Collects the live aggregate state of the bandit from Postgres and
passes it to the ModelVersionRegistry for persistence in MLflow.

Called from:
  1. The versioning router (manual snapshots)
  2. The drift pipeline (automatic snapshot on critical drift)
  3. A scheduled task (nightly snapshot)
"""
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BanditState, User, Interaction
from app.services.versioning.model_registry import model_registry, BanditSnapshot
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_snapshot_counter = 0


async def take_snapshot(
    db: AsyncSession,
    trigger: str = "manual",
    drift_signal: Optional[str] = None,
    notes: str = "",
) -> Optional[dict]:
    """
    Collect live bandit statistics and save a versioned snapshot.
    Returns a dict summary (even if MLflow is unavailable).
    """
    global _snapshot_counter
    _snapshot_counter += 1

    # ── Aggregate bandit state ─────────────────────────────────────────────
    arms_result = await db.execute(
        select(
            func.count(BanditState.id),
            func.avg(BanditState.alpha),
            func.avg(BanditState.beta_param),
            func.count(func.distinct(BanditState.user_id)),
        )
    )
    row = arms_result.fetchone()
    n_arms       = int(row[0] or 0)
    mean_alpha   = float(row[1] or 1.0)
    mean_beta    = float(row[2] or 1.0)
    n_users      = int(row[3] or 0)

    mean_reward = mean_alpha / (mean_alpha + mean_beta) if (mean_alpha + mean_beta) > 0 else 0.5

    # Exploration rate: fraction of arms with < 5 observations
    unexplored_result = await db.execute(
        select(func.count(BanditState.id)).where(
            BanditState.alpha + BanditState.beta_param < 7.0  # 5 obs + 2 prior
        )
    )
    n_unexplored = int(unexplored_result.scalar() or 0)
    exploration_rate = n_unexplored / n_arms if n_arms > 0 else 1.0

    # Total interactions
    interactions_result = await db.execute(select(func.count(Interaction.id)))
    total_interactions = int(interactions_result.scalar() or 0)

    # ── Build snapshot ─────────────────────────────────────────────────────
    version_name = f"bandit-v{_snapshot_counter}-{datetime.utcnow().strftime('%Y%m%d-%H%M')}"

    snapshot = BanditSnapshot(
        version_name=version_name,
        created_at=datetime.utcnow().isoformat(),
        trigger=trigger,
        drift_signal=drift_signal,
        n_users=n_users,
        n_arms_total=n_arms,
        mean_alpha=round(mean_alpha, 4),
        mean_beta=round(mean_beta, 4),
        mean_reward=round(mean_reward, 4),
        exploration_rate=round(exploration_rate, 4),
        total_interactions=total_interactions,
        context_weight=settings.bandit_context_weight,
        candidate_pool=settings.bandit_candidate_pool,
        notes=notes,
    )

    run_id = await model_registry.save_snapshot(snapshot)

    result = {
        "version_name": version_name,
        "run_id": run_id,
        "mlflow_available": model_registry.is_available,
        "trigger": trigger,
        "drift_signal": drift_signal,
        "stats": {
            "n_users": n_users,
            "n_arms_total": n_arms,
            "mean_reward": round(mean_reward, 4),
            "exploration_rate": round(exploration_rate, 4),
            "total_interactions": total_interactions,
        },
        "created_at": snapshot.created_at,
    }

    logger.info(
        f"Snapshot taken: {version_name} "
        f"n_arms={n_arms} mean_reward={mean_reward:.4f} "
        f"exploration={exploration_rate:.3f} mlflow={'ok' if run_id else 'skipped'}"
    )
    return result
