"""
Experiments Router — Phase 4
-----------------------------
Full A/B testing lifecycle via SPRT.

POST /api/experiments                        — create experiment
GET  /api/experiments                        — list all experiments
GET  /api/experiments/{id}                   — get experiment detail + SPRT state
DELETE /api/experiments/{id}                 — delete experiment
GET  /api/experiments/{id}/arm/{user_id}     — get arm assignment for user
POST /api/experiments/{id}/observe          — record an observation manually
GET  /api/experiments/running                — list running experiments only
POST /api/experiments/{id}/conclude         — force-conclude an experiment
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field
from typing import Optional

from app.db import get_db
from app.models import ExperimentObservation, ExperimentAssignment, User
from app.services.experiments.registry import experiment_registry
from app.services.experiments.sprt import ExperimentStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreateExperimentRequest(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9_-]+$", description="Slug ID, e.g. 'bandit-v2-test'")
    name: str
    description: str = ""
    control_name: str = "control"
    treatment_name: str = "treatment"
    alpha: float = Field(default=0.05, ge=0.001, le=0.2, description="Type I error rate")
    beta: float = Field(default=0.20, ge=0.05, le=0.5, description="Type II error rate")
    delta: float = Field(default=0.02, ge=0.001, le=0.5, description="Minimum detectable effect")
    max_samples_per_arm: int = Field(default=10000, ge=100, le=1_000_000)
    traffic_split_pct: int = Field(default=50, ge=5, le=95, description="% of traffic to treatment")


class ObserveRequest(BaseModel):
    user_id: str
    reward: float = Field(..., ge=-1.0, le=1.0)
    action: str = Field(..., pattern="^(add|skip|play|complete)$")


class ConcludeRequest(BaseModel):
    reason: str
    winner: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_experiment(payload: CreateExperimentRequest):
    """
    Create a new SPRT A/B experiment.

    Example use cases:
      - Compare two context_weight values (0.35 vs 0.50)
      - Test a new candidate pool size (500 vs 1000)
      - Validate whether bandit beats popularity baseline
      - Test explanation display on CTR
    """
    try:
        exp = await experiment_registry.create(
            experiment_id=payload.id,
            name=payload.name,
            description=payload.description,
            control_name=payload.control_name,
            treatment_name=payload.treatment_name,
            alpha=payload.alpha,
            beta=payload.beta,
            delta=payload.delta,
            max_samples_per_arm=payload.max_samples_per_arm,
            traffic_split_pct=payload.traffic_split_pct,
        )
        return exp.summary()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("")
async def list_experiments(
    status: Optional[str] = Query(default=None, description="Filter by status")
):
    """List all experiments, optionally filtered by status."""
    experiments = experiment_registry.list_all()
    if status:
        experiments = [e for e in experiments if e.status.value == status]
    return {
        "total": len(experiments),
        "experiments": [e.summary() for e in experiments],
    }


@router.get("/running")
async def list_running():
    """Shortcut: list only running experiments."""
    running = experiment_registry.list_running()
    return {
        "total": len(running),
        "experiments": [e.summary() for e in running],
    }


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: str, db: AsyncSession = Depends(get_db)):
    """
    Full experiment state including SPRT log-lambda, boundaries,
    arm statistics, and DB observation counts.
    """
    exp = experiment_registry.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    # Enrich with DB counts for audit
    obs_result = await db.execute(
        select(
            ExperimentObservation.arm_name,
            func.count(ExperimentObservation.id),
            func.avg(ExperimentObservation.reward),
        )
        .where(ExperimentObservation.experiment_id == experiment_id)
        .group_by(ExperimentObservation.arm_name)
    )
    db_stats = {row[0]: {"count": row[1], "avg_reward": round(float(row[2] or 0), 4)}
                for row in obs_result.fetchall()}

    summary = exp.summary()
    summary["db_observations"] = db_stats
    return summary


@router.get("/{experiment_id}/arm/{user_id}")
async def get_user_arm(experiment_id: str, user_id: str):
    """
    Return the arm assignment for a specific user.
    Deterministic — same user always gets the same arm.
    """
    arm = experiment_registry.assign_arm(user_id, experiment_id)
    if arm is None:
        exp = experiment_registry.get(experiment_id)
        if not exp:
            raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
        return {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "arm": None,
            "reason": f"Experiment is {exp.status.value}, not running",
        }
    return {
        "user_id": user_id,
        "experiment_id": experiment_id,
        "arm": arm,
    }


@router.post("/{experiment_id}/observe")
async def record_observation(
    experiment_id: str,
    payload: ObserveRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record a reward observation for a user in this experiment.
    Also writes an audit row to experiment_observations.

    This is called automatically by the feedback pipeline when
    an experiment_id is active. Can also be called manually for testing.
    """
    exp = experiment_registry.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    arm = experiment_registry.assign_arm(payload.user_id, experiment_id)
    if not arm:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment '{experiment_id}' is not running"
        )

    new_status = await experiment_registry.record_observation(
        experiment_id, payload.user_id, payload.reward
    )

    # Write audit row to DB
    obs = ExperimentObservation(
        experiment_id=experiment_id,
        user_id=payload.user_id,
        arm_name=arm,
        reward=payload.reward,
        action=payload.action,
    )
    db.add(obs)
    await db.commit()

    return {
        "experiment_id": experiment_id,
        "user_id": payload.user_id,
        "arm": arm,
        "reward": payload.reward,
        "new_status": new_status.value if new_status else None,
        "decided": new_status != ExperimentStatus.RUNNING if new_status else False,
    }


@router.post("/{experiment_id}/conclude")
async def force_conclude(experiment_id: str, payload: ConcludeRequest):
    """
    Force-conclude a running experiment (e.g. due to external business decision).
    Sets status to INCONCLUSIVE with a reason note.
    """
    exp = experiment_registry.get(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")

    if exp.status != ExperimentStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Experiment is already {exp.status.value}"
        )

    exp.status = ExperimentStatus.INCONCLUSIVE
    exp.winner = payload.winner
    exp.decided_at = datetime.utcnow().isoformat()
    logger.info(f"Experiment '{experiment_id}' force-concluded: {payload.reason}")

    await experiment_registry._persist(exp)
    return exp.summary()


@router.delete("/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str):
    """Delete an experiment and remove from Redis."""
    if not experiment_registry.get(experiment_id):
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    await experiment_registry.delete(experiment_id)
