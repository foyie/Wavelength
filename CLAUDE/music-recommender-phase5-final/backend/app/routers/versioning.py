"""
Versioning Router — Phase 5
-----------------------------
Model snapshot management and MLflow integration.

POST /api/versioning/snapshot              — take a manual snapshot now
GET  /api/versioning/snapshots             — list all snapshots
GET  /api/versioning/snapshots/{run_id}    — get a specific snapshot
GET  /api/versioning/status                — MLflow connectivity status
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional

from app.db import get_db
from app.services.versioning.model_registry import model_registry
from app.services.versioning.snapshot import take_snapshot

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/versioning", tags=["versioning"])


class SnapshotRequest(BaseModel):
    trigger: str = "manual"
    notes: str = ""


@router.post("/snapshot")
async def create_snapshot(
    payload: SnapshotRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Take a snapshot of the current bandit state.
    Persists aggregate metrics to MLflow (if available).
    Always returns a summary dict even if MLflow is unavailable.
    """
    result = await take_snapshot(
        db=db,
        trigger=payload.trigger,
        notes=payload.notes,
    )
    return result


@router.get("/snapshots")
async def list_snapshots(
    max_results: int = Query(default=20, ge=1, le=100),
):
    """
    List recent model snapshots from MLflow.
    Returns empty list if MLflow is not configured.
    """
    snapshots = await model_registry.list_snapshots(max_results=max_results)
    return {
        "mlflow_available": model_registry.is_available,
        "total": len(snapshots),
        "snapshots": snapshots,
    }


@router.get("/snapshots/{run_id}")
async def get_snapshot(run_id: str):
    """Fetch the full snapshot artifact for a specific MLflow run."""
    snapshot = await model_registry.get_snapshot(run_id)
    if snapshot is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Snapshot '{run_id}' not found or MLflow unavailable",
        )
    return snapshot


@router.get("/status")
async def versioning_status():
    """Check MLflow connectivity and configuration."""
    return {
        "mlflow_available": model_registry.is_available,
        "tracking_uri": model_registry.tracking_uri,
        "experiment_name": model_registry.experiment_name,
        "note": (
            "MLflow is optional. Snapshots still record to logs when unavailable."
            if not model_registry.is_available
            else "MLflow connected and tracking."
        ),
    }
