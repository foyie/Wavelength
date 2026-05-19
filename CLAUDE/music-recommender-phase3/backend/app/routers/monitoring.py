"""
Monitoring Router — Phase 3
----------------------------
Exposes drift detection state, data quality reports, and catalog metrics.

GET /api/monitoring/drift           — full drift detector summary
GET /api/monitoring/drift/adwin     — ADWIN-specific state
GET /api/monitoring/drift/features  — KS test results per audio feature
GET /api/monitoring/quality         — catalog quality metrics
GET /api/monitoring/quality/report  — full feature completeness breakdown
POST /api/monitoring/drift/reset    — reset ADWIN detectors (post-retrain)
POST /api/monitoring/quality/check  — trigger immediate catalog quality scan
"""
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from pydantic import BaseModel
from typing import Optional

from app.db import get_db
from app.models import Song, Interaction, User
from app.services.monitoring.drift_pipeline import drift_pipeline
from app.services.monitoring.ks_drift import feature_drift_monitor
from app.services.monitoring.data_quality import (
    update_catalog_quality_metrics,
    song_feature_validator,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


# ── Pydantic response models ─────────────────────────────────────────────────

class ADWINStatus(BaseModel):
    signal: str
    window_size: int
    current_mean: float
    n_detections: int
    elements_seen: int


class KSFeatureResult(BaseModel):
    feature: str
    statistic: float
    p_value: float
    reference_mean: float
    current_mean: float
    drifted: bool
    n_reference: int
    n_current: int


class DriftSummary(BaseModel):
    total_observations: int
    last_drift_at: Optional[int]
    reward_adwin: dict
    ctr_adwin: dict
    feature_drift: dict
    recent_drift_events: list[dict]


class QualityReport(BaseModel):
    total_songs: int
    songs_with_all_features: int
    completeness_pct: float
    feature_breakdown: dict[str, dict]   # feature -> {present, missing, pct}
    checked_at: str


# ── Drift endpoints ──────────────────────────────────────────────────────────

@router.get("/drift", response_model=DriftSummary)
async def get_drift_summary():
    """
    Full drift detection summary — good to poll from Grafana or ops dashboard.
    Includes ADWIN state, KS test results, and last 10 drift events.
    """
    summary = drift_pipeline.get_summary()
    return DriftSummary(**summary)


@router.get("/drift/adwin")
async def get_adwin_status():
    """
    ADWIN detector state for reward and CTR signals.
    Window size tells you how many recent observations are being tracked.
    A shrinking window means drift was detected and old data was dropped.
    """
    summary = drift_pipeline.get_summary()
    return {
        "reward": summary["reward_adwin"],
        "ctr":    summary["ctr_adwin"],
        "note": (
            "Window size shrinks on drift detection — smaller = more recent drift. "
            "n_detections counts how many times drift was declared since last reset."
        ),
    }


@router.get("/drift/features", response_model=list[KSFeatureResult])
async def get_feature_drift():
    """
    Latest KS test results per audio feature.
    p_value < 0.05 means drift detected for that feature.
    Returns empty list if reference window not yet full.
    """
    if not feature_drift_monitor.is_reference_ready:
        return []

    results = feature_drift_monitor.get_last_results()
    return [
        KSFeatureResult(
            feature=r.feature,
            statistic=r.statistic,
            p_value=r.p_value,
            reference_mean=r.reference_mean,
            current_mean=r.current_mean,
            drifted=r.drifted,
            n_reference=r.n_reference,
            n_current=r.n_current,
        )
        for r in results
    ]


@router.post("/drift/reset")
async def reset_drift_detectors(signal: str = Query(default="all", pattern="^(all|reward|ctr)$")):
    """
    Reset ADWIN detectors after an intentional model update or catalog refresh.
    Prevents false-positive drift alerts from the reset itself.
    signal: 'all' | 'reward' | 'ctr'
    """
    drift_pipeline.reset_adwin(signal=signal)
    return {
        "status": "reset",
        "signal": signal,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "KS reference window preserved — call /quality/reset to reset that separately",
    }


# ── Data quality endpoints ────────────────────────────────────────────────────

@router.get("/quality", response_model=QualityReport)
async def get_quality_summary(db: AsyncSession = Depends(get_db)):
    """
    Quick catalog quality snapshot.
    Returns total songs, complete feature coverage, and per-feature breakdown.
    """
    report = await _build_quality_report(db)
    return report


@router.post("/quality/check")
async def trigger_quality_check(db: AsyncSession = Depends(get_db)):
    """
    Trigger an immediate data quality scan and update Prometheus gauges.
    Useful after catalog updates or data migrations.
    """
    result = await update_catalog_quality_metrics(db)
    report = await _build_quality_report(db)
    return {
        "status": "complete",
        "timestamp": datetime.utcnow().isoformat(),
        "summary": result,
        "report": report,
    }


@router.get("/quality/interactions")
async def get_interaction_stats(
    hours: int = Query(default=24, ge=1, le=168),
    db: AsyncSession = Depends(get_db),
):
    """
    Interaction statistics over the last N hours.
    Useful for spotting anomalies in the feedback stream.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    # Total interactions and breakdown by action
    action_result = await db.execute(
        select(Interaction.action, func.count(Interaction.id))
        .where(Interaction.timestamp >= cutoff)
        .group_by(Interaction.action)
    )
    action_counts = {row[0]: row[1] for row in action_result.fetchall()}

    # Mean reward
    reward_result = await db.execute(
        select(func.avg(Interaction.reward), func.count(Interaction.id))
        .where(Interaction.timestamp >= cutoff)
    )
    avg_reward, total = reward_result.fetchone()

    # Unique users
    users_result = await db.execute(
        select(func.count(func.distinct(Interaction.user_id)))
        .where(Interaction.timestamp >= cutoff)
    )
    unique_users = users_result.scalar() or 0

    total_events = sum(action_counts.values())
    ctr = action_counts.get("add", 0) / total_events if total_events > 0 else 0.0

    return {
        "window_hours": hours,
        "total_interactions": total or 0,
        "unique_users": unique_users,
        "action_breakdown": action_counts,
        "mean_reward": round(float(avg_reward or 0), 4),
        "click_through_rate": round(ctr, 4),
        "add_count": action_counts.get("add", 0),
        "skip_count": action_counts.get("skip", 0),
        "play_count": action_counts.get("play", 0),
        "complete_count": action_counts.get("complete", 0),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

async def _build_quality_report(db: AsyncSession) -> QualityReport:
    """Build a full feature-by-feature quality breakdown."""
    FEATURES = [
        "danceability", "energy", "valence", "tempo",
        "acousticness", "instrumentalness", "liveness",
        "speechiness", "loudness",
    ]

    total_result = await db.execute(select(func.count(Song.id)))
    total = total_result.scalar() or 0

    breakdown = {}
    for feat in FEATURES:
        col = getattr(Song, feat)
        present_result = await db.execute(
            select(func.count(Song.id)).where(col.isnot(None))
        )
        present = present_result.scalar() or 0
        missing = total - present
        breakdown[feat] = {
            "present": present,
            "missing": missing,
            "completeness_pct": round(present / total * 100, 1) if total > 0 else 0.0,
        }

    complete_result = await db.execute(
        select(func.count(Song.id)).where(
            and_(
                Song.danceability.isnot(None),
                Song.energy.isnot(None),
                Song.valence.isnot(None),
                Song.tempo.isnot(None),
                Song.acousticness.isnot(None),
            )
        )
    )
    complete = complete_result.scalar() or 0

    return QualityReport(
        total_songs=total,
        songs_with_all_features=complete,
        completeness_pct=round(complete / total * 100, 1) if total > 0 else 0.0,
        feature_breakdown=breakdown,
        checked_at=datetime.utcnow().isoformat(),
    )
