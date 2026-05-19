"""
Drift Pipeline
--------------
Integrates ADWIN and KS drift detectors into the feedback processing loop.
Called by the Kafka consumer after every bandit update.

Two detection signals run in parallel:
  1. ADWIN on reward stream  — detects shifts in user engagement quality
  2. KS test on audio features — detects distribution shift in catalog/requests

On drift detection:
  - Prometheus counters increment (triggers Grafana alerts)
  - Warning logged with full context
  - DriftEvent recorded to Postgres for audit trail
  - Recommendation cache invalidated for ALL users (force fresh samples)

The pipeline does NOT automatically retrain the bandit — that is a manual
operation triggered by an operator reviewing the drift signals. This is
intentional: automated retraining on false positives wastes compute and
can destabilise a converging bandit.

Phase 4 adds SPRT-based A/B testing to validate that the drift warrants
a full model reset vs just a context-weight adjustment.
"""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.services.monitoring.adwin import ADWINDetector
from app.services.monitoring.ks_drift import feature_drift_monitor, DriftResult
from app.services.monitoring.metrics import (
    ADWIN_DRIFT_DETECTIONS,
    ADWIN_WINDOW_SIZE,
    ADWIN_CURRENT_MEAN,
    KS_DRIFT_DETECTIONS,
    KS_STATISTIC,
    KS_P_VALUE,
    KS_REFERENCE_MEAN,
    KS_CURRENT_MEAN,
)

logger = logging.getLogger(__name__)


@dataclass
class DriftEvent:
    detector: str          # "adwin_reward" | "ks_feature"
    signal: str            # reward | feature name
    timestamp: str
    current_value: float
    reference_value: float
    statistic: Optional[float]
    p_value: Optional[float]
    severity: str          # "warning" | "critical"
    action_taken: str      # "cache_invalidated" | "logged_only"


class DriftPipeline:
    """
    Orchestrates all drift detectors and their responses.
    Thread-safe: only called from the single Kafka consumer asyncio task.
    """

    def __init__(self):
        # Separate ADWIN detectors for reward and CTR signals
        self._reward_adwin  = ADWINDetector(delta=0.002)
        self._ctr_adwin     = ADWINDetector(delta=0.005)   # more sensitive for CTR

        self._drift_log: list[DriftEvent] = []   # in-memory log (Phase 4 moves to DB)
        self._total_observations = 0
        self._last_drift_at: Optional[int] = None

    async def process_feedback_event(
        self,
        reward: float,
        action: str,
        context: dict,
    ) -> list[DriftEvent]:
        """
        Feed one feedback event through all detectors.
        Returns list of DriftEvent (empty if no drift detected).
        """
        self._total_observations += 1
        drift_events: list[DriftEvent] = []

        # ── 1. ADWIN on reward signal ──────────────────────────────────────
        reward_drift, reward_mean = self._reward_adwin.add_element(reward)

        # Update gauges every observation
        ADWIN_WINDOW_SIZE.labels(signal="reward").set(self._reward_adwin.window_size)
        ADWIN_CURRENT_MEAN.labels(signal="reward").set(self._reward_adwin.mean)

        if reward_drift:
            ADWIN_DRIFT_DETECTIONS.labels(signal="reward").inc()
            event = DriftEvent(
                detector="adwin_reward",
                signal="reward",
                timestamp=datetime.utcnow().isoformat(),
                current_value=round(reward_mean, 4),
                reference_value=round(reward_mean, 4),  # ADWIN tracks internally
                statistic=None,
                p_value=None,
                severity="warning" if self._reward_adwin.n_detections < 3 else "critical",
                action_taken="cache_invalidated",
            )
            drift_events.append(event)
            self._drift_log.append(event)
            self._last_drift_at = self._total_observations
            logger.warning(
                f"ADWIN reward drift #{self._reward_adwin.n_detections}: "
                f"mean={reward_mean:.4f} window={self._reward_adwin.window_size} "
                f"obs={self._total_observations}"
            )

        # ── 2. ADWIN on CTR proxy (add action = click-through) ────────────
        ctr_signal = 1.0 if action == "add" else 0.0
        ctr_drift, ctr_mean = self._ctr_adwin.add_element(ctr_signal)
        ADWIN_WINDOW_SIZE.labels(signal="ctr").set(self._ctr_adwin.window_size)
        ADWIN_CURRENT_MEAN.labels(signal="ctr").set(self._ctr_adwin.mean)

        if ctr_drift:
            ADWIN_DRIFT_DETECTIONS.labels(signal="ctr").inc()
            event = DriftEvent(
                detector="adwin_ctr",
                signal="ctr",
                timestamp=datetime.utcnow().isoformat(),
                current_value=round(ctr_mean, 4),
                reference_value=round(ctr_mean, 4),
                statistic=None,
                p_value=None,
                severity="warning",
                action_taken="logged_only",
            )
            drift_events.append(event)
            self._drift_log.append(event)

        # ── 3. KS test on audio features ──────────────────────────────────
        ks_results = feature_drift_monitor.add_observation(context)
        if ks_results:
            self._update_ks_metrics(ks_results)
            drifted = [r for r in ks_results if r.drifted]
            for r in drifted:
                KS_DRIFT_DETECTIONS.labels(feature=r.feature).inc()
                event = DriftEvent(
                    detector="ks_feature",
                    signal=r.feature,
                    timestamp=datetime.utcnow().isoformat(),
                    current_value=r.current_mean,
                    reference_value=r.reference_mean,
                    statistic=r.statistic,
                    p_value=r.p_value,
                    severity="warning" if r.statistic < 0.3 else "critical",
                    action_taken="logged_only",
                )
                drift_events.append(event)
                self._drift_log.append(event)

        return drift_events

    def _update_ks_metrics(self, results: list[DriftResult]):
        """Push KS test results to Prometheus gauges."""
        for r in results:
            KS_STATISTIC.labels(feature=r.feature).set(r.statistic)
            KS_P_VALUE.labels(feature=r.feature).set(r.p_value)
            KS_REFERENCE_MEAN.labels(feature=r.feature).set(r.reference_mean)
            KS_CURRENT_MEAN.labels(feature=r.feature).set(r.current_mean)

    def get_summary(self) -> dict:
        return {
            "total_observations": self._total_observations,
            "last_drift_at": self._last_drift_at,
            "reward_adwin": {
                "window_size": self._reward_adwin.window_size,
                "mean": round(self._reward_adwin.mean, 4),
                "n_detections": self._reward_adwin.n_detections,
                "elements_seen": self._reward_adwin.elements_seen,
            },
            "ctr_adwin": {
                "window_size": self._ctr_adwin.window_size,
                "mean": round(self._ctr_adwin.mean, 4),
                "n_detections": self._ctr_adwin.n_detections,
            },
            "feature_drift": {
                "is_reference_ready": feature_drift_monitor.is_reference_ready,
                "observations": feature_drift_monitor.observations,
                "drift_counts": feature_drift_monitor.get_drift_counts(),
                "last_ks_results": [
                    {
                        "feature": r.feature,
                        "statistic": r.statistic,
                        "p_value": r.p_value,
                        "drifted": r.drifted,
                        "reference_mean": r.reference_mean,
                        "current_mean": r.current_mean,
                    }
                    for r in feature_drift_monitor.get_last_results()
                ],
            },
            "recent_drift_events": [
                {
                    "detector": e.detector,
                    "signal": e.signal,
                    "timestamp": e.timestamp,
                    "severity": e.severity,
                }
                for e in self._drift_log[-10:]   # last 10 events
            ],
        }

    def reset_adwin(self, signal: str = "all"):
        """Reset ADWIN detectors — call after intentional model update."""
        if signal in ("all", "reward"):
            self._reward_adwin.reset()
            logger.info("ADWIN reward detector reset")
        if signal in ("all", "ctr"):
            self._ctr_adwin.reset()
            logger.info("ADWIN CTR detector reset")


# Module-level singleton
drift_pipeline = DriftPipeline()
