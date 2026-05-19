"""
Kolmogorov-Smirnov Feature Drift Monitor
-----------------------------------------
Compares the distribution of audio features in a recent window against
a reference window (baseline computed from the first N interactions).

The KS test asks: "are these two samples drawn from the same distribution?"
A low p-value means the distributions have shifted — drift detected.

We monitor 5 audio features independently:
  danceability, energy, valence, tempo (normalised), acousticness

Why KS test?
  - Non-parametric: makes no assumptions about distribution shape
  - Sensitive to any kind of distributional shift (mean, variance, shape)
  - Low computational cost: O(n log n) for the two-sample variant
  - Well-understood false positive rate via p-value

Architecture:
  - Reference window: first REFERENCE_WINDOW_SIZE feedback events
  - Sliding window:   most recent SLIDING_WINDOW_SIZE feedback events
  - Run KS test every CHECK_INTERVAL events
  - Emit Prometheus counter on drift detection

DriftResult fields:
  - feature: name of the feature that drifted
  - statistic: KS test statistic (max absolute CDF difference, in [0,1])
  - p_value: probability of observing this difference by chance
  - reference_mean: mean of reference distribution
  - current_mean: mean of sliding window
  - drifted: True if p_value < alpha threshold
"""
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

REFERENCE_WINDOW_SIZE = 500
SLIDING_WINDOW_SIZE   = 200
CHECK_INTERVAL        = 50     # run KS test every N new observations
KS_ALPHA              = 0.05   # p-value threshold for drift detection
TEMPO_NORM            = 200.0

MONITORED_FEATURES = ["danceability", "energy", "valence", "tempo_norm", "acousticness"]


@dataclass
class DriftResult:
    feature: str
    statistic: float
    p_value: float
    reference_mean: float
    current_mean: float
    drifted: bool
    n_reference: int
    n_current: int


class FeatureDriftMonitor:
    """
    Tracks per-feature distribution drift using KS tests.
    Maintains a reference window and a sliding window for each feature.
    """

    def __init__(
        self,
        reference_size: int = REFERENCE_WINDOW_SIZE,
        sliding_size: int = SLIDING_WINDOW_SIZE,
        check_interval: int = CHECK_INTERVAL,
        alpha: float = KS_ALPHA,
    ):
        self.reference_size = reference_size
        self.sliding_size = sliding_size
        self.check_interval = check_interval
        self.alpha = alpha

        # Storage: {feature_name: deque of float values}
        self._reference: dict[str, list[float]] = {f: [] for f in MONITORED_FEATURES}
        self._sliding:   dict[str, deque[float]] = {
            f: deque(maxlen=sliding_size) for f in MONITORED_FEATURES
        }

        self._reference_frozen = False  # True once reference window is full
        self._observations = 0
        self._last_results: list[DriftResult] = []
        self._drift_counts: dict[str, int] = {f: 0 for f in MONITORED_FEATURES}

    def add_observation(self, context: dict) -> Optional[list[DriftResult]]:
        """
        Feed one feedback event's audio-feature context.
        Returns list of DriftResult if a KS check was triggered, else None.

        context dict keys: danceability, energy, valence, tempo, acousticness
        """
        self._observations += 1

        # Extract normalised values
        values = {
            "danceability":  context.get("danceability") or 0.5,
            "energy":        context.get("energy") or 0.5,
            "valence":       context.get("valence") or 0.5,
            "tempo_norm":    (context.get("tempo") or 120.0) / TEMPO_NORM,
            "acousticness":  context.get("acousticness") or 0.5,
        }

        # Fill reference window first
        if not self._reference_frozen:
            for feat, val in values.items():
                self._reference[feat].append(val)
            if self._observations >= self.reference_size:
                self._reference_frozen = True
                logger.info(
                    f"FeatureDriftMonitor: reference window frozen "
                    f"({self.reference_size} observations)"
                )
            return None

        # Add to sliding window
        for feat, val in values.items():
            self._sliding[feat].append(val)

        # Run KS check on interval
        if self._observations % self.check_interval == 0:
            results = self._run_ks_tests()
            self._last_results = results
            return results

        return None

    def _run_ks_tests(self) -> list[DriftResult]:
        """Run KS test for each feature and return results."""
        results = []
        for feat in MONITORED_FEATURES:
            ref = np.array(self._reference[feat])
            cur = np.array(list(self._sliding[feat]))

            if len(ref) < 30 or len(cur) < 30:
                continue

            stat, p_value = stats.ks_2samp(ref, cur)
            drifted = p_value < self.alpha

            if drifted:
                self._drift_counts[feat] += 1
                logger.warning(
                    f"Feature drift detected: {feat} "
                    f"KS={stat:.4f} p={p_value:.4f} "
                    f"ref_mean={ref.mean():.3f} cur_mean={cur.mean():.3f}"
                )

            results.append(DriftResult(
                feature=feat,
                statistic=round(float(stat), 4),
                p_value=round(float(p_value), 4),
                reference_mean=round(float(ref.mean()), 4),
                current_mean=round(float(cur.mean()), 4),
                drifted=drifted,
                n_reference=len(ref),
                n_current=len(cur),
            ))

        return results

    def get_last_results(self) -> list[DriftResult]:
        return self._last_results

    def get_drift_counts(self) -> dict[str, int]:
        return dict(self._drift_counts)

    @property
    def is_reference_ready(self) -> bool:
        return self._reference_frozen

    @property
    def observations(self) -> int:
        return self._observations

    def reset_reference(self, new_reference_data: Optional[dict[str, list[float]]] = None):
        """
        Reset reference window — call when intentional distribution shift occurs
        (e.g. after catalog refresh or seasonal change).
        """
        if new_reference_data:
            self._reference = new_reference_data
            self._reference_frozen = True
        else:
            self._reference = {f: [] for f in MONITORED_FEATURES}
            self._reference_frozen = False
        for f in MONITORED_FEATURES:
            self._sliding[f].clear()
        logger.info("FeatureDriftMonitor: reference window reset")


# Module-level singleton
feature_drift_monitor = FeatureDriftMonitor()
