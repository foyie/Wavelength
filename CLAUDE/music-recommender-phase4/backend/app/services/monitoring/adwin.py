"""
ADWIN (ADaptive WINdowing) Drift Detector
------------------------------------------
ADWIN is a change-detection algorithm that maintains a window of recent
observations and splits it to find the point of maximum mean difference.
When a statistically significant shift is found, it signals concept drift.

Why ADWIN over simple rolling averages?
  - Automatically adjusts window size — no fixed lookback parameter
  - Provides statistical guarantees on false-positive rate (controlled by delta)
  - O(log n) amortised time per observation
  - Detects both sudden shifts AND gradual drift

We track two signals:
  1. Reward drift   — is the mean reward changing? (user tastes evolving)
  2. Feature drift  — are audio-feature distributions shifting? (catalog changing)

References:
  Bifet & Gavalda, "Learning from Time-Changing Data with Adaptive Windowing"
  SIAM International Conference on Data Mining, 2007.

Usage:
    detector = ADWINDetector(delta=0.002)
    for reward in stream:
        drift_detected, new_mean = detector.add_element(reward)
        if drift_detected:
            trigger_model_reset()
"""
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ADWINBucket:
    """A compressed bucket storing sum and count for a sub-window."""
    total: float = 0.0
    variance: float = 0.0
    count: int = 0


class ADWINDetector:
    """
    Full ADWIN implementation.

    Parameters
    ----------
    delta : float
        Confidence parameter. Lower = fewer false positives, slower detection.
        Typical range: 0.0001 (conservative) to 0.05 (aggressive).
        Default 0.002 gives ~5% false-positive rate per 1000 observations.
    max_buckets : int
        Maximum number of buckets per bucket row. Controls memory usage.
        Higher = more precise but uses more memory. Default 5 is standard.
    """

    def __init__(self, delta: float = 0.002, max_buckets: int = 5):
        self.delta = delta
        self.max_buckets = max_buckets

        # Internal state
        self._buckets: list[deque[ADWINBucket]] = [deque() for _ in range(32)]
        self._total: float = 0.0
        self._variance: float = 0.0
        self._width: int = 0        # total number of elements in window
        self._n_detections: int = 0

        # Stats exposed for monitoring
        self.mean: float = 0.0
        self.last_drift_at: Optional[int] = None
        self._elements_seen: int = 0

    def add_element(self, value: float) -> tuple[bool, float]:
        """
        Add a new observation. Returns (drift_detected, current_window_mean).
        """
        self._elements_seen += 1
        self._width += 1

        # Insert into bucket at level 0
        bucket = ADWINBucket(total=value, variance=0.0, count=1)
        self._buckets[0].append(bucket)

        # Compress: merge adjacent buckets when a level overflows
        self._compress_buckets()

        # Recompute window totals
        total_sum = sum(b.total for row in self._buckets for b in row)
        total_count = sum(b.count for row in self._buckets for b in row)
        self._total = total_sum
        self._width = total_count
        self.mean = total_sum / total_count if total_count > 0 else 0.0

        # Test for drift
        drift = self._detect_change()
        if drift:
            self._n_detections += 1
            self.last_drift_at = self._elements_seen
            logger.info(
                f"ADWIN drift detected: mean={self.mean:.4f} "
                f"n_obs={self._elements_seen} n_detections={self._n_detections}"
            )

        return drift, self.mean

    def _compress_buckets(self):
        """Merge buckets within each level when capacity exceeded."""
        for level in range(len(self._buckets) - 1):
            row = self._buckets[level]
            if len(row) <= self.max_buckets:
                break
            # Merge oldest two buckets, push merged to next level
            b1 = row.popleft()
            b2 = row.popleft()
            merged_count = b1.count + b2.count
            merged_total = b1.total + b2.total
            # Welford's variance combination
            delta_mean = (b2.total / b2.count - b1.total / b1.count) if (b1.count > 0 and b2.count > 0) else 0.0
            merged_var = b1.variance + b2.variance + (delta_mean ** 2 * b1.count * b2.count / merged_count)
            merged = ADWINBucket(total=merged_total, variance=merged_var, count=merged_count)
            self._buckets[level + 1].append(merged)

    def _detect_change(self) -> bool:
        """
        Hoeffding-bound test: scan all sub-window splits.
        Returns True if any split shows statistically significant mean difference.
        """
        if self._width < 2:
            return False

        total_mean = self._total / self._width if self._width > 0 else 0.0
        epsilon_cut = self._compute_epsilon_cut(self._width, self._width)

        n0 = 0
        sum0 = 0.0

        # Walk through buckets from newest to oldest
        for level_idx, row in enumerate(self._buckets):
            for bucket in row:
                n0 += bucket.count
                sum0 += bucket.total
                n1 = self._width - n0

                if n1 == 0:
                    continue

                mean0 = sum0 / n0
                mean1 = (self._total - sum0) / n1

                # Hoeffding bound for difference of means
                epsilon = self._compute_epsilon_cut(n0, n1)

                if abs(mean0 - mean1) >= epsilon:
                    # Drift detected — shrink window by keeping the newer half
                    self._shrink_window(n0)
                    return True

        return False

    def _compute_epsilon_cut(self, n0: int, n1: int) -> float:
        """Hoeffding bound threshold for the given sub-window sizes."""
        if n0 == 0 or n1 == 0:
            return float("inf")
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 * self._width / self.delta))
        return epsilon

    def _shrink_window(self, elements_to_remove: int):
        """Remove oldest elements_to_remove elements from the window."""
        removed = 0
        # Walk from oldest buckets (highest level, leftmost)
        for level in range(len(self._buckets) - 1, -1, -1):
            row = self._buckets[level]
            while row and removed < elements_to_remove:
                oldest = row[0]
                if removed + oldest.count <= elements_to_remove:
                    self._total -= oldest.total
                    self._width -= oldest.count
                    removed += oldest.count
                    row.popleft()
                else:
                    # Partial removal
                    fraction = (elements_to_remove - removed) / oldest.count
                    self._total -= oldest.total * fraction
                    self._width -= int(oldest.count * fraction)
                    oldest.total -= oldest.total * fraction
                    oldest.count -= int(oldest.count * fraction)
                    removed = elements_to_remove

        self.mean = self._total / self._width if self._width > 0 else 0.0

    def reset(self):
        """Full reset — call after retraining model."""
        self._buckets = [deque() for _ in range(32)]
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self.mean = 0.0

    @property
    def window_size(self) -> int:
        return self._width

    @property
    def n_detections(self) -> int:
        return self._n_detections

    @property
    def elements_seen(self) -> int:
        return self._elements_seen
