"""
Data Quality Validator
-----------------------
Validates every inbound feedback event and song feature record before
they touch the bandit state or training pipeline.

Two validation layers:
  1. FeedbackValidator  — validates action/reward/user/song constraints
  2. SongFeatureValidator — validates audio feature completeness and ranges

Each violation increments a Prometheus counter (for alerting) and
returns a structured ValidationResult so callers can decide whether to
reject, repair, or log.

Validation rules:
  reward            in [-1.0, 1.0]          (hard reject outside this)
  danceability      in [0.0, 1.0]
  energy            in [0.0, 1.0]
  valence           in [0.0, 1.0]
  tempo             in [40.0, 250.0]        (BPM, physiologically plausible)
  acousticness      in [0.0, 1.0]
  instrumentalness  in [0.0, 1.0]
  liveness          in [0.0, 1.0]
  speechiness       in [0.0, 1.0]
  loudness          in [-80.0, 5.0]         (dB)
  duration_ms       > 0

Missing / null features are flagged but don't hard-reject songs
(we impute with medians at scoring time).
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

from app.services.monitoring.metrics import (
    DATA_QUALITY_NULL_FEATURES,
    DATA_QUALITY_INVALID_REWARDS,
    DATA_QUALITY_VALIDATION_FAILURES,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    repaired: dict = field(default_factory=dict)   # field -> repaired value


# ── Feature bounds ────────────────────────────────────────────────────────────

FEATURE_BOUNDS = {
    "danceability":     (0.0, 1.0),
    "energy":           (0.0, 1.0),
    "valence":          (0.0, 1.0),
    "acousticness":     (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness":         (0.0, 1.0),
    "speechiness":      (0.0, 1.0),
    "tempo":            (40.0, 250.0),
    "loudness":         (-80.0, 5.0),
}

# Median fallbacks for imputation when feature is None
FEATURE_MEDIANS = {
    "danceability":     0.58,
    "energy":           0.62,
    "valence":          0.48,
    "acousticness":     0.25,
    "instrumentalness": 0.04,
    "liveness":         0.17,
    "speechiness":      0.08,
    "tempo":            120.0,
    "loudness":         -8.5,
}

VALID_ACTIONS = {"add", "skip", "play", "complete"}
REWARD_BOUNDS = (-1.0, 1.0)


class FeedbackValidator:
    """Validates a feedback event before it enters the bandit pipeline."""

    def validate(
        self,
        user_id: str,
        song_id: str,
        action: str,
        reward: float,
    ) -> ValidationResult:
        result = ValidationResult(valid=True)

        # Action check
        if action not in VALID_ACTIONS:
            result.errors.append(f"Invalid action '{action}', must be one of {VALID_ACTIONS}")
            result.valid = False
            DATA_QUALITY_VALIDATION_FAILURES.labels(check="invalid_action").inc()

        # Reward range
        lo, hi = REWARD_BOUNDS
        if not (lo <= reward <= hi):
            result.errors.append(f"Reward {reward} out of bounds [{lo}, {hi}]")
            result.valid = False
            DATA_QUALITY_INVALID_REWARDS.inc()
            DATA_QUALITY_VALIDATION_FAILURES.labels(check="reward_range").inc()

        # ID presence
        if not user_id or not user_id.strip():
            result.errors.append("user_id is empty")
            result.valid = False

        if not song_id or not song_id.strip():
            result.errors.append("song_id is empty")
            result.valid = False

        return result


class SongFeatureValidator:
    """
    Validates audio features for a song.
    Returns a ValidationResult with any repaired values (median imputation).
    Soft failures (nulls, out-of-range) are repaired and warned;
    hard failures (impossible values) mark the song as invalid.
    """

    def validate_and_repair(self, features: dict) -> ValidationResult:
        result = ValidationResult(valid=True)
        repaired = {}

        for feat, (lo, hi) in FEATURE_BOUNDS.items():
            value = features.get(feat)

            if value is None:
                # Impute with median
                imputed = FEATURE_MEDIANS[feat]
                repaired[feat] = imputed
                result.warnings.append(f"Null {feat} — imputed with median {imputed}")
                DATA_QUALITY_NULL_FEATURES.labels(feature=feat).inc()
                continue

            if not isinstance(value, (int, float)):
                result.errors.append(f"{feat}={value!r} is not numeric — imputing")
                repaired[feat] = FEATURE_MEDIANS[feat]
                DATA_QUALITY_VALIDATION_FAILURES.labels(check="feature_type").inc()
                continue

            if not (lo <= value <= hi):
                # Clamp to bounds with a warning
                clamped = max(lo, min(hi, value))
                repaired[feat] = clamped
                result.warnings.append(
                    f"{feat}={value} out of [{lo}, {hi}] — clamped to {clamped}"
                )
                DATA_QUALITY_VALIDATION_FAILURES.labels(check="feature_range").inc()

        result.repaired = repaired

        # duration check
        duration = features.get("duration_ms")
        if duration is not None and duration <= 0:
            result.warnings.append(f"duration_ms={duration} is non-positive")

        if result.errors:
            result.valid = False

        return result

    def validate_batch(self, songs: list[dict]) -> tuple[list[dict], list[str]]:
        """
        Validate and repair a batch of songs.
        Returns (repaired_songs, list_of_warning_messages).
        """
        repaired_songs = []
        all_warnings = []

        for song in songs:
            vr = self.validate_and_repair(song)
            if vr.valid:
                merged = {**song, **vr.repaired}
                repaired_songs.append(merged)
                all_warnings.extend(vr.warnings)
            else:
                logger.warning(f"Song {song.get('id')} failed validation: {vr.errors}")

        return repaired_songs, all_warnings


# ── Catalog quality reporter ──────────────────────────────────────────────────

async def update_catalog_quality_metrics(db) -> dict:
    """
    Run catalog quality checks and update Prometheus gauges.
    Called on startup and periodically by the background task.
    """
    from sqlalchemy import select, func, and_
    from app.models import Song
    from app.services.monitoring.metrics import SONGS_IN_CATALOG, SONGS_WITH_FEATURES

    total_result = await db.execute(select(func.count(Song.id)))
    total = total_result.scalar() or 0

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

    SONGS_IN_CATALOG.set(total)
    SONGS_WITH_FEATURES.set(complete)

    completeness_pct = (complete / total * 100) if total > 0 else 0
    logger.info(f"Catalog quality: {total} songs, {complete} complete ({completeness_pct:.1f}%)")

    return {"total": total, "complete": complete, "completeness_pct": completeness_pct}


# Singletons
feedback_validator = FeedbackValidator()
song_feature_validator = SongFeatureValidator()
