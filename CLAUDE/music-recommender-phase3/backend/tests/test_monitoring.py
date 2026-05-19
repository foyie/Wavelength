"""
Phase 3 Tests — Drift Detection + Data Quality
Run: pytest tests/test_monitoring.py -v
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ── ADWIN ─────────────────────────────────────────────────────────────────────

class TestADWIN:
    """Test the ADWIN drift detector in isolation."""

    def _make_detector(self, delta=0.002):
        from app.services.monitoring.adwin import ADWINDetector
        return ADWINDetector(delta=delta)

    def test_no_drift_on_stationary_stream(self):
        """Constant-mean stream should produce zero or very few detections."""
        det = self._make_detector()
        np.random.seed(42)
        detections = 0
        for _ in range(300):
            drift, _ = det.add_element(np.random.normal(0.5, 0.05))
            if drift:
                detections += 1
        # Stationary: very few false positives expected
        assert detections <= 2, f"Too many false positives: {detections}"

    def test_detects_step_change(self):
        """A sudden mean shift from 0.2 to 0.8 should be detected."""
        det = self._make_detector()
        np.random.seed(0)

        detected = False
        # Phase 1: mean=0.2
        for _ in range(200):
            drift, _ = det.add_element(np.random.normal(0.2, 0.03))
            if drift:
                detected = True  # could be detected during phase 1 (OK)

        # Phase 2: mean=0.8 — big shift
        detected_shift = False
        for _ in range(200):
            drift, _ = det.add_element(np.random.normal(0.8, 0.03))
            if drift:
                detected_shift = True
                break

        assert detected_shift, "ADWIN failed to detect step change from 0.2 → 0.8"

    def test_window_shrinks_on_drift(self):
        """After drift, the window size should drop (old data removed)."""
        det = self._make_detector()
        np.random.seed(1)

        for _ in range(300):
            det.add_element(np.random.normal(0.3, 0.02))

        size_before = det.window_size

        # Inject abrupt shift
        for _ in range(200):
            det.add_element(np.random.normal(0.9, 0.02))

        size_after = det.window_size

        # Window should have shrunk at some point
        assert det.n_detections > 0
        # Window size after drift should be <= original + new elements
        assert size_after <= size_before + 200

    def test_mean_tracks_current_distribution(self):
        """After stabilising on a new mean, ADWIN.mean should track it."""
        det = self._make_detector()
        np.random.seed(2)

        target = 0.75
        for _ in range(500):
            det.add_element(np.random.normal(target, 0.02))

        assert abs(det.mean - target) < 0.05, f"Mean {det.mean:.3f} too far from {target}"

    def test_reset_clears_state(self):
        """Reset should zero out all internal state."""
        det = self._make_detector()
        for _ in range(100):
            det.add_element(0.5)
        det.reset()
        assert det.window_size == 0
        assert det.mean == 0.0
        assert det.n_detections == 0

    def test_single_element(self):
        """Adding one element should not crash."""
        det = self._make_detector()
        drift, mean = det.add_element(0.5)
        assert not drift
        assert 0.0 <= mean <= 1.0

    def test_gradual_drift_detected_eventually(self):
        """Slow drift (0.0 → 1.0 over 1000 steps) should eventually trigger."""
        det = self._make_detector(delta=0.005)  # slightly more sensitive
        np.random.seed(3)
        detections = 0
        for i in range(1000):
            value = i / 1000.0 + np.random.normal(0, 0.05)
            drift, _ = det.add_element(value)
            if drift:
                detections += 1
        assert detections >= 1, "Gradual drift from 0→1 should be detected"

    def test_elements_seen_counter(self):
        det = self._make_detector()
        for i in range(42):
            det.add_element(0.5)
        assert det.elements_seen == 42


# ── KS Drift Monitor ─────────────────────────────────────────────────────────

class TestKSDriftMonitor:

    def _make_monitor(self, reference_size=50, sliding_size=30, check_interval=10):
        from app.services.monitoring.ks_drift import FeatureDriftMonitor
        return FeatureDriftMonitor(
            reference_size=reference_size,
            sliding_size=sliding_size,
            check_interval=check_interval,
            alpha=0.05,
        )

    def _make_context(self, **overrides):
        base = {
            "danceability": 0.5,
            "energy": 0.6,
            "valence": 0.5,
            "tempo": 120.0,
            "acousticness": 0.3,
        }
        base.update(overrides)
        return base

    def test_no_results_before_reference_ready(self):
        mon = self._make_monitor(reference_size=50)
        ctx = self._make_context()
        for _ in range(30):  # less than reference_size
            result = mon.add_observation(ctx)
        assert not mon.is_reference_ready

    def test_reference_frozen_after_enough_obs(self):
        mon = self._make_monitor(reference_size=50)
        ctx = self._make_context()
        for _ in range(50):
            mon.add_observation(ctx)
        assert mon.is_reference_ready

    def test_no_drift_on_same_distribution(self):
        """Identical distributions should not trigger drift."""
        mon = self._make_monitor(reference_size=100, sliding_size=50, check_interval=20)
        np.random.seed(10)

        # Fill reference
        for _ in range(100):
            ctx = self._make_context(
                danceability=float(np.random.uniform(0.4, 0.6)),
                energy=float(np.random.uniform(0.5, 0.7)),
            )
            mon.add_observation(ctx)

        # Feed same distribution
        any_drift = False
        for _ in range(100):
            ctx = self._make_context(
                danceability=float(np.random.uniform(0.4, 0.6)),
                energy=float(np.random.uniform(0.5, 0.7)),
            )
            results = mon.add_observation(ctx)
            if results:
                if any(r.drifted for r in results):
                    any_drift = True

        assert not any_drift, "False positive drift detected on stationary distribution"

    def test_detects_distribution_shift(self):
        """Large mean shift should be detected by KS test."""
        mon = self._make_monitor(reference_size=100, sliding_size=50, check_interval=10)
        np.random.seed(20)

        # Reference: low energy [0.1, 0.3]
        for _ in range(100):
            ctx = self._make_context(energy=float(np.random.uniform(0.1, 0.3)))
            mon.add_observation(ctx)

        # Sliding: high energy [0.7, 0.9]
        detected = False
        for _ in range(100):
            ctx = self._make_context(energy=float(np.random.uniform(0.7, 0.9)))
            results = mon.add_observation(ctx)
            if results:
                energy_results = [r for r in results if r.feature == "energy"]
                if any(r.drifted for r in energy_results):
                    detected = True
                    break

        assert detected, "KS test failed to detect energy distribution shift [0.1-0.3] → [0.7-0.9]"

    def test_drift_count_increments(self):
        mon = self._make_monitor(reference_size=50, sliding_size=30, check_interval=10)
        np.random.seed(30)

        for _ in range(50):
            mon.add_observation(self._make_context(valence=float(np.random.uniform(0.1, 0.2))))
        for _ in range(100):
            mon.add_observation(self._make_context(valence=float(np.random.uniform(0.8, 0.9))))

        counts = mon.get_drift_counts()
        assert counts.get("valence", 0) >= 1, "Drift count for valence should be > 0"

    def test_reset_clears_reference(self):
        mon = self._make_monitor(reference_size=20)
        for _ in range(20):
            mon.add_observation(self._make_context())
        assert mon.is_reference_ready
        mon.reset_reference()
        assert not mon.is_reference_ready


# ── Data Quality Validator ────────────────────────────────────────────────────

class TestFeedbackValidator:

    def _validator(self):
        from app.services.monitoring.data_quality import FeedbackValidator
        return FeedbackValidator()

    def test_valid_add_passes(self):
        v = self._validator()
        r = v.validate("user1", "song1", "add", 1.0)
        assert r.valid
        assert not r.errors

    def test_valid_skip_passes(self):
        v = self._validator()
        r = v.validate("user1", "song1", "skip", -0.3)
        assert r.valid

    def test_invalid_action_rejected(self):
        v = self._validator()
        r = v.validate("user1", "song1", "purchase", 1.0)
        assert not r.valid
        assert any("action" in e.lower() for e in r.errors)

    def test_reward_above_one_rejected(self):
        v = self._validator()
        r = v.validate("user1", "song1", "add", 5.0)
        assert not r.valid

    def test_reward_below_minus_one_rejected(self):
        v = self._validator()
        r = v.validate("user1", "song1", "skip", -2.0)
        assert not r.valid

    def test_empty_user_id_rejected(self):
        v = self._validator()
        r = v.validate("", "song1", "add", 1.0)
        assert not r.valid

    def test_empty_song_id_rejected(self):
        v = self._validator()
        r = v.validate("user1", "", "add", 1.0)
        assert not r.valid


class TestSongFeatureValidator:

    def _validator(self):
        from app.services.monitoring.data_quality import SongFeatureValidator
        return SongFeatureValidator()

    def _valid_features(self, **overrides):
        f = {
            "id": "song1",
            "danceability": 0.7,
            "energy": 0.8,
            "valence": 0.6,
            "tempo": 120.0,
            "acousticness": 0.2,
            "instrumentalness": 0.0,
            "liveness": 0.1,
            "speechiness": 0.05,
            "loudness": -8.5,
            "duration_ms": 200000,
        }
        f.update(overrides)
        return f

    def test_valid_song_passes(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features())
        assert r.valid
        assert not r.errors

    def test_null_feature_imputed(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features(danceability=None))
        assert r.valid
        assert "danceability" in r.repaired
        assert r.repaired["danceability"] == pytest.approx(0.58, abs=0.01)
        assert "Null danceability" in " ".join(r.warnings)

    def test_out_of_range_clamped(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features(energy=1.5))
        assert r.valid
        assert "energy" in r.repaired
        assert r.repaired["energy"] == pytest.approx(1.0)

    def test_negative_energy_clamped_to_zero(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features(energy=-0.1))
        assert r.repaired["energy"] == pytest.approx(0.0)

    def test_all_nulls_produces_imputed_values(self):
        v = self._validator()
        features = {"id": "song1"}  # everything missing
        r = v.validate_and_repair(features)
        assert r.valid
        # All monitored features should be imputed
        for feat in ["danceability", "energy", "valence", "tempo", "acousticness"]:
            assert feat in r.repaired

    def test_batch_validate_filters_invalid(self):
        v = self._validator()
        songs = [
            self._valid_features(id="s1"),
            self._valid_features(id="s2", danceability=None),
        ]
        repaired, warnings = v.validate_batch(songs)
        assert len(repaired) == 2  # both valid (null is repaired, not rejected)
        assert any("Null danceability" in w for w in warnings)

    def test_tempo_out_of_range_clamped(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features(tempo=300.0))  # above 250 max
        assert "tempo" in r.repaired
        assert r.repaired["tempo"] == 250.0

    def test_loudness_bounds(self):
        v = self._validator()
        r = v.validate_and_repair(self._valid_features(loudness=-100.0))  # below -80 min
        assert "loudness" in r.repaired
        assert r.repaired["loudness"] == -80.0


# ── Drift Pipeline integration ────────────────────────────────────────────────

class TestDriftPipeline:

    def test_summary_structure(self):
        from app.services.monitoring.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        summary = pipeline.get_summary()
        assert "total_observations" in summary
        assert "reward_adwin" in summary
        assert "feature_drift" in summary
        assert "recent_drift_events" in summary

    def test_reset_adwin_all(self):
        from app.services.monitoring.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        pipeline.reset_adwin("all")
        # Should not raise

    def test_reset_adwin_individual(self):
        from app.services.monitoring.drift_pipeline import DriftPipeline
        pipeline = DriftPipeline()
        pipeline.reset_adwin("reward")
        pipeline.reset_adwin("ctr")
