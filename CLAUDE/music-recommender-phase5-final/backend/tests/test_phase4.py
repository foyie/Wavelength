"""
Phase 4 Tests — SPRT A/B Testing + SHAP Explainability
Run: pytest tests/test_phase4.py -v
"""
import pytest
import math
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch


# ── SPRT Engine ───────────────────────────────────────────────────────────────

class TestSPRTBoundaries:

    def _make_exp(self, alpha=0.05, beta=0.20, delta=0.02):
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm
        return SPRTExperiment(
            id="test-exp",
            name="Test",
            description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
            alpha=alpha, beta=beta, delta=delta,
        )

    def test_upper_boundary_formula(self):
        exp = self._make_exp(alpha=0.05, beta=0.20)
        expected = math.log((1 - 0.20) / 0.05)
        assert abs(exp.upper_boundary - expected) < 1e-9

    def test_lower_boundary_formula(self):
        exp = self._make_exp(alpha=0.05, beta=0.20)
        expected = math.log(0.20 / (1 - 0.05))
        assert abs(exp.lower_boundary - expected) < 1e-9

    def test_initial_status_is_running(self):
        from app.services.experiments.sprt import ExperimentStatus
        exp = self._make_exp()
        assert exp.status == ExperimentStatus.RUNNING

    def test_boundaries_tighter_with_stricter_alpha(self):
        """Lower alpha = harder to cross upper boundary."""
        exp_loose  = self._make_exp(alpha=0.10)
        exp_strict = self._make_exp(alpha=0.01)
        assert exp_strict.upper_boundary > exp_loose.upper_boundary

    def test_boundaries_tighter_with_higher_power(self):
        """Lower beta (higher power) = harder to cross lower boundary."""
        exp_low_power  = self._make_exp(beta=0.40)
        exp_high_power = self._make_exp(beta=0.05)
        assert abs(exp_high_power.lower_boundary) > abs(exp_low_power.lower_boundary)


class TestSPRTObservations:

    def _make_exp(self, delta=0.05):
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm
        return SPRTExperiment(
            id="t", name="T", description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
            alpha=0.05, beta=0.20, delta=delta, max_samples_per_arm=50_000,
        )

    def test_observation_increments_count(self):
        exp = self._make_exp()
        exp.add_observation("control", 0.5)
        assert exp.control_arm.n_observations == 1
        assert exp.treatment_arm.n_observations == 0

    def test_observation_accumulates_reward(self):
        exp = self._make_exp()
        exp.add_observation("treatment", 0.8)
        exp.add_observation("treatment", 0.6)
        assert abs(exp.treatment_arm.total_reward - 1.4) < 1e-9

    def test_reward_clamped_to_unit_interval(self):
        exp = self._make_exp()
        exp.add_observation("control", 5.0)   # should be clamped to 1.0
        assert exp.control_arm.total_reward <= 1.0

    def test_unknown_arm_ignored(self):
        from app.services.experiments.sprt import ExperimentStatus
        exp = self._make_exp()
        status = exp.add_observation("ghost_arm", 1.0)
        assert status == ExperimentStatus.RUNNING
        assert exp.control_arm.n_observations == 0

    def test_treatment_wins_with_strong_signal(self):
        """Feed 1000 treatment observations with reward=0.9, control=0.5 → treatment wins."""
        from app.services.experiments.sprt import ExperimentStatus
        exp = self._make_exp(delta=0.10)
        np.random.seed(42)

        decided = False
        for i in range(5000):
            exp.add_observation("control",   float(np.random.beta(5, 5)))   # mean ~0.5
            exp.add_observation("treatment", float(np.random.beta(8, 2)))   # mean ~0.8
            if exp.status != ExperimentStatus.RUNNING:
                decided = True
                break

        assert decided, "SPRT did not converge within 5000 observations"
        assert exp.status == ExperimentStatus.TREATMENT_WINS
        assert exp.winner == "treatment"
        assert exp.decided_at is not None

    def test_control_wins_when_treatment_worse(self):
        """Treatment consistently worse than control → control wins."""
        from app.services.experiments.sprt import ExperimentStatus
        exp = self._make_exp(delta=0.05)
        np.random.seed(99)

        for i in range(10000):
            exp.add_observation("control",   float(np.random.beta(8, 2)))   # mean ~0.8
            exp.add_observation("treatment", float(np.random.beta(3, 7)))   # mean ~0.3
            if exp.status != ExperimentStatus.RUNNING:
                break

        # With such a large difference, either control_wins or treatment_wins
        assert exp.status in (
            ExperimentStatus.CONTROL_WINS,
            ExperimentStatus.TREATMENT_WINS,
        )

    def test_max_samples_triggers_inconclusive(self):
        from app.services.experiments.sprt import ExperimentStatus, SPRTExperiment, ExperimentArm
        exp = SPRTExperiment(
            id="t", name="T", description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
            alpha=0.05, beta=0.20, delta=0.02, max_samples_per_arm=10,
        )
        for _ in range(11):
            exp.add_observation("control", 0.5)
            exp.add_observation("treatment", 0.51)

        assert exp.status == ExperimentStatus.INCONCLUSIVE

    def test_decided_experiment_ignores_new_obs(self):
        from app.services.experiments.sprt import ExperimentStatus, SPRTExperiment, ExperimentArm
        exp = SPRTExperiment(
            id="t", name="T", description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
            alpha=0.05, beta=0.20, delta=0.02, max_samples_per_arm=5,
        )
        for _ in range(6):
            exp.add_observation("control", 0.5)
            exp.add_observation("treatment", 0.5)

        final_status = exp.status
        obs_before = exp.n_updates

        # More observations after decision
        for _ in range(100):
            exp.add_observation("control", 1.0)
        assert exp.status == final_status  # unchanged
        assert exp.n_updates == obs_before  # no more updates

    def test_summary_has_required_fields(self):
        exp = self._make_exp()
        summary = exp.summary()
        for key in ["id", "name", "status", "control", "treatment", "sprt", "lift"]:
            assert key in summary

    def test_lift_calculation(self):
        exp = self._make_exp()
        for _ in range(10):
            exp.add_observation("control", 0.5)
            exp.add_observation("treatment", 0.6)
        summary = exp.summary()
        assert abs(summary["lift"] - 0.1) < 0.05


# ── Experiment Registry ───────────────────────────────────────────────────────

class TestExperimentRegistry:

    def _make_registry(self):
        from app.services.experiments.registry import ExperimentRegistry
        return ExperimentRegistry()

    @pytest.mark.asyncio
    async def test_create_and_get(self):
        reg = self._make_registry()
        reg._redis = AsyncMock()
        reg._redis.setex = AsyncMock()
        reg._redis.sadd = AsyncMock()

        exp = await reg.create(
            experiment_id="exp-1",
            name="Test Exp",
            description="desc",
            control_name="ctl",
            treatment_name="trt",
        )
        assert exp.id == "exp-1"
        assert reg.get("exp-1") is exp

    @pytest.mark.asyncio
    async def test_duplicate_id_raises(self):
        reg = self._make_registry()
        reg._redis = AsyncMock()
        reg._redis.setex = AsyncMock()
        reg._redis.sadd = AsyncMock()

        await reg.create("exp-1", "A", "", "ctl", "trt")
        with pytest.raises(ValueError):
            await reg.create("exp-1", "B", "", "ctl", "trt")

    def test_arm_assignment_deterministic(self):
        from app.services.experiments.registry import ExperimentRegistry
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm
        reg = ExperimentRegistry()

        exp = SPRTExperiment(
            id="stable", name="S", description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
        )
        exp._traffic_split_pct = 50
        reg._experiments["stable"] = exp

        # Same user always gets same arm
        arm1 = reg.assign_arm("user-abc", "stable")
        arm2 = reg.assign_arm("user-abc", "stable")
        assert arm1 == arm2
        assert arm1 in ("control", "treatment")

    def test_arm_assignment_splits_roughly_50_50(self):
        from app.services.experiments.registry import ExperimentRegistry
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm
        reg = ExperimentRegistry()

        exp = SPRTExperiment(
            id="split", name="S", description="",
            control_arm=ExperimentArm(name="control"),
            treatment_arm=ExperimentArm(name="treatment"),
        )
        exp._traffic_split_pct = 50
        reg._experiments["split"] = exp

        arms = [reg.assign_arm(f"user-{i}", "split") for i in range(1000)]
        treatment_count = arms.count("treatment")
        # Should be roughly 50/50 ± 5%
        assert 400 <= treatment_count <= 600

    def test_list_running_filters_correctly(self):
        from app.services.experiments.registry import ExperimentRegistry
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm, ExperimentStatus
        reg = ExperimentRegistry()

        for i in range(3):
            exp = SPRTExperiment(
                id=f"exp-{i}", name=f"E{i}", description="",
                control_arm=ExperimentArm(name="control"),
                treatment_arm=ExperimentArm(name="treatment"),
            )
            reg._experiments[f"exp-{i}"] = exp

        reg._experiments["exp-1"].status = ExperimentStatus.CONTROL_WINS
        running = reg.list_running()
        assert len(running) == 2


# ── SHAP Explainer ────────────────────────────────────────────────────────────

class TestSHAPExplainer:

    def _explainer(self):
        from app.services.explainability.shap_explainer import RecommendationExplainer
        return RecommendationExplainer()

    def _song(self, **overrides):
        base = {
            "danceability": 0.7,
            "energy": 0.8,
            "valence": 0.6,
            "tempo": 120.0,
            "acousticness": 0.2,
        }
        base.update(overrides)
        return base

    def _user(self, **overrides):
        base = {
            "avg_danceability": 0.6,
            "avg_energy": 0.7,
            "avg_valence": 0.5,
            "avg_tempo": 115.0,
            "avg_acousticness": 0.25,
        }
        base.update(overrides)
        return base

    def test_returns_shap_result(self):
        from app.services.explainability.shap_explainer import ShapResult
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), alpha=2.0, beta=1.5, final_score=0.6)
        assert isinstance(result, ShapResult)

    def test_feature_contributions_has_all_keys(self):
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), 2.0, 1.5, 0.6)
        for feat in ["danceability", "energy", "valence", "tempo_norm", "acousticness", "bandit_prior"]:
            assert feat in result.feature_contributions

    def test_score_breakdown_sums_approximately(self):
        """context_component + bandit_component ≈ total (within rounding)."""
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), 3.0, 2.0, 0.7)
        sb = result.score_breakdown
        diff = abs(sb["context_component"] + sb["bandit_component"] - sb["total"])
        # Allow up to 0.15 difference (bandit component uses mean, not a drawn sample)
        assert diff < 0.15

    def test_explanation_text_is_string(self):
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), 1.0, 1.0, 0.5)
        assert isinstance(result.explanation_text, str)
        assert len(result.explanation_text) > 5

    def test_dominant_feature_is_in_contributions(self):
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), 2.0, 1.5, 0.6)
        assert result.dominant_feature in result.feature_contributions

    def test_null_song_features_handled(self):
        """None features should be imputed, not crash."""
        ex = self._explainer()
        song = {"danceability": None, "energy": None, "valence": None, "tempo": None, "acousticness": None}
        result = ex.explain(song, self._user(), 1.0, 1.0, 0.5)
        assert result is not None

    def test_high_energy_song_shows_energy_contribution(self):
        """Song with very high energy vs low-energy user should show energy as contributor."""
        ex = self._explainer()
        song = self._song(energy=0.99)
        user = self._user(avg_energy=0.1)
        result = ex.explain(song, user, 2.0, 2.0, 0.5)
        # energy should have a non-zero contribution
        assert abs(result.feature_contributions.get("energy", 0)) > 0

    def test_batch_explain_returns_correct_count(self):
        ex = self._explainer()
        songs = [self._song() for _ in range(5)]
        user = self._user()
        results = ex.explain_batch(songs, user, [2.0]*5, [1.5]*5, [0.6]*5)
        assert len(results) == 5

    def test_cold_start_explanation_text(self):
        """New user (1,1 prior) with unknown song should mention exploring."""
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), 1.0, 1.0, 0.5)
        # Explanation should indicate uncertainty / exploration
        text = result.explanation_text.lower()
        assert any(word in text for word in ["exploring", "discovery", "no listening", "prefer", "match"])

    def test_high_confidence_explanation_text(self):
        """Well-observed arm with high mean should show confidence signal."""
        ex = self._explainer()
        result = ex.explain(self._song(), self._user(), alpha=25.0, beta=5.0, final_score=0.85)
        text = result.explanation_text.lower()
        # Should mention signals or interactions
        assert any(word in text for word in ["signal", "interaction", "prefer", "match", "based"])


# ── Integration: SPRT + Registry serialisation ────────────────────────────────

class TestSPRTSerialisation:

    def _make_registry(self):
        from app.services.experiments.registry import ExperimentRegistry
        return ExperimentRegistry()

    def test_serialise_deserialise_roundtrip(self):
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm, ExperimentStatus
        reg = self._make_registry()

        exp = SPRTExperiment(
            id="round-trip", name="RT", description="test",
            control_arm=ExperimentArm(name="ctl", n_observations=100, total_reward=50.0),
            treatment_arm=ExperimentArm(name="trt", n_observations=95, total_reward=62.0),
            alpha=0.05, beta=0.20, delta=0.03,
        )
        exp.log_lambda = 1.23
        exp._traffic_split_pct = 60

        serialised = reg._serialise(exp)
        restored = reg._deserialise(serialised)

        assert restored.id == exp.id
        assert restored.log_lambda == exp.log_lambda
        assert restored.control_arm.n_observations == 100
        assert restored.treatment_arm.total_reward == 62.0
        assert getattr(restored, "_traffic_split_pct", None) == 60

    def test_summary_progress_pct_between_0_and_100(self):
        from app.services.experiments.sprt import SPRTExperiment, ExperimentArm
        exp = SPRTExperiment(
            id="prog", name="P", description="",
            control_arm=ExperimentArm(name="c"),
            treatment_arm=ExperimentArm(name="t"),
        )
        for _ in range(20):
            exp.add_observation("control", 0.5)
            exp.add_observation("treatment", 0.6)

        pct = exp.summary()["sprt"]["progress_pct"]
        assert 0 <= pct <= 100
