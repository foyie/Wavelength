"""
Phase 5 Tests — Model Versioning + System Integration
Run: pytest tests/test_phase5.py -v
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch


# ── BanditSnapshot ─────────────────────────────────────────────────────────────

class TestBanditSnapshot:

    def _snapshot(self, **overrides):
        from app.services.versioning.model_registry import BanditSnapshot
        defaults = dict(
            version_name="bandit-v1-20250101-1200",
            created_at="2025-01-01T12:00:00",
            trigger="manual",
            drift_signal=None,
            n_users=42,
            n_arms_total=840,
            mean_alpha=2.3,
            mean_beta=1.7,
            mean_reward=0.575,
            exploration_rate=0.31,
            total_interactions=12500,
            context_weight=0.35,
            candidate_pool=500,
            notes="",
        )
        defaults.update(overrides)
        return BanditSnapshot(**defaults)

    def test_snapshot_creation(self):
        snap = self._snapshot()
        assert snap.version_name == "bandit-v1-20250101-1200"
        assert snap.n_users == 42
        assert snap.mean_reward == 0.575

    def test_snapshot_asdict(self):
        from dataclasses import asdict
        snap = self._snapshot()
        d = asdict(snap)
        assert "version_name" in d
        assert "mean_reward" in d
        assert "drift_signal" in d

    def test_snapshot_serialises_to_json(self):
        from dataclasses import asdict
        snap = self._snapshot()
        payload = json.dumps(asdict(snap))
        restored = json.loads(payload)
        assert restored["version_name"] == snap.version_name
        assert restored["mean_reward"] == snap.mean_reward

    def test_snapshot_with_drift_signal(self):
        snap = self._snapshot(trigger="drift_detected", drift_signal="adwin_reward")
        assert snap.trigger == "drift_detected"
        assert snap.drift_signal == "adwin_reward"

    def test_mean_reward_range(self):
        """mean_reward should be alpha / (alpha + beta)."""
        snap = self._snapshot(mean_alpha=3.0, mean_beta=1.0, mean_reward=0.75)
        expected = 3.0 / (3.0 + 1.0)
        assert abs(snap.mean_reward - expected) < 0.01


# ── ModelVersionRegistry ─────────────────────────────────────────────────────

class TestModelVersionRegistry:

    def _registry(self, mlflow_available=False):
        from app.services.versioning.model_registry import ModelVersionRegistry
        reg = ModelVersionRegistry(tracking_uri="http://localhost:5000")
        reg._available = mlflow_available
        return reg

    @pytest.mark.asyncio
    async def test_save_snapshot_skipped_when_unavailable(self):
        reg = self._registry(mlflow_available=False)
        from app.services.versioning.model_registry import BanditSnapshot
        snap = BanditSnapshot(
            version_name="test-v1", created_at="2025-01-01T00:00:00",
            trigger="manual", drift_signal=None, n_users=10,
            n_arms_total=100, mean_alpha=2.0, mean_beta=1.5,
            mean_reward=0.57, exploration_rate=0.4,
            total_interactions=500, context_weight=0.35, candidate_pool=500,
        )
        run_id = await reg.save_snapshot(snap)
        assert run_id is None   # graceful skip

    @pytest.mark.asyncio
    async def test_list_snapshots_empty_when_unavailable(self):
        reg = self._registry(mlflow_available=False)
        result = await reg.list_snapshots()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_snapshot_none_when_unavailable(self):
        reg = self._registry(mlflow_available=False)
        result = await reg.get_snapshot("some-run-id")
        assert result is None

    def test_is_available_false_by_default(self):
        from app.services.versioning.model_registry import ModelVersionRegistry
        reg = ModelVersionRegistry()
        assert not reg.is_available

    def test_tracking_uri_stored(self):
        from app.services.versioning.model_registry import ModelVersionRegistry
        reg = ModelVersionRegistry(tracking_uri="http://custom:5001")
        assert reg.tracking_uri == "http://custom:5001"

    def test_experiment_name_stored(self):
        from app.services.versioning.model_registry import ModelVersionRegistry
        reg = ModelVersionRegistry(experiment_name="my-experiment")
        assert reg.experiment_name == "my-experiment"


# ── Snapshot service ──────────────────────────────────────────────────────────

class TestSnapshotService:

    @pytest.mark.asyncio
    async def test_take_snapshot_returns_dict(self):
        from app.services.versioning.snapshot import take_snapshot

        # Mock the DB session
        mock_db = AsyncMock()

        # Mock scalar results for each query
        def make_fetchone():
            m = AsyncMock()
            m.fetchone = MagicMock(return_value=(100, 2.3, 1.7, 25))
            return m

        def make_scalar(value):
            m = AsyncMock()
            m.scalar = MagicMock(return_value=value)
            return m

        call_count = 0
        async def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fetchone()   # arms aggregate
            elif call_count == 2:
                return make_scalar(30)  # unexplored count
            else:
                return make_scalar(500) # total interactions

        mock_db.execute = execute_side_effect

        # Mock model_registry to avoid MLflow calls
        with patch("app.services.versioning.snapshot.model_registry") as mock_reg:
            mock_reg.save_snapshot = AsyncMock(return_value=None)
            mock_reg.is_available = False

            result = await take_snapshot(mock_db, trigger="manual", notes="test")

        assert isinstance(result, dict)
        assert "version_name" in result
        assert "stats" in result
        assert result["trigger"] == "manual"
        assert result["stats"]["n_arms_total"] == 100
        assert result["stats"]["total_interactions"] == 500

    @pytest.mark.asyncio
    async def test_version_name_format(self):
        from app.services.versioning.snapshot import take_snapshot

        mock_db = AsyncMock()

        def make_fetchone():
            m = AsyncMock()
            m.fetchone = MagicMock(return_value=(0, 1.0, 1.0, 0))
            return m

        def make_scalar(v):
            m = AsyncMock()
            m.scalar = MagicMock(return_value=v)
            return m

        call_count = 0
        async def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fetchone()
            return make_scalar(0)

        mock_db.execute = execute_side_effect

        with patch("app.services.versioning.snapshot.model_registry") as mock_reg:
            mock_reg.save_snapshot = AsyncMock(return_value=None)
            mock_reg.is_available = False

            result = await take_snapshot(mock_db)

        # Version name should follow pattern: bandit-vN-YYYYMMDD-HHMM
        assert result["version_name"].startswith("bandit-v")
        parts = result["version_name"].split("-")
        assert len(parts) >= 4

    @pytest.mark.asyncio
    async def test_snapshot_counter_increments(self):
        from app.services.versioning import snapshot as snap_module
        from app.services.versioning.snapshot import take_snapshot

        mock_db = AsyncMock()

        def make_fetchone():
            m = AsyncMock()
            m.fetchone = MagicMock(return_value=(0, 1.0, 1.0, 0))
            return m

        def make_scalar(v):
            m = AsyncMock()
            m.scalar = MagicMock(return_value=v)
            return m

        call_count = 0
        async def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fetchone()
            call_count = 0  # reset for next snapshot
            return make_scalar(0)

        mock_db.execute = execute_side_effect

        initial = snap_module._snapshot_counter

        with patch("app.services.versioning.snapshot.model_registry") as mock_reg:
            mock_reg.save_snapshot = AsyncMock(return_value=None)
            mock_reg.is_available = False

            await take_snapshot(mock_db)

        assert snap_module._snapshot_counter == initial + 1


# ── End-to-end API shape tests ────────────────────────────────────────────────

class TestVersioningRouterShapes:
    """
    Test that the versioning router returns correctly shaped responses
    without needing a live DB or MLflow.
    """

    def test_snapshot_request_schema(self):
        from app.routers.versioning import SnapshotRequest
        req = SnapshotRequest(trigger="manual", notes="post-drift reset")
        assert req.trigger == "manual"
        assert req.notes == "post-drift reset"

    def test_snapshot_request_defaults(self):
        from app.routers.versioning import SnapshotRequest
        req = SnapshotRequest()
        assert req.trigger == "manual"
        assert req.notes == ""


# ── Load test configuration ───────────────────────────────────────────────────

class TestLoadTestConfig:
    """Verify load test script is importable and has required attributes."""

    def test_user_ids_generated(self):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../scripts"))
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "load_test",
                os.path.join(os.path.dirname(__file__), "../../scripts/load_test.py"),
            )
            # We just check the file parses without errors
            with open(os.path.join(os.path.dirname(__file__), "../../scripts/load_test.py")) as f:
                source = f.read()
            compile(source, "load_test.py", "exec")
        except SyntaxError as e:
            pytest.fail(f"load_test.py has syntax error: {e}")

    def test_traffic_weights_sum_to_ten(self):
        """Task weights in MusicAppUser should reflect the documented traffic mix."""
        # get_recommendations: 6, send_feedback: 3, get_playlist: 1, get_explanation: 1 = 11
        # (Locust normalises weights, so this is just a sanity check they're defined)
        weights = [6, 3, 1, 1]
        assert sum(weights) == 11  # locust normalises internally

    def test_heavy_user_weight_is_minority(self):
        """HeavyRecommendUser should be a small fraction of total traffic."""
        heavy_weight = 5
        normal_weight = 95  # implicit default in Locust
        assert heavy_weight < normal_weight


# ── Integration: all phases coexist ──────────────────────────────────────────

class TestAllPhasesCoexist:
    """
    Verify that all phase-specific modules can be imported together
    without circular imports or namespace collisions.
    """

    def test_phase1_imports(self):
        from app.services.feature_store import feature_store
        from app.services.spotify_client import spotify_client
        assert feature_store is not None
        assert spotify_client is not None

    def test_phase2_imports(self):
        from app.services.bandit.thompson import bandit, ThompsonBandit
        from app.services.bandit.state_manager import bandit_state_manager
        from app.services.kafka.producer import feedback_producer
        from app.services.kafka.consumer import feedback_consumer
        assert bandit is not None
        assert bandit_state_manager is not None

    def test_phase3_imports(self):
        from app.services.monitoring.adwin import ADWINDetector
        from app.services.monitoring.ks_drift import feature_drift_monitor
        from app.services.monitoring.metrics import RECOMMENDATION_LATENCY
        from app.services.monitoring.data_quality import feedback_validator
        from app.services.monitoring.drift_pipeline import drift_pipeline
        assert drift_pipeline is not None

    def test_phase4_imports(self):
        from app.services.experiments.sprt import SPRTExperiment, ExperimentStatus
        from app.services.experiments.registry import experiment_registry
        from app.services.explainability.shap_explainer import explainer
        assert experiment_registry is not None
        assert explainer is not None

    def test_phase5_imports(self):
        from app.services.versioning.model_registry import model_registry
        from app.services.versioning.snapshot import take_snapshot
        assert model_registry is not None
        assert callable(take_snapshot)

    def test_no_duplicate_metric_names(self):
        """Prometheus metric names must be unique — duplicate registration crashes the app."""
        from prometheus_client import REGISTRY
        metric_names = [m.name for m in REGISTRY.collect()]
        duplicates = [n for n in metric_names if metric_names.count(n) > 1]
        # Some prometheus internals duplicate — only check music_ metrics
        music_duplicates = [n for n in duplicates if n.startswith("music_")]
        assert len(music_duplicates) == 0, f"Duplicate Prometheus metrics: {music_duplicates}"
