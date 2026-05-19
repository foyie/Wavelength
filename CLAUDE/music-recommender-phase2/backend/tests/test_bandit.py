"""
Phase 2 Tests — Thompson Sampling engine + Kafka fallback
Run: pytest tests/ -v
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.bandit.thompson import ThompsonBandit, build_context_scores


# ── ThompsonBandit ────────────────────────────────────────────────────────────

class TestThompsonUpdate:

    def test_add_reward_increases_alpha(self):
        b = ThompsonBandit()
        a, bt = b.update(1.0, 1.0, 1.0)
        assert a == 2.0
        assert abs(bt - 1.0) < 1e-9

    def test_skip_increases_beta(self):
        b = ThompsonBandit()
        a, bt = b.update(1.0, 1.0, 0.0)
        assert abs(a - 1.0) < 1e-9
        assert bt == 2.0

    def test_partial_reward_splits_correctly(self):
        b = ThompsonBandit()
        a, bt = b.update(1.0, 1.0, 0.5)
        assert abs(a - 1.5) < 1e-9
        assert abs(bt - 1.5) < 1e-9

    def test_reward_clamped_above_one(self):
        b = ThompsonBandit()
        a, bt = b.update(1.0, 1.0, 5.0)  # reward > 1 should be clamped
        assert a == 2.0
        assert abs(bt - 1.0) < 1e-9

    def test_reward_clamped_below_zero(self):
        b = ThompsonBandit()
        a, bt = b.update(1.0, 1.0, -3.0)  # reward < 0 should be clamped
        assert abs(a - 1.0) < 1e-9
        assert bt == 2.0

    def test_many_updates_converge_toward_truth(self):
        """After 100 positive signals, mean should be >> 0.5."""
        b = ThompsonBandit()
        alpha, beta = 1.0, 1.0
        for _ in range(100):
            alpha, beta = b.update(alpha, beta, 1.0)
        mean = alpha / (alpha + beta)
        assert mean > 0.95

    def test_many_negative_converges_low(self):
        b = ThompsonBandit()
        alpha, beta = 1.0, 1.0
        for _ in range(100):
            alpha, beta = b.update(alpha, beta, 0.0)
        mean = alpha / (alpha + beta)
        assert mean < 0.05


class TestThompsonSampling:

    def test_sample_scores_shape(self):
        b = ThompsonBandit()
        n = 50
        alphas = np.ones(n)
        betas  = np.ones(n)
        context = np.random.rand(n)
        scores = b.sample_scores(alphas, betas, context)
        assert scores.shape == (n,)

    def test_scores_in_unit_interval(self):
        b = ThompsonBandit()
        n = 200
        alphas = np.random.uniform(1, 10, n)
        betas  = np.random.uniform(1, 10, n)
        context = np.random.rand(n)
        scores = b.sample_scores(alphas, betas, context)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_high_alpha_ranks_higher_on_average(self):
        """Arm with alpha=10, beta=1 should rank above alpha=1, beta=10 most of the time."""
        b = ThompsonBandit()
        wins = 0
        trials = 200
        context = np.array([0.5, 0.5])
        for _ in range(trials):
            scores = b.sample_scores(
                np.array([10.0, 1.0]),
                np.array([1.0, 10.0]),
                context,
            )
            if scores[0] > scores[1]:
                wins += 1
        assert wins > 160  # should win ~95%+ of trials

    def test_rank_returns_correct_count(self):
        b = ThompsonBandit()
        n = 100
        top_k = 20
        alphas = np.random.uniform(1, 5, n)
        betas  = np.random.uniform(1, 5, n)
        context = np.random.rand(n)
        indices = b.rank(alphas, betas, context, top_k=top_k)
        assert len(indices) == top_k
        assert len(set(indices)) == top_k  # no duplicates

    def test_rank_all_returns_all(self):
        b = ThompsonBandit()
        n = 10
        alphas = np.ones(n)
        betas  = np.ones(n)
        context = np.random.rand(n)
        indices = b.rank(alphas, betas, context, top_k=n)
        assert len(indices) == n


class TestContextScores:

    def test_returns_popularity_for_none_user(self):
        songs = [MagicMock(popularity=80, danceability=0.5, energy=0.5,
                           valence=0.5, tempo=120, acousticness=0.5)]
        scores = build_context_scores(songs, user=None)
        assert abs(scores[0] - 0.80) < 0.01

    def test_identical_user_song_gives_high_similarity(self):
        user = MagicMock(
            avg_danceability=0.8, avg_energy=0.9, avg_valence=0.7,
            avg_tempo=140, avg_acousticness=0.2,
        )
        song = MagicMock(
            danceability=0.8, energy=0.9, valence=0.7,
            tempo=140, acousticness=0.2, popularity=50,
        )
        scores = build_context_scores([song], user)
        assert scores[0] > 0.95

    def test_opposite_user_song_gives_low_similarity(self):
        user = MagicMock(
            avg_danceability=0.9, avg_energy=0.9, avg_valence=0.9,
            avg_tempo=180, avg_acousticness=0.1,
        )
        song = MagicMock(
            danceability=0.1, energy=0.1, valence=0.1,
            tempo=60, acousticness=0.9, popularity=50,
        )
        scores = build_context_scores([song], user)
        assert scores[0] < 0.55  # shifted from [-1,1] to [0,1], so 0.5 is neutral

    def test_output_shape(self):
        user = MagicMock(
            avg_danceability=0.5, avg_energy=0.5, avg_valence=0.5,
            avg_tempo=120, avg_acousticness=0.5,
        )
        songs = [
            MagicMock(danceability=0.5, energy=0.5, valence=0.5,
                      tempo=120, acousticness=0.5, popularity=50)
            for _ in range(15)
        ]
        scores = build_context_scores(songs, user)
        assert scores.shape == (15,)


# ── Explain ───────────────────────────────────────────────────────────────────

class TestExplain:

    def test_cold_start_explanation(self):
        b = ThompsonBandit()
        msg = b.explain(1.0, 1.0, 0.6, "Test Song")
        assert "exploring" in msg.lower() or "new" in msg.lower()

    def test_high_confidence_explanation(self):
        b = ThompsonBandit()
        msg = b.explain(50.0, 5.0, 0.8, "Good Song")  # ~90% mean
        assert "strong" in msg.lower() or "match" in msg.lower() or "signal" in msg.lower()

    def test_explanation_is_string(self):
        b = ThompsonBandit()
        msg = b.explain(3.0, 2.0, 0.5, "Any Song")
        assert isinstance(msg, str) and len(msg) > 5


# ── FeedbackEvent ─────────────────────────────────────────────────────────────

class TestFeedbackEvent:

    def test_roundtrip_json(self):
        from app.services.kafka.producer import FeedbackEvent
        event = FeedbackEvent(
            user_id="u1", song_id="s1", action="add",
            reward=1.0, timestamp="2024-01-01T00:00:00",
        )
        restored = FeedbackEvent.from_json(event.to_json())
        assert restored.user_id == event.user_id
        assert restored.reward == event.reward
        assert restored.action == event.action

    def test_json_contains_all_fields(self):
        from app.services.kafka.producer import FeedbackEvent
        import json
        event = FeedbackEvent(
            user_id="u1", song_id="s1", action="skip",
            reward=-0.3, timestamp="2024-01-01T00:00:00",
        )
        d = json.loads(event.to_json())
        assert "user_id" in d
        assert "song_id" in d
        assert "reward" in d
        assert "action" in d
