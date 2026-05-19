"""
Phase 1 unit tests — run with: pytest tests/ -v
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from app.services.recommender import PopularityRecommender, normalise_song, cosine_similarity
from app.schemas import FeedbackRequest


# ── Recommender ──────────────────────────────────────────────────────────────

class MockSong:
    def __init__(self, **kwargs):
        defaults = dict(
            id="song1", name="Test", artist="Artist", album=None,
            album_art_url=None, preview_url=None, duration_ms=200000,
            popularity=80, explicit=False,
            danceability=0.7, energy=0.8, valence=0.6, tempo=120.0,
            acousticness=0.2, genres=[]
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


class MockUser:
    def __init__(self, **kwargs):
        defaults = dict(
            avg_danceability=0.6, avg_energy=0.7, avg_valence=0.5,
            avg_tempo=115.0, avg_acousticness=0.25,
        )
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_cosine_similarity_identical():
    a = np.array([0.5, 0.7, 0.3, 0.6, 0.4])
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-6


def test_cosine_similarity_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_score_without_user():
    rec = PopularityRecommender()
    song = MockSong(popularity=70)
    score = rec.score(song, user=None)
    assert abs(score - 0.70) < 0.01


def test_score_with_user_similar():
    rec = PopularityRecommender()
    # Song closely matches user prefs
    song = MockSong(popularity=50, danceability=0.6, energy=0.7, valence=0.5, tempo=115, acousticness=0.25)
    user = MockUser()
    score = rec.score(song, user)
    # Should be higher than popularity alone due to high similarity
    baseline = 0.50 * 0.6  # popularity_weight * popularity
    assert score > baseline


def test_score_range():
    rec = PopularityRecommender()
    song = MockSong(popularity=100)
    user = MockUser()
    score = rec.score(song, user)
    assert 0.0 <= score <= 1.0


def test_explain_no_user():
    rec = PopularityRecommender()
    song = MockSong(popularity=85)
    explanation = rec.explain(song, user=None)
    assert "85" in explanation


# ── Feedback schema ───────────────────────────────────────────────────────────

def test_feedback_reward_add():
    fb = FeedbackRequest(user_id="u1", song_id="s1", action="add")
    assert fb.reward == 1.0


def test_feedback_reward_skip():
    fb = FeedbackRequest(user_id="u1", song_id="s1", action="skip")
    assert fb.reward == -0.3


def test_feedback_reward_long_play():
    fb = FeedbackRequest(user_id="u1", song_id="s1", action="play", play_duration_ms=60000)
    assert fb.reward == 0.8


def test_feedback_reward_short_play():
    fb = FeedbackRequest(user_id="u1", song_id="s1", action="play", play_duration_ms=2000)
    assert fb.reward == -0.3


def test_feedback_invalid_action():
    with pytest.raises(Exception):
        FeedbackRequest(user_id="u1", song_id="s1", action="buy")
