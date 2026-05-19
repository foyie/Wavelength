from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Song ────────────────────────────────────────────────────────────────────

class SongBase(BaseModel):
    id: str
    name: str
    artist: str
    album: Optional[str] = None
    album_art_url: Optional[str] = None
    preview_url: Optional[str] = None
    duration_ms: Optional[int] = None
    popularity: int = 0
    explicit: bool = False
    danceability: Optional[float] = None
    energy: Optional[float] = None
    valence: Optional[float] = None
    tempo: Optional[float] = None
    acousticness: Optional[float] = None
    genres: list[str] = []


class SongResponse(SongBase):
    model_config = {"from_attributes": True}


class RecommendedSong(SongResponse):
    score: float = 0.0
    rank: int = 0
    explanation: Optional[str] = None   # Phase 4: SHAP-based explanation


# ── User ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    id: str
    display_name: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    display_name: Optional[str]
    total_interactions: int
    created_at: datetime
    last_active: datetime

    model_config = {"from_attributes": True}


# ── Playlist ─────────────────────────────────────────────────────────────────

class PlaylistResponse(BaseModel):
    user_id: str
    songs: list[SongResponse]
    total: int


# ── Feedback ─────────────────────────────────────────────────────────────────

REWARD_MAP = {
    "add": 1.0,
    "complete": 0.8,      # played full song
    "play": 0.5,          # started playing
    "skip": -0.3,
}


class FeedbackRequest(BaseModel):
    user_id: str
    song_id: str
    action: str = Field(..., pattern="^(add|skip|play|complete)$")
    play_duration_ms: Optional[int] = None  # for partial play scoring

    @property
    def reward(self) -> float:
        if self.action == "play" and self.play_duration_ms is not None:
            # Partial play: reward scales with how much was listened to
            if self.play_duration_ms < 5000:
                return -0.3
            elif self.play_duration_ms > 30000:
                return 0.8
            else:
                return 0.5
        return REWARD_MAP.get(self.action, 0.0)


class FeedbackResponse(BaseModel):
    status: str
    user_id: str
    song_id: str
    action: str
    reward: float


# ── Recommendations ───────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    user_id: str
    limit: int = Field(default=20, ge=1, le=100)
    exclude_playlist: bool = True   # don't recommend songs already in playlist


class RecommendationResponse(BaseModel):
    user_id: str
    songs: list[RecommendedSong]
    algorithm: str
    latency_ms: float


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: str
    redis: str
    version: str = "1.0.0"
