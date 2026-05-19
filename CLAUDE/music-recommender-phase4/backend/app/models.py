from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, ForeignKey, Text, JSON, Index
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Song(Base):
    __tablename__ = "songs"

    id = Column(String, primary_key=True)          # Spotify track ID
    name = Column(String, nullable=False)
    artist = Column(String, nullable=False)
    album = Column(String)
    album_art_url = Column(String)
    preview_url = Column(String)
    duration_ms = Column(Integer)
    popularity = Column(Integer, default=0)        # Spotify popularity 0-100
    explicit = Column(Boolean, default=False)

    # Spotify audio features
    danceability = Column(Float)                   # 0.0–1.0
    energy = Column(Float)                         # 0.0–1.0
    key = Column(Integer)                          # 0–11 (pitch class)
    loudness = Column(Float)                       # dB, typically -60–0
    mode = Column(Integer)                         # 0=minor, 1=major
    speechiness = Column(Float)                    # 0.0–1.0
    acousticness = Column(Float)                   # 0.0–1.0
    instrumentalness = Column(Float)               # 0.0–1.0
    liveness = Column(Float)                       # 0.0–1.0
    valence = Column(Float)                        # 0.0–1.0 (musical positiveness)
    tempo = Column(Float)                          # BPM
    genres = Column(JSON, default=list)

    created_at = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("Interaction", back_populates="song")

    __table_args__ = (
        Index("ix_songs_popularity", "popularity"),
        Index("ix_songs_artist", "artist"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    display_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    # Aggregated preference vector (updated online)
    avg_danceability = Column(Float, default=0.5)
    avg_energy = Column(Float, default=0.5)
    avg_valence = Column(Float, default=0.5)
    avg_tempo = Column(Float, default=120.0)
    avg_acousticness = Column(Float, default=0.5)
    total_interactions = Column(Integer, default=0)

    interactions = relationship("Interaction", back_populates="user")
    playlist_items = relationship("PlaylistItem", back_populates="user")


class Interaction(Base):
    """Every user action on a song."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, ForeignKey("songs.id"), nullable=False)
    action = Column(String, nullable=False)        # add | skip | play | complete
    reward = Column(Float, nullable=False)         # computed reward signal
    context = Column(JSON, default=dict)           # snapshot of features at time of interaction
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="interactions")
    song = relationship("Song", back_populates="interactions")

    __table_args__ = (
        Index("ix_interactions_user_id", "user_id"),
        Index("ix_interactions_song_id", "song_id"),
        Index("ix_interactions_timestamp", "timestamp"),
    )


class PlaylistItem(Base):
    """Songs explicitly added to a user's playlist."""
    __tablename__ = "playlist_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, ForeignKey("songs.id"), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    position = Column(Integer, default=0)

    user = relationship("User", back_populates="playlist_items")
    song = relationship("Song")

    __table_args__ = (
        Index("ix_playlist_user_id", "user_id"),
    )


class BanditState(Base):
    """Per-user, per-song bandit parameters. Populated in Phase 2."""
    __tablename__ = "bandit_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    song_id = Column(String, ForeignKey("songs.id"), nullable=False)
    alpha = Column(Float, default=1.0)             # Beta distribution success count
    beta_param = Column(Float, default=1.0)        # Beta distribution failure count
    updated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_bandit_user_song", "user_id", "song_id", unique=True),
    )


class ExperimentAssignment(Base):
    """Tracks which A/B arm each user is in for each experiment."""
    __tablename__ = "experiment_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    arm_name = Column(String, nullable=False)          # "control" | "treatment"
    assigned_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_exp_assignment_exp_user", "experiment_id", "user_id", unique=True),
    )


class ExperimentObservation(Base):
    """
    One reward signal tied to an experiment arm.
    Used to rebuild SPRT state from Postgres (audit trail + replayability).
    """
    __tablename__ = "experiment_observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    arm_name = Column(String, nullable=False)
    reward = Column(Float, nullable=False)
    action = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_exp_obs_exp_id", "experiment_id"),
        Index("ix_exp_obs_user_id", "user_id"),
    )
