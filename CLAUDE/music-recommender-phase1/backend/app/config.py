from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://music:music@localhost:5432/music_recommender"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_feature_ttl: int = 86400  # 24 hours

    # Spotify
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    # App
    app_name: str = "Music Recommender"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    # Recommendation
    default_recommendation_limit: int = 20
    popularity_weight: float = 0.6  # Phase 1: popularity-based baseline

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
