from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://music:music@localhost:5432/music_recommender"

    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_feature_ttl: int = 86400

    # Spotify
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    # App
    app_name: str = "Music Recommender"
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    # Recommendation
    default_recommendation_limit: int = 20
    popularity_weight: float = 0.6

    # Phase 2: Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_feedback_topic: str = "music.feedback"
    kafka_consumer_group: str = "bandit-updater"
    kafka_enabled: bool = True

    # Phase 2: Bandit
    bandit_context_weight: float = 0.35
    bandit_candidate_pool: int = 500

    # Phase 4: Experiments
    experiments_enabled: bool = True

    # Phase 4: Explainability
    explain_by_default: bool = False   # set True to always include SHAP (adds ~3ms)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
