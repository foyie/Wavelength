import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.db import check_db_health, create_tables
from app.schemas import HealthResponse
from app.services.feature_store import feature_store
from app.services.spotify_client import spotify_client
from app.services.kafka.producer import feedback_producer
from app.services.kafka.consumer import feedback_consumer
from app.services.bandit.state_manager import bandit_state_manager
from app.services.monitoring.collector import metrics_collector
from app.services.experiments.registry import experiment_registry
from app.services.versioning.model_registry import model_registry
from app.routers import recommendations, playlist, feedback, monitoring
from app.routers import bandit_admin, experiments, explainability, versioning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Music Recommender API (Phase 5)...")
    await create_tables()
    logger.info("Database tables ensured")

    if settings.kafka_enabled:
        await feedback_producer.start()
        await feedback_consumer.start()
        logger.info("Kafka producer/consumer started")
    else:
        logger.info("Kafka disabled — using direct bandit writes")

    await metrics_collector.start()
    logger.info("Metrics collector started")

    if settings.experiments_enabled:
        await experiment_registry.load_from_redis()
        logger.info("Experiment registry loaded")

    # Phase 5: initialise MLflow model registry
    if settings.mlflow_enabled:
        await model_registry.initialise()
        logger.info(
            f"Model registry: {'enabled' if model_registry.is_available else 'unavailable (MLflow not reachable)'}"
        )

    yield

    logger.info("Shutting down...")
    await metrics_collector.stop()
    if settings.kafka_enabled:
        await feedback_consumer.stop()
        await feedback_producer.stop()
    await bandit_state_manager.close()
    await experiment_registry.close()
    await feature_store.close()
    await spotify_client.close()


app = FastAPI(
    title="Music Recommender API",
    version="5.0.0",
    description=(
        "Adaptive music playlist recommendation engine — "
        "Thompson Sampling · ADWIN/KS drift · SPRT A/B testing · "
        "SHAP explanations · MLflow model versioning"
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(recommendations.router)
app.include_router(playlist.router)
app.include_router(feedback.router)
app.include_router(bandit_admin.router)
app.include_router(monitoring.router)
app.include_router(experiments.router)
app.include_router(explainability.router)
app.include_router(versioning.router)        # Phase 5


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    db_status = await check_db_health()
    redis_status = await feature_store.ping()
    overall = "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded"
    return HealthResponse(
        status=overall, db=db_status, redis=redis_status, version="5.0.0"
    )
