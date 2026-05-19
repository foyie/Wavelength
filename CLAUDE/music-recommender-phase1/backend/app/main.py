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
from app.routers import recommendations, playlist, feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Music Recommender API...")
    await create_tables()
    logger.info("Database tables ensured")
    yield
    logger.info("Shutting down...")
    await feature_store.close()
    await spotify_client.close()


app = FastAPI(
    title="Music Recommender API",
    version="1.0.0",
    description="Adaptive music playlist recommendation engine",
    lifespan=lifespan,
)

# CORS — allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

# Routers
app.include_router(recommendations.router)
app.include_router(playlist.router)
app.include_router(feedback.router)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health():
    db_status = await check_db_health()
    redis_status = await feature_store.ping()
    overall = "healthy" if db_status == "healthy" and redis_status == "healthy" else "degraded"
    return HealthResponse(status=overall, db=db_status, redis=redis_status)
