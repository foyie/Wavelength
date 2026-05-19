# Wavelength — Adaptive Music Recommendation Engine

Production-grade playlist recommender that learns from user interactions.

## Phase 1 — What's built
- FastAPI backend with PostgreSQL + Redis caching (<100ms responses)
- Popularity + user similarity scoring (baseline before bandit)
- Online user preference vectors updated from every feedback action
- React UI: recommendations, playlist, audio previews, add/skip feedback
- Dockerised infrastructure, Prometheus metrics endpoint ready

## Quick start

```bash
# 1. Clone & configure
cp .env.example .env
# Fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
# Get them at: https://developer.spotify.com/dashboard

# 2. Start infrastructure
docker-compose up -d postgres redis

# 3. Start backend (local dev)
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 4. Seed Spotify data (~1000 songs)
python -m scripts.load_spotify_data

# 5. Start frontend (local dev)
cd ../frontend
npm install
npm start
# Opens at http://localhost:3000

# Or run everything in Docker:
docker-compose up --build
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health check |
| GET | `/api/recommend/{user_id}` | Get ranked recommendations |
| GET | `/api/users/{user_id}/playlist` | Get user playlist |
| POST | `/api/feedback` | Record add/skip/play action |
| GET | `/metrics` | Prometheus metrics |

Interactive docs: http://localhost:8000/docs

## Project structure

```
music-recommender/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + lifespan
│   │   ├── config.py          # Settings via pydantic-settings
│   │   ├── db.py              # SQLAlchemy async engine
│   │   ├── models.py          # ORM models
│   │   ├── schemas.py         # Pydantic request/response schemas
│   │   ├── routers/           # recommendations, playlist, feedback
│   │   └── services/
│   │       ├── spotify_client.py   # Spotify Web API client
│   │       ├── feature_store.py    # Redis caching layer
│   │       ├── recommender.py      # Phase 1 baseline (Phase 2: swap for bandit)
│   │       └── feedback.py         # Feedback processing + EMA updates
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.tsx            # Main shell, tab routing, state
│       ├── components/        # SongCard
│       ├── hooks/             # usePlayer (audio preview)
│       ├── services/api.ts    # Typed API client
│       └── types/             # TypeScript interfaces
├── scripts/
│   └── load_spotify_data.py   # Seed ~1000 songs from Spotify
└── docker-compose.yml
```

## Phase roadmap

| Phase | Week | What gets added |
|-------|------|-----------------|
| 1 ✅ | 1 | Spotify data, FastAPI, Redis, React UI, popularity baseline |
| 2 | 2 | Thompson Sampling bandit, Kafka, online learning |
| 3 | 3 | ADWIN drift detection, data quality, Prometheus/Grafana |
| 4 | 4 | A/B testing (SPRT), SHAP explanations, admin panel |
| 5 | 5 | Full Grafana dashboards, MLflow versioning, load tests |

## Running tests

```bash
cd backend
pytest tests/ -v
```
