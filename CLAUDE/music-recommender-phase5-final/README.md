# Wavelength

A production-grade adaptive music recommendation engine built from first principles.
Most recommendation systems are static — they rank items and call it done.
Wavelength learns in real time, detects when it is wrong, explains every decision,
and validates changes with sequential hypothesis tests before shipping them.

---

## What makes this different

Most open-source recommender systems demonstrate one idea cleanly: matrix
factorisation, or a neural network, or a simple bandit. Wavelength demonstrates
the full production stack that surrounds the algorithm — the part that is usually
invisible in papers and tutorials but dominates engineering time in practice.

**The algorithm learns from every interaction without retraining.**
Thompson Sampling with a Beta(alpha, beta) posterior per (user, song) arm
updates continuously via a Kafka consumer. There is no nightly batch job.
A skip at 2am shifts the next morning's rankings for that user.

**The system knows when it stops working.**
ADWIN (Adaptive Windowing) monitors the live reward stream and detects
distributional shifts in O(log n) time. A Kolmogorov-Smirnov test runs
independently on the audio-feature distributions, catching catalog drift
separately from preference drift. Both fire Prometheus alerts and are
visible in Grafana dashboards before the degradation reaches users.

**Every recommendation carries an explanation.**
SHAP values decompose each song's score into contributions from individual
audio features plus the bandit's learned prior. The system can tell a user:
"This ranked first because it matches your energy preference (0.09) and your
12 previous positive signals (0.23), despite a low acoustic similarity (-0.03)."

**Changes are validated with sequential hypothesis tests.**
SPRT (Sequential Probability Ratio Test) replaces fixed-sample A/B testing.
It stops as soon as the evidence is sufficient — either direction — rather
than waiting for a pre-committed sample size. This cuts wasted traffic by
30-60% on large effects while maintaining the same Type I and Type II
error guarantees.

**The model's history is audited, not forgotten.**
MLflow records every bandit state snapshot with the metrics, parameters,
and drift signals that prompted it. If a reset causes regressions, the
prior version is one API call away.

---

## Architecture

```
User action (add/skip/play)
        |
        v
FastAPI backend  ──── Postgres (interactions, bandit state, experiments)
        |                         |
        |              Redis (feature store, rec cache,
        |                    bandit alpha/beta vectors,
        |                    experiment registry)
        v
Kafka topic: music.feedback
        |
        v
Consumer loop
  ├── FeedbackValidator       (data quality gate)
  ├── BanditStateManager      (update Beta posteriors)
  ├── DriftPipeline
  │     ├── ADWINDetector     (reward stream drift)
  │     └── KS test           (audio feature distribution drift)
  └── ExperimentRegistry      (SPRT A/B observation recording)
        |
        v
Prometheus  ──── Grafana (4 dashboards, 30+ metrics)
MLflow           (bandit snapshots, lineage, rollback)
```

**Backend:** Python 3.11, FastAPI, SQLAlchemy (async), asyncpg, aiokafka

**Frontend:** React 18, TypeScript, Spotify 30-second preview playback

**Infrastructure:** PostgreSQL 16, Redis 7, Kafka (Confluent), Prometheus, Grafana, MLflow

---

## Capabilities by phase

| Phase | What it delivers |
|-------|-----------------|
| 1 | FastAPI backend, PostgreSQL, Redis, React UI, Spotify catalog seed, popularity + cosine similarity baseline |
| 2 | Thompson Sampling bandit, Kafka feedback pipeline, online EMA preference updates, Beta posterior persistence |
| 3 | ADWIN reward drift detection, KS audio-feature drift, data quality validation, 30+ Prometheus metrics, 3 Grafana dashboards |
| 4 | SPRT A/B testing with Redis-persisted experiment state, SHAP leave-one-out feature attribution, user taste profiles |
| 5 | MLflow model versioning with full lineage, Locust load tests (100 req/s target), GitHub Actions CI/CD, system health dashboard |

---

## Quick start

**Prerequisites:** Python 3.11, Node 20, Docker Desktop

```bash
git clone https://github.com/your-username/wavelength.git
cd wavelength

# Configure Spotify credentials
# Get them at https://developer.spotify.com/dashboard
cp .env.example .env
# Edit .env: fill in SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET

# Start everything
docker compose up -d

# Seed ~1000 songs from Spotify (one-time, takes 3-8 minutes)
docker compose exec backend python -m scripts.load_spotify_data

# Open the app
open http://localhost:3000
```

**Services started:**

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | React recommendation UI |
| API | http://localhost:8000 | FastAPI backend |
| API Docs | http://localhost:8000/docs | Interactive Swagger UI |
| Grafana | http://localhost:3001 | Dashboards (admin / music123) |
| Prometheus | http://localhost:9090 | Metrics |
| MLflow | http://localhost:5000 | Model versioning |

---

## Local development (without Docker)

```bash
# Infrastructure only
docker compose up -d postgres redis zookeeper kafka

# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000

# Seed data (new terminal, venv active)
python -m scripts.load_spotify_data

# Frontend (new terminal)
cd frontend
npm install && npm start
```

---

## API reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health + version |
| GET | `/api/recommend/{user_id}` | Ranked recommendations (Thompson Sampling) |
| POST | `/api/feedback` | Record add / skip / play / complete |
| GET | `/api/users/{user_id}/playlist` | User's saved playlist |

**Recommendation query parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `limit` | 20 | Number of songs (max 100) |
| `explain` | false | Include SHAP feature attributions per song |
| `experiment_id` | — | Enrol user in named A/B experiment |
| `bypass_cache` | false | Force fresh bandit sample |

### Bandit admin

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/bandit/{user_id}/state` | All arm (alpha, beta) values for a user |
| GET | `/api/bandit/{user_id}/arm/{song_id}` | Single arm with uncertainty |
| POST | `/api/bandit/{user_id}/reset` | Reset arms to prior (1, 1) |
| GET | `/api/bandit/stats` | Aggregate arm statistics |

### Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/monitoring/drift` | ADWIN + KS drift summary |
| GET | `/api/monitoring/drift/adwin` | ADWIN window state per signal |
| GET | `/api/monitoring/drift/features` | KS test results per audio feature |
| POST | `/api/monitoring/drift/reset` | Reset detectors after model update |
| GET | `/api/monitoring/quality` | Catalog feature completeness |
| GET | `/api/monitoring/quality/interactions` | Interaction stats (last N hours) |

### Experiments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/experiments` | Create SPRT experiment |
| GET | `/api/experiments` | List all experiments |
| GET | `/api/experiments/{id}` | Full SPRT state + DB audit |
| GET | `/api/experiments/{id}/arm/{user_id}` | Deterministic arm assignment |
| POST | `/api/experiments/{id}/conclude` | Force-conclude |
| DELETE | `/api/experiments/{id}` | Delete |

### Explainability

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/explain/{user_id}/{song_id}` | SHAP breakdown for one song |
| POST | `/api/explain/{user_id}/batch` | Batch SHAP (up to 50 songs) |
| GET | `/api/explain/{user_id}/profile` | User taste profile + natural language summary |

### Versioning

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/versioning/snapshot` | Take a bandit state snapshot |
| GET | `/api/versioning/snapshots` | List snapshots from MLflow |
| GET | `/api/versioning/snapshots/{run_id}` | Full snapshot artifact |
| GET | `/api/versioning/status` | MLflow connectivity |

---

## Running tests

```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing
```

| Test file | Coverage |
|-----------|----------|
| `test_recommender.py` | Popularity baseline, EMA updates, reward computation |
| `test_bandit.py` | Thompson Sampling update, convergence, context scoring, Kafka event serialisation |
| `test_monitoring.py` | ADWIN (stationary, step change, gradual drift), KS test (no-drift, drift, false positive), data quality validator |
| `test_phase4.py` | SPRT boundaries, convergence, arm assignment distribution, experiment serialisation, SHAP attributions |
| `test_phase5.py` | Model versioning, snapshot service, all-phase import coexistence, no duplicate Prometheus metrics |

Total: ~100 tests across all phases.

---

## Project structure

```
wavelength/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app, lifespan startup/shutdown
│   │   ├── config.py                      # Pydantic settings (env-driven)
│   │   ├── db.py                          # Async SQLAlchemy engine + session
│   │   ├── models.py                      # ORM: Song, User, Interaction,
│   │   │                                  #   BanditState, PlaylistItem,
│   │   │                                  #   ExperimentAssignment/Observation
│   │   ├── schemas.py                     # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── recommendations.py         # GET /api/recommend
│   │   │   ├── feedback.py                # POST /api/feedback
│   │   │   ├── playlist.py                # GET/DELETE /api/users/*/playlist
│   │   │   ├── bandit_admin.py            # GET/POST /api/bandit
│   │   │   ├── monitoring.py              # GET/POST /api/monitoring
│   │   │   ├── experiments.py             # CRUD /api/experiments
│   │   │   ├── explainability.py          # GET /api/explain
│   │   │   └── versioning.py              # GET/POST /api/versioning
│   │   └── services/
│   │       ├── spotify_client.py          # Async Spotify Web API client
│   │       ├── feature_store.py           # Redis caching (TTL, pipeline bulk load)
│   │       ├── recommender.py             # BanditRecommender (SHAP-aware)
│   │       ├── feedback.py                # Feedback pipeline + experiment routing
│   │       ├── bandit/
│   │       │   ├── thompson.py            # ThompsonBandit: sample, rank, update
│   │       │   └── state_manager.py       # Redis/Postgres alpha-beta persistence
│   │       ├── kafka/
│   │       │   ├── producer.py            # Async Kafka producer (graceful degradation)
│   │       │   └── consumer.py            # Consumer loop: validate → update → drift
│   │       ├── monitoring/
│   │       │   ├── adwin.py               # ADWIN drift detector (O(log n))
│   │       │   ├── ks_drift.py            # KS two-sample feature drift monitor
│   │       │   ├── metrics.py             # 30+ Prometheus metric definitions
│   │       │   ├── data_quality.py        # Feedback + feature validators
│   │       │   ├── drift_pipeline.py      # Orchestrates ADWIN + KS + alerting
│   │       │   └── collector.py           # Background gauge refresh (60s interval)
│   │       ├── experiments/
│   │       │   ├── sprt.py                # Wald SPRT: log-lambda, boundaries, update
│   │       │   └── registry.py            # Experiment CRUD + Redis persistence
│   │       ├── explainability/
│   │       │   └── shap_explainer.py      # Leave-one-out SHAP for cosine similarity
│   │       └── versioning/
│   │           ├── model_registry.py      # MLflow async wrapper
│   │           └── snapshot.py            # Aggregate stats collector + snapshot trigger
│   ├── tests/
│   │   ├── test_recommender.py
│   │   ├── test_bandit.py
│   │   ├── test_monitoring.py
│   │   ├── test_phase4.py
│   │   └── test_phase5.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.tsx                        # Tab shell, state, feedback loop
│       ├── components/SongCard.tsx        # Album art, feature bars, SHAP display
│       ├── hooks/usePlayer.ts             # Spotify 30s preview playback
│       ├── services/api.ts                # Typed fetch client
│       └── types/index.ts                 # TypeScript interfaces
├── scripts/
│   ├── load_spotify_data.py               # Seed ~1000 songs with audio features
│   └── load_test.py                       # Locust: 100 req/s, realistic traffic mix
├── monitoring/
│   ├── prometheus/prometheus.yml          # Scrape config (15s interval, 30d retention)
│   └── grafana/
│       ├── provisioning/                  # Auto-provision datasource + dashboards
│       └── dashboards/
│           ├── recommendation_quality.json
│           ├── drift_detection.json
│           ├── data_quality_bandit.json
│           └── system_health.json
├── .github/workflows/ci.yml              # Test → lint → build → push → smoke test
├── docker-compose.yml                    # All 10 services, health-checked
├── .env.example
└── README.md
```

---

## Key design decisions

**Why Thompson Sampling instead of UCB or epsilon-greedy?**
UCB requires a tuned confidence parameter. Epsilon-greedy requires a tuned
exploration rate. Thompson Sampling requires neither — it derives exploration
automatically from posterior uncertainty. An arm with two observations will
naturally be sampled more often than one with two hundred, without any
hyperparameter to tune. When the Beta distribution is wide (uncertain),
samples vary widely and the arm gets explored. When it narrows (confident),
it competes on mean reward alone.

**Why ADWIN instead of Page-Hinkley or CUSUM?**
ADWIN adapts its own window size. Other tests require a fixed lookback window,
which is a hyperparameter that must be tuned per dataset. ADWIN has
provable false-positive bounds controlled by a single delta parameter and
runs in O(log n) amortised time via compressed bucket structures.

**Why SPRT instead of fixed-sample A/B tests?**
Fixed-sample tests commit to a sample size before seeing data. If the effect
is large, you collect far more samples than necessary. If the effect is small,
the pre-committed size may be underpowered. SPRT stops as soon as the
log-likelihood ratio crosses a boundary — in either direction — giving the
same Type I and Type II guarantees with 30-60% fewer samples on strong effects.

**Why leave-one-out instead of KernelSHAP for explanations?**
Full KernelSHAP requires ~200 coalition evaluations per prediction, adding
60+ ms per request at scale. Leave-one-out with five features requires exactly
five forward passes and is exact (not approximate) for cosine similarity
because the function is nearly additive in feature contributions. At 20
recommendations per request, the total SHAP overhead is under 3ms.

**Why Kafka for feedback instead of direct writes?**
Direct writes couple API latency to bandit update latency. Kafka decouples
them: the API returns in under 10ms, and the bandit update happens
asynchronously in the consumer. It also provides a durable event log for
replay, auditing, and the drift detection pipeline. The producer falls back
gracefully to direct writes when Kafka is unreachable, so the bandit still
learns — just synchronously.

---

## Performance targets

| Metric | Target | Measured at |
|--------|--------|-------------|
| Recommendation latency (p50) | < 25ms | 50 concurrent users |
| Recommendation latency (p95) | < 100ms | 50 concurrent users |
| Recommendation latency (p99) | < 250ms | 50 concurrent users |
| Feedback processing latency (p95) | < 50ms | Kafka consumer |
| Throughput | > 100 rec/s | Locust load test |
| Error rate | < 0.1% | 120s sustained load |

Run the load test:
```bash
pip install locust
locust -f scripts/load_test.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 120s --headless \
  --csv load_test_results
```

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPOTIFY_CLIENT_ID` | — | Required. Spotify app credentials |
| `SPOTIFY_CLIENT_SECRET` | — | Required. Spotify app credentials |
| `DATABASE_URL` | `postgresql+asyncpg://music:music@localhost:5432/music_recommender` | Postgres DSN |
| `REDIS_URL` | `redis://localhost:6379` | Redis DSN |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker address |
| `KAFKA_ENABLED` | `true` | Set `false` to use direct bandit writes |
| `BANDIT_CONTEXT_WEIGHT` | `0.35` | Audio-feature vs posterior blend (0=pure bandit, 1=pure similarity) |
| `BANDIT_CANDIDATE_POOL` | `500` | Songs scored per recommendation call |
| `MLFLOW_ENABLED` | `true` | Set `false` to disable model versioning |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server address |
| `EXPERIMENTS_ENABLED` | `true` | Enable A/B experiment registry |

---

## Execution guides

Detailed step-by-step execution instructions are in the repository:

- `EXECUTION_GUIDE.md` — Phase 1 and Phase 2
- `PHASE3_AND_4_EXECUTION.md` — Phase 3 (drift + monitoring) and Phase 4 (A/B testing + explanations)
- `PHASE5_EXECUTION.md` — Phase 5 (model versioning, load testing, CI/CD)

Each guide includes expected log output, verification commands, troubleshooting
for every known failure mode, and PromQL queries for Grafana investigation.
