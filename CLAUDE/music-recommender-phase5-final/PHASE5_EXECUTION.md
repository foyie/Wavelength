# Phase 5 Execution Guide — Model Versioning, Load Testing, CI/CD

Builds on Phase 4. No API contract changes. Adds MLflow sidecar,
Locust load tests, GitHub Actions CI/CD pipeline, and a fourth Grafana
dashboard for system health.

---

## Prerequisites

Phase 4 running and healthy:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","version":"4.0.0",...}
```

---

## Step 1 — Install Phase 5 dependencies

```bash
cd backend
source .venv/bin/activate
pip install mlflow==2.13.0 locust==2.28.0 pytest-cov==5.0.0 ruff==0.4.4

# Verify
python -c "import mlflow; print('mlflow', mlflow.__version__)"
python -c "import locust; print('locust OK')"
```

---

## Step 2 — Stop the backend

```bash
# Ctrl+C in the uvicorn terminal, or:
docker compose stop backend
```

---

## Step 3 — Start MLflow

MLflow needs Postgres as its backend store. Start it alongside your
existing infrastructure:

```bash
# From project root
docker compose up -d mlflow

# Wait for MLflow to be healthy (~20 seconds)
docker compose ps mlflow
# mlflow   Up (healthy)

# Verify MLflow UI is accessible
curl -s http://localhost:5000/health
# OK
```

Open **http://localhost:5000** — you should see the MLflow experiment tracker UI.
The `music-bandit` experiment will be created automatically on first snapshot.

---

## Step 4 — Start the Phase 5 backend

```bash
cd backend
source .venv/bin/activate

KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
MLFLOW_ENABLED=true \
MLFLOW_TRACKING_URI=http://localhost:5000 \
uvicorn app.main:app --reload --port 8000
```

Expected startup log — verify all six lines appear:
```
INFO  Starting Music Recommender API (Phase 5)...
INFO  Database tables ensured
INFO  Kafka producer connected: localhost:9092
INFO  Kafka consumer started: group=bandit-updater topic=music.feedback
INFO  Metrics collector started
INFO  Experiment registry loaded
INFO  Model registry: enabled
INFO  Application startup complete
```

If `Model registry: unavailable (MLflow not reachable)` appears instead of
`enabled`, MLflow is not reachable. Check Step 3. The backend still works —
versioning is gracefully degraded, not a hard dependency.

Confirm version 5.0.0:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"5.0.0"}
```

---

## Step 5 — Run Phase 5 tests

```bash
cd backend
source .venv/bin/activate
pytest tests/test_phase5.py -v
```

Expected: 18 tests passing. Key ones:

```
test_snapshot_creation                      PASSED
test_snapshot_serialises_to_json            PASSED
test_save_snapshot_skipped_when_unavailable PASSED  ← graceful degradation
test_list_snapshots_empty_when_unavailable  PASSED
test_take_snapshot_returns_dict             PASSED
test_version_name_format                    PASSED
test_snapshot_counter_increments            PASSED
test_phase1_imports                         PASSED  ← coexistence check
test_phase2_imports                         PASSED
test_phase3_imports                         PASSED
test_phase4_imports                         PASSED
test_phase5_imports                         PASSED
test_no_duplicate_metric_names              PASSED  ← Prometheus safety check
```

Full suite (all phases):
```bash
pytest tests/ -v --cov=app --cov-report=term-missing
# Expected: ~100 tests, >80% coverage
```

---

## Step 6 — Take your first model snapshot

```bash
# Manual snapshot — captures current bandit aggregate state
curl -X POST http://localhost:8000/api/versioning/snapshot \
  -H "Content-Type: application/json" \
  -d '{"trigger": "manual", "notes": "Phase 5 baseline snapshot"}' \
  | python -m json.tool
```

Expected response:
```json
{
  "version_name": "bandit-v1-20250101-1200",
  "run_id": "abc123...",
  "mlflow_available": true,
  "trigger": "manual",
  "drift_signal": null,
  "stats": {
    "n_users": 1,
    "n_arms_total": 5,
    "mean_reward": 0.5,
    "exploration_rate": 1.0,
    "total_interactions": 0
  },
  "created_at": "2025-01-01T12:00:00"
}
```

If `mlflow_available` is false, the snapshot still logs to stdout and returns
a dict — it just isn't persisted to MLflow.

---

## Step 7 — Verify the snapshot in MLflow UI

1. Open **http://localhost:5000**
2. Click **music-bandit** experiment in the left sidebar
3. You should see one run named `bandit-v1-...`
4. Click the run to see:
   - **Metrics:** mean_reward, exploration_rate, n_arms_total, total_interactions
   - **Parameters:** trigger, context_weight, candidate_pool
   - **Artifacts:** `snapshot.json` with the full state dict

---

## Step 8 — Generate traffic, then take a post-training snapshot

```bash
# Get songs
SONGS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=10&bypass_cache=true" \
  | python -c "import sys,json; print(' '.join(s['id'] for s in json.load(sys.stdin)['songs']))")

# Send 500 feedback events to build up bandit history
echo "Sending 500 feedback events..."
for i in $(seq 1 500); do
  SONG=$(echo $SONGS | tr ' ' '\n' | shuf -n1)
  USER="user-$(( i % 30 ))"
  ACTION=$(python3 -c "import random; print(random.choice(['add','skip','play']))")
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$USER\",\"song_id\":\"$SONG\",\"action\":\"$ACTION\"}" > /dev/null
  if [ $(( i % 100 )) -eq 0 ]; then echo "  $i / 500"; fi
done

# Take post-training snapshot
curl -X POST http://localhost:8000/api/versioning/snapshot \
  -H "Content-Type: application/json" \
  -d '{"trigger": "manual", "notes": "After 500 interactions — baseline established"}' \
  | python -m json.tool
```

---

## Step 9 — List and compare snapshots

```bash
# List all snapshots
curl http://localhost:8000/api/versioning/snapshots | python -m json.tool

# Check MLflow status
curl http://localhost:8000/api/versioning/status | python -m json.tool
```

In the MLflow UI at http://localhost:5000, select both runs and click
**Compare** to see metric differences side by side. The second snapshot
should show a higher `total_interactions` and a lower `exploration_rate`
(more arms have observations now).

---

## Step 10 — Run the Locust load test

The load test simulates realistic traffic: 60% recommendation browsing,
25% feedback, 10% playlist checks, 5% SHAP explanations.

**Interactive mode (with web UI):**
```bash
cd project-root
locust -f scripts/load_test.py --host http://localhost:8000
# Open http://localhost:8089
# Set: Users=50, Spawn rate=5
# Click Start swarming
```

**Headless mode (for CI or benchmarking):**
```bash
locust -f scripts/load_test.py \
  --host http://localhost:8000 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 120s \
  --headless \
  --csv load_test_results

# Results written to:
#   load_test_results_stats.csv
#   load_test_results_failures.csv
#   load_test_results_stats_history.csv
```

**Performance targets to verify:**

| Endpoint | p50 target | p95 target |
|----------|-----------|-----------|
| `GET /api/recommend/{id}` | < 25ms | < 100ms |
| `POST /api/feedback` | < 15ms | < 50ms |
| `GET /api/recommend?explain=true` | < 50ms | < 200ms |
| Error rate | — | < 0.1% |

**Reading the results:**
```bash
# Print summary of key endpoints from CSV
python3 -c "
import csv
with open('load_test_results_stats.csv') as f:
    for row in csv.DictReader(f):
        if row['Name'] != 'Aggregated':
            print(f\"{row['Name']:<50} p50={row['50%']}ms p95={row['95%']}ms err={row['Failure Count']}\")
"
```

---

## Step 11 — View the System Health Grafana dashboard

The fourth dashboard, System Health, was added in Phase 5.

1. Open **http://localhost:3001**
2. Navigate to Dashboards → Music Recommender → System Health
3. Key panels to check after the load test:
   - **Recommendation Latency Heatmap** — should show most requests sub-50ms
   - **API Error Rate** — should be at or near zero
   - **Redis Cache Hit Rate** — should be high (>80%) after warmup
   - **Kafka Consumer Lag** — should stay near zero during load

---

## Step 12 — CI/CD pipeline setup

The GitHub Actions workflow at `.github/workflows/ci.yml` runs on every
push to `main` or `develop`:

1. **Backend tests** — full pytest suite with Postgres and Redis service containers
2. **Backend lint** — Ruff + mypy
3. **Frontend tests** — TypeScript check + production build
4. **Docker build** — builds and pushes backend + frontend images to GHCR
5. **Integration smoke test** — starts backend, hits all major endpoints

**To enable it:**
```bash
# Push to GitHub
git add .
git commit -m "feat: complete wavelength music recommender (all 5 phases)"
git push origin main

# View workflow runs at:
# https://github.com/your-username/wavelength/actions
```

**Run lint locally before pushing:**
```bash
cd backend
source .venv/bin/activate
ruff check app/ --select=E,F,W,I --ignore=E501
```

**Run the full CI suite locally (simulates GitHub Actions):**
```bash
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing --cov-fail-under=80
```

---

## Step 13 — Full Docker deployment (all phases)

To run all 10 services containerised:

```bash
# From project root
docker compose up --build

# All services:
#   postgres    :5432
#   redis       :6379
#   zookeeper   :2181
#   kafka       :9092
#   kafka-init  (one-shot)
#   backend     :8000
#   frontend    :3000
#   prometheus  :9090
#   grafana     :3001
#   mlflow      :5000

# Seed songs (after services are healthy)
docker compose exec backend python -m scripts.load_spotify_data
```

---

## Complete port reference — all phases

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| Frontend | 3000 | http://localhost:3000 | — |
| Backend API | 8000 | http://localhost:8000 | — |
| Swagger Docs | 8000 | http://localhost:8000/docs | — |
| Prometheus Metrics | 8000 | http://localhost:8000/metrics | — |
| Prometheus UI | 9090 | http://localhost:9090 | — |
| Grafana | 3001 | http://localhost:3001 | admin / music123 |
| MLflow | 5000 | http://localhost:5000 | — |
| Kafka | 9092 | localhost:9092 | — |
| Postgres | 5432 | localhost:5432 | music / music |
| Redis | 6379 | localhost:6379 | — |
| Zookeeper | 2181 | localhost:2181 | — |
| Locust UI | 8089 | http://localhost:8089 | — (when running) |

---

## Troubleshooting

**`mlflow` import error on startup**
```bash
source backend/.venv/bin/activate
pip install mlflow==2.13.0
```

**MLflow says `unavailable` despite service being up**
The MLflow container uses Postgres as its backend store. It needs Postgres
to be healthy first. Check:
```bash
docker compose ps mlflow
# Should show Up (healthy)
docker compose logs mlflow | tail -20
# Common error: "could not connect to server" — Postgres not ready yet
# Fix: docker compose restart mlflow
```

**Locust `Connection refused` on load test start**
The backend must be running on port 8000 before starting Locust.
The `on_test_start` hook will fail gracefully (prints a warning) if the
recommendation endpoint returns no songs — the test will still run but
`SONG_IDS` will be empty and the feedback task will skip.

**Load test latencies higher than targets**
Common causes:
- Redis cache cold — first 20-30 seconds of any load test are slower
- No songs seeded — run `scripts/load_spotify_data.py` first
- Kafka consumer lag building up — check Grafana System Health dashboard
- Running on a resource-constrained machine — reduce `--users` to 20

**GitHub Actions `docker login` fails**
The workflow uses `GITHUB_TOKEN` for GHCR. This is automatically available.
If it fails, check that your repository has `packages: write` permission
in the workflow settings (Settings → Actions → General → Workflow permissions).

**Coverage below 80%**
The routers that need a live DB session will not be covered by unit tests.
Add `# pragma: no cover` to the router endpoints or configure coverage to
exclude them in a `pyproject.toml`:
```toml
[tool.coverage.report]
exclude_lines = [
  "pragma: no cover",
  "if TYPE_CHECKING:",
  "raise NotImplementedError",
]
```
