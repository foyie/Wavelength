# Phase 3 Execution Guide — Drift Detection & Data Quality

Builds on the running Phase 2 system. No API or frontend changes.
New: ADWIN + KS drift detectors, data quality validation, Prometheus metrics,
Grafana dashboards, and a monitoring REST API.

---

## Prerequisites

Phase 2 must be running and healthy:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","version":"2.0.0",...}
```

---

## Step 1 — Install new dependency (scipy)

```bash
cd backend
source .venv/bin/activate
pip install scipy==1.13.0
```

---

## Step 2 — Stop the running backend

```bash
# Ctrl+C in the uvicorn terminal, or:
docker compose stop backend
```

---

## Step 3 — Start the Phase 3 backend

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000

# You should now see:
# INFO  Starting Music Recommender API (Phase 3)...
# INFO  Database tables ensured
# INFO  Kafka producer/consumer started
# INFO  MetricsCollector started (interval=60s)
# INFO  Application startup complete
```

Confirm version 3.0.0:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"3.0.0"}
```

---

## Step 4 — Start Prometheus and Grafana

```bash
# From project root
docker compose up -d prometheus grafana

# Verify Prometheus is scraping
curl http://localhost:9090/api/v1/targets | python -m json.tool | grep -A3 '"health"'
# "health": "up"

# Grafana is at http://localhost:3001
# Login: admin / music123
```

Confirm metrics are flowing:
```bash
curl -s http://localhost:8000/metrics | grep music_
# music_recommender_requests_total{...} 0.0
# music_catalog_songs_total 0.0   ← will populate after first metric collection cycle
```

---

## Step 5 — Verify Grafana dashboards

1. Open http://localhost:3001
2. Login: `admin` / `music123`
3. Go to **Dashboards → Music Recommender**
4. Three dashboards should appear:
   - **Recommendation Quality** — request rates, latency, CTR
   - **Drift Detection** — ADWIN windows, KS statistics
   - **Data Quality & Bandit Health** — catalog completeness, exploration rate

Most panels will show "No data" until traffic flows — that's expected.

---

## Step 6 — Verify monitoring API endpoints

```bash
# Drift summary
curl http://localhost:8000/api/monitoring/drift | python -m json.tool
# {
#   "total_observations": 0,
#   "last_drift_at": null,
#   "reward_adwin": {"window_size": 0, "mean": 0.0, "n_detections": 0, ...},
#   "feature_drift": {"is_reference_ready": false, "observations": 0, ...},
#   "recent_drift_events": []
# }

# Feature-level KS results (empty until reference window fills)
curl http://localhost:8000/api/monitoring/drift/features
# []

# Data quality report
curl http://localhost:8000/api/monitoring/quality | python -m json.tool
# {
#   "total_songs": 1032,
#   "songs_with_all_features": 980,
#   "completeness_pct": 94.9,
#   "feature_breakdown": {...},
#   "checked_at": "..."
# }

# Interaction statistics
curl "http://localhost:8000/api/monitoring/quality/interactions?hours=24" | python -m json.tool
```

---

## Step 7 — Generate traffic to trigger drift monitoring

The KS reference window needs 500 observations before it starts testing.
The ADWIN window activates immediately.

Generate synthetic traffic:
```bash
# Get 5 song IDs from recommendations
SONG_IDS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=5&bypass_cache=true" \
  | python -c "
import sys, json
songs = json.load(sys.stdin)['songs']
print(' '.join(s['id'] for s in songs))
")
echo "Songs: $SONG_IDS"

# Send 100 mixed feedback events
for i in $(seq 1 100); do
  SONG=$(echo $SONG_IDS | tr ' ' '\n' | shuf -n1)
  ACTION=$(python -c "import random; print(random.choice(['add','skip','play']))")
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"$ACTION\"}" \
    > /dev/null
done
echo "100 feedback events sent"

# Check ADWIN window has grown
curl http://localhost:8000/api/monitoring/drift/adwin | python -m json.tool
# reward.window_size should now be ~100
```

---

## Step 8 — Observe drift detection in action

To trigger an artificial drift (for testing), inject a sudden behaviour change:

```bash
# Phase 1: 200 skips (negative signal)
SONG=$(echo $SONG_IDS | awk '{print $1}')
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"skip\"}" \
    > /dev/null
done
echo "200 skips sent (negative phase)"

# Phase 2: 200 adds (positive signal — big shift)
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"add\"}" \
    > /dev/null
done
echo "200 adds sent (positive phase)"

# Check drift detections
curl http://localhost:8000/api/monitoring/drift | python -m json.tool
# reward_adwin.n_detections should be > 0
# recent_drift_events should be non-empty
```

Watch the backend logs during this:
```
WARNING  ADWIN drift detected: mean=0.7342 n_obs=347 n_detections=1
```

And in Grafana, the **Drift Detection** dashboard will show:
- ADWIN window size dropping (old data evicted)
- ADWIN detections counter incrementing
- CTR signal shifting

---

## Step 9 — Data quality validation in action

The validator runs on every Kafka message. Test it explicitly:

```bash
# Valid feedback — should pass
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"demo-user-01","song_id":"'$SONG'","action":"add"}'

# Watch the backend log for:
# INFO  Feedback: user=demo-user-01 song=... action=add reward=1.0 kafka=ok

# Trigger quality check manually
curl -X POST http://localhost:8000/api/monitoring/quality/check | python -m json.tool
```

Prometheus metrics to verify validation is working:
```bash
curl -s http://localhost:8000/metrics | grep music_data_quality
# music_data_quality_null_features_total{feature="tempo"} 0.0
# music_data_quality_validation_failures_total{check="..."} 0.0
```

---

## Step 10 — Reset drift detectors after model update

After investigating and addressing drift (e.g. reseeding songs, resetting bandit):

```bash
# Reset ADWIN detectors
curl -X POST "http://localhost:8000/api/monitoring/drift/reset?signal=all"
# {"status": "reset", "signal": "all", ...}

# Reset bandit arms for a user (if warranted)
curl -X POST http://localhost:8000/api/bandit/demo-user-01/reset
```

---

## Step 11 — Run Phase 3 tests

```bash
cd backend
pytest tests/test_monitoring.py -v

# Expected: 29 tests, all passing
# Key tests:
#   test_detects_step_change                    PASSED  ← ADWIN core
#   test_gradual_drift_detected_eventually      PASSED
#   test_detects_distribution_shift             PASSED  ← KS core
#   test_no_drift_on_same_distribution          PASSED  ← false positive check
#   test_null_feature_imputed                   PASSED
#   test_out_of_range_clamped                   PASSED
```

Full suite (all phases):
```bash
pytest tests/ -v
# Expected: ~57 tests passing
```

---

## Step 12 — Full Docker deployment (Phase 3)

```bash
docker compose up --build

# All services:
#   postgres   :5432
#   redis      :6379
#   zookeeper  :2181
#   kafka      :9092
#   kafka-init (one-shot)
#   backend    :8000
#   frontend   :3000
#   prometheus :9090
#   grafana    :3001
```

---

## Prometheus Queries Reference

Useful PromQL queries for Grafana or ad-hoc investigation:

```promql
# Recommendation request rate (per minute)
rate(music_recommender_requests_total[1m]) * 60

# p95 recommendation latency
histogram_quantile(0.95, rate(music_recommender_latency_ms_bucket[5m]))

# Click-through rate
rate(music_feedback_events_total{action="add"}[5m])
/ rate(music_feedback_events_total[5m])

# Drift detection events per hour
increase(music_drift_adwin_detections_total[1h])

# Features with drift (KS p-value < 0.05)
music_drift_ks_p_value < 0.05

# Bandit exploration rate (should drop over time)
music_bandit_exploration_rate

# Null feature rate (should be near 0)
rate(music_data_quality_null_features_total[5m]) * 60

# Feedback processing p95 latency
histogram_quantile(0.95, rate(music_feedback_processing_latency_ms_bucket[5m]))
```

---

## Port Reference (complete, Phases 1–3)

| Service     | Port | URL                              |
|-------------|------|----------------------------------|
| Frontend    | 3000 | http://localhost:3000            |
| Backend API | 8000 | http://localhost:8000            |
| API Docs    | 8000 | http://localhost:8000/docs       |
| Metrics     | 8000 | http://localhost:8000/metrics    |
| Kafka       | 9092 | localhost:9092                   |
| Postgres    | 5432 | localhost:5432                   |
| Redis       | 6379 | localhost:6379                   |
| Zookeeper   | 2181 | localhost:2181                   |
| Prometheus  | 9090 | http://localhost:9090            |
| Grafana     | 3001 | http://localhost:3001            |

---

## Troubleshooting

**"Module 'scipy' not found"**
```bash
source backend/.venv/bin/activate
pip install scipy==1.13.0
```

**KS results always empty**
- The reference window needs 500 observations before it activates.
- Check: `curl http://localhost:8000/api/monitoring/drift | python -m json.tool`
- Look at `feature_drift.observations` — must reach 500.

**Grafana shows "No data" on all panels**
- Verify Prometheus is scraping: http://localhost:9090/targets
- Target should be `UP`. If `DOWN`, check backend is on port 8000.
- Wait 2-3 scrape cycles (30-45 seconds) after startup.

**ADWIN never detects drift even with big shifts**
- The default `delta=0.002` is conservative. For testing, reduce it:
  - Edit `drift_pipeline.py`, change `ADWINDetector(delta=0.002)` to `delta=0.05`
  - Restart backend

**Prometheus scrape target shows wrong host**
- Inside Docker compose, backend is on `backend:8000`, not `localhost:8000`
- The `prometheus.yml` already uses `backend:8000` for the docker target

**MetricsCollector error on startup**
- This is usually a missing DB table or connection issue
- Check: `curl http://localhost:8000/api/health`
- DB must be healthy before collector runs
