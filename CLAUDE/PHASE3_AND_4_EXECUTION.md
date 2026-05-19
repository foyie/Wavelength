# Execution Guide — Phase 3 & Phase 4

This guide assumes Phase 1 and Phase 2 are already running and healthy.
Work through each phase in order — Phase 4 picks up exactly where Phase 3 leaves off.

---

## Before You Start — Verify Phase 2 Is Healthy

```bash
curl http://localhost:8000/api/health
```

Expected output:
```json
{"status":"healthy","db":"healthy","redis":"healthy","version":"2.0.0"}
```

If you see `version: 1.0.0`, you are still on Phase 1 — complete Phase 2 first.
If `db` or `redis` shows `unhealthy`, run:

```bash
# From project root
docker compose up -d postgres redis
# Wait 10 seconds, then retry the health check
```

---

---

# PHASE 3 — Drift Detection, Data Quality, Prometheus, Grafana

**What gets added:**
- ADWIN drift detector on the live reward stream
- Kolmogorov-Smirnov test on audio feature distributions
- Data quality validator on every Kafka message
- 30+ Prometheus metrics exposed at `/metrics`
- 3 auto-provisioned Grafana dashboards
- `GET/POST /api/monitoring/*` REST endpoints

**Infrastructure changes:** Prometheus (`:9090`) and Grafana (`:3001`) added to docker-compose.
No schema migrations needed.

---

## Phase 3 — Step 1: Install scipy

Phase 3 adds one new Python dependency for the KS test.

```bash
cd backend
source .venv/bin/activate
pip install scipy==1.13.0

# Verify
python -c "from scipy import stats; print('scipy OK')"
```

---

## Phase 3 — Step 2: Stop the backend

```bash
# If running locally with uvicorn:
# Press Ctrl+C in the uvicorn terminal

# If running in Docker:
docker compose stop backend
```

---

## Phase 3 — Step 3: Start Kafka (if not already running)

Phase 3 still uses Kafka from Phase 2. If it is already up, skip this step.

```bash
# From project root
docker compose up -d zookeeper kafka

# Wait until Kafka reports healthy (~30 seconds)
docker compose ps kafka
# kafka   Up (healthy)   ← need this before proceeding
```

If you see `Up` but not `(healthy)`, wait another 20 seconds and check again.
Kafka is slow to initialise on first boot.

---

## Phase 3 — Step 4: Start the Phase 3 backend

```bash
# From the backend/ directory with .venv active
cd backend
source .venv/bin/activate

KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000
```

**Expected startup log — verify all four lines appear:**
```
INFO  Starting Music Recommender API (Phase 3)...
INFO  Database tables ensured
INFO  Kafka producer connected: localhost:9092
INFO  Kafka consumer started: group=bandit-updater topic=music.feedback
INFO  MetricsCollector started (interval=60s)
INFO  Application startup complete
```

If `Kafka producer connected` does not appear, Kafka is not reachable.
Check Step 3 and ensure the topic exists:
```bash
docker compose exec kafka \
  kafka-topics --bootstrap-server localhost:9092 --list
# music.feedback   ← must be present
```

**Confirm version 3.0.0:**
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"3.0.0"}
```

---

## Phase 3 — Step 5: Start Prometheus and Grafana

```bash
# From project root
docker compose up -d prometheus grafana

# Verify both are running
docker compose ps prometheus grafana
# prometheus   Up
# grafana      Up
```

**Check Prometheus is scraping the backend:**
```bash
curl -s http://localhost:9090/api/v1/targets \
  | python -c "
import sys, json
targets = json.load(sys.stdin)['data']['activeTargets']
for t in targets:
    print(t['labels'].get('job'), '->', t['health'])
"
# music-recommender-api -> up   ← must be 'up'
```

If the target shows `down`, the most common cause is Docker networking —
Prometheus inside Docker cannot reach `localhost:8000` on your host.
Confirm the `prometheus.yml` uses `backend:8000` (not `localhost:8000`) as the target.
This is already correct in the generated file; only matters if you edited it.

**Verify metrics are being exposed:**
```bash
curl -s http://localhost:8000/metrics | grep "^music_" | head -10
# music_recommender_requests_total{...} 0.0
# music_feedback_events_total{...} 0.0
# music_bandit_arm_updates_total 0.0
```

---

## Phase 3 — Step 6: Open Grafana and verify dashboards load

1. Open **http://localhost:3001**
2. Login: `admin` / `music123`
3. Click **Dashboards** in the left sidebar
4. Open the **Music Recommender** folder

Three dashboards should be present:

| Dashboard | What it shows |
|-----------|--------------|
| Recommendation Quality | Request rate, latency p50/p95/p99, CTR, reward distribution |
| Drift Detection | ADWIN window size, KS statistics per feature, detection counts |
| Data Quality & Bandit Health | Catalog completeness, exploration rate, null feature rate |

All panels will show **No data** at this point — that is expected.
Data appears after traffic flows in the next steps.

---

## Phase 3 — Step 7: Run Phase 3 tests

```bash
cd backend
source .venv/bin/activate
pytest tests/test_monitoring.py -v
```

**Expected: 29 tests passing.** Key ones to watch:

```
test_no_drift_on_stationary_stream          PASSED  ← false positive check
test_detects_step_change                    PASSED  ← ADWIN core
test_gradual_drift_detected_eventually      PASSED  ← ADWIN sensitivity
test_no_drift_on_same_distribution          PASSED  ← KS false positive check
test_detects_distribution_shift             PASSED  ← KS core
test_null_feature_imputed                   PASSED  ← data quality
test_out_of_range_clamped                   PASSED  ← data quality
test_summary_structure                      PASSED  ← drift pipeline
```

If `test_detects_step_change` fails, scipy may not be installed correctly:
```bash
pip install scipy==1.13.0 --force-reinstall
```

---

## Phase 3 — Step 8: Verify monitoring API endpoints

With the backend running, test all new endpoints:

```bash
# 1. Full drift summary
curl http://localhost:8000/api/monitoring/drift | python -m json.tool
```

Expected output shape:
```json
{
  "total_observations": 0,
  "last_drift_at": null,
  "reward_adwin": {
    "window_size": 0,
    "mean": 0.0,
    "n_detections": 0,
    "elements_seen": 0
  },
  "ctr_adwin": {"window_size": 0, "mean": 0.0, "n_detections": 0},
  "feature_drift": {
    "is_reference_ready": false,
    "observations": 0,
    "drift_counts": {"danceability": 0, "energy": 0, "valence": 0, ...}
  },
  "recent_drift_events": []
}
```

```bash
# 2. ADWIN-specific state
curl http://localhost:8000/api/monitoring/drift/adwin | python -m json.tool

# 3. KS test results per feature (empty until 500 observations collected)
curl http://localhost:8000/api/monitoring/drift/features
# []

# 4. Catalog quality report
curl http://localhost:8000/api/monitoring/quality | python -m json.tool
```

Expected quality output:
```json
{
  "total_songs": 1032,
  "songs_with_all_features": 987,
  "completeness_pct": 95.6,
  "feature_breakdown": {
    "danceability": {"present": 1032, "missing": 0, "completeness_pct": 100.0},
    "energy":       {"present": 1032, "missing": 0, "completeness_pct": 100.0},
    ...
  }
}
```

```bash
# 5. Interaction statistics (last 24 hours)
curl "http://localhost:8000/api/monitoring/quality/interactions?hours=24" | python -m json.tool
```

---

## Phase 3 — Step 9: Generate traffic and watch ADWIN activate

ADWIN activates immediately on the first observation.
The KS reference window needs **500 observations** before it starts testing.

**Generate 100 feedback events across multiple users:**
```bash
# Get song IDs to use
SONG_IDS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=5&bypass_cache=true" \
  | python -c "
import sys, json
songs = json.load(sys.stdin)['songs']
print(' '.join(s['id'] for s in songs))
")
echo "Using songs: $SONG_IDS"

# Send 100 mixed feedback events
for i in $(seq 1 100); do
  SONG=$(echo $SONG_IDS | tr ' ' '\n' | shuf -n1)
  ACTION=$(python3 -c "import random; print(random.choice(['add','skip','play']))")
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"$ACTION\"}" \
    > /dev/null
done
echo "Done — 100 events sent"
```

**Check ADWIN has consumed the events:**
```bash
curl http://localhost:8000/api/monitoring/drift/adwin | python -m json.tool
# reward.window_size should now be ~100
# reward.mean should reflect the mix of rewards from add/skip/play
```

---

## Phase 3 — Step 10: Trigger a detectable drift

This simulates what happens when user behaviour shifts — for example, after
a catalog change or a seasonal preference shift.

```bash
SONG=$(echo $SONG_IDS | awk '{print $1}')

# Phase A: 200 skips — establish a low-reward baseline
echo "Sending 200 skips..."
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"skip\"}" \
    > /dev/null
done

# Phase B: 200 adds — abrupt shift to positive reward
echo "Sending 200 adds..."
for i in $(seq 1 200); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"add\"}" \
    > /dev/null
done

echo "Done — checking drift state..."
curl http://localhost:8000/api/monitoring/drift | python -m json.tool
```

**What to look for in the output:**
- `reward_adwin.n_detections` > 0
- `reward_adwin.window_size` smaller than total events sent (old data evicted)
- `recent_drift_events` contains entries with `"detector": "adwin_reward"`

**In the backend logs you should see:**
```
WARNING  ADWIN drift detected: mean=0.7123 n_obs=312 n_detections=1
```

**In Grafana (Drift Detection dashboard):**
- ADWIN Window Size panel shows a visible drop (the eviction event)
- ADWIN detections counter increments

---

## Phase 3 — Step 11: Fill the KS reference window

The KS test needs 500 observations. Send them across multiple users so the
feature distribution is varied:

```bash
echo "Sending 500 events to fill KS reference window..."
for i in $(seq 1 500); do
  USER="user-$(( i % 20 ))"
  SONG=$(echo $SONG_IDS | tr ' ' '\n' | shuf -n1)
  ACTION=$(python3 -c "import random; print(random.choice(['add','skip','play','complete']))")
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$USER\",\"song_id\":\"$SONG\",\"action\":\"$ACTION\"}" \
    > /dev/null
  if [ $(( i % 100 )) -eq 0 ]; then
    echo "  $i / 500 sent"
  fi
done

# Check KS reference is now ready
curl http://localhost:8000/api/monitoring/drift \
  | python -c "
import sys, json
d = json.load(sys.stdin)
fd = d['feature_drift']
print('Reference ready:', fd['is_reference_ready'])
print('Observations:', fd['observations'])
"
# Reference ready: True
# Observations: 500+
```

Now check KS results:
```bash
curl http://localhost:8000/api/monitoring/drift/features | python -m json.tool
# Returns list of per-feature KS results
# drifted: false for all (same distribution — expected)
```

---

## Phase 3 — Step 12: Trigger quality check and reset detectors

```bash
# Manually trigger a catalog quality scan
curl -X POST http://localhost:8000/api/monitoring/quality/check | python -m json.tool

# After a model update or catalog refresh, reset ADWIN
curl -X POST "http://localhost:8000/api/monitoring/drift/reset?signal=all"
# {"status": "reset", "signal": "all", ...}

# Optionally reset a user's bandit arms if drift was severe
curl -X POST http://localhost:8000/api/bandit/demo-user-01/reset
# {"status": "reset", "user_id": "demo-user-01", "arms_reset": N}
```

---

## Phase 3 — Step 13: Grafana live data check

After completing Steps 9–11 you should have enough traffic to see data in Grafana.

1. Open **http://localhost:3001** → Dashboards → Music Recommender
2. **Recommendation Quality** dashboard:
   - "Feedback Events by Action" should show bars for add/skip/play
   - "Active Users (24h)" should show > 0
3. **Drift Detection** dashboard:
   - "ADWIN Window Size" should show a time-series with values
   - If you completed Step 10, you'll see a visible dip (drift eviction)
4. **Data Quality & Bandit Health**:
   - "Catalog: Total Songs" should show your seeded count (~1000)
   - "Bandit: Exploration Rate" should be near 1.0 (all arms unexplored)

**Useful ad-hoc PromQL queries to run in Grafana Explore:**
```promql
# Total feedback events received
music_feedback_events_total

# Current ADWIN reward mean
music_drift_adwin_current_mean{signal="reward"}

# KS p-value per feature (after reference window fills)
music_drift_ks_p_value

# Catalog completeness
music_catalog_songs_with_features / music_catalog_songs_total * 100
```

---

## Phase 3 — Step 14: Full Docker deployment (optional)

To run everything containerised instead of mixing local/Docker:

```bash
# From project root — builds and starts all 9 services
docker compose up --build

# Tail Phase 3 backend logs
docker compose logs -f backend | grep -E "INFO|WARNING|ERROR"

# Run seed script inside the container (if not done in Phase 1)
docker compose exec backend python -m scripts.load_spotify_data
```

All services and ports:
```
postgres    localhost:5432
redis       localhost:6379
zookeeper   localhost:2181
kafka       localhost:9092
backend     localhost:8000   (API + /metrics)
frontend    localhost:3000
prometheus  localhost:9090
grafana     localhost:3001
```

---

## Phase 3 Troubleshooting

**`ImportError: No module named 'scipy'`**
```bash
source backend/.venv/bin/activate
pip install scipy==1.13.0
```

**KS results always return `[]`**
The reference window is not yet full. Check observation count:
```bash
curl http://localhost:8000/api/monitoring/drift \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['feature_drift']['observations'])"
# Must reach 500 before KS tests run
```
Run Step 11 to fill it.

**ADWIN never fires even with obvious shift**
The default delta (`0.002`) is conservative. To test more aggressively:
```python
# In backend/app/services/monitoring/drift_pipeline.py, change:
self._reward_adwin = ADWINDetector(delta=0.002)
# to:
self._reward_adwin = ADWINDetector(delta=0.05)
```
Restart the backend, then repeat Step 10.

**Prometheus target shows `DOWN`**
```bash
# Check Prometheus config is using the right host
cat monitoring/prometheus/prometheus.yml | grep targets
# For Docker: should be backend:8000
# For local dev: should be host.docker.internal:8000 or localhost:8000 depending on setup
```
When running Prometheus in Docker and backend locally, use:
```yaml
static_configs:
  - targets: ["host.docker.internal:8000"]
```

**Grafana login fails**
Default credentials are `admin` / `music123` (set in docker-compose.yml).
If changed, reset with:
```bash
docker compose exec grafana grafana-cli admin reset-admin-password music123
```

**MetricsCollector throws on startup**
Usually a DB connection issue. The collector queries the `songs` and `bandit_states`
tables — both must exist. Run:
```bash
curl http://localhost:8000/api/health
# If db shows unhealthy, postgres is not reachable
docker compose up -d postgres
```

---

---

# PHASE 4 — SPRT A/B Testing + SHAP Explanations

**What gets added:**
- SPRT (Sequential Probability Ratio Test) A/B testing engine
- Experiment registry with Redis persistence across restarts
- Per-song SHAP feature attribution (leave-one-out perturbation)
- Automatic experiment observation routing from every feedback event
- `GET/POST /api/experiments/*` REST endpoints
- `GET/POST /api/explain/*` REST endpoints
- `explain=true` and `experiment_id` query params on `/api/recommend`

**Infrastructure changes:** None. Same Postgres, Redis, Kafka, Prometheus, Grafana.
Two new DB tables (`experiment_assignments`, `experiment_observations`) created
automatically on startup via `create_tables()`.

**New pip dependencies:** None. Everything needed was already installed in Phase 3.

---

## Phase 4 — Step 1: Stop the backend

```bash
# Ctrl+C in the uvicorn terminal, or:
docker compose stop backend
```

---

## Phase 4 — Step 2: Start the Phase 4 backend

```bash
cd backend
source .venv/bin/activate

KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000
```

**Expected startup log — verify all five lines appear:**
```
INFO  Starting Music Recommender API (Phase 4)...
INFO  Database tables ensured
INFO  Kafka producer connected: localhost:9092
INFO  Kafka consumer started: group=bandit-updater topic=music.feedback
INFO  Metrics collector started
INFO  Experiment registry loaded
INFO  Application startup complete
```

The `Experiment registry loaded` line is new in Phase 4 — it loads any
experiments persisted in Redis from previous runs.

**Confirm version 4.0.0:**
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"4.0.0"}
```

**Confirm new routes exist:**
```bash
curl -s http://localhost:8000/openapi.json \
  | python -c "
import sys, json
paths = list(json.load(sys.stdin)['paths'].keys())
new = [p for p in paths if 'experiment' in p or 'explain' in p]
print('\n'.join(sorted(new)))
"
# /api/experiments
# /api/experiments/running
# /api/experiments/{experiment_id}
# /api/experiments/{experiment_id}/arm/{user_id}
# /api/experiments/{experiment_id}/conclude
# /api/experiments/{experiment_id}/observe
# /api/explain/{user_id}/batch
# /api/explain/{user_id}/profile
# /api/explain/{user_id}/{song_id}
```

---

## Phase 4 — Step 3: Run Phase 4 tests

```bash
cd backend
source .venv/bin/activate
pytest tests/test_phase4.py -v
```

**Expected: 32 tests passing.** Key ones:

```
test_upper_boundary_formula                  PASSED  ← SPRT math
test_lower_boundary_formula                  PASSED
test_treatment_wins_with_strong_signal       PASSED  ← SPRT convergence
test_control_wins_when_treatment_worse       PASSED
test_max_samples_triggers_inconclusive       PASSED  ← safety cap
test_decided_experiment_ignores_new_obs      PASSED  ← immutability
test_arm_assignment_deterministic            PASSED  ← same user, same arm
test_arm_assignment_splits_roughly_50_50     PASSED  ← hash distribution
test_serialise_deserialise_roundtrip         PASSED  ← Redis persistence
test_returns_shap_result                     PASSED  ← SHAP output
test_feature_contributions_has_all_keys      PASSED
test_high_energy_song_shows_energy_contribution PASSED
test_null_song_features_handled              PASSED  ← edge case
test_cold_start_explanation_text             PASSED
```

**Full suite (all four phases):**
```bash
pytest tests/ -v
# Expected: ~89 tests passing
```

---

## Phase 4 — Step 4: Verify empty experiment state

```bash
# Should return empty list on a fresh start
curl http://localhost:8000/api/experiments | python -m json.tool
# {"total": 0, "experiments": []}

curl http://localhost:8000/api/experiments/running | python -m json.tool
# {"total": 0, "experiments": []}
```

---

## Phase 4 — Step 5: Create your first experiment

Create an experiment to test whether showing audio-feature explanations
increases the add-to-playlist rate:

```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "id": "shap-ctr-test",
    "name": "SHAP Explanation CTR Test",
    "description": "Does showing why a song was recommended increase adds?",
    "control_name": "no-explanation",
    "treatment_name": "with-explanation",
    "alpha": 0.05,
    "beta": 0.20,
    "delta": 0.02,
    "max_samples_per_arm": 5000,
    "traffic_split_pct": 50
  }' | python -m json.tool
```

Expected response:
```json
{
  "id": "shap-ctr-test",
  "name": "SHAP Explanation CTR Test",
  "status": "running",
  "winner": null,
  "created_at": "2025-...",
  "decided_at": null,
  "n_updates": 0,
  "sprt": {
    "log_lambda": 0.0,
    "upper_boundary": 2.7726,
    "lower_boundary": -2.0794,
    "alpha": 0.05,
    "beta": 0.2,
    "delta": 0.02,
    "progress_pct": 0.0
  },
  "control":   {"name": "no-explanation", "n_observations": 0, "mean_reward": 0.0},
  "treatment": {"name": "with-explanation", "n_observations": 0, "mean_reward": 0.0},
  "lift": 0.0,
  "lift_pct": 0.0
}
```

**Understanding the SPRT boundaries:**
- `upper_boundary = 2.7726` — cross this and treatment wins
- `lower_boundary = -2.0794` — cross this and control wins
- `log_lambda` starts at 0 and moves toward one of the two boundaries

---

## Phase 4 — Step 6: Verify arm assignment

```bash
# Check arm for demo-user-01
curl http://localhost:8000/api/experiments/shap-ctr-test/arm/demo-user-01
# {"user_id":"demo-user-01","experiment_id":"shap-ctr-test","arm":"treatment"}

# The assignment is deterministic — call it 5 times and get the same answer
for i in $(seq 1 5); do
  curl -s http://localhost:8000/api/experiments/shap-ctr-test/arm/demo-user-01 \
    | python -c "import sys,json; print(json.load(sys.stdin)['arm'])"
done
# treatment
# treatment
# treatment
# treatment
# treatment   ← always the same

# Check that different users split roughly 50/50
TREATMENT=0; CONTROL=0
for i in $(seq 1 100); do
  ARM=$(curl -s "http://localhost:8000/api/experiments/shap-ctr-test/arm/user-$i" \
    | python -c "import sys,json; print(json.load(sys.stdin)['arm'])")
  if [ "$ARM" = "treatment" ]; then TREATMENT=$((TREATMENT+1))
  else CONTROL=$((CONTROL+1)); fi
done
echo "Treatment: $TREATMENT / Control: $CONTROL (out of 100)"
# Treatment: 52 / Control: 48   ← approximately 50/50
```

---

## Phase 4 — Step 7: Test SHAP explanations on recommendations

**Without explanations (default behaviour, fast path):**
```bash
curl "http://localhost:8000/api/recommend/demo-user-01?limit=3&bypass_cache=true" \
  | python -m json.tool | grep -A1 '"explanation"'
# "explanation": "New to your taste — exploring for you · 50% audio match"
# No "shap" field present
```

**With SHAP feature attributions (`explain=true`):**
```bash
curl "http://localhost:8000/api/recommend/demo-user-01?limit=3&bypass_cache=true&explain=true" \
  | python -m json.tool
```

Each song in the response now includes a `shap` block:
```json
{
  "name": "As It Was",
  "artist": "Harry Styles",
  "score": 0.6142,
  "rank": 1,
  "explanation": "Matches your energy · based on 0 interactions",
  "shap": {
    "feature_contributions": {
      "danceability": 0.0412,
      "energy":       0.0891,
      "valence":      0.0203,
      "tempo_norm":  -0.0091,
      "acousticness":-0.0156,
      "bandit_prior": 0.3250
    },
    "dominant_feature": "energy",
    "score_breakdown": {
      "context_component": 0.1678,
      "bandit_component":  0.3250,
      "total":             0.6142
    }
  }
}
```

**Reading SHAP values:**
- Positive value → this feature pushed the score up
- Negative value → this feature pulled the score down
- `bandit_prior` → contribution from the learned Beta posterior (experience)
- `dominant_feature` → the single largest absolute contributor

---

## Phase 4 — Step 8: Test the on-demand explanation endpoints

```bash
# Get a song ID to explain
SONG_ID=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=1&bypass_cache=true" \
  | python -c "import sys,json; print(json.load(sys.stdin)['songs'][0]['id'])")

echo "Explaining song: $SONG_ID"
```

**Single-song explanation:**
```bash
curl "http://localhost:8000/api/explain/demo-user-01/$SONG_ID" | python -m json.tool
```

Expected response:
```json
{
  "user_id": "demo-user-01",
  "song": {"id": "...", "name": "...", "artist": "..."},
  "explanation_text": "Matches your energy and danceability",
  "dominant_feature": "energy",
  "dominant_feature_label": "Energy",
  "feature_contributions": {
    "Energy": 0.0891,
    "Danceability": 0.0412,
    "Mood/Positivity": 0.0203,
    "Tempo": -0.0091,
    "Acoustic Feel": -0.0156,
    "Learned Preference": 0.3250
  },
  "score_breakdown": {"context_component": 0.168, "bandit_component": 0.325, "total": 0.614},
  "bandit_info": {
    "alpha": 1.0,
    "beta": 1.0,
    "mean_reward": 0.5,
    "n_observations": 0,
    "uncertainty": 0.25
  }
}
```

**User taste profile:**
```bash
curl "http://localhost:8000/api/explain/demo-user-01/profile" | python -m json.tool
```

Expected:
```json
{
  "user_id": "demo-user-01",
  "total_interactions": 0,
  "taste_summary": "Balanced taste across features — still learning your preferences",
  "preference_vector": {
    "danceability": 0.5, "energy": 0.5, "valence": 0.5,
    "tempo": 120.0, "acousticness": 0.5
  },
  "vs_population_avg": {
    "danceability": -0.08, "energy": -0.12, "valence": 0.02,
    "tempo": 0.0, "acousticness": 0.25
  }
}
```

After sending some feedback, `taste_summary` will update to reflect learned preferences.

**Batch explanation (up to 50 songs):**
```bash
# Get 5 song IDs
SONG_IDS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=5&bypass_cache=true" \
  | python -c "
import sys, json
ids = [s['id'] for s in json.load(sys.stdin)['songs']]
print(json.dumps(ids))
")

curl -X POST "http://localhost:8000/api/explain/demo-user-01/batch" \
  -H "Content-Type: application/json" \
  -d "{\"song_ids\": $SONG_IDS}" | python -m json.tool
```

---

## Phase 4 — Step 9: Send feedback and watch experiment accumulate observations

Every feedback event automatically routes to all running experiments.

```bash
# Get songs
SONGS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=10&bypass_cache=true" \
  | python -c "
import sys, json
print(' '.join(s['id'] for s in json.load(sys.stdin)['songs']))
")

# Send 200 feedback events across 50 different users
echo "Sending 200 feedback events..."
for i in $(seq 1 200); do
  SONG=$(echo $SONGS | tr ' ' '\n' | shuf -n1)
  USER="user-$(( i % 50 ))"
  ACTION=$(python3 -c "import random; print(random.choice(['add','skip','play']))")

  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$USER\",\"song_id\":\"$SONG\",\"action\":\"$ACTION\"}" \
    > /dev/null

  if [ $(( i % 50 )) -eq 0 ]; then
    LAMBDA=$(curl -s http://localhost:8000/api/experiments/shap-ctr-test \
      | python -c "import sys,json; d=json.load(sys.stdin); print(f\"lambda={d['sprt']['log_lambda']:.3f} progress={d['sprt']['progress_pct']}%\")")
    echo "  [$i events] $LAMBDA"
  fi
done
```

**Check experiment state after traffic:**
```bash
curl http://localhost:8000/api/experiments/shap-ctr-test | python -m json.tool
```

You will see:
- `n_updates` > 0
- `control.n_observations` and `treatment.n_observations` both populated
- `sprt.log_lambda` has moved away from 0.0
- `sprt.progress_pct` > 0

---

## Phase 4 — Step 10: Watch SPRT converge to a decision

To see SPRT converge quickly, send strong positive signals so treatment clearly wins.
In production this happens organically over days; here we accelerate it.

```bash
SONG=$(echo $SONGS | awk '{print $1}')

echo "Sending 1000 'add' events to force SPRT convergence..."
for i in $(seq 1 1000); do
  USER="user-$(( i % 100 ))"

  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"$USER\",\"song_id\":\"$SONG\",\"action\":\"add\"}" \
    > /dev/null

  # Check status every 100 events
  if [ $(( i % 100 )) -eq 0 ]; then
    STATUS=$(curl -s http://localhost:8000/api/experiments/shap-ctr-test \
      | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"status={d['status']} lambda={d['sprt']['log_lambda']:.3f} boundary={d['sprt']['upper_boundary']:.3f}\")
")
    echo "  [$i events] $STATUS"
    # Stop checking once decided
    if echo "$STATUS" | grep -q "treatment_wins\|control_wins\|inconclusive"; then
      echo "  EXPERIMENT DECIDED — stopping early"
      break
    fi
  fi
done

# Final experiment result
echo ""
echo "=== EXPERIMENT RESULT ==="
curl http://localhost:8000/api/experiments/shap-ctr-test \
  | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Status: {d['status']}\")
print(f\"Winner: {d['winner']}\")
print(f\"Decided at: {d['decided_at']}\")
print(f\"Control mean reward:   {d['control']['mean_reward']}\")
print(f\"Treatment mean reward: {d['treatment']['mean_reward']}\")
print(f\"Lift: {d['lift_pct']}%\")
print(f\"Total observations: {d['n_updates']}\")
"
```

---

## Phase 4 — Step 11: Create a second experiment (algorithm comparison)

The most production-relevant use case — compare bandit algorithm variants:

```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "id": "context-weight-test",
    "name": "Context Weight Comparison",
    "description": "Test whether higher audio-feature weighting (0.55) beats current (0.35)",
    "control_name": "context-0.35",
    "treatment_name": "context-0.55",
    "alpha": 0.05,
    "beta": 0.20,
    "delta": 0.03,
    "max_samples_per_arm": 10000,
    "traffic_split_pct": 50
  }' | python -m json.tool

# Confirm both experiments running
curl http://localhost:8000/api/experiments/running \
  | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"{d['total']} running experiments:\")
for e in d['experiments']:
    print(f\"  {e['id']} — {e['status']} — {e['n_updates']} updates\")
"
```

---

## Phase 4 — Step 12: Force-conclude and clean up

```bash
# Force-conclude the second experiment (e.g., deadline hit)
curl -X POST "http://localhost:8000/api/experiments/context-weight-test/conclude" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Sprint ended — shipping without conclusion", "winner": null}' \
  | python -m json.tool

# Verify it is now inconclusive
curl http://localhost:8000/api/experiments/context-weight-test \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['status'], '/', d['decided_at'])"
# inconclusive / 2025-...

# Delete a concluded experiment to clean up
curl -X DELETE http://localhost:8000/api/experiments/context-weight-test
# HTTP 204 No Content — no response body

# Confirm deleted
curl http://localhost:8000/api/experiments | python -m json.tool
# total should now be 1 (only shap-ctr-test remains)
```

---

## Phase 4 — Step 13: Verify experiment persistence across restart

Experiments are stored in Redis and survive backend restarts.

```bash
# Stop the backend
# (Ctrl+C or docker compose stop backend)

# Start again
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000

# Look for this in the startup log:
# INFO  Experiment registry loaded   ← experiments restored from Redis

# Verify experiment survived
curl http://localhost:8000/api/experiments/shap-ctr-test \
  | python -c "
import sys, json
d = json.load(sys.stdin)
print(f\"Survived restart: status={d['status']} updates={d['n_updates']}\")
"
```

---

## Phase 4 — Step 14: Taste profile after interactions

After sending feedback, the user's preference vector has been updated via EMA.
The profile now reflects their actual taste:

```bash
# Send 20 adds for high-energy songs
SONG=$(echo $SONGS | awk '{print $1}')
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG\",\"action\":\"add\"}" > /dev/null
done

# Profile should now show updated taste
curl "http://localhost:8000/api/explain/demo-user-01/profile" \
  | python -c "
import sys, json
d = json.load(sys.stdin)
print('Taste summary:', d['taste_summary'])
print('Energy pref:', d['preference_vector']['energy'])
print('vs avg:', d['vs_population_avg']['energy'])
print('Total interactions:', d['total_interactions'])
"
# Taste summary: You tend to prefer high-energy music
# Energy pref: 0.71
# vs avg: 0.09
# Total interactions: 20
```

---

## Phase 4 Troubleshooting

**`"Experiment 'X' already exists"` on POST**
```bash
# Delete it first
curl -X DELETE http://localhost:8000/api/experiments/X
# Then recreate
```

**SPRT `log_lambda` stays at 0.0 after many events**
The SPRT lambda only updates when both arms have at least one observation.
With 50/50 split, this happens naturally. Check counts:
```bash
curl http://localhost:8000/api/experiments/shap-ctr-test \
  | python -c "
import sys,json; d=json.load(sys.stdin)
print('Control obs:', d['control']['n_observations'])
print('Treatment obs:', d['treatment']['n_observations'])
"
# Both should be > 0 after traffic
```
If one arm has 0 observations, try sending feedback with a variety of users
(different user IDs get different arms via the hash assignment).

**SHAP explanations all say "exploring" or "no listening history"**
The user has no interactions — bandit prior is `alpha=1, beta=1, n_obs=0`.
Send a few feedback events first:
```bash
SONG_ID=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=1" \
  | python -c "import sys,json; print(json.load(sys.stdin)['songs'][0]['id'])")
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG_ID\",\"action\":\"add\"}"
```

**Experiments not surviving restart**
Redis must be running and accessible. Verify:
```bash
redis-cli keys "experiment:*"
redis-cli smembers "experiments:index"
```
If empty after creating an experiment, Redis persistence is broken.
Check: `docker compose ps redis` — must be `Up`.

**`experiment_observations` table doesn't exist**
The new tables are created automatically by `create_tables()` on startup.
If you see this error the startup is failing silently.
Check full logs:
```bash
docker compose logs backend 2>&1 | grep -E "ERROR|error|Exception"
```

**`/api/explain/{user_id}/profile` returns 404**
The user does not exist in the DB yet. Create them first:
```bash
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"id":"demo-user-01","display_name":"Demo"}'
```

---

## Complete API Reference — Phases 3 and 4

### Phase 3: Monitoring

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/monitoring/drift` | Full ADWIN + KS drift summary |
| GET | `/api/monitoring/drift/adwin` | ADWIN state (reward + CTR signals) |
| GET | `/api/monitoring/drift/features` | KS test results per audio feature |
| POST | `/api/monitoring/drift/reset` | Reset ADWIN detectors post-update |
| GET | `/api/monitoring/quality` | Catalog feature completeness report |
| POST | `/api/monitoring/quality/check` | Trigger immediate quality scan |
| GET | `/api/monitoring/quality/interactions` | Interaction stats (last N hours) |

### Phase 4: Experiments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/experiments` | Create SPRT experiment |
| GET | `/api/experiments` | List all (filter by `?status=running`) |
| GET | `/api/experiments/running` | List running experiments only |
| GET | `/api/experiments/{id}` | Full SPRT state + DB audit counts |
| GET | `/api/experiments/{id}/arm/{user_id}` | Deterministic arm assignment |
| POST | `/api/experiments/{id}/observe` | Manual observation (for testing) |
| POST | `/api/experiments/{id}/conclude` | Force-conclude |
| DELETE | `/api/experiments/{id}` | Delete + remove from Redis |

### Phase 4: Explainability

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/explain/{user_id}/{song_id}` | SHAP breakdown for one song |
| POST | `/api/explain/{user_id}/batch` | Batch SHAP (≤50 songs) |
| GET | `/api/explain/{user_id}/profile` | User taste profile + taste summary |

### Updated: Recommendations

| Param | Default | Description |
|-------|---------|-------------|
| `explain` | `false` | Add `shap` block to each song (Phase 4) |
| `experiment_id` | `null` | Label recs with experiment arm (Phase 4) |
| `bypass_cache` | `false` | Force fresh bandit sample (Phases 2–4) |

---

## Port Reference — All Phases

| Service | Port | URL |
|---------|------|-----|
| Frontend | 3000 | http://localhost:3000 |
| Backend API | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |
| Prometheus metrics | 8000 | http://localhost:8000/metrics |
| Prometheus UI | 9090 | http://localhost:9090 |
| Grafana | 3001 | http://localhost:3001 (admin / music123) |
| Kafka | 9092 | localhost:9092 |
| Postgres | 5432 | localhost:5432 |
| Redis | 6379 | localhost:6379 |
| Zookeeper | 2181 | localhost:2181 |
