# Phase 4 Execution Guide — SPRT A/B Testing + SHAP Explanations

Builds on the running Phase 3 system. No infrastructure changes — same
Postgres, Redis, Kafka, Prometheus, Grafana. Two new service layers and
two new router groups added.

---

## Prerequisites

Phase 3 running and healthy:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","version":"3.0.0",...}
```

No new pip dependencies needed — scipy, numpy, scikit-learn already installed.

---

## Step 1 — Stop the backend

```bash
# Ctrl+C in uvicorn terminal, or:
docker compose stop backend
```

---

## Step 2 — Start Phase 4 backend

```bash
cd backend
source .venv/bin/activate

KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000

# You should see:
# INFO  Starting Music Recommender API (Phase 4)...
# INFO  Database tables ensured
# INFO  Kafka producer/consumer started
# INFO  Metrics collector started
# INFO  Experiment registry loaded
# INFO  Application startup complete
```

Confirm version 4.0.0:
```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"4.0.0"}
```

New routes in interactive docs: http://localhost:8000/docs
- `/api/experiments` — A/B experiment management
- `/api/explain` — SHAP feature attribution

---

## Step 3 — Run Phase 4 tests

```bash
pytest tests/test_phase4.py -v

# Expected: 32 tests passing
# Key tests:
#   test_upper_boundary_formula             PASSED
#   test_treatment_wins_with_strong_signal  PASSED  ← SPRT convergence
#   test_max_samples_triggers_inconclusive  PASSED
#   test_arm_assignment_splits_roughly_50_50 PASSED
#   test_serialise_deserialise_roundtrip    PASSED
#   test_returns_shap_result                PASSED
#   test_dominant_feature_is_in_contributions PASSED
#   test_high_energy_song_shows_energy_contribution PASSED

# Full suite (all phases):
pytest tests/ -v
# Expected: ~89 tests passing
```

---

## Step 4 — Create your first A/B experiment

The canonical Phase 4 use case: test whether showing SHAP explanations
in the UI improves click-through rate.

```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "id": "shap-explanation-ctr",
    "name": "SHAP Explanation CTR Test",
    "description": "Does showing audio feature explanations increase add-to-playlist rate?",
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
  "id": "shap-explanation-ctr",
  "status": "running",
  "sprt": {
    "log_lambda": 0.0,
    "upper_boundary": 2.7726,
    "lower_boundary": -2.0794,
    "progress_pct": 0.0
  },
  "control": {"name": "no-explanation", "n_observations": 0},
  "treatment": {"name": "with-explanation", "n_observations": 0}
}
```

---

## Step 5 — Verify user arm assignment

```bash
# Check which arm a user is in (deterministic)
curl http://localhost:8000/api/experiments/shap-explanation-ctr/arm/demo-user-01
# {"user_id": "demo-user-01", "experiment_id": "shap-explanation-ctr", "arm": "treatment"}

# Same user always gets the same arm
curl http://localhost:8000/api/experiments/shap-explanation-ctr/arm/demo-user-01
# arm is the same ↑

# Different users split roughly 50/50
for i in $(seq 1 10); do
  curl -s http://localhost:8000/api/experiments/shap-explanation-ctr/arm/user-$i \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d['arm'])"
done
# control
# treatment
# treatment
# control
# ...  (roughly 50/50)
```

---

## Step 6 — SHAP explanations in recommendations

```bash
# Without explanations (default, fast)
curl "http://localhost:8000/api/recommend/demo-user-01?limit=3&bypass_cache=true" \
  | python -m json.tool | grep explanation

# With SHAP feature attributions (adds ~3ms)
curl "http://localhost:8000/api/recommend/demo-user-01?limit=3&bypass_cache=true&explain=true" \
  | python -m json.tool
```

With `explain=true`, each song now has a `shap` field:
```json
{
  "name": "Blinding Lights",
  "explanation": "Matches your energy and danceability · 12 positive signals",
  "shap": {
    "feature_contributions": {
      "danceability": 0.0821,
      "energy": 0.1134,
      "valence": 0.0203,
      "tempo_norm": 0.0041,
      "acousticness": -0.0312,
      "bandit_prior": 0.2275
    },
    "dominant_feature": "energy",
    "score_breakdown": {
      "context_component": 0.1678,
      "bandit_component": 0.2275,
      "total": 0.6142
    }
  }
}
```

---

## Step 7 — On-demand explanation for any song

```bash
# Get a song ID from recommendations
SONG_ID=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=1&bypass_cache=true" \
  | python -c "import sys,json; print(json.load(sys.stdin)['songs'][0]['id'])")

# Full explanation with bandit arm info
curl "http://localhost:8000/api/explain/demo-user-01/$SONG_ID" | python -m json.tool
# {
#   "explanation_text": "Matches your energy and danceability",
#   "dominant_feature": "energy",
#   "dominant_feature_label": "Energy",
#   "feature_contributions": {
#     "Energy": 0.1134,
#     "Danceability": 0.0821,
#     ...
#   },
#   "bandit_info": {
#     "alpha": 1.0, "beta": 1.0,
#     "mean_reward": 0.5,
#     "n_observations": 0,
#     "uncertainty": 0.25
#   }
# }

# User taste profile
curl "http://localhost:8000/api/explain/demo-user-01/profile" | python -m json.tool
# {
#   "taste_summary": "You tend to prefer high-energy and danceable music",
#   "preference_vector": {...},
#   "vs_population_avg": {...}
# }

# Batch explanations (up to 50 songs)
curl -X POST "http://localhost:8000/api/explain/demo-user-01/batch" \
  -H "Content-Type: application/json" \
  -d "{\"song_ids\": [\"$SONG_ID\"]}" | python -m json.tool
```

---

## Step 8 — Watch SPRT converge with synthetic traffic

Generate traffic and watch the experiment make a decision:

```bash
# Get song IDs
SONGS=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=10&bypass_cache=true" \
  | python -c "
import sys, json
songs = json.load(sys.stdin)['songs']
print(' '.join(s['id'] for s in songs))
")

# Send 500 add events — treatment should win (high reward)
for i in $(seq 1 500); do
  SONG=$(echo $SONGS | tr ' ' '\n' | shuf -n1)

  # All users get their feedback routed to the experiment automatically
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{
      \"user_id\": \"user-$(( RANDOM % 50 ))\",
      \"song_id\": \"$SONG\",
      \"action\": \"add\"
    }" > /dev/null

  # Check experiment every 50 events
  if [ $(( i % 50 )) -eq 0 ]; then
    STATUS=$(curl -s http://localhost:8000/api/experiments/shap-explanation-ctr \
      | python -c "import sys,json; d=json.load(sys.stdin); print(d['status'], 'lambda='+str(d['sprt']['log_lambda']))")
    echo "[$i events] $STATUS"
  fi
done

# Final result
curl http://localhost:8000/api/experiments/shap-explanation-ctr | python -m json.tool
```

---

## Step 9 — Create a bandit algorithm comparison experiment

The most common production use case: compare two algorithm variants.

```bash
# Test higher context_weight (more audio-feature driven) vs current bandit
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
    "id": "context-weight-test",
    "name": "Context Weight 0.35 vs 0.55",
    "description": "Does higher context weighting improve CTR for new users?",
    "control_name": "weight-0.35",
    "treatment_name": "weight-0.55",
    "alpha": 0.05,
    "beta": 0.20,
    "delta": 0.03,
    "max_samples_per_arm": 10000,
    "traffic_split_pct": 50
  }' | python -m json.tool

# List all experiments
curl http://localhost:8000/api/experiments | python -m json.tool

# List only running
curl http://localhost:8000/api/experiments/running | python -m json.tool
```

---

## Step 10 — Force-conclude and clean up

```bash
# Conclude an experiment before it reaches max_samples
curl -X POST "http://localhost:8000/api/experiments/context-weight-test/conclude" \
  -H "Content-Type: application/json" \
  -d '{"reason": "Ship date moved up — not enough time to collect samples", "winner": null}'

# Verify concluded
curl http://localhost:8000/api/experiments/context-weight-test \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d['decided_at'])"

# Delete experiment
curl -X DELETE http://localhost:8000/api/experiments/context-weight-test
# 204 No Content
```

---

## API Reference — New Phase 4 Endpoints

### Experiments

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/experiments` | Create SPRT experiment |
| GET | `/api/experiments` | List all experiments |
| GET | `/api/experiments/running` | List running only |
| GET | `/api/experiments/{id}` | Full experiment state |
| GET | `/api/experiments/{id}/arm/{user_id}` | Get user's arm assignment |
| POST | `/api/experiments/{id}/observe` | Manual observation recording |
| POST | `/api/experiments/{id}/conclude` | Force-conclude |
| DELETE | `/api/experiments/{id}` | Delete experiment |

### Explainability

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/explain/{user_id}/{song_id}` | Explain one recommendation |
| POST | `/api/explain/{user_id}/batch` | Explain batch (≤50 songs) |
| GET | `/api/explain/{user_id}/profile` | User taste profile |

### Recommendations (updated)

| Param | Default | Description |
|-------|---------|-------------|
| `explain` | false | Include SHAP feature_contributions per song |
| `experiment_id` | null | Enrol user in named experiment |

---

## SPRT Parameters Guide

| Parameter | What it controls | Tighten if... |
|-----------|-----------------|---------------|
| `alpha` (0.05) | False positive rate — declaring winner when there isn't one | You can tolerate fewer false alarms |
| `beta` (0.20) | False negative rate — missing a real winner | You need high power |
| `delta` (0.02) | Minimum effect size you care about | Small effects matter (lower = more samples) |
| `max_samples_per_arm` | Safety cap to force conclusion | You have traffic constraints |

Rule of thumb: with `alpha=0.05, beta=0.20, delta=0.02`, expect ~4,000 samples per arm for a 50/50 split on a 2pp lift.

---

## Troubleshooting

**"Experiment already exists"**
```bash
# Delete and recreate
curl -X DELETE http://localhost:8000/api/experiments/my-exp
```

**SPRT never decides**
- Increase `delta` (lower MDE = more samples needed)
- Reduce `alpha` and `beta` for faster but less reliable decisions
- Check that feedback events are actually flowing: `curl http://localhost:8000/api/experiments/my-exp | python -m json.tool`

**SHAP explanations all say "exploring"**
- User has no interactions yet — bandit prior is (1,1)
- Send some feedback events first: `curl -X POST /api/feedback ...`

**Experiment arm assignment not 50/50**
- This is expected with small N — hash distribution stabilises at N≥200
- Check with 1000 synthetic users (see Step 5)

**Experiment observations not persisting across restarts**
- Experiments are stored in Redis — ensure Redis is running
- Check: `redis-cli keys "experiment:*"`
