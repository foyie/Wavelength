# Execution Guide — Phase 1 & Phase 2

## Prerequisites

| Tool           | Min version | Install                                    |
| -------------- | ----------- | ------------------------------------------ |
| Python         | 3.11        | `brew install python@3.11` or pyenv      |
| Node.js        | 20          | `brew install node`                      |
| Docker Desktop | 4.x         | https://docker.com/products/docker-desktop |
| Git            | any         | pre-installed on most systems              |

---

## PHASE 1 — Execution Steps

### Step 1 — Spotify credentials

1. Go to https://developer.spotify.com/dashboard
2. Create an app (any name, any redirect URI, e.g. `http://localhost`)
3. Copy the **Client ID** and **Client Secret**

```bash
cp .env.example .env
# Edit .env and fill in:
#   SPOTIFY_CLIENT_ID=abc123...
#   SPOTIFY_CLIENT_SECRET=def456...
```

---

### Step 2 — Start infrastructure (Postgres + Redis only)

```bash
# Start only Phase 1 services — no Kafka yet
docker compose up -d postgres redis

# Verify both are healthy
docker compose ps
# postgres   Up (healthy)
# redis      Up (healthy)
```

Wait ~10 seconds for Postgres to initialise on first run.

---

### Step 3 — Backend setup

```bash
cd backend

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify Spotify credentials work before seeding
python -c "
import asyncio, os
from dotenv import load_dotenv
load_dotenv('../.env')
from app.services.spotify_client import spotify_client
async def test():
    tracks = await spotify_client.search_tracks('pop hits 2023', limit=1)
    print('Spotify OK —', tracks[0]['name'] if tracks else 'no results')
    await spotify_client.close()
asyncio.run(test())
"
```

Expected output: `Spotify OK — <some song name>`

---

### Step 4 — Start the backend

```bash
# Still in backend/ with .venv active
uvicorn app.main:app --reload --port 8000

# You should see:
# INFO  Database tables ensured
# INFO  Application startup complete
# INFO  Uvicorn running on http://0.0.0.0:8000
```

Check the health endpoint:

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"1.0.0"}
```

Interactive API docs: http://localhost:8000/docs

---

### Step 5 — Seed song data from Spotify

**This is a one-time step.** Takes 3–8 minutes (Spotify rate limits).

```bash
# In a new terminal, backend/ with .venv active
python -m scripts.load_spotify_data

# Progress output:
# INFO  Fetching track IDs from Spotify search...
# INFO  Query 'year:2023 genre:pop': 87 unique tracks so far
# ...
# INFO  Upserting 1032 songs to database...
# INFO  Populating Redis feature cache...
# INFO  Done! 1032 songs loaded.
```

Verify the seed:

```bash
# Check song count
python -c "
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select, func
from app.models import Song
from app.config import get_settings

async def count():
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    async with async_sessionmaker(engine)() as s:
        n = (await s.execute(select(func.count(Song.id)))).scalar()
        print(f'{n} songs in database')
    await engine.dispose()
asyncio.run(count())
"
```

---

### Step 6 — Start the frontend

```bash
cd frontend
npm install
npm start
# Opens http://localhost:3000 automatically
```

The app will show "For You" recommendations immediately.
Try adding and skipping songs — you'll see the preference vector updating.

---

### Step 7 — Smoke test Phase 1

```bash
# Create a test user
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"id":"test-user","display_name":"Tester"}'

# Get recommendations
curl "http://localhost:8000/api/recommend/test-user?limit=5" | python -m json.tool

# Send add feedback
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","song_id":"<any id from recs>","action":"add"}'

# Send skip feedback
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test-user","song_id":"<another id>","action":"skip"}'

# Get fresh recs (should reflect preference)
curl "http://localhost:8000/api/recommend/test-user?limit=5&bypass_cache=true" | python -m json.tool
```

---

### Step 8 — Run Phase 1 tests

```bash
cd backend
pytest tests/test_recommender.py -v

# Expected: 9 passed
```

---

## PHASE 2 — Execution Steps

Phase 2 **replaces** `recommender.py` and `feedback.py` in-place.
The frontend and all API contracts are unchanged.

---

### Step 1 — Confirm Phase 1 is working

Before upgrading, verify Phase 1 is healthy:

```bash
curl http://localhost:8000/api/health
# status should be "healthy"
```

---

### Step 2 — Stop the running backend

```bash
# In the terminal running uvicorn, press Ctrl+C
# Or if using docker compose:
docker compose stop backend
```

---

### Step 3 — Start Kafka infrastructure

The Phase 2 `docker-compose.yml` adds Zookeeper + Kafka. Kafka takes ~30 seconds to start.

```bash
# From project root
docker compose up -d zookeeper kafka

# Wait for Kafka to be healthy
docker compose ps
# zookeeper  Up
# kafka      Up (healthy)   ← wait for this

# Create the feedback topic (or let kafka-init do it)
docker compose run --rm kafka-init

# Verify topic exists
docker compose exec kafka \
  kafka-topics --bootstrap-server localhost:9092 --list
# music.feedback
```

---

### Step 4 — Install Phase 2 Python dependency

```bash
cd backend
source .venv/bin/activate
pip install aiokafka==0.11.0
```

---

### Step 5 — Start the Phase 2 backend

The Phase 2 files are already in place (recommender.py, feedback.py, config.py, main.py are all updated).

```bash
# Still in backend/ with .venv active
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 \
KAFKA_ENABLED=true \
uvicorn app.main:app --reload --port 8000

# You should now see:
# INFO  Starting Music Recommender API (Phase 2)...
# INFO  Database tables ensured
# INFO  Kafka producer connected: localhost:9092
# INFO  Kafka consumer started: group=bandit-updater topic=music.feedback
# INFO  Application startup complete
```

Check the health endpoint confirms version 2.0.0:

```bash
curl http://localhost:8000/api/health
# {"status":"healthy","db":"healthy","redis":"healthy","version":"2.0.0"}
```

---

### Step 6 — Verify Thompson Sampling is live

```bash
# Recommendations now come from the bandit
curl "http://localhost:8000/api/recommend/demo-user-01?limit=3&bypass_cache=true" | python -m json.tool

# algorithm field should show "thompson_sampling" (not "popularity_baseline")
# explanation field will say "New to your taste — exploring for you..."
```

---

### Step 7 — Test the online learning loop

This is the core of Phase 2 — watch the bandit update in real-time:

```bash
# Terminal 1: watch backend logs
uvicorn app.main:app --reload --port 8000 2>&1 | grep -E "Bandit|feedback|Kafka"

# Terminal 2: send repeated positive signals for one song
SONG_ID=$(curl -s "http://localhost:8000/api/recommend/demo-user-01?limit=1&bypass_cache=true" \
  | python -c "import sys,json; d=json.load(sys.stdin); print(d['songs'][0]['id'])")

echo "Targeting song: $SONG_ID"

for i in $(seq 1 10); do
  curl -s -X POST http://localhost:8000/api/feedback \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"demo-user-01\",\"song_id\":\"$SONG_ID\",\"action\":\"add\"}"
  echo " → feedback $i sent"
  sleep 0.5
done

# Check the arm state — alpha should be growing
curl "http://localhost:8000/api/bandit/demo-user-01/arm/$SONG_ID" | python -m json.tool
# {
#   "alpha": 11.0,   ← was 1.0, grew by 1.0 per add (reward=1.0)
#   "beta": 1.0,
#   "mean_reward": 0.9167,
#   "n_observations": 10,
#   "uncertainty": 0.0765
# }
```

---

### Step 8 — Inspect full bandit state

```bash
# See all arms for a user, sorted by mean reward
curl "http://localhost:8000/api/bandit/demo-user-01/state" | python -m json.tool

# Aggregate health stats
curl "http://localhost:8000/api/bandit/stats" | python -m json.tool
# {
#   "total_users_with_state": 1,
#   "total_arms": 10,
#   "avg_observations_per_arm": 4.5,
#   "total_interactions": 45
# }
```

---

### Step 9 — Run Phase 2 tests

```bash
cd backend
pytest tests/test_bandit.py -v

# Expected: 19 passed
# Key tests:
#   test_add_reward_increases_alpha       PASSED
#   test_many_updates_converge_toward_truth  PASSED
#   test_high_alpha_ranks_higher_on_average  PASSED
#   test_identical_user_song_gives_high_similarity  PASSED
#   test_roundtrip_json                   PASSED
```

Run all tests together:

```bash
pytest tests/ -v
# Expected: 28 passed (9 from Phase 1 + 19 from Phase 2)
```

---

### Step 10 — Full Docker deployment (Phase 2)

To run everything in Docker (recommended for staging):

```bash
# From project root
docker compose up --build

# Services started:
#   postgres   :5432
#   redis      :6379
#   zookeeper  :2181
#   kafka      :9092
#   kafka-init (one-shot topic creation)
#   backend    :8000
#   frontend   :3000

# Tail logs
docker compose logs -f backend

# Run seed script inside the container
docker compose exec backend python -m scripts.load_spotify_data
```

---

## Kafka Without Docker (local dev)

If you want Kafka locally without Docker:

```bash
# macOS
brew install kafka
brew services start zookeeper
brew services start kafka

# Create topic
kafka-topics --bootstrap-server localhost:9092 --create \
  --topic music.feedback --partitions 3

# Then start backend with:
KAFKA_BOOTSTRAP_SERVERS=localhost:9092 uvicorn app.main:app --reload
```

---

## Running Without Kafka (fallback mode)

If you want Phase 2 bandit without setting up Kafka, set `KAFKA_ENABLED=false`.
Bandit updates happen synchronously in the request instead of asynchronously.

```bash
KAFKA_ENABLED=false uvicorn app.main:app --reload --port 8000
```

All bandit functionality works identically — only the delivery mechanism changes.

---

## Port Reference

| Service            | Port | URL                           |
| ------------------ | ---- | ----------------------------- |
| Frontend           | 3000 | http://localhost:3000         |
| Backend API        | 8000 | http://localhost:8000         |
| API Docs           | 8000 | http://localhost:8000/docs    |
| Prometheus metrics | 8000 | http://localhost:8000/metrics |
| Kafka broker       | 9092 | localhost:9092                |
| Postgres           | 5432 | localhost:5432                |
| Redis              | 6379 | localhost:6379                |
| Zookeeper          | 2181 | localhost:2181                |

---

## Troubleshooting

**`Spotify OK` fails in Step 3**

- Double-check `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` in `.env`
- Make sure you have no trailing spaces

**Postgres connection refused**

```bash
docker compose ps postgres   # check it's Up
docker compose logs postgres  # look for init errors
```

**Kafka consumer won't start**

```bash
docker compose ps kafka   # must show (healthy)
docker compose logs kafka | tail -20
# Common issue: Zookeeper not ready yet — wait 30s and retry
```

**`bandit_states` table missing**

```bash
# Force table creation
curl -X POST http://localhost:8000/api/users \
  -H "Content-Type: application/json" \
  -d '{"id":"init-user"}'
# The lifespan handler creates all tables on startup
```

**Recommendations not changing after feedback**

```bash
# Always use bypass_cache=true when testing learning
curl "http://localhost:8000/api/recommend/USER_ID?bypass_cache=true"
```

**aiokafka import error**

```bash
source backend/.venv/bin/activate
pip install aiokafka==0.11.0
```
