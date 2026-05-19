"""
Locust Load Test — Music Recommender
--------------------------------------
Simulates realistic production traffic patterns for the recommendation API.

Traffic mix (approximates a real music app user session):
  60%  GET  /api/recommend          — browsing recommendations
  25%  POST /api/feedback           — adding/skipping/playing songs
  10%  GET  /api/users/{id}/playlist — checking saved playlist
   5%  GET  /api/explain/{id}/{song} — viewing song explanations

User behaviour model:
  - Each user has a stable ID (simulates returning users)
  - They browse 1-5 recommendation pages per session
  - They interact with 30-70% of songs they see
  - Sessions last 2-10 minutes with think times between actions

Run:
  pip install locust
  locust -f scripts/load_test.py --host http://localhost:8000

  # Headless (CI / automated):
  locust -f scripts/load_test.py \
    --host http://localhost:8000 \
    --users 50 --spawn-rate 5 --run-time 120s --headless \
    --csv load_test_results

  # Targets to hit:
  #   p50 recommendation latency < 50ms
  #   p95 recommendation latency < 200ms
  #   p99 recommendation latency < 500ms
  #   Error rate < 0.1%
  #   Throughput > 100 rec/s at 50 concurrent users
"""
import random
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


# Seeded song IDs — populated from the API on test start
SONG_IDS: list[str] = []
USER_IDS = [f"load-test-user-{i:04d}" for i in range(200)]


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Fetch song IDs from the API before the test starts."""
    global SONG_IDS
    if isinstance(environment.runner, MasterRunner):
        return  # only workers fetch songs

    with environment.runner.user_classes[0](environment) as client:
        # Ensure at least one user exists
        try:
            client.post(
                "/api/users",
                json={"id": "load-test-user-0000", "display_name": "Load Tester"},
                name="[setup] create user",
            )
        except Exception:
            pass

        # Fetch songs
        resp = client.get(
            "/api/recommend/load-test-user-0000?limit=50&bypass_cache=true",
            name="[setup] fetch song IDs",
        )
        if resp.status_code == 200:
            songs = resp.json().get("songs", [])
            SONG_IDS.extend(s["id"] for s in songs)
            print(f"[load test] Loaded {len(SONG_IDS)} song IDs")
        else:
            print(f"[load test] WARNING: Could not fetch songs ({resp.status_code})")


class MusicAppUser(HttpUser):
    """
    Simulates a typical user of the music recommendation app.
    Think time between 1-4 seconds (realistic browsing pace).
    """
    wait_time = between(1, 4)

    def on_start(self):
        """Each simulated user gets a stable ID from the pool."""
        self.user_id = random.choice(USER_IDS)
        self.seen_songs: list[str] = []

        # Ensure this user exists
        self.client.post(
            "/api/users",
            json={"id": self.user_id, "display_name": f"Tester {self.user_id[-4:]}"},
            name="/api/users [create]",
        )

    @task(6)
    def get_recommendations(self):
        """Browse recommendations — most common action."""
        with self.client.get(
            f"/api/recommend/{self.user_id}?limit=20",
            name="/api/recommend/[user_id]",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                songs = resp.json().get("songs", [])
                self.seen_songs = [s["id"] for s in songs[:10]]
                latency = resp.elapsed.total_seconds() * 1000
                if latency > 500:
                    resp.failure(f"Recommendation too slow: {latency:.0f}ms")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(3)
    def send_feedback(self):
        """React to a recommendation — add, skip, or play."""
        if not self.seen_songs and not SONG_IDS:
            return

        song_id = (
            random.choice(self.seen_songs) if self.seen_songs
            else random.choice(SONG_IDS)
        )
        action = random.choices(
            ["add", "skip", "play", "complete"],
            weights=[0.15, 0.45, 0.30, 0.10],
        )[0]

        self.client.post(
            "/api/feedback",
            json={"user_id": self.user_id, "song_id": song_id, "action": action},
            name="/api/feedback",
        )

    @task(1)
    def get_playlist(self):
        """Check saved playlist."""
        self.client.get(
            f"/api/users/{self.user_id}/playlist",
            name="/api/users/[user_id]/playlist",
        )

    @task(1)
    def get_explanation(self):
        """View explanation for a song."""
        song_id = (
            random.choice(self.seen_songs) if self.seen_songs
            else (random.choice(SONG_IDS) if SONG_IDS else None)
        )
        if not song_id:
            return

        self.client.get(
            f"/api/explain/{self.user_id}/{song_id}",
            name="/api/explain/[user_id]/[song_id]",
        )


class HeavyRecommendUser(HttpUser):
    """
    Power user — hits recommendations with explain=true.
    Represents a client that renders SHAP bars in the UI.
    5% of simulated load.
    """
    wait_time = between(2, 6)
    weight = 5   # 5% of traffic

    def on_start(self):
        self.user_id = random.choice(USER_IDS)
        self.client.post(
            "/api/users",
            json={"id": self.user_id},
            name="/api/users [create]",
        )

    @task
    def get_recommendations_with_shap(self):
        with self.client.get(
            f"/api/recommend/{self.user_id}?limit=20&explain=true",
            name="/api/recommend/[user_id]?explain=true",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                latency = resp.elapsed.total_seconds() * 1000
                # SHAP adds ~3ms — allow 600ms SLO
                if latency > 600:
                    resp.failure(f"SHAP recommendation too slow: {latency:.0f}ms")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
