"""
Prometheus Metrics Registry
-----------------------------
All custom metrics for the music recommender, organised by subsystem.

Naming convention: music_<subsystem>_<metric>_<unit>
  subsystem: recommender | bandit | feedback | drift | data_quality | kafka

Metric types:
  Counter    — monotonically increasing (events, errors, detections)
  Gauge      — current value (window size, active users, latency snapshot)
  Histogram  — distribution of values (latencies, reward distributions)
  Summary    — (not used — prefer Histogram for percentile queries in Grafana)

These metrics are exposed at GET /metrics and scraped by Prometheus.
Grafana dashboards query them via PromQL.

Key dashboards they feed (configured in monitoring/grafana/dashboards/):
  1. Recommendation Quality  — CTR, reward rate, latency
  2. Bandit Health           — arm convergence, exploration rate
  3. Data Quality            — feature distributions, null rates
  4. Drift Detection         — ADWIN & KS signals, detection counts
  5. System Health           — Kafka lag, DB connections, Redis hits
"""
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

# Use default registry so prometheus-fastapi-instrumentator picks them up
# without needing a separate /metrics endpoint.

# ── Recommendation metrics ────────────────────────────────────────────────────

RECOMMENDATION_REQUESTS = Counter(
    "music_recommender_requests_total",
    "Total recommendation requests",
    ["user_type", "cache_hit"],  # user_type: new|returning, cache_hit: true|false
)

RECOMMENDATION_LATENCY = Histogram(
    "music_recommender_latency_ms",
    "End-to-end recommendation latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

RECOMMENDATION_ALGORITHM = Counter(
    "music_recommender_algorithm_total",
    "Recommendations served by algorithm",
    ["algorithm"],  # thompson_sampling | thompson_sampling_cached | fallback
)

CANDIDATE_POOL_SIZE = Gauge(
    "music_recommender_candidate_pool_size",
    "Number of songs in the candidate pool for the last recommendation call",
)

# ── Feedback / reward metrics ─────────────────────────────────────────────────

FEEDBACK_EVENTS = Counter(
    "music_feedback_events_total",
    "Total feedback events received",
    ["action"],  # add | skip | play | complete
)

REWARD_DISTRIBUTION = Histogram(
    "music_feedback_reward",
    "Distribution of reward values from feedback events",
    buckets=[-0.5, -0.3, 0.0, 0.3, 0.5, 0.8, 1.0, 1.1],
)

FEEDBACK_LATENCY = Histogram(
    "music_feedback_processing_latency_ms",
    "Time to process a feedback event (including bandit update)",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

KAFKA_PUBLISH_SUCCESS = Counter(
    "music_kafka_publish_total",
    "Kafka publish attempts",
    ["status"],  # success | fallback | error
)

KAFKA_CONSUMER_LAG = Gauge(
    "music_kafka_consumer_lag",
    "Approximate Kafka consumer lag (messages behind latest offset)",
)

# ── Bandit health metrics ─────────────────────────────────────────────────────

BANDIT_ARM_UPDATES = Counter(
    "music_bandit_arm_updates_total",
    "Total bandit arm posterior updates",
)

BANDIT_MEAN_REWARD = Gauge(
    "music_bandit_mean_reward_per_user",
    "Rolling mean reward across all active arms for sampled users",
)

BANDIT_EXPLORATION_RATE = Gauge(
    "music_bandit_exploration_rate",
    "Fraction of recommendations going to arms with < 5 observations (exploration)",
)

BANDIT_ALPHA_DISTRIBUTION = Histogram(
    "music_bandit_alpha_values",
    "Distribution of alpha values across all bandit arms",
    buckets=[1, 2, 3, 5, 10, 20, 50, 100],
)

# ── Drift detection metrics ───────────────────────────────────────────────────

ADWIN_DRIFT_DETECTIONS = Counter(
    "music_drift_adwin_detections_total",
    "Total reward drift detections by ADWIN detector",
    ["signal"],  # reward | ctr
)

ADWIN_WINDOW_SIZE = Gauge(
    "music_drift_adwin_window_size",
    "Current ADWIN adaptive window size (number of observations)",
    ["signal"],
)

ADWIN_CURRENT_MEAN = Gauge(
    "music_drift_adwin_current_mean",
    "Current mean value in the ADWIN window",
    ["signal"],
)

KS_DRIFT_DETECTIONS = Counter(
    "music_drift_ks_detections_total",
    "KS test drift detections per feature",
    ["feature"],  # danceability | energy | valence | tempo_norm | acousticness
)

KS_STATISTIC = Gauge(
    "music_drift_ks_statistic",
    "Most recent KS test statistic for each feature (0=identical, 1=maximally different)",
    ["feature"],
)

KS_P_VALUE = Gauge(
    "music_drift_ks_p_value",
    "Most recent KS test p-value for each feature",
    ["feature"],
)

KS_REFERENCE_MEAN = Gauge(
    "music_drift_ks_reference_mean",
    "Reference window mean for each audio feature",
    ["feature"],
)

KS_CURRENT_MEAN = Gauge(
    "music_drift_ks_current_mean",
    "Sliding window mean for each audio feature",
    ["feature"],
)

# ── Data quality metrics ──────────────────────────────────────────────────────

DATA_QUALITY_NULL_FEATURES = Counter(
    "music_data_quality_null_features_total",
    "Null/missing audio feature values encountered during validation",
    ["feature"],
)

DATA_QUALITY_INVALID_REWARDS = Counter(
    "music_data_quality_invalid_rewards_total",
    "Feedback events with out-of-range reward values",
)

DATA_QUALITY_VALIDATION_FAILURES = Counter(
    "music_data_quality_validation_failures_total",
    "Total data validation failures",
    ["check"],  # reward_range | feature_range | duplicate_feedback
)

SONGS_IN_CATALOG = Gauge(
    "music_catalog_songs_total",
    "Total songs in the database catalog",
)

SONGS_WITH_FEATURES = Gauge(
    "music_catalog_songs_with_features",
    "Songs in catalog that have complete audio features",
)

# ── System / infrastructure metrics ──────────────────────────────────────────

ACTIVE_USERS_24H = Gauge(
    "music_active_users_24h",
    "Users with at least one interaction in the last 24 hours",
)

DB_POOL_SIZE = Gauge(
    "music_db_pool_size",
    "Current SQLAlchemy async connection pool size",
)

REDIS_CACHE_HITS = Counter(
    "music_redis_cache_hits_total",
    "Redis recommendation cache hits",
    ["cache"],  # recs | features
)

REDIS_CACHE_MISSES = Counter(
    "music_redis_cache_misses_total",
    "Redis recommendation cache misses",
    ["cache"],
)
