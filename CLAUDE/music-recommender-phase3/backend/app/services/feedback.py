"""
Phase 2 Feedback Service
-------------------------
Primary path:  publish to Kafka → consumer updates bandit async
Fallback path: if Kafka unavailable, direct bandit update + DB write
Both paths still update the user preference EMA vector.
"""
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import User, Song, Interaction, PlaylistItem
from app.services.feature_store import feature_store
from app.services.kafka.producer import feedback_producer, FeedbackEvent
from app.services.bandit.thompson import bandit
from app.services.bandit.state_manager import bandit_state_manager

logger = logging.getLogger(__name__)
EMA_ALPHA = 0.1


async def process_feedback(
    user_id: str,
    song_id: str,
    action: str,
    reward: float,
    db: AsyncSession,
):
    # Ensure user
    user_result = await db.execute(select(User).where(User.id == user_id))
    user = user_result.scalar_one_or_none()
    if user is None:
        user = User(id=user_id, display_name=f"User {user_id[:8]}")
        db.add(user)
        await db.flush()

    # Load song
    song_result = await db.execute(select(Song).where(Song.id == song_id))
    song = song_result.scalar_one_or_none()
    if song is None:
        logger.warning(f"Feedback for unknown song {song_id} — ignoring")
        return

    # Record interaction
    db.add(Interaction(
        user_id=user_id, song_id=song_id, action=action, reward=reward,
        context={
            "danceability": song.danceability, "energy": song.energy,
            "valence": song.valence, "tempo": song.tempo,
            "acousticness": song.acousticness, "popularity": song.popularity,
        },
    ))

    # Add to playlist
    if action == "add":
        existing = await db.execute(
            select(PlaylistItem).where(
                PlaylistItem.user_id == user_id, PlaylistItem.song_id == song_id,
            )
        )
        if existing.scalar_one_or_none() is None:
            db.add(PlaylistItem(user_id=user_id, song_id=song_id))

    # Update EMA preference vector
    if reward > 0:
        a = EMA_ALPHA
        user.avg_danceability = _ema(user.avg_danceability, song.danceability, a)
        user.avg_energy       = _ema(user.avg_energy,       song.energy,       a)
        user.avg_valence      = _ema(user.avg_valence,      song.valence,      a)
        user.avg_tempo        = _ema(user.avg_tempo,        song.tempo,        a)
        user.avg_acousticness = _ema(user.avg_acousticness, song.acousticness, a)

    user.total_interactions += 1
    user.last_active = datetime.utcnow()
    await db.commit()

    # Publish to Kafka (primary bandit update path)
    event = FeedbackEvent(
        user_id=user_id, song_id=song_id, action=action, reward=reward,
        timestamp=datetime.utcnow().isoformat(),
        context={"danceability": song.danceability, "energy": song.energy, "valence": song.valence},
    )
    published = await feedback_producer.publish(event)

    if not published:
        # Fallback: direct bandit update
        logger.debug(f"Kafka unavailable — direct bandit update for user={user_id}")
        current_alpha, current_beta = await bandit_state_manager.get_single_arm(user_id, song_id, db)
        new_alpha, new_beta = bandit.update(current_alpha, current_beta, reward)
        await bandit_state_manager.update_arm(user_id, song_id, new_alpha, new_beta, db)

    await feature_store.invalidate_user_features(user_id)
    await feature_store.invalidate_recommendations(user_id)
    logger.info(f"Feedback: user={user_id} song={song_id} action={action} reward={reward} kafka={'ok' if published else 'fallback'}")


def _ema(current, new_val, alpha):
    if current is None: return new_val or 0.5
    if new_val is None: return current
    return alpha * new_val + (1 - alpha) * current
