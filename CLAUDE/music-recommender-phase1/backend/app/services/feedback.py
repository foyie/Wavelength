"""
Feedback service for Phase 1.
In Phase 2 this is replaced by a Kafka consumer that also updates bandit posteriors.
For now it directly updates the user preference vector in Postgres and invalidates Redis.
"""
import logging
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import User, Song, Interaction, PlaylistItem
from app.services.feature_store import feature_store

logger = logging.getLogger(__name__)

# Exponential moving average decay — how quickly user prefs adapt
EMA_ALPHA = 0.1


async def process_feedback(
    user_id: str,
    song_id: str,
    action: str,
    reward: float,
    db: AsyncSession,
):
    """
    1. Record the interaction.
    2. Update the user preference vector (EMA on audio features).
    3. If action==add, add song to playlist.
    4. Invalidate Redis caches so next recommendation is fresh.
    """

    # Ensure user exists
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
    interaction = Interaction(
        user_id=user_id,
        song_id=song_id,
        action=action,
        reward=reward,
        context={
            "danceability": song.danceability,
            "energy": song.energy,
            "valence": song.valence,
            "tempo": song.tempo,
            "acousticness": song.acousticness,
            "popularity": song.popularity,
        },
    )
    db.add(interaction)

    # Add to playlist
    if action == "add":
        existing = await db.execute(
            select(PlaylistItem).where(
                PlaylistItem.user_id == user_id,
                PlaylistItem.song_id == song_id,
            )
        )
        if existing.scalar_one_or_none() is None:
            playlist_item = PlaylistItem(user_id=user_id, song_id=song_id)
            db.add(playlist_item)

    # Update user preference vector via EMA (only on positive feedback)
    if reward > 0:
        alpha = EMA_ALPHA
        user.avg_danceability = _ema(user.avg_danceability, song.danceability, alpha)
        user.avg_energy = _ema(user.avg_energy, song.energy, alpha)
        user.avg_valence = _ema(user.avg_valence, song.valence, alpha)
        user.avg_tempo = _ema(user.avg_tempo, song.tempo, alpha)
        user.avg_acousticness = _ema(user.avg_acousticness, song.acousticness, alpha)

    user.total_interactions += 1
    user.last_active = datetime.utcnow()

    await db.commit()

    # Invalidate caches
    await feature_store.invalidate_user_features(user_id)
    await feature_store.invalidate_recommendations(user_id)

    logger.info(f"Feedback processed: user={user_id} song={song_id} action={action} reward={reward}")


def _ema(current: float | None, new_value: float | None, alpha: float) -> float:
    """Exponential moving average update."""
    if current is None:
        return new_value or 0.5
    if new_value is None:
        return current
    return alpha * new_value + (1 - alpha) * current
