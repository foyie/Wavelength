"""
Bandit Admin Router — Phase 2 debug & inspection endpoints.
These are not user-facing; protect with API key in production.

GET  /api/bandit/{user_id}/state          — view all arm parameters for a user
GET  /api/bandit/{user_id}/arm/{song_id}  — view a single arm
POST /api/bandit/{user_id}/reset          — reset all arms to prior (1, 1)
GET  /api/bandit/stats                    — aggregate bandit health stats
"""
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from app.db import get_db
from app.models import BanditState, Song, User, Interaction
from app.services.bandit.state_manager import bandit_state_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/bandit", tags=["bandit-admin"])


class ArmState(BaseModel):
    song_id: str
    song_name: str
    artist: str
    alpha: float
    beta: float
    mean_reward: float        # alpha / (alpha + beta)
    n_observations: int       # alpha + beta - 2 (subtract prior)
    uncertainty: float        # std dev of Beta distribution


class BanditUserState(BaseModel):
    user_id: str
    total_arms: int
    arms: list[ArmState]


class BanditStats(BaseModel):
    total_users_with_state: int
    total_arms: int
    avg_observations_per_arm: float
    total_interactions: int


def beta_std(alpha: float, beta: float) -> float:
    """Standard deviation of Beta(alpha, beta)."""
    ab = alpha + beta
    return ((alpha * beta) / (ab * ab * (ab + 1))) ** 0.5


@router.get("/{user_id}/state", response_model=BanditUserState)
async def get_user_bandit_state(user_id: str, db: AsyncSession = Depends(get_db)):
    """View all arm parameters for a user. Useful for debugging convergence."""
    result = await db.execute(
        select(BanditState, Song)
        .join(Song, BanditState.song_id == Song.id)
        .where(BanditState.user_id == user_id)
        .order_by(BanditState.alpha.desc())
    )
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No bandit state for user {user_id}")

    arms = [
        ArmState(
            song_id=bs.song_id,
            song_name=song.name,
            artist=song.artist,
            alpha=bs.alpha,
            beta=bs.beta_param,
            mean_reward=round(bs.alpha / (bs.alpha + bs.beta_param), 4),
            n_observations=max(0, int(bs.alpha + bs.beta_param - 2)),
            uncertainty=round(beta_std(bs.alpha, bs.beta_param), 4),
        )
        for bs, song in rows
    ]

    return BanditUserState(user_id=user_id, total_arms=len(arms), arms=arms)


@router.get("/{user_id}/arm/{song_id}", response_model=ArmState)
async def get_single_arm(user_id: str, song_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(BanditState, Song)
        .join(Song, BanditState.song_id == Song.id)
        .where(BanditState.user_id == user_id, BanditState.song_id == song_id)
    )
    row = result.first()
    if not row:
        # Return default prior if no data yet
        song_result = await db.execute(select(Song).where(Song.id == song_id))
        song = song_result.scalar_one_or_none()
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")
        return ArmState(
            song_id=song_id, song_name=song.name, artist=song.artist,
            alpha=1.0, beta=1.0, mean_reward=0.5, n_observations=0, uncertainty=0.25,
        )

    bs, song = row
    return ArmState(
        song_id=bs.song_id,
        song_name=song.name,
        artist=song.artist,
        alpha=bs.alpha,
        beta=bs.beta_param,
        mean_reward=round(bs.alpha / (bs.alpha + bs.beta_param), 4),
        n_observations=max(0, int(bs.alpha + bs.beta_param - 2)),
        uncertainty=round(beta_std(bs.alpha, bs.beta_param), 4),
    )


@router.post("/{user_id}/reset", status_code=200)
async def reset_user_bandit(user_id: str, db: AsyncSession = Depends(get_db)):
    """Reset all arms to default prior. Useful for A/B testing cold-start."""
    result = await db.execute(
        select(BanditState).where(BanditState.user_id == user_id)
    )
    rows = result.scalars().all()
    n = len(rows)
    for row in rows:
        row.alpha = 1.0
        row.beta_param = 1.0
    await db.commit()

    # Invalidate Redis
    from app.services.feature_store import feature_store
    await feature_store.invalidate_recommendations(user_id)

    return {"status": "reset", "user_id": user_id, "arms_reset": n}


@router.get("/stats", response_model=BanditStats)
async def get_bandit_stats(db: AsyncSession = Depends(get_db)):
    """Aggregate health stats — good to put on a Grafana dashboard."""
    users_result = await db.execute(
        select(func.count(func.distinct(BanditState.user_id)))
    )
    total_users = users_result.scalar() or 0

    arms_result = await db.execute(select(func.count(BanditState.id)))
    total_arms = arms_result.scalar() or 0

    avg_obs_result = await db.execute(
        select(func.avg(BanditState.alpha + BanditState.beta_param - 2))
    )
    avg_obs = float(avg_obs_result.scalar() or 0)

    interactions_result = await db.execute(
        select(func.count(Interaction.id))
    )
    total_interactions = interactions_result.scalar() or 0

    return BanditStats(
        total_users_with_state=total_users,
        total_arms=total_arms,
        avg_observations_per_arm=round(avg_obs, 2),
        total_interactions=total_interactions,
    )
