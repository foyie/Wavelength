import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.schemas import FeedbackRequest, FeedbackResponse
from app.services.feedback import process_feedback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
async def record_feedback(
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Record a user action on a song.
    Phase 1: Direct DB update + EMA preference update.
    Phase 2: Also publishes to Kafka for bandit updates.
    """
    reward = payload.reward  # computed by pydantic property
    try:
        await process_feedback(
            user_id=payload.user_id,
            song_id=payload.song_id,
            action=payload.action,
            reward=reward,
            db=db,
        )
    except Exception as e:
        logger.error(f"Feedback processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to record feedback")

    return FeedbackResponse(
        status="recorded",
        user_id=payload.user_id,
        song_id=payload.song_id,
        action=payload.action,
        reward=reward,
    )
