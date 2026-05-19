import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db import get_db
from app.models import User, PlaylistItem, Song
from app.schemas import PlaylistResponse, SongResponse, UserCreate, UserResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["playlist"])


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.id == payload.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="User already exists")
    user = User(id=payload.id, display_name=payload.display_name)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.get("/users/{user_id}/playlist", response_model=PlaylistResponse)
async def get_playlist(user_id: str, db: AsyncSession = Depends(get_db)):
    """Return songs explicitly added to the user's playlist."""
    result = await db.execute(
        select(PlaylistItem, Song)
        .join(Song, PlaylistItem.song_id == Song.id)
        .where(PlaylistItem.user_id == user_id)
        .order_by(PlaylistItem.added_at.desc())
    )
    rows = result.fetchall()
    songs = [SongResponse.model_validate(song) for _, song in rows]
    return PlaylistResponse(user_id=user_id, songs=songs, total=len(songs))


@router.delete("/users/{user_id}/playlist/{song_id}", status_code=204)
async def remove_from_playlist(
    user_id: str, song_id: str, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(PlaylistItem).where(
            PlaylistItem.user_id == user_id,
            PlaylistItem.song_id == song_id,
        )
    )
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Song not in playlist")
    await db.delete(item)
    await db.commit()
