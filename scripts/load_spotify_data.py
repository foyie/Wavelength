#!/usr/bin/env python3
"""
Seed the database with ~1000 songs from Spotify.
Run: python -m scripts.load_spotify_data
Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env
"""
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select

from app.config import get_settings
from app.models import Base, Song
from app.services.spotify_client import spotify_client, parse_track_to_dict
from app.services.feature_store import feature_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

# Queries designed to surface diverse, popular tracks
SEED_QUERIES = [
    "year:2023 genre:pop",
    "year:2023 genre:hip-hop",
    "year:2023 genre:rock",
    "year:2022 genre:r&b",
    "year:2022 genre:electronic",
    "year:2023 genre:indie",
    "year:2022 genre:latin",
    "year:2023 genre:country",
    "top hits 2023",
    "viral hits 2022",
    "best songs 2021",
    "year:2024 genre:pop",
]


async def load_songs():
    engine = create_async_engine(settings.database_url)
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    all_track_ids = []
    seen = set()

    logger.info("Fetching track IDs from Spotify search...")
    for query in SEED_QUERIES:
        try:
            for offset in range(0, 100, 50):  # 2 pages of 50 = 100 per query
                tracks = await spotify_client.search_tracks(query, limit=50, offset=offset)
                for t in tracks:
                    if t and t.get("id") and t["id"] not in seen:
                        seen.add(t["id"])
                        all_track_ids.append(t["id"])
            logger.info(f"Query '{query}': {len(seen)} unique tracks so far")
        except Exception as e:
            logger.error(f"Error searching '{query}': {e}")
        await asyncio.sleep(0.1)  # rate limit courtesy

    logger.info(f"Total unique track IDs: {len(all_track_ids)}")

    # Fetch full metadata and audio features in batches
    all_songs = []
    batch_size = 50

    for i in range(0, len(all_track_ids), batch_size):
        batch_ids = all_track_ids[i:i + batch_size]
        try:
            tracks = await spotify_client.get_tracks(batch_ids)
            features_list = await spotify_client.get_audio_features(batch_ids)

            features_map = {f["id"]: f for f in features_list if f and f.get("id")}

            for track in tracks:
                if not track or not track.get("id"):
                    continue
                features = features_map.get(track["id"], {})
                song_dict = parse_track_to_dict(track, features)
                all_songs.append(song_dict)

            logger.info(f"Fetched batch {i // batch_size + 1}: {len(all_songs)} songs total")
        except Exception as e:
            logger.error(f"Error fetching batch {i}: {e}")
        await asyncio.sleep(0.2)

    logger.info(f"Upserting {len(all_songs)} songs to database...")
    async with SessionLocal() as session:
        for song_dict in all_songs:
            existing = await session.execute(
                select(Song).where(Song.id == song_dict["id"])
            )
            if existing.scalar_one_or_none():
                continue

            song = Song(
                id=song_dict["id"],
                name=song_dict["name"],
                artist=song_dict["artist"],
                album=song_dict.get("album"),
                album_art_url=song_dict.get("album_art_url"),
                preview_url=song_dict.get("preview_url"),
                duration_ms=song_dict.get("duration_ms"),
                popularity=song_dict.get("popularity", 0),
                explicit=song_dict.get("explicit", False),
                danceability=song_dict.get("danceability"),
                energy=song_dict.get("energy"),
                key=song_dict.get("key"),
                loudness=song_dict.get("loudness"),
                mode=song_dict.get("mode"),
                speechiness=song_dict.get("speechiness"),
                acousticness=song_dict.get("acousticness"),
                instrumentalness=song_dict.get("instrumentalness"),
                liveness=song_dict.get("liveness"),
                valence=song_dict.get("valence"),
                tempo=song_dict.get("tempo"),
                genres=[],
            )
            session.add(song)

        await session.commit()

    logger.info("Populating Redis feature cache...")
    await feature_store.set_song_features_bulk(all_songs)

    logger.info(f"Done! {len(all_songs)} songs loaded.")
    await spotify_client.close()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(load_songs())
