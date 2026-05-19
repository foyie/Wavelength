"""
Spotify Web API client using Client Credentials flow.
No user login needed — we fetch catalog data, not personal playlists.
"""
import base64
import json
import logging
from typing import Optional
import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


class SpotifyClient:
    def __init__(self):
        self._token: Optional[str] = None
        self._http: Optional[httpx.AsyncClient] = None

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

    async def _get_token(self) -> str:
        """Fetch or refresh a client credentials token."""
        if self._token:
            return self._token

        credentials = f"{settings.spotify_client_id}:{settings.spotify_client_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        http = await self._get_http()
        resp = await http.post(
            SPOTIFY_AUTH_URL,
            headers={"Authorization": f"Basic {encoded}"},
            data={"grant_type": "client_credentials"},
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        logger.info("Spotify token refreshed")
        return self._token

    async def _get(self, path: str, params: dict = None) -> dict:
        token = await self._get_token()
        http = await self._get_http()
        resp = await http.get(
            f"{SPOTIFY_API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
        )
        if resp.status_code == 401:
            # Token expired — refresh and retry once
            self._token = None
            token = await self._get_token()
            resp = await http.get(
                f"{SPOTIFY_API_BASE}{path}",
                headers={"Authorization": f"Bearer {token}"},
                params=params or {},
            )
        resp.raise_for_status()
        return resp.json()

    # ── Catalog methods ─────────────────────────────────────────────────────

    async def search_tracks(self, query: str, limit: int = 50, offset: int = 0) -> list[dict]:
        data = await self._get("/search", {
            "q": query,
            "type": "track",
            "limit": min(limit, 50),
            "offset": offset,
        })
        return data.get("tracks", {}).get("items", [])

    async def get_audio_features(self, track_ids: list[str]) -> list[dict]:
        """Batch fetch audio features for up to 100 tracks."""
        results = []
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            data = await self._get("/audio-features", {"ids": ",".join(batch)})
            results.extend(data.get("audio_features", []))
        return results

    async def get_tracks(self, track_ids: list[str]) -> list[dict]:
        """Batch fetch track metadata."""
        results = []
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i + 50]
            data = await self._get("/tracks", {"ids": ",".join(batch)})
            results.extend(data.get("tracks", []))
        return results

    async def get_playlist_tracks(self, playlist_id: str, limit: int = 100) -> list[dict]:
        data = await self._get(f"/playlists/{playlist_id}/tracks", {
            "limit": min(limit, 100),
            "fields": "items(track(id,name,artists,album,duration_ms,popularity,preview_url,explicit))",
        })
        return [item["track"] for item in data.get("items", []) if item.get("track")]

    async def get_featured_playlists(self, limit: int = 20) -> list[dict]:
        data = await self._get("/browse/featured-playlists", {"limit": limit})
        return data.get("playlists", {}).get("items", [])

    async def close(self):
        if self._http and not self._http.is_closed:
            await self._http.aclose()


def parse_track_to_dict(track: dict, features: dict = None) -> dict:
    """Normalise a Spotify track object into our schema format."""
    artists = track.get("artists", [])
    artist_name = ", ".join(a["name"] for a in artists) if artists else "Unknown"
    album = track.get("album", {})
    images = album.get("images", [])
    album_art = images[0]["url"] if images else None

    result = {
        "id": track["id"],
        "name": track["name"],
        "artist": artist_name,
        "album": album.get("name"),
        "album_art_url": album_art,
        "preview_url": track.get("preview_url"),
        "duration_ms": track.get("duration_ms"),
        "popularity": track.get("popularity", 0),
        "explicit": track.get("explicit", False),
        "genres": [],
    }

    if features:
        result.update({
            "danceability": features.get("danceability"),
            "energy": features.get("energy"),
            "key": features.get("key"),
            "loudness": features.get("loudness"),
            "mode": features.get("mode"),
            "speechiness": features.get("speechiness"),
            "acousticness": features.get("acousticness"),
            "instrumentalness": features.get("instrumentalness"),
            "liveness": features.get("liveness"),
            "valence": features.get("valence"),
            "tempo": features.get("tempo"),
        })

    return result


# Singleton
spotify_client = SpotifyClient()
