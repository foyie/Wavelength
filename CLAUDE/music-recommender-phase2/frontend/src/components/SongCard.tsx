import React, { useState } from 'react';
import type { Song, RecommendedSong } from '../types';

interface SongCardProps {
  song: Song | RecommendedSong;
  isPlaying?: boolean;
  onPlay?: () => void;
  onAdd?: () => void;
  onSkip?: () => void;
  onRemove?: () => void;
  showScore?: boolean;
  animationDelay?: number;
  variant?: 'recommendation' | 'playlist';
}

function formatDuration(ms: number | null): string {
  if (!ms) return '--:--';
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function EnergyBar({ value, label }: { value: number | null; label: string }) {
  if (value === null) return null;
  return (
    <div className="feature-bar">
      <span className="feature-label">{label}</span>
      <div className="feature-track">
        <div className="feature-fill" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

export function SongCard({
  song,
  isPlaying,
  onPlay,
  onAdd,
  onSkip,
  onRemove,
  showScore,
  animationDelay = 0,
  variant = 'recommendation',
}: SongCardProps) {
  const [expanded, setExpanded] = useState(false);
  const isRec = 'score' in song;

  return (
    <div
      className={`song-card ${isPlaying ? 'song-card--playing' : ''}`}
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      {/* Album art */}
      <div className="song-art" onClick={onPlay}>
        {song.album_art_url ? (
          <img src={song.album_art_url} alt={song.album || song.name} />
        ) : (
          <div className="song-art__placeholder">♪</div>
        )}
        {song.preview_url && (
          <div className={`song-art__overlay ${isPlaying ? 'song-art__overlay--visible' : ''}`}>
            <button className="play-btn" aria-label={isPlaying ? 'Pause' : 'Play preview'}>
              {isPlaying ? '⏸' : '▶'}
            </button>
          </div>
        )}
        {isPlaying && <div className="song-art__pulse" />}
      </div>

      {/* Info */}
      <div className="song-info">
        <div className="song-meta">
          <p className="song-name">{song.name}</p>
          <p className="song-artist">{song.artist}</p>
          {song.album && <p className="song-album">{song.album}</p>}
        </div>

        <div className="song-tags">
          {song.explicit && <span className="tag tag--explicit">E</span>}
          {song.popularity > 0 && (
            <span className="tag tag--popularity">
              {'★'.repeat(Math.ceil(song.popularity / 25))}
            </span>
          )}
          <span className="tag tag--duration">{formatDuration(song.duration_ms)}</span>
        </div>

        {isRec && (song as RecommendedSong).explanation && (
          <p className="song-explanation">
            {(song as RecommendedSong).explanation}
          </p>
        )}

        {/* Feature bars toggle */}
        {(song.energy !== null || song.danceability !== null) && (
          <button
            className="features-toggle"
            onClick={() => setExpanded((e) => !e)}
          >
            {expanded ? 'Hide features ▲' : 'Audio features ▼'}
          </button>
        )}

        {expanded && (
          <div className="feature-bars">
            <EnergyBar value={song.energy} label="Energy" />
            <EnergyBar value={song.danceability} label="Dance" />
            <EnergyBar value={song.valence} label="Mood" />
            <EnergyBar value={song.acousticness} label="Acoustic" />
          </div>
        )}
      </div>

      {/* Score badge (Phase 1 debug, hidden in prod) */}
      {showScore && isRec && (
        <div className="song-score">
          <span className="score-rank">#{(song as RecommendedSong).rank}</span>
          <span className="score-value">{((song as RecommendedSong).score * 100).toFixed(0)}</span>
        </div>
      )}

      {/* Actions */}
      <div className="song-actions">
        {variant === 'recommendation' && (
          <>
            <button
              className="action-btn action-btn--add"
              onClick={onAdd}
              title="Add to playlist"
            >
              +
            </button>
            <button
              className="action-btn action-btn--skip"
              onClick={onSkip}
              title="Not interested"
            >
              ×
            </button>
          </>
        )}
        {variant === 'playlist' && (
          <button
            className="action-btn action-btn--remove"
            onClick={onRemove}
            title="Remove from playlist"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
}
