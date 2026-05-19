import React, { useState, useEffect, useCallback } from 'react';
import { api } from './services/api';
import { SongCard } from './components/SongCard';
import { usePlayer } from './hooks/usePlayer';
import type { RecommendedSong, Song } from './types';
import './App.css';

// Phase 1: demo user ID. Phase 2+ will add real auth.
const DEMO_USER_ID = 'demo-user-01';

type Tab = 'recommendations' | 'playlist';

export default function App() {
  const [tab, setTab] = useState<Tab>('recommendations');
  const [recommendations, setRecommendations] = useState<RecommendedSong[]>([]);
  const [playlist, setPlaylist] = useState<Song[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [algorithm, setAlgorithm] = useState('');
  const [toast, setToast] = useState<string | null>(null);
  const [skippedIds, setSkippedIds] = useState<Set<string>>(new Set());
  const { playingId, play } = usePlayer();

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const ensureUser = async () => {
    try {
      await api.getUser(DEMO_USER_ID);
    } catch {
      await api.createUser(DEMO_USER_ID, 'Demo Listener');
    }
  };

  const fetchRecommendations = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);
    try {
      const data = await api.getRecommendations(DEMO_USER_ID, 20);
      const visible = data.songs.filter((s) => !skippedIds.has(s.id));
      setRecommendations(visible);
      setLatencyMs(data.latency_ms);
      setAlgorithm(data.algorithm);
    } catch (e) {
      showToast('Failed to load recommendations');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [skippedIds]);

  const fetchPlaylist = useCallback(async () => {
    try {
      const data = await api.getPlaylist(DEMO_USER_ID);
      setPlaylist(data.songs);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    (async () => {
      await ensureUser();
      await Promise.all([fetchRecommendations(), fetchPlaylist()]);
    })();
  }, []);

  const handleAdd = async (song: RecommendedSong) => {
    setRecommendations((prev) => prev.filter((s) => s.id !== song.id));
    setPlaylist((prev) => [song, ...prev]);
    showToast(`Added "${song.name}" to playlist`);
    await api.sendFeedback({ user_id: DEMO_USER_ID, song_id: song.id, action: 'add' });
    // Refresh recs after adding (learns from feedback)
    fetchRecommendations(true);
  };

  const handleSkip = async (song: RecommendedSong) => {
    setSkippedIds((prev) => new Set([...prev, song.id]));
    setRecommendations((prev) => prev.filter((s) => s.id !== song.id));
    await api.sendFeedback({ user_id: DEMO_USER_ID, song_id: song.id, action: 'skip' });
  };

  const handleRemove = async (song: Song) => {
    setPlaylist((prev) => prev.filter((s) => s.id !== song.id));
    await api.removeFromPlaylist(DEMO_USER_ID, song.id);
    showToast(`Removed "${song.name}"`);
  };

  const handlePlay = (song: Song) => {
    if (song.preview_url) {
      play(song.id, song.preview_url);
      api.sendFeedback({ user_id: DEMO_USER_ID, song_id: song.id, action: 'play' });
    } else {
      showToast('No preview available for this track');
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="brand-icon">⟐</span>
            <div>
              <h1 className="brand-title">WAVELENGTH</h1>
              <p className="brand-sub">Adaptive Playlist Engine</p>
            </div>
          </div>

          <div className="header-stats">
            {latencyMs !== null && (
              <div className="stat-pill">
                <span className="stat-dot stat-dot--green" />
                <span>{latencyMs.toFixed(0)}ms</span>
              </div>
            )}
            {algorithm && (
              <div className="stat-pill">
                <span className="stat-dot stat-dot--blue" />
                <span>{algorithm.replace(/_/g, ' ')}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Tabs */}
      <nav className="app-tabs">
        <button
          className={`tab ${tab === 'recommendations' ? 'tab--active' : ''}`}
          onClick={() => setTab('recommendations')}
        >
          For You
          <span className="tab-count">{recommendations.length}</span>
        </button>
        <button
          className={`tab ${tab === 'playlist' ? 'tab--active' : ''}`}
          onClick={() => {
            setTab('playlist');
            fetchPlaylist();
          }}
        >
          My Playlist
          <span className="tab-count">{playlist.length}</span>
        </button>
      </nav>

      {/* Content */}
      <main className="app-main">
        {tab === 'recommendations' && (
          <div className="panel">
            <div className="panel-header">
              <p className="panel-desc">
                Songs ranked for you · Swipe to shape your taste
              </p>
              <button
                className="refresh-btn"
                onClick={() => fetchRecommendations(true)}
                disabled={refreshing}
              >
                {refreshing ? '⟳ Refreshing...' : '⟳ Refresh'}
              </button>
            </div>

            {loading ? (
              <div className="loading-grid">
                {Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} className="skeleton-card" style={{ animationDelay: `${i * 60}ms` }} />
                ))}
              </div>
            ) : (
              <div className="songs-grid">
                {recommendations.map((song, i) => (
                  <SongCard
                    key={song.id}
                    song={song}
                    isPlaying={playingId === song.id}
                    onPlay={() => handlePlay(song)}
                    onAdd={() => handleAdd(song)}
                    onSkip={() => handleSkip(song)}
                    showScore={false}
                    animationDelay={i * 40}
                    variant="recommendation"
                  />
                ))}
                {recommendations.length === 0 && !loading && (
                  <div className="empty-state">
                    <span className="empty-icon">✦</span>
                    <p>All caught up! Refresh for more tracks.</p>
                    <button className="refresh-btn" onClick={() => fetchRecommendations()}>
                      Load more
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {tab === 'playlist' && (
          <div className="panel">
            <div className="panel-header">
              <p className="panel-desc">
                {playlist.length} song{playlist.length !== 1 ? 's' : ''} · Your curated collection
              </p>
            </div>

            {playlist.length === 0 ? (
              <div className="empty-state">
                <span className="empty-icon">♪</span>
                <p>Your playlist is empty. Add songs from recommendations!</p>
                <button className="refresh-btn" onClick={() => setTab('recommendations')}>
                  Browse recommendations
                </button>
              </div>
            ) : (
              <div className="songs-grid">
                {playlist.map((song, i) => (
                  <SongCard
                    key={song.id}
                    song={song}
                    isPlaying={playingId === song.id}
                    onPlay={() => handlePlay(song)}
                    onRemove={() => handleRemove(song)}
                    animationDelay={i * 30}
                    variant="playlist"
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      {/* Toast */}
      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
