export interface Song {
  id: string;
  name: string;
  artist: string;
  album: string | null;
  album_art_url: string | null;
  preview_url: string | null;
  duration_ms: number | null;
  popularity: number;
  explicit: boolean;
  danceability: number | null;
  energy: number | null;
  valence: number | null;
  tempo: number | null;
  acousticness: number | null;
  genres: string[];
}

export interface RecommendedSong extends Song {
  score: number;
  rank: number;
  explanation: string | null;
}

export interface RecommendationResponse {
  user_id: string;
  songs: RecommendedSong[];
  algorithm: string;
  latency_ms: number;
}

export interface PlaylistResponse {
  user_id: string;
  songs: Song[];
  total: number;
}

export interface FeedbackRequest {
  user_id: string;
  song_id: string;
  action: 'add' | 'skip' | 'play' | 'complete';
  play_duration_ms?: number;
}

export interface UserResponse {
  id: string;
  display_name: string | null;
  total_interactions: number;
  created_at: string;
  last_active: string;
}
