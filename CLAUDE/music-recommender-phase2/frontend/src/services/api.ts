import type {
  RecommendationResponse,
  PlaylistResponse,
  FeedbackRequest,
  UserResponse,
} from '../types';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${res.status}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  // Users
  createUser: (id: string, displayName?: string) =>
    request<UserResponse>('/api/users', {
      method: 'POST',
      body: JSON.stringify({ id, display_name: displayName }),
    }),

  getUser: (userId: string) =>
    request<UserResponse>(`/api/users/${userId}`),

  // Recommendations
  getRecommendations: (userId: string, limit = 20) =>
    request<RecommendationResponse>(
      `/api/recommend/${userId}?limit=${limit}&exclude_playlist=true`
    ),

  // Playlist
  getPlaylist: (userId: string) =>
    request<PlaylistResponse>(`/api/users/${userId}/playlist`),

  removeFromPlaylist: (userId: string, songId: string) =>
    request<void>(`/api/users/${userId}/playlist/${songId}`, { method: 'DELETE' }),

  // Feedback
  sendFeedback: (payload: FeedbackRequest) =>
    request<{ status: string }>('/api/feedback', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  // Health
  health: () => request<{ status: string; db: string; redis: string }>('/api/health'),
};
