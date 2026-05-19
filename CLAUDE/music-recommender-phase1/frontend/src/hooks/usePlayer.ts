import { useState, useRef, useCallback } from 'react';

export function usePlayer() {
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const play = useCallback((songId: string, previewUrl: string) => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }

    if (playingId === songId) {
      setPlayingId(null);
      return;
    }

    const audio = new Audio(previewUrl);
    audio.volume = 0.6;
    audioRef.current = audio;

    audio.play().catch(console.error);
    setPlayingId(songId);

    audio.addEventListener('ended', () => setPlayingId(null));
  }, [playingId]);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setPlayingId(null);
  }, []);

  return { playingId, play, stop };
}
