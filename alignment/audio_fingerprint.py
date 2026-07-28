"""
Audio fingerprinting for locating where a short clip was cut from within a
longer source video.

Uses librosa's onset-strength envelope (a 1-D transient-energy signal over
time) rather than raw loudness, because it stays comparatively robust to
loudness normalization / background music a clipper's edit might add on top
of the original stream audio, since it emphasizes transients (speech/impact
onsets) rather than absolute level.

Audio is extracted directly to mono WAV at TARGET_SR via yt-dlp's ffmpeg
postprocessor (rather than downloading full-quality audio and resampling in
Python), keeping cached files small — a source VOD's audio at 11025 Hz mono
is ~1.3MB/min, so even a 2+ hour VOD is well under 200MB.
"""

import logging
import subprocess
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SR = 11025
HOP_LENGTH = 512
FPS = TARGET_SR / HOP_LENGTH  # ~21.5 envelope frames/sec


def download_audio(video_id: str, url: str, cache_dir: Path) -> Path | None:
    """Downloads and extracts mono WAV audio at TARGET_SR, cached by video_id.

    Failures (e.g. age-gated videos requiring authenticated cookies we don't
    have) are cached too via a `.failed` sentinel, so a video that can't be
    fetched isn't re-attempted on every single clip that brute-forces
    through the candidate pool.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{video_id}.wav"
    failed_marker = cache_dir / f"{video_id}.failed"
    if out_path.exists():
        return out_path
    if failed_marker.exists():
        return None

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {TARGET_SR} -ac 1",
        "-o", str(cache_dir / f"{video_id}.%(ext)s"),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not out_path.exists():
        logger.warning("Audio download failed for %s:\n%s", video_id, result.stderr.strip()[-500:])
        failed_marker.write_text(result.stderr.strip()[-2000:], encoding="utf-8")
        return None
    return out_path


def onset_envelope(audio_path: Path, cache_path: Path | None = None) -> np.ndarray | None:
    """Onset-strength envelope for the given audio file, cached to .npy if a
    cache_path is given (VODs are long — this shouldn't be recomputed)."""
    if cache_path and cache_path.exists():
        return np.load(cache_path)

    try:
        y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    except Exception as exc:
        logger.warning("Failed to load audio %s: %s", audio_path, exc)
        return None

    if len(y) < HOP_LENGTH * 4:
        logger.warning("Audio too short to fingerprint: %s", audio_path)
        return None

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, env)
    return env


def get_envelope(video_id: str, url: str, audio_cache_dir: Path, envelope_cache_dir: Path) -> np.ndarray | None:
    """Full pipeline: download (if needed) -> compute envelope (if needed), both cached by video_id."""
    envelope_cache_path = envelope_cache_dir / f"{video_id}.npy"
    if envelope_cache_path.exists():
        return np.load(envelope_cache_path)

    audio_path = download_audio(video_id, url, audio_cache_dir)
    if audio_path is None:
        return None
    return onset_envelope(audio_path, cache_path=envelope_cache_path)
