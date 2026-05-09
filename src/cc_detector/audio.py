"""
Audio extraction and loading.

Uses imageio-ffmpeg's bundled binary — no system FFmpeg required.
All downstream models (YAMNet, Silero VAD, librosa) expect 16 kHz mono.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16_000   # YAMNet + Silero VAD both expect 16 kHz mono
_FFMPEG   = imageio_ffmpeg.get_ffmpeg_exe()

SUPPORTED_VIDEO: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".ts", ".m2ts"}
)
SUPPORTED_AUDIO: frozenset[str] = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
)


class MediaError(RuntimeError):
    """Raised when media extraction or audio loading fails."""


def is_video(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_AUDIO


def extract_audio(media_path: Path, out_wav: Path) -> Path:
    """
    Extract 16 kHz mono WAV from any video or audio file.

    Uses the imageio-ffmpeg bundled binary — callers need not install
    system FFmpeg.  Raises MediaError on failure.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _FFMPEG, "-y",
        "-i",  str(media_path),
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        "-f",  "wav",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(
            f"FFmpeg failed on {media_path.name}:\n"
            + (result.stderr[-600:] or "(no stderr)")
        )
    return out_wav


def load_mono_f32(wav_path: Path) -> tuple[np.ndarray, int]:
    """
    Load a WAV file as float32 mono numpy array.
    Returns (samples, sample_rate).
    Normalises amplitude to [-1, 1] if needed (YAMNet expects this range).
    """
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Normalise if raw PCM was decoded outside [-1, 1]
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    return audio, int(sr)
