"""Silero VAD speech suppression — lazy-loaded singleton."""
from __future__ import annotations
import torch
from .audio import TARGET_SR

_vad_model = None
_read_audio = None
_get_ts = None


def _load() -> None:
    global _vad_model, _read_audio, _get_ts
    if _vad_model is not None:
        return
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
        verbose=False,
    )
    get_ts, _, read_audio, *_ = utils
    _vad_model  = model
    _get_ts     = get_ts
    _read_audio = read_audio


def get_speech_intervals(
    wav_path: str,
    sr: int = TARGET_SR,
    threshold: float = 0.50,
    min_silence_ms: int = 300,
) -> list[tuple[float, float]]:
    _load()
    wav  = _read_audio(wav_path, sampling_rate=sr)
    hits = _get_ts(
        wav, _vad_model,
        sampling_rate=sr,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
    )
    return [(h["start"] / sr, h["end"] / sr) for h in hits]


def is_speech(
    timestamp: float,
    intervals: list[tuple[float, float]],
    tolerance: float = 0.35,
) -> bool:
    return any(
        (s - tolerance) <= timestamp <= (e + tolerance)
        for s, e in intervals
    )
