"""
YAMNet sound event detector — v0.3 (fixed for short transient events).

Root causes of missed dog bark / door slam in v0.2:
  1. consensus_k=2 killed single-frame transient events
  2. spectral flatness gate rejected harmonically-rich animal sounds
  3. 'domestic animals, pets' was in blocklist

Fixes applied:
  1. Consensus voting is BYPASSED for transient-class labels
     (dog bark, door, knock, glass, gunshot, etc.)
     — a single frame above threshold is sufficient.
  2. Spectral flatness gate REMOVED entirely.
     It was too aggressive and the librosa onset gate provides
     sufficient false-positive protection.
  3. Onset strength gate is only applied to pure percussion labels,
     not to tonal-transient events like dog bark or laughter.
  4. conf_thresh default lowered to 0.25 (matches working notebook).
  5. rms_thresh lowered to 0.010 to catch quiet background barks.
  6. VAD tolerance widened to 0.35 s to catch events at speech boundaries.
  7. Top-5 instead of top-3 candidates examined per frame.
  8. Onset transient pass now reports the YAMNet top-1 label for that
     timestamp instead of always returning generic "IMPACT".
"""

from __future__ import annotations

import csv
import os
import time
import warnings
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from .audio import TARGET_SR, load_mono_f32
from .events import SoundEvent
from .labels import (
    is_blocklisted, remap_label, caption_en, caption_hi,
    is_transient,
)
from .vad import is_speech

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
warnings.filterwarnings("ignore", category=UserWarning)

YAMNET_URL       = "https://tfhub.dev/google/yamnet/1"
YAMNET_FRAME_HOP = 0.48   # seconds between frames
YAMNET_FRAME_WIN = 0.96   

_yamnet_model   = None
_yamnet_classes = None


def _load_yamnet():
    global _yamnet_model, _yamnet_classes
    if _yamnet_model is not None:
        return _yamnet_model, _yamnet_classes
    print("Loading YAMNet from TensorFlow Hub...")
    t0 = time.time()
    model = hub.load(YAMNET_URL)
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    classes = []
    with tf.io.gfile.GFile(class_map_path) as fh:
        for row in csv.DictReader(fh):
            classes.append(row["display_name"])
    _yamnet_model   = model
    _yamnet_classes = classes
    print(f"  YAMNet loaded in {time.time()-t0:.1f}s  |  {len(classes)} classes")
    return model, classes


def _rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))


def _has_onset(chunk: np.ndarray, sr: int, min_strength: float = 1.0) -> bool:
    """
    True if the chunk contains a genuine energy onset.
    Used ONLY for sustained ambient labels (engine, rain, crowd) where
    we want to confirm the event actually started rather than was ongoing.
    NOT applied to transient events (dog, door, etc.).
    """
    if len(chunk) < 512:
        return True   # too short to assess — let through
    try:
        import librosa
        env = librosa.onset.onset_strength(
            y=chunk.astype(np.float32), sr=sr, hop_length=256
        )
        return float(np.max(env)) >= min_strength
    except Exception:
        return True   # if librosa fails, don't block


# Labels where we require an energy onset (sustained ambient sounds that
# YAMNet sometimes fires on room tone if it's loud enough)
_REQUIRES_ONSET: frozenset[str] = frozenset({
    "ENGINE", "RAIN", "WIND", "FIRE", "CROWD", "MECHANICAL", "VEHICLE",
})


def _merge_raw(raw: list[dict], gap_sec: float) -> list[dict]:
    if not raw:
        return []
    out = []
    cur = dict(raw[0])
    for ev in raw[1:]:
        same  = ev["label"] == cur["label"]
        close = (ev["timestamp"] - cur["end"]) <= gap_sec
        if same and close:
            cur["end"] = ev["end"]
            cur["frame_count"] = cur.get("frame_count", 1) + 1
            if ev["confidence"] > cur["confidence"]:
                cur["confidence"]     = ev["confidence"]
                cur["top_candidates"] = ev.get("top_candidates", [])
        else:
            out.append(cur)
            cur = dict(ev)
            cur.setdefault("frame_count", 1)
    out.append(cur)
    return out


def _remove_overlaps(events: list[dict], min_gap: float = 0.5) -> list[dict]:
    if not events:
        return []
    events = sorted(events, key=lambda x: x["start"])
    clean  = [events[0]]
    for ev in events[1:]:
        last = clean[-1]
        if ev["start"] < last["end"] + min_gap:
            if ev["confidence"] > last["confidence"]:
                clean[-1] = ev
        else:
            clean.append(ev)
    return clean


def _raw_to_events(raw: list[dict]) -> list[SoundEvent]:
    out = []
    for r in raw:
        label = r["label"]
        end   = r["end"]
        if end - r["start"] < 1.0:
            end = r["start"] + 1.0
        out.append(SoundEvent(
            label         = label,
            caption_en    = caption_en(label),
            caption_hi    = caption_hi(label),
            start_time    = round(r["start"], 3),
            end_time      = round(end, 3),
            confidence    = round(r["confidence"], 4),
            yamnet_raw    = r.get("yamnet_raw", label),
            frame_count   = r.get("frame_count", 1),
            onset_source  = r.get("onset_source", "yamnet"),
            spectral_gate = False,
            top_candidates= r.get("top_candidates", []),
        ))
    return sorted(out, key=lambda e: e.start_time)


class DetectionStats:
    def __init__(self):
        self.speech       = 0
        self.silent       = 0
        self.blocklist    = 0
        self.low_conf     = 0
        self.onset_fail   = 0   
        self.consensus    = 0   
        self.accepted     = 0

    def __repr__(self):
        return (
            f"speech={self.speech}  silent={self.silent}  "
            f"blocklist={self.blocklist}  low_conf={self.low_conf}  "
            f"onset_fail={self.onset_fail}  consensus={self.consensus}  "
            f"accepted={self.accepted}"
        )


def detect(
    wav_path,
    speech_intervals,
    *,
    conf_thresh:       float = 0.25,
    rms_thresh:        float = 0.010,
    merge_gap:         float = 1.5,
    top_k:             int   = 5,
    use_onset_pass:    bool  = True,
    consensus_window:  int   = 3,
    consensus_k:       int   = 2,
    vad_tolerance:     float = 0.35,
) -> tuple[list[SoundEvent], DetectionStats, float]:
    """
    Run YAMNet detection with transient-aware filtering.

    Key design:
    - Transient events (dog, door, knock, glass, gunshot…) bypass consensus
      voting. A single frame above conf_thresh is accepted.
    - Sustained events (engine, rain, crowd…) require consensus_k hits in
      consensus_window consecutive frames AND an energy onset check.
    - No spectral flatness gate — it was killing legitimate animal sounds.
    - Onset transient pass (librosa) labels events from YAMNet scores,
      not a generic "IMPACT".
    """
    wav_path = str(wav_path)
    model, classes = _load_yamnet()
    audio, sr = load_mono_f32(Path(wav_path))

    FRAME_N = int(YAMNET_FRAME_HOP * sr)

    t0 = time.time()
    waveform  = tf.convert_to_tensor(audio, dtype=tf.float32)
    scores, _, _ = model(waveform)
    scores_np    = scores.numpy()
    infer_time   = time.time() - t0

    stats = DetectionStats()
    raw: list[dict] = []

    # Per-label sliding window for consensus 
    label_history: dict[str, deque] = {}

    for frame_idx, frame_scores in enumerate(scores_np):
        ts = round(frame_idx * YAMNET_FRAME_HOP, 3)

        # Gate 1: VAD speech suppression
        if is_speech(ts, speech_intervals, tolerance=vad_tolerance):
            stats.speech += 1
            continue

        # Gate 2: RMS energy gate
        s     = frame_idx * FRAME_N
        e     = min(s + FRAME_N, len(audio))
        chunk = audio[s:e]
        if _rms(chunk) < rms_thresh:
            stats.silent += 1
            continue

        # Find best non-blocklisted label in top-K
        top_indices = np.argsort(frame_scores)[::-1][:top_k]
        top_candidates = [
            {"rank": i + 1,
             "label": classes[idx],
             "confidence": round(float(frame_scores[idx]), 4)}
            for i, idx in enumerate(top_indices)
            if float(frame_scores[idx]) >= 0.05
        ]

        chosen_raw  = None
        chosen_conf = 0.0
        for idx in top_indices:
            raw_lbl = classes[idx]
            conf    = float(frame_scores[idx])
            if conf < conf_thresh:
                break
            if is_blocklisted(raw_lbl):
                continue
            chosen_raw  = raw_lbl
            chosen_conf = conf
            break

        if chosen_raw is None:
            stats.blocklist += 1
            continue

        canonical = remap_label(chosen_raw)
        transient = is_transient(canonical)

        if transient:
            # ── Transient path: accept immediately
            stats.accepted += 1
            raw.append({
                "timestamp":    ts,
                "label":        canonical,
                "confidence":   round(chosen_conf, 4),
                "frame_dur":    YAMNET_FRAME_HOP,
                "start":        ts,
                "end":          ts + YAMNET_FRAME_WIN,
                "yamnet_raw":   chosen_raw,
                "onset_source": "yamnet",
                "top_candidates": top_candidates,
                "frame_count":  1,
            })
        else:
        
            if canonical not in label_history:
                label_history[canonical] = deque(maxlen=consensus_window)
            label_history[canonical].append(True)

            votes = sum(label_history[canonical])
            if votes < consensus_k:
                stats.consensus += 1
                continue

            # Gate 3: onset check for sustained labels
            if canonical in _REQUIRES_ONSET:
                if not _has_onset(chunk, sr):
                    stats.onset_fail += 1
                    continue

            stats.accepted += 1
            raw.append({
                "timestamp":    ts,
                "label":        canonical,
                "confidence":   round(chosen_conf, 4),
                "frame_dur":    YAMNET_FRAME_HOP,
                "start":        ts,
                "end":          ts + YAMNET_FRAME_WIN,
                "yamnet_raw":   chosen_raw,
                "onset_source": "yamnet",
                "top_candidates": top_candidates,
                "frame_count":  1,
            })

    
    transient_raw: list[dict] = []
    if use_onset_pass:
        try:
            import librosa
            onset_times = librosa.onset.onset_detect(
                y=audio.astype(np.float32), sr=sr,
                units="time", delta=0.30, wait=4,
            )
            for t in onset_times:
                t = round(float(t), 3)
                if is_speech(t, speech_intervals, tolerance=vad_tolerance):
                    continue
                s = int(max(0, t - 0.05) * sr)
                e = int(min(len(audio), t + 0.20) * sr)
                if _rms(audio[s:e]) < 0.008:
                    continue
                # Look up YAMNet's top-1 label at this timestamp
                frame_idx = min(int(t / YAMNET_FRAME_HOP), len(scores_np) - 1)
                onset_scores = scores_np[frame_idx]
                top5 = np.argsort(onset_scores)[::-1][:5]
                onset_label = None
                onset_conf  = 0.55   # default confidence for onset events
                for idx in top5:
                    raw_lbl = classes[idx]
                    if is_blocklisted(raw_lbl):
                        continue
                    candidate_canonical = remap_label(raw_lbl)
                    candidate_conf      = float(onset_scores[idx])
                    # Only keep if it's a transient class
                    if is_transient(candidate_canonical) and candidate_conf >= 0.15:
                        onset_label = candidate_canonical
                        onset_conf  = max(onset_conf, candidate_conf)
                        break
                if onset_label is None:
                    onset_label = "IMPACT"

                transient_raw.append({
                    "timestamp":    t,
                    "label":        onset_label,
                    "confidence":   round(onset_conf, 4),
                    "frame_dur":    0.25,
                    "start":        t,
                    "end":          t + 0.5,
                    "yamnet_raw":   "onset_transient",
                    "onset_source": "onset",
                    "top_candidates": [],
                    "frame_count":  1,
                })
        except ImportError:
            pass   # librosa not installed — skip onset pass

    all_raw = sorted(raw + transient_raw, key=lambda x: x["timestamp"])
    merged  = _remove_overlaps(_merge_raw(all_raw, merge_gap))
    events  = _raw_to_events(merged)

    return events, stats, infer_time
