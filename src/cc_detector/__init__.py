"""
cc_detector — Intelligent Closed Caption Suggestion Tool.
Module 1 MVP: YAMNet-based non-speech sound event detection.

Improvements over baseline YAMNet approaches:
  - 5-gate filtering pipeline (speech, RMS, spectral, harmonic, blocklist)
  - Top-K consensus voting across frames before accepting an event
  - Temporal smoothing with a sliding-window majority vote
  - Librosa spectral harmonic gating suppresses tonal music artefacts
  - Librosa onset detection catches short transients (<0.2s) YAMNet misses
  - Dual SRT output: English + Hindi with tuple-based keyword matching
  - JSON + CSV + SRT export with per-event metadata debug fields
"""

__version__ = "0.2.0"

