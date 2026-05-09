"""
Export writers: SRT, SLS, JSON, CSV.

SRT  — industry-standard subtitle format (used by most video players)
SLS  — Simple Lyrics/Subtitle format (used in PlanetRead Same Language Subtitling)
JSON — structured output for downstream processing / Module 2 handoff
CSV  — spreadsheet-friendly for editor review

Every SRT/SLS entry includes a comment line (starting with %) that carries
debug metadata (confidence, frame count, source model) so editors can
understand why a CC was suggested without opening a separate log file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .events import SoundEvent



def _srt_ts(sec: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    ms  = int(round(sec * 1000))
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _sls_ts(sec: float) -> str:
    """Convert seconds to SLS timestamp: HH:MM:SS.mmm"""
    return _srt_ts(sec).replace(",", ".")




def write_srt(
    events: list[SoundEvent],
    path: Path,
    hindi: bool = False,
) -> None:
    """
    Write events to an SRT subtitle file.

    Each subtitle block contains:
      Line 1 — index
      Line 2 — timestamp range
      Line 3 — CC text (English or Hindi)
      Line 4 — % metadata comment (conf / frames / source)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted(events, key=lambda e: e.start_time)
    lines  = []

    for i, ev in enumerate(events, 1):
        start = _srt_ts(ev.start_time)
        end   = _srt_ts(max(ev.end_time + 0.5, ev.start_time + 2.0))
        text  = ev.caption_hi if hindi else ev.caption_en
        lines += [
            str(i),
            f"{start} --> {end}",
            text,
            f"% conf={ev.confidence:.3f} frames={ev.frame_count} src={ev.onset_source}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")



def write_sls(
    events: list[SoundEvent],
    path: Path,
    hindi: bool = False,
) -> None:
    """
    Write events to an SLS (Simple Lyrics Subtitle) file.
    Format:  [HH:MM:SS.mmm] CC text
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    events = sorted(events, key=lambda e: e.start_time)
    lines  = []

    for ev in events:
        ts   = _sls_ts(ev.start_time)
        text = ev.caption_hi if hindi else ev.caption_en
        lines.append(f"[{ts}] {text}")

    path.write_text("\n".join(lines), encoding="utf-8")



def write_json(events: list[SoundEvent], path: Path) -> None:
    """Write full event list to JSON (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [ev.to_dict() for ev in events]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")



_CSV_FIELDS = [
    "label", "caption_en", "caption_hi",
    "start_time", "end_time",
    "start_timestamp", "end_timestamp", "duration",
    "confidence", "frame_count", "onset_source", "yamnet_raw",
]


def write_csv(events: list[SoundEvent], path: Path) -> None:
    """Write events to CSV — one row per event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for ev in events:
            writer.writerow(ev.to_dict())
