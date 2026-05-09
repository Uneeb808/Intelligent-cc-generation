"""
Core data model: SoundEvent dataclass.

Every detected event flowing through the pipeline is represented as a
SoundEvent.  The to_dict() method produces the export-ready payload for
JSON, CSV, and SRT writers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


def _fmt(seconds: float) -> str:
    """HH:MM:SS.mmm timestamp string."""
    ms  = int(round(seconds * 1000))
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass
class SoundEvent:
    label:          str    
    caption_en:     str    
    caption_hi:     str    
    start_time:     float  
    end_time:       float 
    confidence:     float  
    yamnet_raw:     str    

    frame_count:    int   = 1     
    onset_source:   str   = "yamnet" 
    spectral_gate:  bool  = False  
    top_candidates: list  = field(default_factory=list) 

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def start_ts(self) -> str:
        return _fmt(self.start_time)

    @property
    def end_ts(self) -> str:
        return _fmt(self.end_time)

    def to_dict(self) -> dict:
        return {
            "label":          self.label,
            "caption_en":     self.caption_en,
            "caption_hi":     self.caption_hi,
            "start_time":     round(self.start_time, 3),
            "end_time":       round(self.end_time,   3),
            "start_timestamp": self.start_ts,
            "end_timestamp":   self.end_ts,
            "duration":        round(self.duration, 3),
            "confidence":      round(self.confidence, 4),
            "frame_count":     self.frame_count,
            "onset_source":    self.onset_source,
            "yamnet_raw":      self.yamnet_raw,
        }
