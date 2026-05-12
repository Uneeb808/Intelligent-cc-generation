"""Spectral helpers — kept minimal after v0.3 fixes removed the flatness gate."""
from __future__ import annotations
import numpy as np

def rms(chunk: np.ndarray) -> float:
    return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
