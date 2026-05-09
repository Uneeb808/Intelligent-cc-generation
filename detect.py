"""
Root-level entry point.

    python detect.py --input video.mp4 [options]

This simply re-exports the CLI main() so the tool can be run from
the project root without installing the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cc_detector.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
