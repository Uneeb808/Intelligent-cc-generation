"""Command-line interface for the Intelligent CC Suggestion Tool."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from .audio import extract_audio, is_video, is_audio, MediaError
from .export import write_srt, write_sls, write_json, write_csv
from .vad import get_speech_intervals
from .yamnet import detect


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cc_detector",
        description=(
            "Intelligent CC Suggestion Tool — Module 1\n"
            "YAMNet-based non-speech sound event detection → SRT/SLS/JSON/CSV"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, type=Path, metavar="FILE",
                   help="Input video or audio file")
    p.add_argument("--json", type=Path, default=Path("outputs/events.json"))
    p.add_argument("--csv",  type=Path, default=Path("outputs/events.csv"))
    p.add_argument("--srt",  type=Path, default=Path("outputs/cc_english.srt"))
    p.add_argument("--srt-hi", type=Path, default=Path("outputs/cc_hindi.srt"))
    p.add_argument("--sls",    type=Path, default=None)
    p.add_argument("--sls-hi", type=Path, default=None)
    p.add_argument("--keep-audio", type=Path, default=None)

    p.add_argument("--min-confidence", type=float, default=0.25,
                   help="YAMNet confidence threshold (default: 0.25)")
    p.add_argument("--rms-threshold", type=float, default=0.010,
                   help="Silence gate (default: 0.010)")
    p.add_argument("--merge-gap", type=float, default=1.5,
                   help="Merge gap seconds (default: 1.5)")
    p.add_argument("--top-k", type=int, default=5,
                   help="Top-K labels per frame (default: 5)")
    p.add_argument("--vad-threshold", type=float, default=0.50,
                   help="Silero VAD threshold (default: 0.50)")
    p.add_argument("--consensus-window", type=int, default=3)
    p.add_argument("--consensus-k", type=int, default=2)
    p.add_argument("--no-onset-pass", action="store_true")
    p.add_argument("--block-label", action="append", default=[], metavar="LABEL")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"[ERROR] Input not found: {args.input}", file=sys.stderr)
        return 1
    if not (is_video(args.input) or is_audio(args.input)):
        print(f"[ERROR] Unsupported format: {args.input.suffix}", file=sys.stderr)
        return 1

    if args.block_label:
        from . import labels as lbl_module
        extra = frozenset(
            part.strip().lower()
            for val in args.block_label
            for part in val.split(",") if part.strip()
        )
        lbl_module.BLOCKLIST = lbl_module.BLOCKLIST | extra

    t_start = time.time()
    try:
        with TemporaryDirectory() as tmpdir:
            wav_path = args.keep_audio or Path(tmpdir) / "audio.wav"

            print(f"[1/4] Extracting audio from: {args.input.name}")
            extract_audio(args.input, wav_path)

            print("[2/4] Running Silero VAD (speech suppression)...")
            speech_intervals = get_speech_intervals(
                str(wav_path), threshold=args.vad_threshold
            )
            print(f"       {len(speech_intervals)} speech segment(s) found")

            print("[3/4] Running YAMNet sound event detection...")
            events, stats, infer_time = detect(
                wav_path, speech_intervals,
                conf_thresh      = args.min_confidence,
                rms_thresh       = args.rms_threshold,
                merge_gap        = args.merge_gap,
                top_k            = args.top_k,
                use_onset_pass   = not args.no_onset_pass,
                consensus_window = args.consensus_window,
                consensus_k      = args.consensus_k,
                vad_tolerance    = 0.35,
            )
            print(f"       YAMNet inference: {infer_time:.2f}s")
            print(f"       Gate stats: {stats}")
            print(f"       {len(events)} CC event(s) detected")

            print("[4/4] Exporting outputs...")
            write_json(events, args.json)
            write_csv(events,  args.csv)
            write_srt(events,  args.srt,    hindi=False)
            write_srt(events,  args.srt_hi, hindi=True)
            if args.sls:    write_sls(events, args.sls,    hindi=False)
            if args.sls_hi: write_sls(events, args.sls_hi, hindi=True)

    except MediaError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr); return 1
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        import traceback; traceback.print_exc(); return 1

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print("  CC DETECTION COMPLETE")
    print("=" * 60)
    print(f"  Events detected  : {len(events)}")
    print(f"  Total wall time  : {elapsed:.1f}s")
    print(f"  YAMNet inference : {infer_time:.1f}s")
    print()
    print("  Output files:")
    print(f"    JSON  : {args.json}")
    print(f"    CSV   : {args.csv}")
    print(f"    SRT   : {args.srt}")
    print(f"    SRT   : {args.srt_hi}  (Hindi)")
    if args.sls:    print(f"    SLS   : {args.sls}")
    if args.sls_hi: print(f"    SLS   : {args.sls_hi}  (Hindi)")
    print()

    if events:
        print(f"  {'#':<4} {'Start':>8} {'End':>8}  {'Label':<22} {'Conf':>6}  {'Frames':>6}  {'Src':<8}  Hindi CC")
        print("  " + "─" * 85)
        for i, ev in enumerate(events, 1):
            print(f"  {i:<4} {ev.start_time:>7.2f}s {ev.end_time:>7.2f}s  "
                  f"{ev.label:<22} {ev.confidence:>6.3f}  {ev.frame_count:>6}  "
                  f"{ev.onset_source:<8}  {ev.caption_hi}")
    return 0
