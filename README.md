# Intelligent CC Suggestion Tool — Module 1

**PlanetRead DMP 2026** | Sound Event Detection → SRT/SLS Output

A Python backend pipeline that accepts any video or audio file and produces
closed-caption suggestions for meaningful non-speech audio events —
honking, laughter, glass breaking, explosions, applause — without
over-captioning ambient sounds.

---

## What's Different Here vs. Baseline YAMNet

| Feature | Basic YAMNet | This implementation |
|---|---|---|
| False positive filter | confidence threshold only | **5-gate pipeline** |
| Near-silence "snake" FP | manual blocklist | RMS gate + blocklist |
| Music mislabelled as explosion | blocklist | **Spectral flatness gate** |
| No-onset sustained noise | threshold | **Onset strength gate** |
| Single-frame spurious hits | none | **Temporal consensus voting** |
| Short clicks/knocks (<0.2s) | missed | **librosa transient pass** |
| SRT only | yes | **SRT + SLS + JSON + CSV** |
| Hindi labels | keyword map | Extended tuple-based keyword matching |
| Debug metadata per event | none | confidence + frames + source in SRT |

---

## Setup

No system FFmpeg required — `imageio-ffmpeg` bundles its own binary.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Run on a Video

```bash
python detect.py \
  --input samples/hindi_clip.mp4 \
  --json  outputs/events.json \
  --csv   outputs/events.csv \
  --srt   outputs/cc_english.srt \
  --srt-hi outputs/cc_hindi.srt
```

### All options

```
--input FILE              Input video or audio file
--json PATH               JSON output (default: outputs/events.json)
--csv  PATH               CSV output  (default: outputs/events.csv)
--srt  PATH               English SRT (default: outputs/cc_english.srt)
--srt-hi PATH             Hindi SRT   (default: outputs/cc_hindi.srt)
--sls  PATH               English SLS (optional)
--sls-hi PATH             Hindi SLS   (optional)
--keep-audio PATH         Save extracted WAV (optional)

--min-confidence FLOAT    YAMNet confidence threshold (default: 0.28)
--rms-threshold  FLOAT    Silence gate threshold (default: 0.015)
--merge-gap      SEC      Gap to merge same-label events (default: 1.5)
--top-k          INT      Top-K labels per frame (default: 3)
--consensus-window INT    Look-back window for voting (default: 3)
--consensus-k    INT      Minimum votes to accept (default: 2)
--vad-threshold  FLOAT    Silero VAD threshold (default: 0.50)

--no-spectral-gates       Disable spectral + onset gates (faster)
--no-onset-pass           Disable librosa transient detection
--block-label LABEL       Suppress extra YAMNet label (repeatable)
```

---

## Output Example

```
events.json:
[
  {
    "label": "GUNSHOT",
    "caption_en": "[gunshot]",
    "caption_hi": "[गोली की आवाज़]",
    "start_time": 9.6,
    "end_time": 12.48,
    "confidence": 0.9488,
    "frame_count": 6,
    "onset_source": "yamnet",
    "yamnet_raw": "Gunshot, gunfire"
  }
]
```

```
cc_english.srt:
1
00:00:09,600 --> 00:00:12,480
[gunshot]
% conf=0.949 frames=6 src=yamnet
```

---

## Project Structure

```
intelligent-cc-tool/
├── src/
│   └── cc_detector/
│       ├── __init__.py       version
│       ├── __main__.py       entry point
│       ├── cli.py            argparse CLI (all options)
│       ├── events.py         SoundEvent dataclass
│       ├── labels.py         blocklist + remapping + Hindi map
│       ├── audio.py          FFmpeg extraction, WAV loading
│       ├── vad.py            Silero VAD speech detection
│       ├── spectral.py       RMS, spectral flatness, onset gates
│       ├── yamnet.py         5-gate detection + consensus voting
│       └── export.py         SRT / SLS / JSON / CSV writers
├── detect.py                 root runner
├── requirements.txt
└── README.md
```

---

## Limitations (Module 1 scope)

- Module 2 (visual speaker-reaction scoring) not yet integrated.
- YAMNet trained on English-centric AudioSet — India-specific sounds
  (dhol, shehnai) may have lower accuracy.
- First run requires internet to download YAMNet from TF Hub (~12 MB).

## Next Steps (Module 2)

- Extract video frames at each event timestamp using OpenCV.
- Run MediaPipe Pose + FaceMesh to detect speaker reactions.
- Combine: `final = 0.6 × audio_conf + 0.4 × visual_reaction_score`.
- Emit CC only when `final > 0.50`.
