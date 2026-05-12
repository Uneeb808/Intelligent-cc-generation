# PR: Intelligent CC Suggestion Tool — Module 1 Complete

## Summary

This PR delivers a fully working **Module 1** (Sound Event Detection → SRT/SLS output) and lays the architectural groundwork for Module 2 (Visual Reaction Detection). The pipeline accepts any video or audio file and produces closed-caption suggestions for meaningful non-speech audio events — without over-captioning ambient sounds.

**What this PR includes:**
- Full YAMNet-based detection pipeline with a transient-aware 3-path filter (not just a confidence threshold)
- English + Hindi SRT/SLS/JSON/CSV export — no translation API, fully offline
- Silero VAD speech suppression so speech frames never become false CC events
- librosa onset pass to catch short transients (<0.2s) YAMNet's window misses
- Architectural groundwork for Module 2 visual reaction scoring

---

## Pipeline Architecture

```
INPUT VIDEO
    ├──▶ AUDIO EXTRACTION (imageio-ffmpeg, no system install needed)
    │           │
    │    ┌──────┴──────┐
    │    │ Silero VAD  │──▶ speech intervals (suppressed from detection)
    │    └─────────────┘
    │           │
    │    ┌──────▼──────────────────────────────────┐
    │    │  YAMNet  ·  RMS gate  ·  Blocklist     │
    │    │                                         │
    │    │  Transient? ──YES──▶ accept immediately │
    │    │      │               (dog bark, gunshot,│
    │    │      NO              door slam, glass…) │
    │    │      ▼                                  │
    │    │  Consensus voting (2/3 frames)          │
    │    │  + onset check (engine, rain, crowd)    │
    │    └──────────────────┬──────────────────────┘
    │                       │
    │           librosa onset pass (catches <0.2s events)
    │                       │
    │           Merge · Deduplicate · Sort
    │
    └──▶ SRT (EN + HI)  ·  SLS  ·  JSON  ·  CSV
```

---

## Run

```bash
python detect.py --input video.mp4 --srt outputs/cc_en.srt --srt-hi outputs/cc_hi.srt
```

📎 **Colab links:** https://colab.research.google.com/drive/1aAbBrZBw1xg8ASqS98lyCewVWRSZb_Bj?usp=sharing,
https://colab.research.google.com/drive/15kpMJkWYWQO0sBoJZhYFMqcRBbLLzVMy?usp=sharing
 

---

## Research: Benchmark Across 5 Model Families

Before settling on YAMNet as the production solution, we benchmarked five model families. Here's what we found.

---

### WAV2CLIP + CLAP — Not viable

Both models embed audio into CLIP/text space and score against text prompts via cosine similarity. In theory, free-form labels; in practice:

- **CLAP (HTSAT-base)** had repeated checkpoint/architecture mismatches — `laion_clap`'s `load_ckpt()` silently builds a different model width depending on `enable_fusion`, causing a `RuntimeError` on every load attempt across three configurations
- **WAV2CLIP** loaded but produced inconsistent, low-confidence labelling — it lives in CLIP's *visual* embedding space, which wasn't built for diverse audio
- **Verdict:** The cosine-similarity approach is brittle without a dedicated audio backbone. Not worth pursuing further.

---

### PANNs CNN14 — Better mAP, but wrong fit for CC

PANNs CNN14 (mAP 0.385 vs YAMNet's 0.306) is technically a stronger AudioSet model. A fair benchmark was run with blocklist off and thresholds matched to YAMNet sensitivity.

**The problem:** PANNs is trained on AudioSet's full 527-class hierarchy including very broad meta-classes — `"Music"`, `"Animal"`, `"Sound"`. On a real video, over 1500 frames fired on these broad labels. They're not wrong, but they're not CC-worthy. With blocklist on, too many real events get suppressed as collateral; with it off, the output is noisy.

PANNs' higher mAP comes from scoring those broad categories well. For CC specifically — where you want narrow, specific, actionable events — the broad-label training is a liability, not an asset. **YAMNet's narrower 521-class set, which looks like a weakness on paper, is actually an advantage here.**

---

### Qwen2-Audio-7B — Most promising, fine-tune path forward

Qwen2-Audio is a 7B audio-language model (Whisper-large-v2 encoder + LLM). Instead of cosine similarity or fixed class indices, it reasons about audio in natural language and returns structured JSON.

**What stood out:**
- Contextual descriptions, not bare labels: `"glass breaking, likely a fight scene"` — directly useful for CC editors reviewing output
- Native Hindi/multilingual support — no separate translation step needed
- Zero-shot on any category; the label bank is a prompt, not a fixed classifier
- Self-reported confidence calibrated better than embedding-similarity scores

**Limitation:** 7B params needs ~14GB VRAM at full precision; we ran 4-bit quantized on a T4 (fits in 15GB). Inference is ~3–5× slower than YAMNet.

**Fine-tuning is the path forward.** Qwen2-Audio can be adapted to Indian content without starting from scratch:
1. Start from AudioSet classes YAMNet is trained on as the base — strong prior already exists
2. Augment with clips for underrepresented India-specific sounds: dhol, shehnai, auto-rickshaw horn, switch/click sounds, crowd chanting
3. Map new classes to existing AudioSet parents where possible (dhol → `Drum`, shehnai → `Wind instrument`) so existing weights transfer
4. Fine-tune only the output mapping / last few layers — audio encoder is already strong
5. A ~1000-clip augmented dataset fine-tunes in 2–3 hours on a T4

---

### Full Comparison Table

| | YAMNet | PANNs CNN14 | CLAP | WAV2CLIP | Qwen2-Audio |
|---|---|---|---|---|---|
| AudioSet mAP | 0.306 | 0.385 | ~0.47 | ~0.40 | LLM-based |
| Parameters | 3.7M | 81M | 87M | ~60M | 7B |
| Label type | Fixed 521 | Fixed 527 | Free text | Free text | Free text + reasoning |
| Hindi support | manual map | manual map | via prompt | via prompt | native |
| India-specific sounds | poor | poor | moderate | poor | best (zero-shot) |
| False positive control | gates + blocklist | blocklist too aggressive | mAP ceiling | inconsistent | LLM reasoning |
| Speed | fastest | fast | fast | moderate | slow |
| Offline | ✅ | ✅ | ✅ | ✅ | ✅ (quantized) |
| **Verdict** | ✅ **production** | ❌ noisy for CC | ❌ load errors | ❌ low quality | 🔬 fine-tune target |



cc @abinash-sketch @keerthiseelan-planetread
