# Clipper Highlights

`clipper-highlights` is a Python project for turning long gameplay recordings into ranked highlight clips.

It uses:

- `ffmpeg` to extract audio and export final clips
- `faster-whisper` for speech-to-text with timestamps
- `librosa` to detect adaptive audio events instead of a brittle fixed dB threshold
- transcript keyword matching to identify likely combat or hype moments
- a deterministic candidate generator to build windows around those moments
- an optional Gemini ranking step to choose the strongest final clips

## Why this architecture

The pipeline is split into two stages on purpose:

1. Candidate generation is deterministic and cheap.
2. Final clip selection is semantic and optional.

That keeps the LLM from doing the entire job and makes the system easier to tune, debug, and benchmark.

## Project layout

```text
src/clipper_highlights/
  audio.py          # audio extraction + adaptive spike detection
  transcription.py  # faster-whisper integration
  candidates.py     # keyword matching + window generation
  llm.py            # Gemini or heuristic ranking
  exporter.py       # ffmpeg clip export
  pipeline.py       # end-to-end orchestration
  cli.py            # typer CLI
```

## Requirements

- Python 3.11+
- `ffmpeg` available on `PATH`

## Install

```bash
cd ~/github/personal/clipper-highlights
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

Create a starter config:

```bash
clipper-highlights init-config config.yaml
```

Run the full pipeline:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --game tarkov \
  --output-dir runs/tarkov-session-01
```

If you want deterministic ranking only:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --no-llm
```

## Output

Each run writes:

- `artifacts/audio.wav`
- `artifacts/transcript.json`
- `artifacts/audio_spikes.json`
- `artifacts/timeline_bundle.json`
- `artifacts/ranked_clips.json`
- `clips/*.mp4`

## Gemini support

Set a Gemini API key if you want semantic ranking:

```bash
export GEMINI_API_KEY=your_key_here
```

The LLM stage receives a compact JSON summary of top candidate windows and must return JSON only. If Gemini fails or no API key is present, the pipeline falls back to heuristic ranking.

## Tuning ideas

- Add game-specific HUD OCR or kill-feed detection
- Feed separate mic and game-audio tracks instead of a mixed track
- Replace keyword rules with a lightweight event classifier
- Store accepted and rejected clips to train a learned ranker later
- Add scene-cut snapping before export
