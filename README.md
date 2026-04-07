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
git clone https://github.com/drshotyou/clipper-highlights.git
cd clipper-highlights
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Usage Guide

The project exposes two commands:

- `clipper-highlights init-config <path>` to generate a starter YAML config
- `clipper-highlights run <video>` to process a recording and export clips

The shortest local workflow is:

```bash
cd clipper-highlights
source .venv/bin/activate
clipper-highlights init-config config.yaml
clipper-highlights run /path/to/session.mp4 --config config.yaml --output-dir runs/session-01
```

If you do not want to install the package into the current shell, this also works from the repo root:

```bash
PYTHONPATH=src python3 -m clipper_highlights init-config config.yaml
PYTHONPATH=src python3 -m clipper_highlights run /path/to/session.mp4 --config config.yaml
```

## Setup Steps

### 1. Generate a config file

Create a starter config:

```bash
clipper-highlights init-config config.yaml
```

That config controls:

- the Whisper model and transcription settings
- adaptive audio detection thresholds
- transcript keywords and weights
- Gemini ranking behavior
- export codec settings

### 2. Tune the config for the target game

The first fields worth editing are:

- `transcription.hotwords`: game names, map names, weapon names, teammate names
- `candidate.keywords`: phrases that usually indicate a fight, clutch, reaction, or payoff moment
- `llm.provider`: set to `none` for deterministic-only ranking or `gemini` for semantic reranking

For example, for Tarkov you would usually want terms like `pmc`, `scav`, `extract`, and common squad callouts.

### 3. Run the pipeline

Basic local run:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --game tarkov \
  --output-dir runs/tarkov-session-01
```

If `--output-dir` is omitted, the default is:

```text
runs/<input-video-stem>
```

### 4. Review the generated artifacts

Each run writes intermediate files as well as final clips, which makes debugging easier:

- `artifacts/audio.wav`
- `artifacts/transcript.json`
- `artifacts/audio_spikes.json`
- `artifacts/timeline_bundle.json`
- `artifacts/ranked_clips.json`
- `clips/*.mp4`

The two most useful debugging artifacts are:

- `timeline_bundle.json` to inspect merged transcript/audio evidence and candidate windows
- `ranked_clips.json` to inspect the final chosen windows before or after export

## Multitrack Audio

If your recording contains multiple audio streams, the pipeline can now select specific stream indices instead of blindly taking the first audio track.

Example OBS-style layout:

- stream `0`: Discord
- stream `1`: game audio
- stream `2`: mic

If you want to exclude Discord and use only game + mic, set:

```yaml
audio:
  stream_indices: [1, 2]
```

If you want to analyze just one track, use a single index:

```yaml
audio:
  stream_indices: [2]
```

Behavior:

- no `stream_indices`: ffmpeg picks the default audio stream
- one stream index: that specific stream is extracted
- multiple stream indices: those streams are mixed together before transcription and audio analysis

## Common Run Variants

Run without Gemini:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --no-llm
```

Generate artifacts only and skip clip export:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --no-export
```

Force recomputation of cached artifacts:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --force
```

Temporarily add keyword overrides without editing the YAML file:

```bash
clipper-highlights run /path/to/session.mp4 \
  --config config.yaml \
  --keyword "pmc=3.0" \
  --keyword "one shot=2.0"
```

## Recommended First Run

If you are giving this to someone else, the most reliable first-run path is:

1. Generate `config.yaml`.
2. Set `llm.provider` to `none`.
3. Run one short recording first, not a huge full-session archive.
4. Inspect `artifacts/timeline_bundle.json`.
5. Tune `transcription.hotwords` and `candidate.keywords`.
6. Re-run with `--force`.
7. Enable Gemini only after the deterministic candidate windows already look reasonable.

## Justfile

If you use `just`, the repo includes shortcuts for the common local and Docker workflows:

```bash
just
just test
just build
just build-gpu
just init-config
just cli run /path/to/session.mp4 --config config.yaml --no-llm
just docker init-config /workspace/config.yaml
just docker run /data/session.mp4 --config /workspace/config.yaml --output-dir /workspace/runs/session-01
just docker-gpu run /data/session.mp4 --config /workspace/config.yaml --output-dir /workspace/runs/session-01
```

## Docker

The repo includes a CPU-first Docker setup with `ffmpeg` and persistent model caches.

### Platform Notes

The Docker setup is intended to work on Linux, macOS, and Windows with Docker Desktop.

The container itself is Linux-based, but that is fine on Windows because Docker Desktop runs Linux containers.

For Windows users:

- use Docker Desktop, not a native Python install, if you want the simplest path
- use the provided `.env` file instead of inline environment-variable syntax
- if you set `CLIPPER_MEDIA_DIR` to an absolute Windows path, use forward slashes like `C:/Users/you/Videos/recordings`
- GPU support in Docker Desktop is only available on Windows with the WSL2 backend and an NVIDIA GPU with up-to-date WSL2-compatible drivers. Source: [Docker GPU support on Windows](https://docs.docker.com/desktop/features/gpu/)

### First-Time Docker Setup

Copy the example environment file:

Linux/macOS:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

CMD:

```cmd
copy .env.example .env
```

Then edit `.env` if needed:

- leave `CLIPPER_MEDIA_DIR=./data` if your videos will live inside the repo’s `data/` folder
- set `CLIPPER_MEDIA_DIR` to another host directory if your recordings live elsewhere
- set `GEMINI_API_KEY` if you want Gemini reranking
- set `OPENAI_API_KEY` if you want OpenAI reranking

### CPU Docker Run

Build the image:

```bash
cd clipper-highlights
docker compose build
```

Create a starter config through the container:

```bash
docker compose run --rm clipper-highlights init-config /workspace/config.yaml
```

Run the pipeline with media mounted at `/data`:

```bash
mkdir -p data runs
cp /path/to/session.mp4 data/

docker compose run --rm \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  clipper-highlights run /data/session.mp4 \
  --config /workspace/config.yaml \
  --game tarkov \
  --output-dir /workspace/runs/tarkov-session-01
```

### GPU Docker Run

There are two separate Docker images in this repo:

- `Dockerfile` for CPU
- `Dockerfile.gpu` for NVIDIA GPU

The GPU path is specifically for `faster-whisper`. Audio feature extraction with `librosa` is still CPU-bound, and ffmpeg export remains CPU-based by default.

Build the GPU image:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml build
```

Run with the GPU override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm clipper-highlights run /data/session.mp4 \
  --config /workspace/config.yaml \
  --game tarkov \
  --output-dir /workspace/runs/tarkov-session-01
```

Recommended config for GPU transcription:

```yaml
transcription:
  device: auto
  compute_type: int8_float16
```

If GPU detection is working, the startup logs should now show:

```text
Loading Whisper model 'medium.en' on device 'cuda'
```

If it still says `cpu`, check:

1. Docker Desktop GPU support is enabled and working on the host.
2. `docker run --rm --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark` works on the machine. Source: [Docker GPU support on Windows](https://docs.docker.com/desktop/features/gpu/)
3. You used the GPU compose override or the GPU Dockerfile, not the default CPU image.
4. `transcription.device` is `auto` or `cuda`, not `cpu`.

If your recordings live outside the repo, update `.env` instead of using shell-specific inline syntax:

```env
CLIPPER_MEDIA_DIR=/absolute/path/to/recordings
```

Windows example:

```env
CLIPPER_MEDIA_DIR=C:/Users/you/Videos/recordings
```

Then run the same Docker command:

```bash
docker compose run --rm clipper-highlights run /data/session.mp4 \
  --config /workspace/config.yaml \
  --output-dir /workspace/runs/session-01
```

You can also use plain `docker run`:

```bash
docker build -t clipper-highlights .

docker run --rm \
  -v /absolute/path/to/clipper-highlights:/workspace \
  -v /absolute/path/to/recordings:/data \
  -v clipper-highlights-cache:/cache \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  clipper-highlights run /data/session.mp4 \
  --config /workspace/config.yaml \
  --output-dir /workspace/runs/session-01
```

Notes:

- The default container setup runs on CPU.
- Whisper model downloads are cached in the named `clipper-cache` volume.
- The compose service mounts the repo into `/workspace`, so local code changes are visible without rebuilding the image unless dependencies change.
- The long-form bind mount syntax in `compose.yaml` is used to avoid Windows drive-letter parsing problems.

## LLM Provider Setup

The project supports these ranking modes:

- `none`: deterministic heuristic ranking only
- `gemini`: Google Gemini JSON ranking
- `openai`: OpenAI Responses API JSON ranking

### Gemini

Set a Gemini API key if you want Gemini ranking:

```bash
export GEMINI_API_KEY=your_key_here
```

Example config:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash
  api_key_env: GEMINI_API_KEY
```

### OpenAI

Set an OpenAI API key if you want OpenAI ranking:

```bash
export OPENAI_API_KEY=your_key_here
```

Example config:

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
```

The OpenAI integration uses the Responses API with Structured Outputs so the ranker returns schema-validated JSON rather than free-form text. Source: [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs)

If Gemini or OpenAI fails, or no API key is present, the pipeline falls back to heuristic ranking.

## Tuning ideas

- Add game-specific HUD OCR or kill-feed detection
- Feed separate mic and game-audio tracks instead of a mixed track
- Replace keyword rules with a lightweight event classifier
- Store accepted and rejected clips to train a learned ranker later
- Add scene-cut snapping before export
