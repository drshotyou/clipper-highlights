from __future__ import annotations

import subprocess
from pathlib import Path

from clipper_highlights.config import AudioConfig
from clipper_highlights.models import AudioSpike


def extract_audio(video_path: Path, output_path: Path, config: AudioConfig) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
    return output_path


def analyze_audio(audio_path: Path, config: AudioConfig) -> list[AudioSpike]:
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
    rms = librosa.feature.rms(
        y=y,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )[0]
    rms = np.maximum(rms, 1e-10)
    dbfs = librosa.amplitude_to_db(rms, ref=1.0)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.hop_length)
    onset = _match_length(onset, len(dbfs))
    times = librosa.frames_to_time(
        np.arange(len(dbfs)),
        sr=sr,
        hop_length=config.hop_length,
    )

    energy_z = _robust_zscore(dbfs)
    delta_z = _robust_zscore(np.clip(np.diff(dbfs, prepend=dbfs[0]), a_min=0.0, a_max=None))
    onset_z = _robust_zscore(onset)
    combined = (
        config.energy_weight * energy_z
        + config.delta_weight * delta_z
        + config.onset_weight * onset_z
    )

    active = (dbfs >= config.min_dbfs) & (combined >= config.combined_threshold)
    hop_duration = config.hop_length / sr

    spikes: list[AudioSpike] = []
    start_idx: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = index
            continue

        if not is_active and start_idx is not None:
            spikes.append(
                _build_spike(
                    start_idx,
                    index - 1,
                    times=times,
                    dbfs=dbfs,
                    onset=onset,
                    combined=combined,
                    hop_duration=hop_duration,
                )
            )
            start_idx = None

    if start_idx is not None:
        spikes.append(
            _build_spike(
                start_idx,
                len(active) - 1,
                times=times,
                dbfs=dbfs,
                onset=onset,
                combined=combined,
                hop_duration=hop_duration,
            )
        )

    return _merge_nearby_spikes(spikes)


def _build_spike(
    start_idx: int,
    end_idx: int,
    *,
    times,
    dbfs,
    onset,
    combined,
    hop_duration: float,
) -> AudioSpike:
    import numpy as np

    peak_index = start_idx + int(np.argmax(combined[start_idx : end_idx + 1]))
    return AudioSpike(
        start=float(times[start_idx]),
        end=float(times[end_idx] + hop_duration),
        peak_dbfs=float(dbfs[peak_index]),
        combined_score=float(combined[peak_index]),
        onset_peak=float(onset[peak_index]),
    )


def _merge_nearby_spikes(spikes: list[AudioSpike], gap_seconds: float = 0.35) -> list[AudioSpike]:
    if not spikes:
        return []

    merged: list[AudioSpike] = [spikes[0]]
    for spike in spikes[1:]:
        current = merged[-1]
        if spike.start - current.end > gap_seconds:
            merged.append(spike)
            continue

        merged[-1] = AudioSpike(
            start=current.start,
            end=max(current.end, spike.end),
            peak_dbfs=max(current.peak_dbfs, spike.peak_dbfs),
            combined_score=max(current.combined_score, spike.combined_score),
            onset_peak=max(current.onset_peak, spike.onset_peak),
        )
    return merged


def _match_length(values, target_length: int):
    import numpy as np

    if len(values) == target_length:
        return values
    if len(values) > target_length:
        return values[:target_length]
    return np.pad(values, (0, target_length - len(values)))


def _robust_zscore(values):
    import numpy as np

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = max(mad * 1.4826, 1e-6)
    return (values - median) / scale
