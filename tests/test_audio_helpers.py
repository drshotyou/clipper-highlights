import numpy as np

from pathlib import Path

from clipper_highlights.audio import _build_extract_audio_command, _match_length, _merge_nearby_spikes, _robust_zscore
from clipper_highlights.config import AudioConfig
from clipper_highlights.models import AudioSpike


def test_match_length_truncates_and_pads():
    values = np.array([1.0, 2.0, 3.0])

    truncated = _match_length(values, 2)
    padded = _match_length(values, 5)

    assert truncated.tolist() == [1.0, 2.0]
    assert padded.tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_merge_nearby_spikes_combines_adjacent_ranges():
    spikes = [
        AudioSpike(start=1.0, end=1.4, peak_dbfs=-6.0, combined_score=2.0, onset_peak=1.0),
        AudioSpike(start=1.6, end=2.0, peak_dbfs=-3.0, combined_score=4.0, onset_peak=2.0),
        AudioSpike(start=3.0, end=3.2, peak_dbfs=-8.0, combined_score=1.5, onset_peak=0.8),
    ]

    merged = _merge_nearby_spikes(spikes, gap_seconds=0.35)

    assert len(merged) == 2
    assert merged[0].start == 1.0
    assert merged[0].end == 2.0
    assert merged[0].peak_dbfs == -3.0
    assert merged[0].combined_score == 4.0


def test_robust_zscore_handles_constant_values():
    values = np.array([5.0, 5.0, 5.0])

    scores = _robust_zscore(values)

    assert scores.tolist() == [0.0, 0.0, 0.0]


def test_build_extract_audio_command_defaults_to_ffmpeg_selected_stream():
    command = _build_extract_audio_command(
        Path("session.mkv"),
        Path("audio.wav"),
        AudioConfig(sample_rate=16000),
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "session.mkv",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "audio.wav",
    ]


def test_build_extract_audio_command_maps_one_specific_stream():
    command = _build_extract_audio_command(
        Path("session.mkv"),
        Path("audio.wav"),
        AudioConfig(sample_rate=16000, stream_indices=[2]),
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "session.mkv",
        "-map",
        "0:a:2",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "audio.wav",
    ]


def test_build_extract_audio_command_mixes_multiple_streams():
    command = _build_extract_audio_command(
        Path("session.mkv"),
        Path("audio.wav"),
        AudioConfig(sample_rate=16000, stream_indices=[1, 2]),
    )

    assert command == [
        "ffmpeg",
        "-y",
        "-i",
        "session.mkv",
        "-filter_complex",
        "[0:a:1][0:a:2]amix=inputs=2:normalize=0[aout]",
        "-map",
        "[aout]",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "audio.wav",
    ]
