import numpy as np

from clipper_highlights.audio import _match_length, _merge_nearby_spikes, _robust_zscore
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
