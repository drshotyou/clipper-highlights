from clipper_highlights.candidates import generate_candidate_windows
from clipper_highlights.config import CandidateConfig
from clipper_highlights.models import AudioSpike, TranscriptSegment


def test_candidate_generation_merges_audio_and_keywords():
    config = CandidateConfig(
        keywords={"pmc": 2.0, "i'm down": 2.5},
        pre_roll_seconds=2.0,
        post_roll_seconds=2.0,
        merge_gap_seconds=4.0,
        crossreference_window_seconds=5.0,
        min_candidate_score=0.5,
    )
    transcript = [
        TranscriptSegment(start=10.0, end=12.0, text="pmc on me"),
        TranscriptSegment(start=40.0, end=42.0, text="nothing happening here"),
    ]
    spikes = [
        AudioSpike(start=11.0, end=11.8, peak_dbfs=-8.0, combined_score=3.0, onset_peak=4.0),
    ]

    candidates = generate_candidate_windows(transcript, spikes, config)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.start <= 8.5
    assert candidate.end >= 13.5
    assert candidate.score > 5.0
    assert "Audio and transcript evidence aligned in time" in candidate.reasons
