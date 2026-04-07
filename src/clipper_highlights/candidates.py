from __future__ import annotations

import re

from clipper_highlights.config import CandidateConfig
from clipper_highlights.models import AudioSpike, CandidateWindow, Evidence, TranscriptSegment


def generate_candidate_windows(
    transcript_segments: list[TranscriptSegment],
    audio_spikes: list[AudioSpike],
    config: CandidateConfig,
) -> list[CandidateWindow]:
    windows: list[CandidateWindow] = []
    for spike in audio_spikes:
        windows.append(
            CandidateWindow(
                candidate_id="pending",
                start=max(0.0, spike.start - config.pre_roll_seconds),
                end=spike.end + config.post_roll_seconds,
                score=spike.combined_score * config.audio_event_weight,
                reasons=[f"Audio event score {spike.combined_score:.2f}"],
                evidences=[
                    Evidence(
                        kind="audio",
                        start=spike.start,
                        end=spike.end,
                        score=spike.combined_score * config.audio_event_weight,
                        label="audio_spike",
                        detail=f"peak_dbfs={spike.peak_dbfs:.2f}",
                    )
                ],
            )
        )

    for segment in transcript_segments:
        segment_hits = _keyword_hits(segment.text, config.keywords)
        if not segment_hits:
            continue

        hit_score = sum(weight * count for _, weight, count in segment_hits) * config.keyword_event_weight
        labels = ", ".join(keyword for keyword, _, _ in segment_hits)
        windows.append(
            CandidateWindow(
                candidate_id="pending",
                start=max(0.0, segment.start - config.pre_roll_seconds),
                end=segment.end + config.post_roll_seconds,
                score=hit_score,
                reasons=[f"Transcript keywords: {labels}"],
                evidences=[
                    Evidence(
                        kind="keyword",
                        start=segment.start,
                        end=segment.end,
                        score=weight * count * config.keyword_event_weight,
                        label=keyword,
                        detail=f"occurrences={count}",
                    )
                    for keyword, weight, count in segment_hits
                ],
            )
        )

    merged = _merge_windows(windows, config)
    for index, window in enumerate(merged):
        window.candidate_id = f"cand_{index:03d}"
        window.transcript_excerpt = _excerpt_for_window(
            transcript_segments,
            window.start,
            window.end,
        )
        _apply_length_bounds(window, config)

    ranked = sorted(merged, key=lambda item: item.score, reverse=True)
    return ranked[: config.max_candidates]


def _keyword_hits(text: str, keywords: dict[str, float]) -> list[tuple[str, float, int]]:
    lowered = text.lower()
    hits: list[tuple[str, float, int]] = []
    for keyword, weight in keywords.items():
        occurrences = len(re.findall(re.escape(keyword.lower()), lowered))
        if occurrences:
            hits.append((keyword, weight, occurrences))
    return hits


def _merge_windows(windows: list[CandidateWindow], config: CandidateConfig) -> list[CandidateWindow]:
    if not windows:
        return []

    ordered = sorted(windows, key=lambda item: (item.start, item.end))
    merged: list[CandidateWindow] = [ordered[0]]

    for candidate in ordered[1:]:
        current = merged[-1]
        if candidate.start - current.end > config.merge_gap_seconds:
            merged.append(candidate)
            continue

        current.start = min(current.start, candidate.start)
        current.end = max(current.end, candidate.end)
        current.score += candidate.score
        current.reasons.extend(reason for reason in candidate.reasons if reason not in current.reasons)
        current.evidences.extend(candidate.evidences)

    filtered: list[CandidateWindow] = []
    for candidate in merged:
        if candidate.score < config.min_candidate_score:
            continue

        if _has_crossreference(candidate, config):
            candidate.score += config.crossreference_boost
            candidate.reasons.append("Audio and transcript evidence aligned in time")

        filtered.append(candidate)

    return filtered


def _has_crossreference(candidate: CandidateWindow, config: CandidateConfig) -> bool:
    audio_events = [item for item in candidate.evidences if item.kind == "audio"]
    keyword_events = [item for item in candidate.evidences if item.kind == "keyword"]
    for audio in audio_events:
        for keyword in keyword_events:
            if abs(_center(audio.start, audio.end) - _center(keyword.start, keyword.end)) <= config.crossreference_window_seconds:
                return True
    return False


def _excerpt_for_window(
    transcript_segments: list[TranscriptSegment],
    start: float,
    end: float,
) -> str:
    lines = [
        segment.text
        for segment in transcript_segments
        if not (segment.end < start or segment.start > end)
    ]
    return " ".join(lines).strip()


def _apply_length_bounds(candidate: CandidateWindow, config: CandidateConfig) -> None:
    duration = candidate.end - candidate.start
    if duration < config.min_clip_seconds:
        pad = (config.min_clip_seconds - duration) / 2
        candidate.start = max(0.0, candidate.start - pad)
        candidate.end = candidate.end + pad
        duration = candidate.end - candidate.start

    if duration <= config.max_clip_seconds:
        return

    anchor = _weighted_anchor(candidate.evidences)
    half = config.max_clip_seconds / 2
    candidate.start = max(0.0, anchor - half)
    candidate.end = candidate.start + config.max_clip_seconds


def _weighted_anchor(evidences: list[Evidence]) -> float:
    total_weight = sum(max(item.score, 0.001) for item in evidences)
    weighted_sum = sum(_center(item.start, item.end) * max(item.score, 0.001) for item in evidences)
    return weighted_sum / total_weight if total_weight else 0.0


def _center(start: float, end: float) -> float:
    return (start + end) / 2
