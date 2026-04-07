from __future__ import annotations

import json
import os
from importlib import resources

import requests

from clipper_highlights.config import LLMConfig
from clipper_highlights.models import CandidateWindow, RankedClip, TimelineBundle


def rank_candidates(bundle: TimelineBundle, config: LLMConfig) -> list[RankedClip]:
    candidates = bundle.candidate_windows[: config.top_candidate_windows]
    if not candidates:
        return []

    if config.provider.lower() == "gemini":
        try:
            return GeminiRanker(config).rank(bundle, candidates)
        except Exception:
            return HeuristicRanker(config).rank(bundle, candidates)

    return HeuristicRanker(config).rank(bundle, candidates)


class HeuristicRanker:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def rank(self, bundle: TimelineBundle, candidates: list[CandidateWindow]) -> list[RankedClip]:
        ranked: list[RankedClip] = []
        for candidate in candidates[: self.config.max_output_clips]:
            ranked.append(
                RankedClip(
                    candidate_id=candidate.candidate_id,
                    start=candidate.start,
                    end=candidate.end,
                    score=round(candidate.score, 3),
                    title=_heuristic_title(candidate),
                    reason=" | ".join(candidate.reasons[:3]) or "High combined heuristic score",
                )
            )
        return ranked


class GeminiRanker:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def rank(self, bundle: TimelineBundle, candidates: list[CandidateWindow]) -> list[RankedClip]:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key in ${self.config.api_key_env}")

        prompt = _render_prompt(bundle, candidates, self.config.max_output_clips)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.model}:generateContent?key={api_key}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
        response.raise_for_status()

        data = response.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(_extract_json(text))
        return _coerce_clips(parsed, candidates, self.config.max_output_clips)


def _render_prompt(
    bundle: TimelineBundle,
    candidates: list[CandidateWindow],
    max_output_clips: int,
) -> str:
    prompt_template = resources.files("clipper_highlights.prompts").joinpath("rank_candidates.txt").read_text()
    candidate_payload = []
    for candidate in candidates:
        candidate_payload.append(
            {
                "candidate_id": candidate.candidate_id,
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "score": round(candidate.score, 3),
                "reasons": candidate.reasons[:5],
                "transcript_excerpt": candidate.transcript_excerpt[:1200],
                "evidences": [
                    {
                        "kind": evidence.kind,
                        "start": round(evidence.start, 3),
                        "end": round(evidence.end, 3),
                        "score": round(evidence.score, 3),
                        "label": evidence.label,
                        "detail": evidence.detail,
                    }
                    for evidence in candidate.evidences[:12]
                ],
            }
        )

    return prompt_template.format(
        source_video=bundle.source_video,
        game=bundle.game or "unknown",
        max_output_clips=max_output_clips,
        candidates_json=json.dumps(candidate_payload, indent=2),
    )


def _coerce_clips(
    payload: dict,
    candidates: list[CandidateWindow],
    max_output_clips: int,
) -> list[RankedClip]:
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    clips: list[RankedClip] = []
    for raw_clip in payload.get("clips", [])[:max_output_clips]:
        candidate = candidate_map.get(raw_clip.get("candidate_id", ""))
        start = float(raw_clip.get("start", candidate.start if candidate else 0.0))
        end = float(raw_clip.get("end", candidate.end if candidate else start))
        if end <= start:
            continue
        clips.append(
            RankedClip(
                candidate_id=raw_clip.get("candidate_id"),
                start=start,
                end=end,
                score=float(raw_clip.get("score", candidate.score if candidate else 0.0)),
                title=raw_clip.get("title", candidate.candidate_id if candidate else "clip"),
                reason=raw_clip.get("reason", "Selected by Gemini"),
            )
        )
    return clips


def _heuristic_title(candidate: CandidateWindow) -> str:
    labels = [item.label for item in candidate.evidences if item.kind == "keyword"]
    if labels:
        joined = ", ".join(dict.fromkeys(labels))
        return f"Combat moment: {joined[:80]}"
    return "Audio spike highlight"


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return stripped
