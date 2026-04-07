from __future__ import annotations

import json
import os
from importlib import resources

import requests

from clipper_highlights.config import LLMConfig
from clipper_highlights.models import CandidateWindow, RankedClip, TimelineBundle

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}

DEFAULT_API_KEY_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def rank_candidates(bundle: TimelineBundle, config: LLMConfig) -> list[RankedClip]:
    candidates = bundle.candidate_windows[: config.top_candidate_windows]
    if not candidates:
        return []

    if config.provider.lower() == "gemini":
        try:
            return GeminiRanker(config).rank(bundle, candidates)
        except Exception:
            return HeuristicRanker(config).rank(bundle, candidates)

    if config.provider.lower() == "openai":
        try:
            return OpenAIRanker(config).rank(bundle, candidates)
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
        api_key = os.getenv(_resolve_api_key_env("gemini", self.config.api_key_env))
        if not api_key:
            raise RuntimeError("Missing Gemini API key")

        prompt = _render_prompt(bundle, candidates, self.config.max_output_clips)
        model = _resolve_model("gemini", self.config.model)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
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


class OpenAIRanker:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def rank(self, bundle: TimelineBundle, candidates: list[CandidateWindow]) -> list[RankedClip]:
        api_key = os.getenv(_resolve_api_key_env("openai", self.config.api_key_env))
        if not api_key:
            raise RuntimeError("Missing OpenAI API key")

        prompt = _render_prompt(bundle, candidates, self.config.max_output_clips)
        model = _resolve_model("openai", self.config.model)
        response = requests.post(
            _resolve_openai_url(self.config.base_url),
            headers=_openai_headers(api_key, self.config),
            json={
                "model": model,
                "store": False,
                "input": [
                    {
                        "role": "system",
                        "content": "Return only a JSON object matching the provided schema.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "ranked_clips",
                        "schema": _clip_response_schema(),
                        "strict": True,
                    }
                },
            },
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = _extract_openai_response_text(data)
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


def _resolve_model(provider: str, configured_model: str | None) -> str:
    if configured_model:
        normalized = configured_model.strip()
        if normalized and not (
            provider == "openai" and normalized.startswith("gemini")
        ):
            return normalized
    return DEFAULT_MODELS[provider]


def _resolve_api_key_env(provider: str, configured_api_key_env: str | None) -> str:
    if configured_api_key_env:
        normalized = configured_api_key_env.strip()
        if normalized and not (
            provider == "openai" and normalized == "GEMINI_API_KEY"
        ):
            return normalized
    return DEFAULT_API_KEY_ENVS[provider]


def _resolve_openai_url(base_url: str | None) -> str:
    if not base_url:
        return "https://api.openai.com/v1/responses"

    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    return f"{normalized}/responses"


def _openai_headers(api_key: str, config: LLMConfig) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if config.organization:
        headers["OpenAI-Organization"] = config.organization
    if config.project:
        headers["OpenAI-Project"] = config.project
    return headers


def _clip_response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "clips": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "score": {"type": "number"},
                        "title": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "start", "end", "score", "title", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["clips"],
        "additionalProperties": False,
    }


def _extract_openai_response_text(payload: dict) -> str:
    output_text = payload.get("output_text")
    if output_text:
        return output_text

    output = payload.get("output", [])
    for item in output:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                return text

    raise RuntimeError("OpenAI response did not contain output text")
