import os

import pytest

from clipper_highlights.config import LLMConfig
from clipper_highlights.llm import GeminiRanker, OpenAIRanker, _extract_json, rank_candidates
from clipper_highlights.models import CandidateWindow, Evidence, TimelineBundle


def _bundle() -> TimelineBundle:
    candidate = CandidateWindow(
        candidate_id="cand_001",
        start=10.0,
        end=18.0,
        score=7.5,
        reasons=["Audio event score 4.0", "Transcript keywords: pmc"],
        transcript_excerpt="pmc on me, one shot",
        evidences=[
            Evidence(kind="audio", start=11.0, end=12.0, score=4.0, label="audio_spike"),
            Evidence(kind="keyword", start=10.0, end=13.0, score=3.5, label="pmc"),
        ],
    )
    return TimelineBundle(
        source_video="session.mp4",
        audio_path="audio.wav",
        game="tarkov",
        candidate_windows=[candidate],
    )


def test_rank_candidates_uses_heuristics_when_provider_is_none():
    clips = rank_candidates(_bundle(), LLMConfig(provider="none", max_output_clips=1))

    assert len(clips) == 1
    assert clips[0].candidate_id == "cand_001"
    assert clips[0].title.startswith("Combat moment:")


def test_rank_candidates_falls_back_when_gemini_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    clips = rank_candidates(_bundle(), LLMConfig(provider="gemini", max_output_clips=1))

    assert len(clips) == 1
    assert clips[0].candidate_id == "cand_001"
    assert clips[0].reason


def test_rank_candidates_falls_back_when_openai_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    clips = rank_candidates(_bundle(), LLMConfig(provider="openai", max_output_clips=1))

    assert len(clips) == 1
    assert clips[0].candidate_id == "cand_001"
    assert clips[0].reason


def test_extract_json_removes_markdown_fences():
    fenced = """```json
{"clips": []}
```"""

    assert _extract_json(fenced) == '{"clips": []}'


def test_gemini_ranker_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": '{"clips":[{"candidate_id":"cand_001","title":"Best fight","reason":"Strong multi-signal event","score":9.1}]}'
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, json, timeout):
        assert "generateContent" in url
        assert timeout == 60
        assert json["generationConfig"]["responseMimeType"] == "application/json"
        return FakeResponse()

    monkeypatch.setattr("clipper_highlights.llm.requests.post", fake_post)

    clips = GeminiRanker(LLMConfig(provider="gemini")).rank(_bundle(), _bundle().candidate_windows)

    assert len(clips) == 1
    assert clips[0].title == "Best fight"
    assert clips[0].start == 10.0
    assert clips[0].end == 18.0


def test_openai_ranker_parses_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output_text": '{"clips":[{"candidate_id":"cand_001","start":10.0,"end":18.0,"score":9.2,"title":"OpenAI fight","reason":"Good clip"}]}'
            }

    def fake_post(url, headers, json, timeout):
        assert url == "https://api.openai.com/v1/responses"
        assert headers["Authorization"] == "Bearer test-key"
        assert json["model"] == "gpt-4o-mini"
        assert json["text"]["format"]["type"] == "json_schema"
        assert timeout == 60
        return FakeResponse()

    monkeypatch.setattr("clipper_highlights.llm.requests.post", fake_post)

    clips = OpenAIRanker(LLMConfig(provider="openai")).rank(_bundle(), _bundle().candidate_windows)

    assert len(clips) == 1
    assert clips[0].title == "OpenAI fight"
    assert clips[0].start == 10.0
    assert clips[0].end == 18.0
