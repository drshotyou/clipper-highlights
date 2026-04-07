from clipper_highlights.config import ProjectConfig


def test_project_config_load_defaults_when_no_path_is_given():
    config = ProjectConfig.load()

    assert config.transcription.model == "small.en"
    assert config.audio.sample_rate == 16000
    assert config.llm.provider == "none"
    assert "pmc" in config.candidate.keywords


def test_project_config_round_trips_yaml_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
transcription:
  model: medium.en
  hotwords:
    - pmc
candidate:
  keywords:
    pmc: 3.0
llm:
  provider: gemini
  max_output_clips: 3
""".strip()
    )

    config = ProjectConfig.load(config_path)

    assert config.transcription.model == "medium.en"
    assert config.transcription.hotwords == ["pmc"]
    assert config.candidate.keywords == {"pmc": 3.0}
    assert config.llm.provider == "gemini"
    assert config.llm.max_output_clips == 3

    roundtrip_path = tmp_path / "roundtrip.yaml"
    roundtrip_path.write_text(config.dump_yaml())
    roundtrip = ProjectConfig.load(roundtrip_path)

    assert roundtrip.model_dump() == config.model_dump()
