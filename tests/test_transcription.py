from clipper_highlights.transcription import _resolve_device


def test_resolve_device_returns_explicit_value_unchanged():
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_uses_ctranslate2_cuda_when_available(monkeypatch):
    class FakeCT2:
        @staticmethod
        def get_cuda_device_count():
            return 1

    monkeypatch.setitem(_resolve_device.__globals__, "ctranslate2", None)
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ctranslate2":
            return FakeCT2
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert _resolve_device("auto") == "cuda"


def test_resolve_device_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch):
    class FakeCT2:
        @staticmethod
        def get_cuda_device_count():
            return 0

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ctranslate2":
            return FakeCT2
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    assert _resolve_device("auto") == "cpu"
