from __future__ import annotations

from pathlib import Path
from typing import Callable

from clipper_highlights.config import TranscriptionConfig
from clipper_highlights.models import TranscriptSegment, WordSegment


def transcribe_audio(
    audio_path: Path,
    config: TranscriptionConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> list[TranscriptSegment]:
    from faster_whisper import WhisperModel

    device = _resolve_device(config.device)
    _emit(progress_callback, f"Loading Whisper model '{config.model}' on device '{device}'")
    model = WhisperModel(
        config.model,
        device=device,
        compute_type=config.compute_type,
    )

    initial_prompt = config.initial_prompt
    if config.hotwords:
        hotword_prompt = ", ".join(config.hotwords)
        if initial_prompt:
            initial_prompt = f"{initial_prompt}\nKey vocabulary: {hotword_prompt}"
        else:
            initial_prompt = f"Key vocabulary: {hotword_prompt}"

    _emit(progress_callback, "Starting transcription")
    segments, _ = model.transcribe(
        str(audio_path),
        language=config.language,
        beam_size=config.beam_size,
        vad_filter=config.vad_filter,
        word_timestamps=config.word_timestamps,
        initial_prompt=initial_prompt,
        hotwords=", ".join(config.hotwords) if config.hotwords else None,
    )

    transcript: list[TranscriptSegment] = []
    last_reported_second = 0.0
    for segment in segments:
        words = [
            WordSegment(
                start=float(word.start),
                end=float(word.end),
                text=word.word.strip(),
                probability=float(word.probability) if word.probability is not None else None,
            )
            for word in (segment.words or [])
            if word.start is not None and word.end is not None
        ]
        transcript.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=segment.text.strip(),
                avg_logprob=float(segment.avg_logprob) if segment.avg_logprob is not None else None,
                no_speech_prob=float(segment.no_speech_prob)
                if segment.no_speech_prob is not None
                else None,
                words=words,
            )
        )
        current_second = float(segment.end)
        if len(transcript) == 1 or current_second - last_reported_second >= 30.0:
            _emit(
                progress_callback,
                f"Transcribed {len(transcript)} segments through {current_second:.1f}s of audio",
            )
            last_reported_second = current_second
    _emit(progress_callback, f"Transcription complete: {len(transcript)} segments")
    return transcript


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass

    return "cpu"


def _emit(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)
