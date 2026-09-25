import os
import sys

from faster_whisper import WhisperModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

_model = None


def _load() -> WhisperModel:
    global _model

    if _model is None:
        _model = WhisperModel(
            config.STT_MODEL,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )

    return _model


def transcribe_audio(audio_path: str) -> str:
    segments, _ = _load().transcribe(str(audio_path))
    text = " ".join(segment.text for segment in segments)
    return text.strip()
