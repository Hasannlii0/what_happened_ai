import os
import sys
from pathlib import Path

from gtts import gTTS

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def synthesize_speech(text: str) -> Path:
    config.TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.TTS_OUTPUT_DIR / "latest.mp3"
    gTTS(text=text, lang="en").save(str(output_path))
    return output_path
