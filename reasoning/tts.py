import os
import uuid

from gtts import gTTS

OUTPUT_DIR = "reasoning/tts_output"


def synthesize_speech(text: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename
