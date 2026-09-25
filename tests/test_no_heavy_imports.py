import sys

HEAVY = [
    "torch",
    "ultralytics",
    "transformers",
    "cv2",
    "faster_whisper",
    "gtts",
    "mlflow",
    "reasoning.vlm_qa",
    "reasoning.stt",
    "perception.detect_track",
    "perception.keyframes",
    "api.main",
]


def test_suite_never_pulls_in_a_model_or_video_dependency():
    """Collection imports every test module, so by the time this runs the whole
    suite's import graph is in sys.modules."""
    assert [name for name in HEAVY if name in sys.modules] == []
