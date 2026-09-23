import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _path(env_var: str, default: str) -> Path:
    """Resolve against the repo root so subprocesses and directly-run modules
    agree on paths regardless of their working directory."""
    value = Path(os.environ.get(env_var, default))
    return value if value.is_absolute() else REPO_ROOT / value


VIDEO_PATH = _path("WH_VIDEO_PATH", "test_video.mp4")
DETECTIONS_JSON = _path("WH_DETECTIONS_JSON", "perception/detections.json")
ANNOTATED_VIDEO = _path("WH_ANNOTATED_VIDEO", "perception/output_annotated.mp4")
KEYFRAMES_DIR = _path("WH_KEYFRAMES_DIR", "perception/keyframes")
EVIDENCE_DIR = _path("WH_EVIDENCE_DIR", "perception/evidence")
TTS_OUTPUT_DIR = _path("WH_TTS_OUTPUT_DIR", "reasoning/tts_output")
DETECT_TRACK_SCRIPT = _path("WH_DETECT_TRACK_SCRIPT", "perception/detect_track.py")
YOLO_RUNS_DIR = _path("WH_YOLO_RUNS_DIR", "runs")

YOLO_MODEL = os.environ.get("WH_YOLO_MODEL", "yolov8n.pt")
YOLO_RUN_NAME = os.environ.get("WH_YOLO_RUN_NAME", "track")
TRACKER = os.environ.get("WH_TRACKER", "botsort.yaml")
CONF_THRESHOLD = float(os.environ.get("WH_CONF_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.environ.get("WH_IOU_THRESHOLD", "0.5"))
MLFLOW_EXPERIMENT = os.environ.get("WH_MLFLOW_EXPERIMENT", "what_happened_ai")

KEYFRAME_COUNT = int(os.environ.get("WH_KEYFRAME_COUNT", "3"))
# Qwen2-VL turns a frame into roughly (w/28)*(h/28) tokens, and encoding those
# dominates CPU inference, so width trades directly against latency.
KEYFRAME_RESIZE_WIDTH = int(os.environ.get("WH_KEYFRAME_RESIZE_WIDTH", "336"))

VLM_MODEL = os.environ.get("WH_VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
VLM_DEVICE = os.environ.get("WH_VLM_DEVICE", "")
VLM_DTYPE = os.environ.get("WH_VLM_DTYPE", "")
VLM_MAX_NEW_TOKENS = int(os.environ.get("WH_VLM_MAX_NEW_TOKENS", "300"))
STT_MODEL = os.environ.get("WH_STT_MODEL", "base")
STT_DEVICE = os.environ.get("WH_STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.environ.get("WH_STT_COMPUTE_TYPE", "int8")

PROXIMITY_RATIO = float(os.environ.get("WH_PROXIMITY_RATIO", "0.12"))
MIN_TRACK_SECONDS = float(os.environ.get("WH_MIN_TRACK_SECONDS", "0.4"))
ABANDON_SECONDS = float(os.environ.get("WH_ABANDON_SECONDS", "2.0"))
EVIDENCE_WIDTH = int(os.environ.get("WH_EVIDENCE_WIDTH", "240"))
# The annotated video is only ever a preview, so it is capped here rather than
# re-encoded, searched and streamed at the source resolution (4K from a phone).
ANNOTATED_MAX_SIDE = int(os.environ.get("WH_ANNOTATED_MAX_SIDE", "1280"))

MAX_UPLOAD_BYTES = int(os.environ.get("WH_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
DETECT_TIMEOUT_SEC = int(os.environ.get("WH_DETECT_TIMEOUT_SEC", "600"))

API_URL = os.environ.get("API_URL", "http://localhost:8000")
