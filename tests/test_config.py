import importlib
import os
from pathlib import Path

import pytest

import config


@pytest.fixture(autouse=True)
def restore_config_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
    importlib.reload(config)


def reload_without_overrides(monkeypatch):
    for name in list(os.environ):
        if name.startswith("WH_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("API_URL", raising=False)
    return importlib.reload(config)


DEFAULT_PATHS = {
    "VIDEO_PATH": "test_video.mp4",
    "DETECTIONS_JSON": "perception/detections.json",
    "ANNOTATED_VIDEO": "perception/output_annotated.mp4",
    "KEYFRAMES_DIR": "perception/keyframes",
    "EVIDENCE_DIR": "perception/evidence",
    "TTS_OUTPUT_DIR": "reasoning/tts_output",
    "DETECT_TRACK_SCRIPT": "perception/detect_track.py",
    "YOLO_RUNS_DIR": "runs",
}


@pytest.mark.parametrize("name,relative", DEFAULT_PATHS.items())
def test_default_paths_match_the_historical_literals(monkeypatch, name, relative):
    cfg = reload_without_overrides(monkeypatch)

    assert getattr(cfg, name) == cfg.REPO_ROOT / relative


def test_default_tunables_match_the_historical_literals(monkeypatch):
    cfg = reload_without_overrides(monkeypatch)

    assert cfg.YOLO_MODEL == "yolov8n.pt"
    assert cfg.YOLO_RUN_NAME == "track"
    assert cfg.TRACKER == "botsort.yaml"
    assert cfg.CONF_THRESHOLD == 0.5
    assert cfg.IOU_THRESHOLD == 0.5
    assert cfg.MLFLOW_EXPERIMENT == "what_happened_ai"
    assert cfg.KEYFRAME_COUNT == 3
    assert cfg.KEYFRAME_RESIZE_WIDTH == 336
    assert cfg.VLM_MODEL == "Qwen/Qwen2-VL-2B-Instruct"
    assert cfg.VLM_DEVICE == ""
    assert cfg.VLM_DTYPE == ""
    assert cfg.STT_MODEL == "base"
    assert cfg.STT_DEVICE == "cpu"
    assert cfg.STT_COMPUTE_TYPE == "int8"
    assert cfg.PROXIMITY_RATIO == 0.12
    assert cfg.MIN_TRACK_SECONDS == 0.4
    assert cfg.ABANDON_SECONDS == 2.0
    assert cfg.EVIDENCE_WIDTH == 240
    assert cfg.MAX_UPLOAD_BYTES == 500 * 1024 * 1024
    assert cfg.DETECT_TIMEOUT_SEC == 600
    assert cfg.API_URL == "http://localhost:8000"


def test_repo_root_is_the_directory_holding_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = reload_without_overrides(monkeypatch)

    assert cfg.REPO_ROOT == Path(cfg.__file__).resolve().parent
    assert (cfg.REPO_ROOT / "schema.py").exists()


def test_paths_ignore_the_process_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = reload_without_overrides(monkeypatch)

    assert cfg.VIDEO_PATH.is_absolute()
    assert tmp_path not in cfg.VIDEO_PATH.parents
    assert cfg.VIDEO_PATH == cfg.REPO_ROOT / "test_video.mp4"


def test_relative_env_override_resolves_against_the_repo_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WH_VIDEO_PATH", "clips/other.mp4")
    cfg = importlib.reload(config)

    assert cfg.VIDEO_PATH == cfg.REPO_ROOT / "clips" / "other.mp4"


def test_absolute_env_override_is_used_as_is(monkeypatch, tmp_path):
    absolute = tmp_path / "elsewhere.mp4"
    monkeypatch.setenv("WH_VIDEO_PATH", str(absolute))
    cfg = importlib.reload(config)

    assert cfg.VIDEO_PATH == absolute


def test_numeric_env_overrides_are_coerced(monkeypatch):
    monkeypatch.setenv("WH_PROXIMITY_RATIO", "0.25")
    monkeypatch.setenv("WH_KEYFRAME_COUNT", "7")
    monkeypatch.setenv("WH_MAX_UPLOAD_BYTES", "1024")
    cfg = importlib.reload(config)

    assert cfg.PROXIMITY_RATIO == 0.25
    assert cfg.KEYFRAME_COUNT == 7
    assert cfg.MAX_UPLOAD_BYTES == 1024


def test_api_url_env_var_is_unprefixed(monkeypatch):
    monkeypatch.setenv("API_URL", "http://api:8000")
    cfg = importlib.reload(config)

    assert cfg.API_URL == "http://api:8000"
