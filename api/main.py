import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

import config
from reasoning.event_extractor import extract_events
from reasoning.summarizer import summarize
from reasoning.tts import synthesize_speech
from schema import AnalysisResult, DetectionLog

logger = logging.getLogger(__name__)

app = FastAPI()

PIPELINE_LOCK = threading.Lock()
BUSY_MESSAGE = "Another upload or analysis is already running. Try again in a moment."


def _is_decodable_video(path: Path) -> bool:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or cap.get(cv2.CAP_PROP_FPS) <= 0:
            return False
        return cap.read()[0]
    finally:
        cap.release()


def _last_error_line(result: subprocess.CompletedProcess) -> str:
    lines = (result.stderr or "").strip().splitlines()
    if lines:
        return lines[-1][:200]
    return f"detector exited with code {result.returncode}"


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    content_type = file.content_type or ""
    # Reject only a header that is present and clearly not video. An absent one
    # and the generic binary type fall through to the decode probe, which is the
    # real check; curl and httpx both send octet-stream for a perfectly good mp4.
    if (
        content_type
        and content_type != "application/octet-stream"
        and not content_type.startswith("video/")
    ):
        return JSONResponse(
            status_code=415,
            content={"error": f"Expected a video file, got {content_type}."},
        )

    if not PIPELINE_LOCK.acquire(blocking=False):
        return JSONResponse(status_code=409, content={"error": BUSY_MESSAGE})

    fd, tmp_name = tempfile.mkstemp(dir=config.VIDEO_PATH.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        written = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
                    return JSONResponse(
                        status_code=413,
                        content={"error": f"Video is larger than the {limit_mb} MB limit."},
                    )
                out.write(chunk)

        if not _is_decodable_video(tmp_path):
            return JSONResponse(
                status_code=415,
                content={"error": "That file could not be decoded as a video."},
            )

        os.replace(tmp_path, config.VIDEO_PATH)
    finally:
        tmp_path.unlink(missing_ok=True)
        PIPELINE_LOCK.release()

    return {"status": "uploaded", "filename": file.filename}


@app.post("/analyze")
def analyze():
    if not PIPELINE_LOCK.acquire(blocking=False):
        return JSONResponse(status_code=409, content={"error": BUSY_MESSAGE})
    try:
        return _run_analysis()
    finally:
        PIPELINE_LOCK.release()


def _run_analysis():
    if not config.VIDEO_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet."})

    config.DETECTIONS_JSON.unlink(missing_ok=True)
    config.ANNOTATED_VIDEO.unlink(missing_ok=True)

    try:
        result = subprocess.run(
            [sys.executable, str(config.DETECT_TRACK_SCRIPT)],
            cwd=config.REPO_ROOT,
            stderr=subprocess.PIPE,
            text=True,
            timeout=config.DETECT_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        logger.error("detect_track.py timed out after %ss", config.DETECT_TIMEOUT_SEC)
        return JSONResponse(
            status_code=504,
            content={"error": "Detection timed out. Try a shorter video."},
        )

    if result.returncode != 0:
        logger.error(
            "detect_track.py exited %s:\n%s", result.returncode, result.stderr
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Detection failed while processing the video.",
                "detail": _last_error_line(result),
            },
        )

    if not config.DETECTIONS_JSON.exists():
        logger.error("detect_track.py succeeded but wrote no detections file")
        return JSONResponse(
            status_code=500, content={"error": "Detection produced no results."}
        )

    try:
        with open(config.DETECTIONS_JSON) as f:
            detection_log = DetectionLog(**json.load(f))
    except (OSError, json.JSONDecodeError, ValidationError) as e:
        logger.exception("could not read %s", config.DETECTIONS_JSON)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Detection output could not be read.",
                "detail": type(e).__name__,
            },
        )

    event_log = extract_events(detection_log)

    try:
        from perception.keyframes import extract_keyframes
        from reasoning.vlm_qa import describe_scene

        keyframe_paths = extract_keyframes(str(config.VIDEO_PATH))
        if not keyframe_paths:
            raise RuntimeError("no frames could be decoded from the video")
        summary_text = describe_scene(keyframe_paths)
        summary_source = "vlm"
    except Exception:
        logger.exception("scene description failed, falling back to the template")
        summary_text = summarize(event_log)
        summary_source = "template"

    annotated_name = (
        config.ANNOTATED_VIDEO.name if config.ANNOTATED_VIDEO.exists() else None
    )

    analysis_result = AnalysisResult(
        video_id=event_log.video_id,
        events=event_log.events,
        summary=summary_text,
        summary_source=summary_source,
        annotated_video_path=annotated_name,
    )

    return analysis_result.model_dump()


@app.get("/annotated_video")
def annotated_video():
    if not config.ANNOTATED_VIDEO.exists():
        return JSONResponse(
            status_code=404, content={"error": "No annotated video available."}
        )
    return FileResponse(config.ANNOTATED_VIDEO, media_type="video/mp4")


@app.post("/ask")
def ask_question(question: str):
    if not question.strip():
        return JSONResponse(status_code=400, content={"error": "Ask a question first."})

    if not config.VIDEO_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet."})

    try:
        from perception.keyframes import extract_keyframes
        from reasoning.vlm_qa import ask_vlm

        keyframe_paths = extract_keyframes(str(config.VIDEO_PATH))
        if not keyframe_paths:
            return JSONResponse(
                status_code=422,
                content={"error": "Could not read any frames from the video."},
            )
        answer = ask_vlm(keyframe_paths, question)
    except Exception as e:
        logger.exception("could not answer question")
        return JSONResponse(
            status_code=503,
            content={
                "error": "The vision model could not answer that.",
                "detail": type(e).__name__,
            },
        )

    return {"question": question, "answer": answer}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as out:
            shutil.copyfileobj(file.file, out)

        from reasoning.stt import transcribe_audio

        text = transcribe_audio(tmp_name)
    except Exception as e:
        logger.exception("transcription failed")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Could not transcribe the recording.",
                "detail": type(e).__name__,
            },
        )
    finally:
        os.remove(tmp_name)

    return {"text": text}


@app.post("/speak")
def speak(text: str):
    if not text.strip():
        return JSONResponse(status_code=400, content={"error": "Nothing to speak."})

    try:
        # Read the bytes now instead of streaming the path: the file has a fixed
        # name, so a second /speak would overwrite it mid-send.
        audio = synthesize_speech(text).read_bytes()
    except Exception as e:
        logger.exception("speech synthesis failed")
        return JSONResponse(
            status_code=503,
            content={
                "error": "Could not generate audio.",
                "detail": type(e).__name__,
            },
        )

    return Response(content=audio, media_type="audio/mpeg")
