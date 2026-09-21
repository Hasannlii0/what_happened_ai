import os
import shutil
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from perception.keyframes import extract_keyframes
from reasoning.event_extractor import extract_events
from reasoning.stt import transcribe_audio
from reasoning.summarizer import summarize
from reasoning.tts import synthesize_speech
from reasoning.vlm_qa import ask_vlm, describe_scene
from schema import AnalysisResult, DetectionLog

app = FastAPI()

LATEST_EVENT_LOG = {"data": None}


@app.post("/upload")
def upload_video(file: UploadFile = File(...)):
    save_path = "test_video.mp4"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "path": save_path}


@app.post("/analyze")
def analyze():
    if not os.path.exists("test_video.mp4"):
        return JSONResponse(
            status_code=404, content={"error": "No video uploaded yet."}
        )

    for path in ["perception/detections.json", "perception/output_annotated.mp4"]:
        if os.path.exists(path):
            os.remove(path)

    result = subprocess.run(
        [sys.executable, "perception/detect_track.py"], capture_output=True, text=True
    )

    if result.returncode != 0:
        return JSONResponse(
            status_code=500, content={"error": f"Detection failed: {result.stderr}"}
        )

    detections_path = "perception/detections.json"
    if not os.path.exists(detections_path):
        return JSONResponse(
            status_code=500,
            content={"error": "detect_track.py ran but produced no output."},
        )

    with open(detections_path) as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    LATEST_EVENT_LOG["data"] = event_log

    try:
        keyframe_paths = extract_keyframes("test_video.mp4")
        summary_text = describe_scene(keyframe_paths)
    except Exception as e:
        summary_text = summarize(event_log)
        print(f"VLM summary failed, using template: {e}")

    annotated_path = "perception/output_annotated.mp4"
    if not os.path.exists(annotated_path):
        annotated_path = None

    analysis_result = AnalysisResult(
        video_id=event_log.video_id,
        events=event_log.events,
        summary=summary_text,
        annotated_video_path=annotated_path,
    )

    return analysis_result.model_dump()


@app.post("/ask")
def ask_question(question: str):
    if LATEST_EVENT_LOG["data"] is None:
        return JSONResponse(status_code=400, content={"error": "Run /analyze first."})

    try:
        keyframe_paths = extract_keyframes("test_video.mp4")
        answer = ask_vlm(keyframe_paths, question)
    except Exception as e:
        answer = f"Could not generate an answer: {e}"

    return {"question": question, "answer": answer}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        text = transcribe_audio(temp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"text": text}


@app.post("/speak")
def speak(text: str):
    try:
        audio_path = synthesize_speech(text)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"audio_path": audio_path}
