import sys
import os
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import json

from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from schema import DetectionLog, AnalysisResult
from reasoning.event_extractor import extract_events
from reasoning.summarizer import summarize
from reasoning.llm_qa import ask, generate_summary

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
        return JSONResponse(status_code=404, content={"error": "No video uploaded yet."})

    result = subprocess.run(
        [sys.executable, "perception/detect_track.py"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {result.stderr}"})

    detections_path = "perception/detections.json"
    if not os.path.exists(detections_path):
        return JSONResponse(status_code=500, content={"error": "detect_track.py ran but produced no output."})

    with open(detections_path) as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    LATEST_EVENT_LOG["data"] = event_log

    try:
        summary_text = generate_summary(event_log)
    except Exception:
        summary_text = summarize(event_log)

    annotated_path = "perception/output_annotated.mp4"
    if not os.path.exists(annotated_path):
        annotated_path = None

    analysis_result = AnalysisResult(
        video_id=event_log.video_id,
        events=event_log.events,
        summary=summary_text,
        annotated_video_path=annotated_path
    )

    return analysis_result.model_dump()


@app.post("/ask")
def ask_question(question: str):
    if LATEST_EVENT_LOG["data"] is None:
        return JSONResponse(status_code=400, content={"error": "Run /analyze first."})

    try:
        answer = ask(LATEST_EVENT_LOG["data"], question)
    except Exception as e:
        answer = f"Could not generate an answer: {e}"

    return {"question": question, "answer": answer}