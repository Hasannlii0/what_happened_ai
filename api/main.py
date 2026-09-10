import sys
import os
import shutil
import subprocess
import time
import logging

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("analyze")

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

    for path in ["perception/detections.json", "perception/output_annotated.mp4"]:
        if os.path.exists(path):
            os.remove(path)

    t0 = time.time()
    log.info("STAGE 1/4: running YOLO detection + tracking (detect_track.py)...")
    result = subprocess.run(
        [sys.executable, "perception/detect_track.py"],
        capture_output=True,
        text=True
    )
    t1 = time.time()
    log.info(f"STAGE 1/4 done in {t1 - t0:.1f}s")

    if result.returncode != 0:
        return JSONResponse(status_code=500, content={"error": f"Detection failed: {result.stderr}"})

    detections_path = "perception/detections.json"
    if not os.path.exists(detections_path):
        return JSONResponse(status_code=500, content={"error": "detect_track.py ran but produced no output."})

    with open(detections_path) as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)

    log.info("STAGE 2/4: extracting events from detections...")
    event_log = extract_events(detection_log)
    t2 = time.time()
    log.info(f"STAGE 2/4 done in {t2 - t1:.1f}s")

    LATEST_EVENT_LOG["data"] = event_log

    log.info("STAGE 3/4: generating summary (flan-t5-large — first call also downloads/loads the model, ~3GB)...")
    try:
        summary_text = generate_summary(event_log)
    except Exception as e:
        log.warning(f"LLM summary failed ({e}), falling back to template summarizer")
        summary_text = summarize(event_log)
    t3 = time.time()
    log.info(f"STAGE 3/4 done in {t3 - t2:.1f}s")

    annotated_path = "perception/output_annotated.mp4"
    if not os.path.exists(annotated_path):
        annotated_path = None

    log.info("STAGE 4/4: assembling response...")
    analysis_result = AnalysisResult(
        video_id=event_log.video_id,
        events=event_log.events,
        summary=summary_text,
        annotated_video_path=annotated_path
    )
    log.info(f"TOTAL /analyze time: {time.time() - t0:.1f}s "
             f"(detection={t1-t0:.1f}s, events={t2-t1:.1f}s, summary={t3-t2:.1f}s)")

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