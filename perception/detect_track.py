import json
import shutil
import glob
import cv2
from ultralytics import YOLO
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from schema import Detection, DetectionLog

VIDEO_PATH = "test_video.mp4"
OUTPUT_JSON = "perception/detections.json"
OUTPUT_VIDEO = "perception/output_annotated.mp4"

def run():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    model = YOLO("yolov8n.pt")

    results = model.track(
        source=VIDEO_PATH,
        persist=True,
        tracker="bytetrack.yaml",
        save=True,
        conf=0.4
    )

    detections = []
    for frame_idx, r in enumerate(results):
        if r.boxes.id is None:
            continue
        for box, track_id, cls, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf):
            detections.append(Detection(
                frame=frame_idx,
                timestamp=frame_idx / fps,
                track_id=int(track_id),
                class_name=model.names[int(cls)],
                bbox=box.tolist(),
                confidence=float(conf)
            ))

    log = DetectionLog(
        video_id="test_video",
        fps=fps,
        detections=detections
    )

    with open(OUTPUT_JSON, "w") as f:
        f.write(log.model_dump_json(indent=2))

    save_dir = model.predictor.save_dir
    saved_videos = glob.glob(f"{save_dir}/*.mp4") + glob.glob(f"{save_dir}/*.avi")

    import subprocess

    if saved_videos:
        raw_path = saved_videos[0]
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            OUTPUT_VIDEO
        ], check=True)
        print(f"Annotated video re-encoded and saved to {OUTPUT_VIDEO}")
    else:
        print(f"WARNING: no annotated video found in {save_dir}")

    print(f"Saved {len(detections)} detections to {OUTPUT_JSON}")

if __name__ == "__main__":
    run()