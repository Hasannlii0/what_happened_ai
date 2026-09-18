import json
import sys 
import os 
import shutil
import glob
import subprocess
import cv2
import mlflow
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import Detection, DetectionLog

VIDEO_PATH = "test_video.mp4"
OUTPUT_JSON = "perception/detections.json"
OUTPUT_VIDEO = "perception/output_annotated.mp4"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
TRACKER = "botsort.yaml"


def run():
    mlflow.set_experiment("what_happened_ai")

    with mlflow.start_run():
        mlflow.log_param("conf_threshold", CONF_THRESHOLD)
        mlflow.log_param("iou_threshold", IOU_THRESHOLD)
        mlflow.log_param("tracker", TRACKER)
        mlflow.log_param("model", "yolov8n")

        cap = cv2.VideoCapture(VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        cap.release()

        mlflow.log_param("video_fps", fps)
        mlflow.log_param("video_duration_sec", round(duration, 2))

        model = YOLO("yolov8n.pt")

        results = model.track(
            source=VIDEO_PATH,
            persist=True,
            tracker=TRACKER,
            save=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD
        )

        detections = []
        unique_track_ids = set()

        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue
            for box, track_id, cls, conf in zip(r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf):
                tid = int(track_id)
                unique_track_ids.add(tid)
                detections.append(Detection(
                    frame=frame_idx,
                    timestamp=frame_idx / fps,
                    track_id=tid,
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

        mlflow.log_metric("num_detections", len(detections))
        mlflow.log_metric("num_unique_tracks", len(unique_track_ids))
        mlflow.log_artifact(OUTPUT_JSON)

        save_dir = model.predictor.save_dir
        saved_videos = glob.glob(f"{save_dir}/*.mp4") + glob.glob(f"{save_dir}/*.avi")

        if saved_videos:
            raw_path = saved_videos[0]
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                OUTPUT_VIDEO
            ], check=True)
            print(f"Annotated video re-encoded and saved to {OUTPUT_VIDEO}")
            mlflow.log_artifact(OUTPUT_VIDEO)
        else:
            print(f"WARNING: no annotated video found in {save_dir}")

        print(f"Saved {len(detections)} detections ({len(unique_track_ids)} unique tracks) to {OUTPUT_JSON}")


if __name__ == "__main__":
    run()