import os
import subprocess
import sys
from pathlib import Path

import cv2
import mlflow
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from schema import Detection, DetectionLog


def probe_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    if fps <= 0:
        raise ValueError(f"video reports no frame rate, cannot time events: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"video reports no frame size: {path}")

    return fps, width, height, frame_count


def _scale_filter(side):
    # Shrink the longer side to the cap, never upscale; -2 keeps the aspect
    # ratio and an even dimension, which yuv420p requires.
    return (
        f"scale=w='if(gte(iw,ih),min(iw,{side}),-2)'"
        f":h='if(gte(iw,ih),-2,min(ih,{side}))'"
    )


def make_working_copy(width, height):
    """A copy of the upload no larger than WORK_MAX_SIDE, or the upload itself.

    YOLO letterboxes every frame to 640px, so a 4K source buys no accuracy. It
    only makes decoding, BoT-SORT's per-frame optical flow and the annotated
    video several times more expensive: 25s of tracking at 4K against 7s at
    1280 on the same clip, with identical detections.
    """
    side = config.WORK_MAX_SIDE
    if max(width, height) <= side:
        return config.VIDEO_PATH, width, height

    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(config.VIDEO_PATH),
        "-vf", _scale_filter(side),
        # One output frame per input frame, so frame indices -- and every
        # timestamp derived from them -- still match the original upload.
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-an",
        str(config.WORK_VIDEO),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        result = None
    if result is None or result.returncode != 0:
        print(
            "WARNING: could not downscale the upload, tracking at full resolution.",
            file=sys.stderr,
        )
        return config.VIDEO_PATH, width, height

    _, work_width, work_height, _ = probe_video(config.WORK_VIDEO)
    return config.WORK_VIDEO, work_width, work_height


def reencode_annotated(save_dir, source_stem):
    # Anything already at this path is a previous run's video and must never be
    # served as this one's.
    config.ANNOTATED_VIDEO.unlink(missing_ok=True)

    # The run directory is reused across runs (exist_ok=True), so match the name
    # YOLO gave this run's output -- its input's stem -- rather than whatever
    # video happens to be newest.
    candidates = [
        p
        for p in save_dir.glob(f"{source_stem}.*")
        if p.suffix in {".mp4", ".avi"}
    ]
    if not candidates:
        print(f"WARNING: no annotated video found in {save_dir}", file=sys.stderr)
        return

    raw_path = max(candidates, key=os.path.getmtime)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_path),
        "-vf",
        _scale_filter(config.WORK_MAX_SIDE),
        "-vcodec",
        "libx264",
        # The default "medium" preset spends most of its time on compression a
        # local preview never needs; veryfast is several times quicker.
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(config.ANNOTATED_VIDEO),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            "WARNING: ffmpeg was not found on PATH, so no playable annotated video "
            "was produced. Install ffmpeg to enable the annotated video.",
            file=sys.stderr,
        )
        return

    if result.returncode != 0:
        config.ANNOTATED_VIDEO.unlink(missing_ok=True)
        print(
            f"WARNING: ffmpeg failed with exit code {result.returncode}, so no "
            f"annotated video was produced:\n{result.stderr[-2000:]}",
            file=sys.stderr,
        )
        return

    print(f"Annotated video re-encoded and saved to {config.ANNOTATED_VIDEO}")


def run():
    fps, width, height, frame_count = probe_video(config.VIDEO_PATH)
    duration = frame_count / fps if frame_count > 0 else 0
    source, work_width, work_height = make_working_copy(width, height)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        mlflow.log_param("conf_threshold", config.CONF_THRESHOLD)
        mlflow.log_param("iou_threshold", config.IOU_THRESHOLD)
        mlflow.log_param("tracker", config.TRACKER)
        mlflow.log_param("model", config.YOLO_MODEL)
        mlflow.log_param("video_fps", fps)
        mlflow.log_param("video_duration_sec", round(duration, 2))
        mlflow.log_param("work_resolution", f"{work_width}x{work_height}")

        model = YOLO(config.YOLO_MODEL)

        results = model.track(
            source=str(source),
            stream=True,
            persist=True,
            tracker=config.TRACKER,
            save=True,
            project=str(config.YOLO_RUNS_DIR),
            name=config.YOLO_RUN_NAME,
            exist_ok=True,
            iou=config.IOU_THRESHOLD,
        )

        detections = []
        unique_track_ids = set()

        for frame_idx, r in enumerate(results):
            if r.boxes.id is None:
                continue
            for box, track_id, cls, conf in zip(
                r.boxes.xyxy, r.boxes.id, r.boxes.cls, r.boxes.conf
            ):
                # The tracker runs at its own low confidence so it can bridge
                # occlusions instead of splitting one person into two tracks;
                # the log keeps only the confident boxes.
                if float(conf) < config.CONF_THRESHOLD:
                    continue
                tid = int(track_id)
                unique_track_ids.add(tid)
                detections.append(
                    Detection(
                        frame=frame_idx,
                        timestamp=frame_idx / fps,
                        track_id=tid,
                        class_name=model.names[int(cls)],
                        bbox=box.tolist(),
                        confidence=float(conf),
                    )
                )

        # Boxes are in the working copy's pixels, so the declared frame size
        # must be too, or the proximity threshold is computed on the wrong scale.
        log = DetectionLog(
            video_id=config.VIDEO_PATH.stem,
            fps=fps,
            frame_width=work_width,
            frame_height=work_height,
            detections=detections,
        )

        with open(config.DETECTIONS_JSON, "w") as f:
            f.write(log.model_dump_json(indent=2))

        mlflow.log_metric("num_detections", len(detections))
        mlflow.log_metric("num_unique_tracks", len(unique_track_ids))
        mlflow.log_artifact(str(config.DETECTIONS_JSON))

        reencode_annotated(Path(model.predictor.save_dir), Path(source).stem)
        if source == config.WORK_VIDEO:
            config.WORK_VIDEO.unlink(missing_ok=True)

        print(
            f"Saved {len(detections)} detections ({len(unique_track_ids)} unique tracks) to {config.DETECTIONS_JSON}"
        )


if __name__ == "__main__":
    run()
