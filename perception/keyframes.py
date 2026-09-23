import os
import sys
from pathlib import Path

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _sample_indices(total_frames, num_frames):
    count = min(num_frames, total_frames)
    return sorted({int((i + 0.5) * total_frames / count) for i in range(count)})


def _frames_by_seek(cap, indices):
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def _frames_by_scan(cap, num_frames, stride):
    frames = []
    idx = 0
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    return frames


def extract_keyframes(
    video_path,
    output_dir=config.KEYFRAMES_DIR,
    num_frames=config.KEYFRAME_COUNT,
    resize_width=config.KEYFRAME_RESIZE_WIDTH,
    timestamps=None,
):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot read video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        if timestamps and total_frames > 0 and fps > 0:
            indices = sorted(
                {min(int(t * fps), total_frames - 1) for t in timestamps if t >= 0}
            )
            frames = _frames_by_seek(cap, indices)

        # Seeking to a requested timestamp can come back empty, so fall through
        # to an even sample rather than leaving the caller with no frames.
        if not frames and total_frames > 0:
            frames = _frames_by_seek(cap, _sample_indices(total_frames, num_frames))
        if not frames:
            # Fragmented and variable-frame-rate containers report no frame count
            # and seek unreliably, so walk the stream instead of jumping in it.
            frames = _frames_by_scan(cap, num_frames, max(int(fps), 1))
    finally:
        cap.release()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("frame_*.jpg"):
        os.remove(stale)

    saved_paths = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (resize_width, max(int(h * resize_width / w), 1)))
        path = output_dir / f"frame_{i}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"could not write keyframe: {path}")
        saved_paths.append(str(path))

    if not saved_paths:
        raise ValueError(f"no decodable frames in video: {video_path}")

    return saved_paths


def extract_evidence(
    video_path,
    timestamps,
    output_dir=config.EVIDENCE_DIR,
    width=config.EVIDENCE_WIDTH,
):
    """One frame per timestamp, aligned by index; None where a frame could not
    be read, so a single bad seek does not shift every later event's image."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("event_*.jpg"):
        os.remove(stale)

    paths = [None] * len(timestamps)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot read video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            return paths

        wanted = {}
        for i, t in enumerate(timestamps):
            if t >= 0:
                index = int(t * fps)
                if total_frames > 0:
                    index = min(index, total_frames - 1)
                wanted.setdefault(index, []).append(i)

        # One forward pass decodes every frame exactly once. Seeking per event
        # instead makes H.264 re-decode from the previous keyframe each time,
        # which took 65s for 95 events on a 4K clip.
        frame_index = 0
        last_wanted = max(wanted, default=-1)
        while frame_index <= last_wanted and cap.grab():
            if frame_index in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(frame, (width, max(int(h * width / w), 1)))
                    for i in wanted[frame_index]:
                        candidate = output_dir / f"event_{i}.jpg"
                        if cv2.imwrite(str(candidate), frame):
                            paths[i] = str(candidate)
            frame_index += 1
    finally:
        cap.release()

    return paths


if __name__ == "__main__":
    paths = extract_keyframes(config.VIDEO_PATH)
    print(f"Saved {len(paths)} keyframes: {paths}")
