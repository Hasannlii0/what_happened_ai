import cv2
import os
def extract_keyframes(video_path, output_dir="perception/keyframes", num_frames=3, resize_width=512):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    saved_paths = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            new_h = int(h * (resize_width / w))
            frame = cv2.resize(frame, (resize_width, new_h))
            path = f"{output_dir}/frame_{idx}.jpg"
            cv2.imwrite(path, frame)
            saved_paths.append(path)

    cap.release()
    return saved_paths

if __name__ == "__main__":
    paths = extract_keyframes("test_video.mp4")
    print(f"Saved {len(paths)} keyframes: {paths}")