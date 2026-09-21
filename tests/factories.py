from schema import Detection, DetectionLog

FPS = 30.0


def det(frame, track_id, class_name, bbox, confidence=0.9):
    return Detection(
        frame=frame,
        timestamp=frame / FPS,
        track_id=track_id,
        class_name=class_name,
        bbox=bbox,
        confidence=confidence,
    )


def make_log(detections, width=640, height=480, video_id="v"):
    return DetectionLog(
        video_id=video_id,
        fps=FPS,
        frame_width=width,
        frame_height=height,
        detections=detections,
    )


def kinds(event_log):
    return [e.event for e in event_log.events]


def tuples(event_log):
    return [(e.event, e.subject, e.object) for e in event_log.events]
