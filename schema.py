from pydantic import BaseModel


class Detection(BaseModel):
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    bbox: list[float]
    confidence: float


class DetectionLog(BaseModel):
    video_id: str
    fps: float
    frame_width: int | None = None
    frame_height: int | None = None
    detections: list[Detection]


class Event(BaseModel):
    t: float
    event: str
    subject: str
    object: str | None = None
    location: str | None = None


class EventLog(BaseModel):
    video_id: str
    events: list[Event]


class AnalysisResult(BaseModel):
    video_id: str
    events: list[Event]
    summary: str
    annotated_video_path: str | None = None
