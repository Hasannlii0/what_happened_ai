from pydantic import BaseModel, Field, conlist


class Detection(BaseModel):
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    bbox: conlist(float, min_length=4, max_length=4)
    confidence: float


class DetectionLog(BaseModel):
    video_id: str
    fps: float = Field(gt=0)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
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
    summary_source: str
    annotated_video_path: str | None = None
