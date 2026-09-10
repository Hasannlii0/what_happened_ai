from pydantic import BaseModel
from typing import Optional, List


class Detection(BaseModel):
    frame: int
    timestamp: float
    track_id: int
    class_name: str
    bbox: List[float]
    confidence: float


class DetectionLog(BaseModel):
    video_id: str
    fps: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    detections: List[Detection]


class Event(BaseModel):
    t: float
    event: str
    subject: str
    object: Optional[str] = None
    location: Optional[str] = None


class EventLog(BaseModel):
    video_id: str
    events: List[Event]


class AnalysisResult(BaseModel):
    video_id: str
    events: List[Event]
    summary: str
    annotated_video_path: Optional[str] = None