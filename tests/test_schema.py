import json

import pytest
from pydantic import ValidationError

from schema import AnalysisResult, Detection, DetectionLog, Event, EventLog

VALID_DETECTION = {
    "frame": 0,
    "timestamp": 0.0,
    "track_id": 1,
    "class_name": "person",
    "bbox": [0.0, 0.0, 10.0, 10.0],
    "confidence": 0.9,
}


def detection_log(**overrides):
    payload = {
        "video_id": "v",
        "fps": 30.0,
        "frame_width": 640,
        "frame_height": 480,
        "detections": [VALID_DETECTION],
    }
    payload.update(overrides)
    return DetectionLog(**payload)


def test_valid_detection_log_round_trips():
    log = detection_log()

    assert log.detections[0].bbox == [0.0, 0.0, 10.0, 10.0]
    assert log.frame_width == 640


@pytest.mark.parametrize("bbox", [[1.0, 2.0], [], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
def test_bbox_must_hold_exactly_four_numbers(bbox):
    with pytest.raises(ValidationError):
        Detection(**{**VALID_DETECTION, "bbox": bbox})


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_non_positive_fps_is_rejected(fps):
    with pytest.raises(ValidationError):
        detection_log(fps=fps)


@pytest.mark.parametrize("field", ["frame_width", "frame_height"])
def test_frame_dimensions_are_required_and_positive(field):
    with pytest.raises(ValidationError):
        detection_log(**{field: 0})

    payload = {
        "video_id": "v",
        "fps": 30.0,
        "frame_width": 640,
        "frame_height": 480,
        "detections": [],
    }
    del payload[field]
    with pytest.raises(ValidationError):
        DetectionLog(**payload)


def test_detection_missing_a_required_key_is_rejected():
    broken = {k: v for k, v in VALID_DETECTION.items() if k != "track_id"}

    with pytest.raises(ValidationError):
        detection_log(detections=[broken])


def test_analysis_result_requires_a_summary_source():
    with pytest.raises(ValidationError):
        AnalysisResult(video_id="v", events=[], summary="s")

    result = AnalysisResult(video_id="v", events=[], summary="s", summary_source="vlm")
    assert result.annotated_video_path is None


def test_event_object_and_location_default_to_none():
    event = Event(t=0.0, event="enter", subject="Person_1")

    assert event.object is None
    assert event.location is None
    assert EventLog(video_id="v", events=[event]).events == [event]


def test_shipped_fake_detections_fixture_validates(request):
    path = request.config.rootpath / "reasoning" / "fake_detections.json"
    data = json.loads(path.read_text())

    DetectionLog(**data)
