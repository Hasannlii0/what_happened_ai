import pytest

from reasoning.event_extractor import NARRATIVE_ORDER, extract_events
from tests.factories import det, kinds, make_log, tuples

PERSON_BBOX = [100.0, 100.0, 200.0, 300.0]
NEAR_BBOX = [160.0, 180.0, 200.0, 220.0]
FAR_BBOX = [600.0, 400.0, 620.0, 440.0]


def test_empty_detections_produce_empty_event_log():
    log = extract_events(make_log([], video_id="empty"))

    assert log.video_id == "empty"
    assert log.events == []


def test_person_track_produces_one_enter_and_one_exit():
    log = extract_events(
        make_log([det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60)])
    )

    assert tuples(log) == [
        ("enter", "Person_1", None),
        ("exit", "Person_1", None),
    ]
    assert [e.t for e in log.events] == [0.0, 2.0]


def test_object_appearing_next_to_person_is_placed_then_picked_up():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 2, "backpack", NEAR_BBOX) for f in (30, 60)]

    log = extract_events(make_log(detections))

    assert tuples(log) == [
        ("enter", "Person_1", None),
        ("place", "Person_1", "Backpack_2"),
        ("pick_up", "Person_1", "Backpack_2"),
        ("exit", "Person_1", None),
    ]
    assert [e.t for e in log.events] == [0.0, 1.0, 2.0, 3.0]


def test_object_far_from_person_is_not_placed():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 2, "backpack", FAR_BBOX) for f in (30, 60)]

    log = extract_events(make_log(detections))

    assert kinds(log) == ["enter", "exit"]


def test_single_frame_object_produces_no_interaction_events():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(30, 2, "bottle", NEAR_BBOX)]

    log = extract_events(make_log(detections))

    assert kinds(log) == ["enter", "exit"]


def test_a_high_confidence_flicker_is_still_discarded():
    """A false positive arrives with respectable confidence, so only its
    lifetime distinguishes it from a real subject."""
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 9, "traffic light", FAR_BBOX, confidence=0.62) for f in (40, 42)]

    log = extract_events(make_log(detections))

    assert "Trafficlight_9" not in [e.object for e in log.events]
    assert kinds(log) == ["enter", "exit"]


def test_a_brief_person_track_produces_no_enter_or_exit():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 7, "person", FAR_BBOX) for f in (50, 52)]

    log = extract_events(make_log(detections))

    assert [e.subject for e in log.events] == ["Person_1", "Person_1"]


def test_min_track_seconds_is_tunable():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 7, "person", FAR_BBOX) for f in (50, 56)]

    strict = extract_events(make_log(detections), min_track_seconds=1.0)
    lenient = extract_events(make_log(detections), min_track_seconds=0.1)

    assert "Person_7" not in [e.subject for e in strict.events]
    assert "Person_7" in [e.subject for e in lenient.events]


def test_object_present_from_first_frame_is_not_placed_or_picked_up():
    frames = (0, 30, 60, 90)
    detections = [det(f, 1, "person", PERSON_BBOX) for f in frames]
    detections += [det(f, 2, "chair", NEAR_BBOX) for f in frames]

    log = extract_events(make_log(detections))

    assert "place" not in kinds(log)
    assert "pick_up" not in kinds(log)


def test_approach_fires_once_per_rising_edge_not_once_per_frame():
    far = [0.0, 0.0, 100.0, 100.0]
    near = [450.0, 350.0, 550.0, 450.0]
    detections = [det(f, 1, "person", far) for f in (0, 6, 12)]
    detections += [det(f, 1, "person", near) for f in (18, 24, 30)]
    detections += [
        det(f, 2, "chair", [480.0, 380.0, 520.0, 420.0])
        for f in (0, 6, 12, 18, 24, 30)
    ]

    log = extract_events(make_log(detections))
    approaches = [e for e in log.events if e.event == "approach"]

    assert len(approaches) == 1
    assert approaches[0].subject == "Person_1"
    assert approaches[0].object == "Chair_2"
    assert approaches[0].t == pytest.approx(18 / 30)


def test_approach_fires_again_after_the_person_moves_away_and_returns():
    far = [0.0, 0.0, 100.0, 100.0]
    near = [450.0, 350.0, 550.0, 450.0]
    positions = {0: far, 6: far, 12: far, 18: near, 24: far, 30: near}
    detections = [det(f, 1, "person", b) for f, b in positions.items()]
    detections += [
        det(f, 2, "chair", [480.0, 380.0, 520.0, 420.0])
        for f in (0, 6, 12, 18, 24, 30)
    ]

    log = extract_events(make_log(detections))
    approaches = [e for e in log.events if e.event == "approach"]

    assert [e.t for e in approaches] == pytest.approx([18 / 30, 30 / 30])


def test_placement_does_not_also_emit_an_approach_for_the_same_moment():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 2, "backpack", NEAR_BBOX) for f in (30, 60)]

    log = extract_events(make_log(detections))

    assert "approach" not in kinds(log)


def test_proximity_threshold_scales_with_declared_frame_size():
    detections = [det(f, 1, "person", [50.0, 150.0, 150.0, 250.0]) for f in (0, 30, 60, 90)]
    detections += [det(f, 2, "bottle", [230.0, 180.0, 270.0, 220.0]) for f in (30, 60)]

    small = extract_events(make_log(detections, width=640, height=480))
    large = extract_events(make_log(detections, width=1920, height=1080))

    assert kinds(small) == ["enter", "exit"]
    assert "place" in kinds(large)
    assert kinds(small) != kinds(large)


def test_track_is_named_by_its_majority_class_not_its_first_frame():
    detections = [
        det(0, 1, "handbag", PERSON_BBOX),
        det(30, 1, "person", PERSON_BBOX),
        det(60, 1, "person", PERSON_BBOX),
        det(90, 1, "person", PERSON_BBOX),
        det(30, 2, "suitcase", NEAR_BBOX),
        det(60, 2, "backpack", NEAR_BBOX),
        det(90, 2, "backpack", NEAR_BBOX),
    ]

    log = extract_events(make_log(detections))

    assert tuples(log) == [
        ("enter", "Person_1", None),
        ("place", "Person_1", "Backpack_2"),
        ("exit", "Person_1", None),
    ]


def test_events_at_the_same_timestamp_follow_narrative_order():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 60, 90)]
    detections += [det(f, 2, "backpack", NEAR_BBOX) for f in (90, 120)]

    log = extract_events(make_log(detections))

    assert [(e.event, e.t) for e in log.events] == [
        ("enter", 0.0),
        ("place", 3.0),
        ("exit", 3.0),
    ]


def test_events_are_sorted_by_timestamp():
    detections = [det(f, 1, "person", PERSON_BBOX) for f in (0, 30, 60, 90)]
    detections += [det(f, 3, "person", [400.0, 100.0, 500.0, 300.0]) for f in (30, 120)]
    detections += [det(f, 2, "backpack", NEAR_BBOX) for f in (30, 60)]
    detections += [det(f, 4, "chair", FAR_BBOX) for f in (0, 60, 90)]

    log = extract_events(make_log(detections))
    keys = [(e.t, NARRATIVE_ORDER[e.event]) for e in log.events]

    assert keys == sorted(keys)
    assert len(log.events) > 4
