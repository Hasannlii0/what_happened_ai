from reasoning.event_extractor import extract_events
from reasoning.summarizer import summarize
from schema import Event, EventLog
from tests.factories import det, make_log


def log(*events):
    return EventLog(video_id="v", events=list(events))


def test_enter_and_exit_render_sentences():
    text = summarize(
        log(
            Event(t=0.0, event="enter", subject="Person_1"),
            Event(t=12.34, event="exit", subject="Person_1"),
        )
    )

    assert text == "Person_1 entered the room at 0.0s. Person_1 left the room at 12.3s."


def test_place_and_pick_up_name_the_object():
    text = summarize(
        log(
            Event(t=1.0, event="place", subject="Person_1", object="Backpack_2"),
            Event(t=2.0, event="pick_up", subject="Person_1", object="Backpack_2"),
        )
    )

    assert "Person_1 placed Backpack_2 at 1.0s." in text
    assert "Person_1 picked up Backpack_2 at 2.0s." in text


def test_approach_falls_back_to_the_object_when_no_location_is_set():
    text = summarize(
        log(Event(t=3.0, event="approach", subject="Person_1", object="Chair_4"))
    )

    assert text == "Person_1 approached Chair_4 at 3.0s."


def test_approach_prefers_location_over_object():
    text = summarize(
        log(
            Event(
                t=3.0,
                event="approach",
                subject="Person_1",
                object="Chair_4",
                location="the door",
            )
        )
    )

    assert text == "Person_1 approached the door at 3.0s."


def test_unknown_event_kind_is_reported_rather_than_dropped():
    text = summarize(
        log(
            Event(t=0.0, event="enter", subject="Person_1"),
            Event(t=1.0, event="teleported", subject="Person_1"),
            Event(t=2.0, event="exit", subject="Person_1"),
        )
    )

    assert "Person_1 entered the room at 0.0s." in text
    assert "Person_1 left the room at 2.0s." in text
    assert "teleported" in text


def test_empty_event_log_still_produces_a_sentence():
    assert summarize(log()).strip() != ""


def test_summary_of_extracted_events_reads_as_a_report():
    detections = [det(f, 1, "person", [100.0, 100.0, 200.0, 300.0]) for f in (0, 30, 60, 90)]
    detections += [det(f, 2, "backpack", [160.0, 180.0, 200.0, 220.0]) for f in (30, 60)]

    text = summarize(extract_events(make_log(detections)))

    assert text == (
        "Person_1 entered the room at 0.0s. "
        "Person_1 placed Backpack_2 at 1.0s. "
        "Person_1 picked up Backpack_2 at 2.0s. "
        "Person_1 left the room at 3.0s."
    )
