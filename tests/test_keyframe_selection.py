from reasoning.event_extractor import select_keyframe_times
from reasoning.summarizer import format_event_context
from schema import Event, EventLog


def log(*events):
    return EventLog(video_id="v", events=list(events))


def ev(t, kind, subject="Person_1", obj=None):
    return Event(t=t, event=kind, subject=subject, object=obj)


def test_object_interactions_are_preferred_over_entrances():
    chosen = select_keyframe_times(
        log(
            ev(1.0, "enter"),
            ev(5.0, "place", obj="Bag_2"),
            ev(9.0, "exit"),
        ),
        count=1,
    )

    assert chosen == [5.0]


def test_an_unattended_object_outranks_its_placement():
    chosen = select_keyframe_times(
        log(ev(1.0, "place", obj="Bag_2"), ev(4.0, "abandon", obj="Bag_2")),
        count=1,
    )

    assert chosen == [4.0]


def test_pick_up_outranks_approach():
    chosen = select_keyframe_times(
        log(ev(2.0, "approach", obj="Bag_2"), ev(6.0, "pick_up", obj="Bag_2")),
        count=1,
    )

    assert chosen == [6.0]


def test_results_come_back_in_chronological_order():
    chosen = select_keyframe_times(
        log(
            ev(8.0, "pick_up", obj="Bag_2"),
            ev(2.0, "place", obj="Bag_2"),
            ev(5.0, "approach", obj="Bag_2"),
        ),
        count=3,
    )

    assert chosen == [2.0, 5.0, 8.0]


def test_near_simultaneous_events_do_not_both_get_a_frame():
    chosen = select_keyframe_times(
        log(
            ev(4.00, "place", obj="Bag_2"),
            ev(4.05, "pick_up", obj="Bag_2"),
            ev(9.00, "exit"),
        ),
        count=3,
        min_gap=0.5,
    )

    assert 4.05 not in chosen
    assert chosen == [4.0, 9.0]


def test_selection_is_capped_at_the_requested_count():
    events = [ev(float(i), "approach", obj=f"Bag_{i}") for i in range(10)]

    assert len(select_keyframe_times(log(*events), count=3)) == 3


def test_no_events_yields_no_timestamps():
    assert select_keyframe_times(log(), count=3) == []


def test_event_context_renders_timestamps_subjects_and_objects():
    text = format_event_context(
        log(ev(3.8, "enter", "Person_3"), ev(9.1, "pick_up", "Person_3", "Bottle_16"))
    )
    lines = text.splitlines()

    assert "3.8s" in lines[0]
    assert "Person_3 enter" in lines[0]
    assert "Person_3 pick_up Bottle_16" in lines[1]


def test_event_context_is_empty_for_an_empty_log():
    assert format_event_context(log()) == ""
