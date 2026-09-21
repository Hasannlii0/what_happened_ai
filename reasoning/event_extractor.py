import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from schema import DetectionLog, Event, EventLog

NARRATIVE_ORDER = {"enter": 0, "approach": 1, "place": 2, "pick_up": 3, "exit": 4}


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_distance(b1, b2):
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def group_by_track(detections):
    tracks = {}
    for d in detections:
        tracks.setdefault(d.track_id, []).append(d)
    for tid in tracks:
        tracks[tid].sort(key=lambda d: d.frame)
    return tracks


def track_class_name(track):
    """Majority class over the whole track: YOLO regularly mislabels the first
    frame of a track, and that one frame would otherwise name it forever."""
    return Counter(d.class_name for d in track).most_common(1)[0][0]


def index_by_frame(tracks):
    by_frame = {}
    for ds in tracks.values():
        for d in ds:
            by_frame.setdefault(d.frame, []).append(d)
    return by_frame


def closest_person_at_frame(person_detections, bbox):
    closest_person = None
    closest_dist = float("inf")
    for pd in person_detections:
        dist = bbox_distance(pd.bbox, bbox)
        if dist < closest_dist:
            closest_dist = dist
            closest_person = pd.track_id
    return closest_person, closest_dist


def extract_events(detection_log: DetectionLog, proximity_ratio=None) -> EventLog:
    """proximity_ratio: fraction of the frame diagonal treated as "close enough"
    for a person/object interaction, so behavior holds across resolutions."""
    if proximity_ratio is None:
        proximity_ratio = config.PROXIMITY_RATIO

    tracks = group_by_track(detection_log.detections)
    class_names = {tid: track_class_name(ds) for tid, ds in tracks.items()}

    persons = {tid: ds for tid, ds in tracks.items() if class_names[tid] == "person"}
    objects = {
        tid: ds
        for tid, ds in tracks.items()
        if class_names[tid] != "person" and ds[0].frame != ds[-1].frame
    }

    diagonal = (detection_log.frame_width**2 + detection_log.frame_height**2) ** 0.5
    proximity_threshold = diagonal * proximity_ratio

    persons_by_frame = index_by_frame(persons)
    objects_by_frame = index_by_frame(objects)
    last_frame = max((d.frame for d in detection_log.detections), default=0)

    events = []
    handled = set()

    for tid, ds in persons.items():
        events.append(Event(t=ds[0].timestamp, event="enter", subject=f"Person_{tid}"))
        events.append(Event(t=ds[-1].timestamp, event="exit", subject=f"Person_{tid}"))

    for otid, ods in objects.items():
        first_seen = ods[0]
        last_seen = ods[-1]
        obj_name = f"{class_names[otid].capitalize()}_{otid}"

        # An object visible from frame 0 was already there; nobody placed it.
        if first_seen.frame > 0:
            placer, place_dist = closest_person_at_frame(
                persons_by_frame.get(first_seen.frame, []), first_seen.bbox
            )
            if placer is not None and place_dist < proximity_threshold:
                events.append(
                    Event(
                        t=first_seen.timestamp,
                        event="place",
                        subject=f"Person_{placer}",
                        object=obj_name,
                    )
                )
                handled.add((placer, otid, first_seen.frame))

        # An object still on screen at the end was not carried off.
        if last_seen.frame < last_frame:
            picker, pick_dist = closest_person_at_frame(
                persons_by_frame.get(last_seen.frame, []), last_seen.bbox
            )
            if picker is not None and pick_dist < proximity_threshold:
                events.append(
                    Event(
                        t=last_seen.timestamp,
                        event="pick_up",
                        subject=f"Person_{picker}",
                        object=obj_name,
                    )
                )
                handled.add((picker, otid, last_seen.frame))

    # Rising edge only, and skipping pairs already reported as place/pick_up on
    # this frame, or every placement would also emit a duplicate approach.
    was_near = {}
    for frame in sorted(persons_by_frame):
        ods = objects_by_frame.get(frame)
        if not ods:
            continue
        for pd in persons_by_frame[frame]:
            for od in ods:
                pair = (pd.track_id, od.track_id)
                near = bbox_distance(pd.bbox, od.bbox) < proximity_threshold
                if near and not was_near.get(pair, False):
                    if (pd.track_id, od.track_id, frame) not in handled:
                        events.append(
                            Event(
                                t=pd.timestamp,
                                event="approach",
                                subject=f"Person_{pd.track_id}",
                                object=f"{class_names[od.track_id].capitalize()}_{od.track_id}",
                            )
                        )
                was_near[pair] = near

    events.sort(key=lambda e: (e.t, NARRATIVE_ORDER.get(e.event, len(NARRATIVE_ORDER))))

    return EventLog(video_id=detection_log.video_id, events=events)


# An object changing hands is the moment worth looking at; a person merely
# entering is not. Evenly spaced sampling misses the former almost every time.
KEYFRAME_PRIORITY = {"place": 0, "pick_up": 1, "approach": 2, "enter": 3, "exit": 4}


def select_keyframe_times(event_log, count, min_gap=0.5):
    """Timestamps of the most informative moments, in chronological order.

    min_gap keeps two events at the same instant from spending the whole budget
    on one moment.
    """
    ranked = sorted(
        event_log.events,
        key=lambda e: (KEYFRAME_PRIORITY.get(e.event, len(KEYFRAME_PRIORITY)), e.t),
    )

    chosen = []
    for event in ranked:
        if len(chosen) >= count:
            break
        if all(abs(event.t - picked) >= min_gap for picked in chosen):
            chosen.append(event.t)

    return sorted(chosen)


if __name__ == "__main__":
    with open(config.REPO_ROOT / "reasoning" / "fake_detections.json") as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    print(event_log.model_dump_json(indent=2))
