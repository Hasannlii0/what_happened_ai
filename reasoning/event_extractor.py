import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import DetectionLog, Event, EventLog


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


def extract_events(detection_log: DetectionLog, proximity_threshold=150) -> EventLog:
    tracks = group_by_track(detection_log.detections)
    events = []

    persons = {}
    objects = {}

    for tid, ds in tracks.items():
        if ds[0].class_name == "person":
            persons[tid] = ds
        else:
            objects[tid] = ds

    for tid, ds in persons.items():
        events.append(Event(
            t=ds[0].timestamp,
            event="enter",
            subject=f"Person_{tid}"
        ))
        events.append(Event(
            t=ds[-1].timestamp,
            event="exit",
            subject=f"Person_{tid}"
        ))

    for otid, ods in objects.items():
        first_seen = ods[0]
        closest_person = None
        closest_dist = float("inf")

        for ptid, pds in persons.items():
            for pd in pds:
                if pd.frame == first_seen.frame:
                    dist = bbox_distance(pd.bbox, first_seen.bbox)
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_person = ptid

        if closest_person is not None and closest_dist < proximity_threshold:
            events.append(Event(
                t=first_seen.timestamp,
                event="place",
                subject=f"Person_{closest_person}",
                object=f"{first_seen.class_name.capitalize()}_{otid}"
            ))

    events.sort(key=lambda e: e.t)

    return EventLog(video_id=detection_log.video_id, events=events)


if __name__ == "__main__":
    with open("reasoning/fake_detections.json") as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    print(event_log.model_dump_json(indent=2))