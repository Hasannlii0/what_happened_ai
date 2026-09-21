import json
import os
import sys

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


def closest_person_at_frame(persons, frame, bbox):
    closest_person = None
    closest_dist = float("inf")
    for ptid, pds in persons.items():
        for pd in pds:
            if pd.frame == frame:
                dist = bbox_distance(pd.bbox, bbox)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_person = ptid
    return closest_person, closest_dist


def extract_events(detection_log: DetectionLog, proximity_ratio=0.12) -> EventLog:
    """
    proximity_ratio: fraction of the frame diagonal used as the "close enough"
    distance for person/object interactions. Using a ratio of the frame size
    instead of a fixed pixel value (the previous proximity_threshold=150)
    keeps behavior consistent across different video resolutions.
    """
    tracks = group_by_track(detection_log.detections)
    events = []

    persons = {}
    objects = {}

    for tid, ds in tracks.items():
        if ds[0].class_name == "person":
            persons[tid] = ds
        else:
            objects[tid] = ds

    if detection_log.frame_width and detection_log.frame_height:
        diagonal = (detection_log.frame_width**2 + detection_log.frame_height**2) ** 0.5
        proximity_threshold = diagonal * proximity_ratio
    else:
        # Fallback for logs without frame dimensions (e.g. older data / tests)
        proximity_threshold = 150

    for tid, ds in persons.items():
        events.append(Event(t=ds[0].timestamp, event="enter", subject=f"Person_{tid}"))
        events.append(Event(t=ds[-1].timestamp, event="exit", subject=f"Person_{tid}"))

    for otid, ods in objects.items():
        first_seen = ods[0]
        last_seen = ods[-1]
        obj_name = f"{first_seen.class_name.capitalize()}_{otid}"

        # "place": object first appears close to a person -> that person
        # likely placed/brought it into frame.
        placer, place_dist = closest_person_at_frame(
            persons, first_seen.frame, first_seen.bbox
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

        # "pick_up": object's track disappears while last seen close to a
        # person -> that person likely picked it up and carried it off.
        picker, pick_dist = closest_person_at_frame(
            persons, last_seen.frame, last_seen.bbox
        )
        if (
            picker is not None
            and pick_dist < proximity_threshold
            and last_seen.frame != first_seen.frame
        ):
            events.append(
                Event(
                    t=last_seen.timestamp,
                    event="pick_up",
                    subject=f"Person_{picker}",
                    object=obj_name,
                )
            )

    # "approach": a person's track comes within the proximity threshold of an
    # object track partway through the video (not just at first appearance).
    for ptid, pds in persons.items():
        already_approached = set()
        for pd in pds:
            for otid, ods in objects.items():
                if otid in already_approached:
                    continue
                for od in ods:
                    if od.frame == pd.frame:
                        if bbox_distance(pd.bbox, od.bbox) < proximity_threshold:
                            events.append(
                                Event(
                                    t=pd.timestamp,
                                    event="approach",
                                    subject=f"Person_{ptid}",
                                    object=f"{od.class_name.capitalize()}_{otid}",
                                )
                            )
                            already_approached.add(otid)
                        break

    events.sort(key=lambda e: e.t)

    return EventLog(video_id=detection_log.video_id, events=events)


if __name__ == "__main__":
    with open("reasoning/fake_detections.json") as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    print(event_log.model_dump_json(indent=2))
