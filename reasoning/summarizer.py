import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from schema import EventLog


def summarize(event_log: EventLog) -> str:
    lines = []

    for e in event_log.events:
        if e.event == "enter":
            lines.append(f"{e.subject} entered the room at {e.t:.1f}s.")
        elif e.event == "exit":
            lines.append(f"{e.subject} left the room at {e.t:.1f}s.")
        elif e.event == "place":
            lines.append(f"{e.subject} placed {e.object} at {e.t:.1f}s.")
        elif e.event == "pick_up":
            lines.append(f"{e.subject} picked up {e.object} at {e.t:.1f}s.")
        elif e.event == "approach":
            target = e.location or e.object or "something"
            lines.append(f"{e.subject} approached {target} at {e.t:.1f}s.")
        else:
            lines.append(f"{e.subject} {e.event} at {e.t:.1f}s.")

    if not lines:
        return "No people or objects were detected in this clip."

    return " ".join(lines)


def format_event_context(event_log: EventLog) -> str:
    """The event log as a compact timestamped table, for grounding a VLM prompt."""
    lines = []
    for e in event_log.events:
        target = f" {e.object}" if e.object else ""
        lines.append(f"{e.t:6.1f}s  {e.subject} {e.event}{target}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json

    from event_extractor import extract_events

    import config
    from schema import DetectionLog

    with open(config.REPO_ROOT / "reasoning" / "fake_detections.json") as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)
    print(summarize(event_log))
