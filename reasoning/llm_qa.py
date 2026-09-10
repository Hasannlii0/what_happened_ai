import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
from pathlib import Path



root_dir = str(Path(__file__).resolve().parent.parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from schema import EventLog

# Lazy-loaded: loading flan-t5-large (~3GB) at import time made the API
# process slow to start (and reload the model on every restart) even for
# requests that never call /ask. It's now loaded once, on first use.
_MODEL_NAME = os.environ.get("QA_MODEL_NAME", "google/flan-t5-large")
_tokenizer = None
_model = None

# Beam search (num_beams=4) is noticeably slow on CPU for a "large" model.
# Override with e.g. SUMMARY_NUM_BEAMS=1 for much faster (greedy) generation,
# or leave the default if summary quality matters more than speed.
_SUMMARY_NUM_BEAMS = int(os.environ.get("SUMMARY_NUM_BEAMS", "2"))


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = T5Tokenizer.from_pretrained(_MODEL_NAME)
        _model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)
    return _tokenizer, _model


def build_context(event_log: EventLog) -> str:
    lines = []
    people = set()
    objects = set()
    for e in event_log.events:
        line = f"At {e.t:.1f}s, {e.subject} {e.event}"
        if e.object:
            line += f" {e.object}"
            objects.add(e.object)
        if e.location:
            line += f" at {e.location}"
        lines.append(line)
        if e.subject.startswith("Person"):
            people.add(e.subject)

    summary_facts = f"Total unique people detected: {len(people)}. Total unique objects detected: {len(objects)}."
    return summary_facts + " " + ". ".join(lines) + "."


def ask(event_log: EventLog, question: str) -> str:
    tokenizer, model = _get_model()
    context = build_context(event_log)
    prompt = f"Event log: {context}\n\nQuestion: {question}\nAnswer based only on the event log:"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_summary(event_log: EventLog) -> str:
    tokenizer, model = _get_model()
    context = build_context(event_log)
    prompt = f"Write a detailed paragraph describing everything that happened, in chronological order, based on this event log: {context}"
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_new_tokens=250, min_new_tokens=60, num_beams=_SUMMARY_NUM_BEAMS)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    
    from event_extractor import extract_events
    from schema import DetectionLog

    with open("reasoning/fake_detections.json") as f:
        data = json.load(f)

    detection_log = DetectionLog(**data)
    event_log = extract_events(detection_log)

    print("EVENT LOG:")
    print(event_log.model_dump_json(indent=2))

    print("\nSUMMARY:")
    print(generate_summary(event_log))

    print("\nQ&A TEST:")
    print(ask(event_log, "Did anyone leave an object?"))
    print(ask(event_log, "How many people entered the room?"))
    print(ask(event_log, "Who entered the room first?"))
    print(ask(event_log, 'What are the names of the people?'))