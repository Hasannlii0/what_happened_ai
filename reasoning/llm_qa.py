import os
import json
from huggingface_hub import InferenceClient

from pathlib import Path
import sys


root_dir = str(Path(__file__).resolve().parent.parent)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from schema import EventLog

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def build_context(event_log: EventLog) -> str:
    lines = []
    for e in event_log.events:
        line = f"At {e.t:.1f}s, {e.subject} {e.event}"
        if e.object:
            line += f" {e.object}"
        if e.location:
            line += f" at {e.location}"
        lines.append(line)
    return ". ".join(lines) + "."


def query_hf(prompt: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.3
    )
    return completion.choices[0].message.content.strip()


def ask(event_log: EventLog, question: str) -> str:
    context = build_context(event_log)
    prompt = f"You are an AI video analyst. Given this event log, answer the question using only the information provided. Be clear and specific.\n\nEvent log: {context}\n\nQuestion: {question}"
    return query_hf(prompt)


def generate_summary(event_log: EventLog) -> str:
    context = build_context(event_log)
    prompt = f"You are an AI video analyst. Given this event log, write a clear, natural-language paragraph describing everything that happened, in chronological order.\n\nEvent log: {context}"
    return query_hf(prompt)


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
    print(ask(event_log, "Who entered the room first?"))