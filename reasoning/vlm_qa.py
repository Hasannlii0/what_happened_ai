import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

_model = None
_processor = None
_device = None


def _load():
    global _model, _processor, _device

    if _model is None:
        device = config.VLM_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        # float32 weights for a 2B model are ~8GB, which does not fit a default
        # Docker memory budget. bfloat16 halves that and keeps float32's exponent
        # range, so it does not overflow the way float16 does on CPU.
        dtype = (
            getattr(torch, config.VLM_DTYPE)
            if config.VLM_DTYPE
            else (torch.float16 if device.startswith("cuda") else torch.bfloat16)
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL, torch_dtype=dtype, device_map=device
        )
        processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
        _model, _processor, _device = model, processor, device

    return _model, _processor, _device


def build_prompt(question: str, events_text: str = "", frame_times=None) -> str:
    """Wrap a question in what the detector already knows.

    The model sees a handful of stills with no timeline, so it cannot place
    anything in time or count people it never saw. The event log supplies both.
    """
    parts = []
    if events_text:
        parts.append(
            "A person/object detector already tracked this clip and logged "
            "these events, with timestamps in seconds:\n" + events_text
        )
    if frame_times:
        parts.append(
            "The images are frames captured at: "
            + ", ".join(f"{t:.1f}s" for t in frame_times)
        )
    parts.append(question)
    return "\n\n".join(parts)


def ask_vlm(
    image_paths: list[str],
    question: str,
    events_text: str = "",
    frame_times=None,
) -> str:
    model, processor, device = _load()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append(
        {"type": "text", "text": build_prompt(question, events_text, frame_times)}
    )

    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=config.VLM_MAX_NEW_TOKENS,
        # A 2B model asked about several stills will otherwise loop the same
        # sentence until it runs out of budget.
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def describe_scene(
    image_paths: list[str], events_text: str = "", frame_times=None
) -> str:
    return ask_vlm(
        image_paths,
        "Write a short factual paragraph, at most four sentences, saying what "
        "happens in this clip. Describe the people and objects and what they do, "
        "using the logged events for timing. Do not number the frames, do not "
        "mention the camera, and do not repeat yourself.",
        events_text,
        frame_times,
    )


if __name__ == "__main__":
    from perception.keyframes import extract_keyframes

    paths = extract_keyframes(config.VIDEO_PATH)

    print("SCENE DESCRIPTION:")
    print(describe_scene(paths))

    print("\nQ&A TEST:")
    print(ask_vlm(paths, "Did anyone place an object on a table?"))
    print(ask_vlm(paths, "How many people appear across these frames?"))
