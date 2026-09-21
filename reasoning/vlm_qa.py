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
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL, torch_dtype=dtype, device_map=device
        )
        processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
        _model, _processor, _device = model, processor, device

    return _model, _processor, _device


def ask_vlm(image_paths: list[str], question: str) -> str:
    model, processor, device = _load()

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": question})

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

    generated_ids = model.generate(**inputs, max_new_tokens=200)
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


def describe_scene(image_paths: list[str]) -> str:
    return ask_vlm(
        image_paths,
        "Describe what you see across these frames from a video, in chronological order, focusing on people, objects, and actions.",
    )


if __name__ == "__main__":
    from perception.keyframes import extract_keyframes

    paths = extract_keyframes(config.VIDEO_PATH)

    print("SCENE DESCRIPTION:")
    print(describe_scene(paths))

    print("\nQ&A TEST:")
    print(ask_vlm(paths, "Did anyone place an object on a table?"))
    print(ask_vlm(paths, "How many people appear across these frames?"))
