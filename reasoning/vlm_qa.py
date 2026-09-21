import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map=device
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def ask_vlm(image_paths: list[str], question: str) -> str:
    torch.cuda.empty_cache()

    content = [{"type": "image", "image": path} for path in image_paths]
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
    from keyframes import extract_keyframes

    paths = extract_keyframes(
        "../test_video.mp4" if os.getcwd().endswith("reasoning") else "test_video.mp4"
    )

    print("SCENE DESCRIPTION:")
    print(describe_scene(paths))

    print("\nQ&A TEST:")
    print(ask_vlm(paths, "Did anyone place an object on a table?"))
    print(ask_vlm(paths, "How many people appear across these frames?"))
