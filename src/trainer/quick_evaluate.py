# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from PIL import Image
import requests
from io import BytesIO

import torch
from datasets import load_dataset

from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
)


def format_message(sample: str, system_message: str):
    content = [
        {
            "type": "text",
            "text": sample["query"],
        },
    ]
    if "image" in sample and sample["image"] is not None:
        if isinstance(sample["image"], list):
            for img in sample["image"]:
                content.append({"type": "image", "image": img})
        else:
            content.insert(0, {"type": "image", "image": sample["image"]})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {"role": "user", "content": content},
    ]


def generate_text_from_sample(
    model,
    processor: Qwen2VLProcessor,
    messages,
    max_new_tokens=32,
    device="cuda",
    debug=False,
    **generation_kwargs
):
    # Preparation for inference
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
    )
    inputs = inputs.to(device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text


REDDIT_SYSTEM_MESSAGE = """You are an helpful assistant specialized in the Naruto universe, combining deep knowledge of the manga and anime with advanced visual analysis capabilities.
Your task is to analyze the provided image and respond to queries with concise answers"""


def get_visual_qa():
    ds = load_dataset("SteveTran/naruto-visual-qa", split="character")
    responses = ds["answer"]
    ds_qa = ds.map(
        lambda d: {
            "query": d["question"]
            + ". Return a name of character and explain though process",
        },
    )
    return [format_message(row, REDDIT_SYSTEM_MESSAGE) for row in ds_qa], responses


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, default="unsloth/Qwen2-VL-2B-Instruct")
    args = parser.parse_args()

    model_id = args.model_id
    print("Model ID: ", model_id)
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    if "unsloth" in model_id:
        from unsloth import FastVisionModel

        model, tokenizer = FastVisionModel.from_pretrained(model_id)
        model = FastVisionModel.for_inference(model)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Test quick evaluation")
    url = "https://static.wikia.nocookie.net/naruto/images/2/21/Profile_Jiraiya.PNG/revision/latest?cb=20160115173538"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    sample = {"query": "Who is in this picture?", "image": img}
    conv = format_message(sample, REDDIT_SYSTEM_MESSAGE)
    model_inputs = generate_text_from_sample(
        model=model,
        processor=processor,
        messages=conv,
        max_new_tokens=128,
        temperature=0.7,
        debug=True,
        do_sample=True,
        repetition_penalty=1.1,
    )
    print("Quick input: ", model_inputs)

    sample = {"query": "Who is famous in name Copy Ninja?"}
    conv = format_message(sample, REDDIT_SYSTEM_MESSAGE)
    model_inputs = generate_text_from_sample(
        model=model,
        processor=processor,
        messages=conv,
        max_new_tokens=128,
        repetition_penalty=1.1,
    )
    print("Quick input: ", model_inputs)

    ds_visual_qa, responses = get_visual_qa()
    for i in range(0, 20, 1):
        model_inputs = generate_text_from_sample(
            model=model,
            processor=processor,
            messages=ds_visual_qa[i],
            max_new_tokens=128,
            repetition_penalty=1.1,
            temperature=0.7,
            do_sample=True,
        )
        print("Prediction: {}, Actual: {}".format(model_inputs, responses[i]))
