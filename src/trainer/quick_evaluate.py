# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from PIL import Image
import requests
from io import BytesIO
from datasets import load_dataset

import torch
from unsloth import FastVisionModel
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLProcessor


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
    processor,
    sample,
    max_new_tokens=32,
    device="cuda",
    debug=False,
    **generation_kwargs
):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,  # Use the sample without the system message
    )
    if debug:
        print(text_input)

    # Process the visual input from the sample
    if "image" in sample and sample["image"] is not None:
        image_inputs, _ = process_vision_info(sample)
    else:
        image_inputs = None

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
        truncation=True,
        padding="longest",
    ).to(
        device
    )  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(
        **model_inputs, max_new_tokens=max_new_tokens, **generation_kwargs
    )

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]  # Return the first decoded output text


REDDIT_SYSTEM_MESSAGE = "You are an expert assistant specialized in the Naruto universe, combining deep knowledge of the manga and anime with advanced visual analysis capabilities"


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
    ArgumentParser.add_argument(
        "--model_id", type=str, default="unsloth/Qwen2-VL-2B-Instruct"
    )
    args = ArgumentParser().parse_args()

    model_id = args.model_id
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    model, tokenizer = FastVisionModel.from_pretrained(model_id)
    model = FastVisionModel.for_inference(model)

    url = "https://static.wikia.nocookie.net/naruto/images/2/21/Profile_Jiraiya.PNG/revision/latest?cb=20160115173538"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    sample = {"query": "Who is in this picture?", "image": img}
    conv = format_message(sample, REDDIT_SYSTEM_MESSAGE)
    model_inputs = generate_text_from_sample(
        model=model,
        processor=processor,
        sample=conv,
        max_new_tokens=128,
        temperature=0.7,
        debug=True,
        do_sample=True,
        repetition_penalty=1.1,
    )
    print("Quick input: ", model_inputs)

    ds_visual_qa, responses = get_visual_qa()
    for i in range(-10, -1, 1):
        model_inputs = generate_text_from_sample(
            model=model,
            processor=processor,
            sample=ds_visual_qa[i],
            max_new_tokens=128,
            repetition_penalty=1.1,
        )
        print("Prediction: {}, Actual: {}".format(model_inputs, responses[i]))
