from functools import partial
import torch
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastVisionModel
from accelerate import Accelerator
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

from qwen_vl_utils import process_vision_info

REDDIT_SYSTEM_MESSAGE = """You are an expert Vision Language Model specialized in the Naruto universe, combining deep knowledge of the manga and anime with advanced visual analysis capabilities"""
REDDIT_GUIDELINE = """
# Guideline
Identify the type of question being asked and then provide an appropriately structured response.
## General Guidelines:
1. First state the identified question type
2. Follow the corresponding response structure
3. Maintain focus on visual evidence
4. Use canonical Naruto terminology
5. Keep responses clear and well-organized

## Visual Analysis Requirements:
- Identify characters, jutsu, locations, and symbols
- Recognize art style (manga/anime/fan art)
- Detect text and integrate with visual context
- Parse action sequences and combat dynamics
- Interpret emotional expressions and character interactions
## Question Type Identification:
When receiving a query, first classify it into one of these categories:
### Discussion Questions
- Identified by: Open-ended prompts, requests for opinions, "what do you think about", "let's talk about"
- Response Structure:
  - Begin with a clear stance or observation
  - Support with visual evidence from the image
  - Connect to broader Naruto themes and storylines
  - Encourage further discussion with thoughtful insights

### Direct Questions
- Identified by: Who, what, when, where, why, how queries
- Response Structure:
  - Provide direct answer first
  - Support with specific visual evidence
  - Add brief relevant context if necessary
  - Use precise Naruto terminology

### VS Battle Questions
- Identified by: "Who would win", "vs", "stronger", "compare", "fight between"
- Response Structure:
  - State the combatants and battle context
  - Analyze visual evidence of abilities shown
  - Compare demonstrated powers and techniques
  - Provide reasoning for outcome prediction

### Theory Questions
- Identified by: "Could it be", "what if", "theory about", "possible explanation"
- Response Structure:
  - Acknowledge the theoretical premise
  - Analyze visual evidence supporting/opposing
  - Connect with established Naruto lore
  - Evaluate theory's plausibility

### Analysis Questions
- Identified by: "Explain", "analyze", "break down", "detail about"
- Response Structure:
  - State the focus of analysis
  - Break down visual elements systematically
  - Connect with Naruto mechanics/lore
  - Provide technical insights

### Image Captioning Requests
- Identified by: "Describe", "what's in this image", "caption this"
- Response Structure:
  - Begin with scene overview
  - Identify key characters/elements
  - Note significant actions/emotions
  - Include relevant Naruto-specific context
"""

DEFAULT_SYSTEM_PROMPT = "You are an helpful assistant, who are expert in Naruto manga"
DEFAULT_PROMPT_FORMAT = """{} {}"""
ALPACA_PROMPT_FORMAT = """
# Instruction
{}
# Input
{}
# Output
{}
"""


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

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {"role": "user", "content": content},
    ]
    if "response" in sample:
        conversation.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": sample["response"]}],
            }
        )

    return conversation


def get_text_dataset():
    # First, fine tuning text dataset for the model
    ds_sft_qa = load_dataset("SteveTran/naruto-instruction-prompts")
    ds_sft_qa_wiki = (
        ds_sft_qa["wiki"]
        .map(
            lambda row: {
                "query": DEFAULT_PROMPT_FORMAT.format(row["instruction"], row["input"]),
                "response": row["response"],
            },
            batched=False,
        )
        .shuffle()
    )
    ds_sft_qa_analyze = (
        ds_sft_qa["analyze"]
        .map(
            lambda row: {
                "query": DEFAULT_PROMPT_FORMAT.format(row["instruction"], row["input"]),
                "response": row["response"],
            },
            batched=False,
        )
        .shuffle()
    )

    ds_reddit = load_dataset("SteveTran/naruto-vision-prompts")
    ds_sft_reddit = ds_reddit["train"].filter(lambda d: d["image"] is None, num_proc=4)

    ds_sft_qa = (
        [format_message(row, DEFAULT_SYSTEM_PROMPT) for row in ds_sft_qa_wiki]
        + [format_message(row, DEFAULT_SYSTEM_PROMPT) for row in ds_sft_qa_analyze]
        + [format_message(row, REDDIT_SYSTEM_MESSAGE) for row in ds_sft_reddit]
    )

    return ds_sft_qa


def get_visual_dataset():
    ds_reddit = load_dataset("SteveTran/naruto-vision-prompts")
    ds_sft_reddit = (
        ds_reddit["train"]
        .filter(lambda d: d["image"] is not None, num_proc=4)
        .shuffle()
    )

    return [format_message(row, REDDIT_SYSTEM_MESSAGE) for row in ds_sft_reddit]


def generate_text_from_sample(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
    sample,
    max_new_tokens=32,
    device="cuda",
    **generation_kwargs,
):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample,
        tokenize=False,
        add_generation_prompt=True,  # Use the sample without the system message
    )

    # Process the visual input from the sample
    if "image" not in sample or sample["image"] is None:
        image_inputs = None
    else:
        image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )  # Move inputs to the specified device
    # return model_inputs

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


def evaluate_visual_metrics(model, processor):
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    ds = load_dataset("SteveTran/naruto-visual-qa", split="character")
    responses = ds["answer"]
    ds_qa = ds.map(
        lambda d: {
            "query": d["question"] + "\n Give an answer only, no furthur explanation",
        },
    )
    examples = [
        format_message(ds_qa[i], REDDIT_SYSTEM_MESSAGE) for i in range(len(ds_qa))
    ]
    answers = [
        generate_text_from_sample(model, processor, sample, max_new_tokens=512)
        for sample in tqdm(examples)
    ]

    bleu_score = bleu.compute(references=responses, predictions=answers)
    rough_score = rouge.compute(references=responses, predictions=answers)

    return bleu_score, rough_score