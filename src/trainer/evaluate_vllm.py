import random
import torch
import jinja2
import evaluate
from tqdm import tqdm, trange
from datasets import load_dataset
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


acc_metric = evaluate.load("accuracy")
map_to_int = {chr(i): i - 65 for i in range(65, 91)}

prompt_template = jinja2.Template(
    """
Answer the following multiple choice question about Naruto manga and anime series. Return answer (single letter) and explaination if possible
{% for example_question in fewshot_questions %}{{example_question}}
{% endfor %}
Question: {{question}}
{% for choice in choices %}{{choice["choice"]}}) {{choice["answer"]}}
{% endfor %}
Answer:
"""
)
# prompt_template = jinja2.Template(
#     """
# ### Instruction
# Answer the following are multiple choice questions about Naruto manga. Return answer (single letter) and explaination if possible
# ### Input
# {% for example_question in fewshot_questions %}{{example_question}}
# {% endfor %}
# {{question}}
# {% for choice in choices %}{{choice["choice"]}}) {{choice["answer"]}}
# {% endfor %}
# ### Response
# """
# )
fewshot_questions = [
    """
Question: How many times did Naruto fail the graduation test from the Academy?
A) 1
B) 2
C) 4
D) None of these are correct
E) All of these are correct
Answer:D. Naruto failed his graduation exam 3 times. He failed the graduation exam three times, but the first two failures were because he used the Shadow Clone Jutsu to create multiple clones, which was against the rules of the exam. The third time he failed was due to his lack of skills.
""",
    """
Question: What is the name of Naruto’s son in the sequel series “Boruto: Naruto Next Generations”?
Choices:
A) Mitsuki
B) Konohamaru
C) Boruto Uzumaki
D) None of these are correct
Answer:C. Boruto Uzumaki is the son of Naruto Uzumaki and Hinata Hyuga. He is the main protagonist of the sequel series “Boruto: Naruto Next Generations.” Boruto is a member of Team Konohamaru, and he is a talented ninja who aspires to become a Hokage like his father.
""",
]


def answer_binary_question(
    model,
    question,
    tokenizer,
    prompt_template=prompt_template,
    fewshot_questions=fewshot_questions,
):
    prompt = prompt_template.render(
        question=question["question"],
        choices=question.get("choices", ""),
        fewshot_questions=random.choices(fewshot_questions, k=1),
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():  # Disable gradient calculations
        logits = model(**inputs).logits[0, -1].detach().cpu()
        probs = torch.softmax(logits, dim=-1)
        answer, prob = (
            tokenizer.decode(torch.argmax(logits, dim=-1), skip_special_tokens=True),
            probs.max().item(),
        )
        answer = answer.strip()

    del probs, logits, inputs  # Explicitly delete variables
    torch.cuda.empty_cache()  # Clear CUDA cache

    return answer, prob


def compute_score(model, tokenizer, qa_list, disable_tqdm=False):
    predictions_tuple = [
        answer_binary_question(model=model, tokenizer=tokenizer, question=qa_list[i])
        for i in trange(len(qa_list), disable=disable_tqdm)
    ]
    predictions = [map_to_int.get(answer, -1) for answer, _ in predictions_tuple]
    references = [map_to_int[qa["answer"]] for qa in qa_list]
    scores = [score for _, score in predictions_tuple]
    acc = acc_metric.compute(predictions=predictions, references=references)
    return predictions, acc, scores


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
        generate_text_from_sample(model, processor, sample, max_new_tokens=32)
        for sample in tqdm(examples)
    ]

    score = 0.0
    for i in range(len(ds_qa)):
        response_tokens = set(responses[i].split())
        answer_tokens = set(answers[i].split())
        score += len(response_tokens & answer_tokens) / len(answer_tokens)

    return score / len(ds_qa)
