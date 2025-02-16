from functools import partial
import logging
import os
from typing import Any, Dict, List, NewType, Optional, Tuple
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
from datasets import load_dataset
from unsloth import FastVisionModel
from accelerate import Accelerator
from transformers import (
    set_seed,
    BitsAndBytesConfig,
    HfArgumentParser,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    AutoTokenizer,
)

from trl import SFTTrainer, SFTConfig as AllSFTConfig
from qwen_vl_utils import process_vision_info
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Or completely disable
torch._dynamo.config.disable = True
DataClassType = NewType("DataClassType", Any)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    base_model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization with PEFT adapters."
            )
        },
    )
    model_name: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    model_code_revision: str = field(
        default=None, metadata={"help": "The branch of the IFT model"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code when loading a model."}
    )
    use_flash_attention_2: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`"
            )
        },
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": ("Whether to use PEFT or not for training.")},
    )
    lora_r: Optional[int] = field(
        default=0,
        metadata={"help": ("LoRA R value.")},
    )
    lora_alpha: Optional[int] = field(
        default=32,
        metadata={"help": ("LoRA alpha.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": ("LoRA dropout.")},
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    lora_modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("Model layers to unfreeze & train")},
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})

    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: bool = field(
        default=False, metadata={"help": "use nested quantization"}
    )
    bnb_4bit_quant_storage: Optional[str] = field(
        default="uint8",
        metadata={"help": "storage type to pack the quanitzed 4-bit prarams."},
    )

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    chat_template: Optional[str] = field(
        default=None, metadata={"help": "The chat template to use."}
    )
    dataset_mixer: Optional[Dict[str, float]] = field(
        default=None,
        metadata={
            "help": ("Datasets and their proportions to be used for training ift/rl.")
        },
    )
    text_column: Optional[str] = field(
        default="text",
        metadata={
            "help": "The column name to use for the text in the dataset (only used for continued pretraining)."
        },
    )
    dataset_splits: Optional[List[str]] = field(
        default_factory=lambda: ["train", "test"],
        metadata={"help": ("List of train test splits to use in the dataset")},
    )
    dataset_configs: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    truncation_side: Optional[str] = field(
        default=None, metadata={"help": "Truncation side to use for the tokenizer."}
    )
    auto_insert_empty_system_msg: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template."
            )
        },
    )

    train_dataset_path: Optional[str] = field(
        default=os.environ.get("SM_CHANNEL_TRAIN", ""),
        metadata={"help": ("The path to the training dataset.")},
    )


@dataclass
class SFTConfig:
    """
    Arguments related to the training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    Also used for the continued pretraining task.
    """

    output_data_dir: str = field(
        metadata={"help": ("The output data directory.")},
    )

    model_dir: str = field(
        metadata={"help": ("The model directory.")},
    )

    model_checkpoint_dir: str = field(
        metadata={"help": ("The model checkpoint directory.")},
    )
    dataset_kwargs: Optional[Dict[str, Any]] = field(
        default=None, metadata={"help": "Dataset kwargs for the SFTTrainer"}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "Used by TRL for reward model training, which tries to read this parameter in init."
            )
        },
    )
    logging_first_step: bool = field(
        default=True,
        metadata={
            "help": ("Whether to log and evaluate the first global_step or not.")
        },
    )
    optim: Optional[str] = field(default="adamw_torch_fused")
    train_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": ("The batch size for training.")},
    )
    epochs: Optional[int] = field(
        default=3, metadata={"help": ("The number of epochs to train for.")}
    )
    checkpoint_save_steps: Optional[int] = field(
        default=100, metadata={"help": ("The number of steps to save the model.")}
    )

    logging_steps: Optional[int] = field(
        default=100, metadata={"help": ("The number of steps to log the model.")}
    )

    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": ("The weight decay to use.")}
    )

    text_lr: Optional[float] = field(
        default=2e-5, metadata={"help": ("The learning rate to use.")}
    )

    visual_lr: Optional[float] = field(
        default=2e-5, metadata={"help": ("The learning rate to use.")}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": ("The number of gradient accumulation steps.")}
    )

    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": ("Whether to resume from a checkpoint.")}
    )

    warmup_ratio: Optional[float] = field(
        default=0.1, metadata={"help": ("The warmup ratio.")}
    )
    warmup_steps: Optional[int] = field(
        default=None, metadata={"help": ("The warmup steps.")}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": ("The learning rate scheduler type.")}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": ("The attention implementation.")},
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": ("The deepspeed config file.")}
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": ("The local rank.")}
    )

    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": ("The gradient checkpointing.")}
    )
    fsdp: Optional[str] = field(default="", metadata={"help": ("The fsdp.")})
    fsdp_config: Optional[str] = field(
        default=None, metadata={"help": ("The fsdp config.")}
    )

    evaluate_before_training: Optional[bool] = field(
        default=False, metadata={"help": ("Evaluate before training.")}
    )


def get_checkpoint(output_dir: str):
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
    return last_checkpoint


def get_quantization_config(model_args: ModelArguments) -> BitsAndBytesConfig | None:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.bnb_4bit_quant_storage,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


def parse_args() -> Tuple[ModelArguments, DataArguments, SFTConfig]:
    parser = HfArgumentParser((ModelArguments, DataArguments, SFTConfig))
    return parser.parse_args_into_dataclasses()


REDDIT_SYSTEM_MESSAGE = """You are an helpful assistant specialized in the Naruto universe, combining deep knowledge of the manga and anime with advanced visual analysis capabilities.
Your task is to analyze the provided image and respond to queries with concise answers"""

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
                "role": "assistant",
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


def main(seed: int = 3407):
    # Set seed for reproducibility
    set_seed(seed)
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_args, data_args, training_args = parse_args()

    # EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    logger.info("*** Load pretrained model ***")

    train_text_dataset = get_text_dataset()
    train_image_dataset = get_visual_dataset()

    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name)

    if model_args.lora_r > 0:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        print("Load unsloth model")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=model_args.model_name,
            max_seq_length=training_args.max_seq_length,
            load_in_4bit=model_args.load_in_4bit,
            dtype=torch_dtype,
        )
        if training_args.evaluate_before_training:
            print("Evaluate before training")
            print(evaluate_visual_metrics(model, processor))
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=True,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers
            r=model_args.lora_r,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=model_args.lora_alpha,  # Recommended alpha == r at least
            lora_dropout=model_args.lora_dropout,  # Supports any, but = 0 is optimized
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )
        model.print_trainable_parameters()
        FastVisionModel.for_training(model)
    else:
        from liger_kernel.transformers import monkey_patch

        quantization_config = get_quantization_config(model_args)
        model_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map=(
                get_kbit_device_map() if quantization_config is not None else "auto"
            ),
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        monkey_patch.apply_liger_kernel_to_qwen2_vl()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name,
            use_fast=True,
            trust_remote_code=True,
        )
        if training_args.evaluate_before_training:
            print("Evaluate before training")
            print(evaluate_visual_metrics(model, processor))

    # Create a data collator to encode text and image pairs
    def collate_fn(examples, contain_image=True):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False, add_generation_prompt=False)
            for example in examples
        ]  # Prepare texts for processing

        if contain_image:
            image_inputs = [
                process_vision_info(example)[0] for example in examples
            ]  # Process the images to extract inputs
        else:
            image_inputs = None

        # Tokenize the texts and process the images
        batch = processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
            max_length=training_args.max_seq_length,
            truncation=True,
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = (
            -100
        )  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(
            processor, Qwen2VLProcessor
        ):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [
                151652,
                151653,
                151655,
            ]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch

    collate_text_fn = partial(collate_fn, contain_image=False)
    collate_image_fn = partial(collate_fn, contain_image=True)

    # Stage 1 - Text training
    # Evaluate before training

    resume_from_checkpoint = get_checkpoint(
        os.path.join(training_args.model_checkpoint_dir, "text")
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_text_dataset,
        data_collator=collate_text_fn,
        max_seq_length=training_args.max_seq_length,
        args=AllSFTConfig(
            per_device_train_batch_size=training_args.train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_ratio=training_args.warmup_ratio,
            logging_dir=os.path.join(training_args.output_data_dir, "text"),
            num_train_epochs=training_args.epochs,
            learning_rate=training_args.text_lr,
            logging_steps=training_args.checkpoint_save_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=seed,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            output_dir=os.path.join(training_args.model_checkpoint_dir, "text"),
            save_strategy="steps",
            save_steps=training_args.checkpoint_save_steps,
            restore_callback_states_from_checkpoint=True,
            deepspeed=training_args.deepspeed,
            gradient_checkpointing=training_args.gradient_checkpointing,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
            fsdp=training_args.fsdp,
            fsdp_config=training_args.fsdp_config,
            report_to="tensorboard",
        ),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    resume_from_checkpoint = get_checkpoint(
        os.path.join(training_args.model_checkpoint_dir, "image")
    )
    # Stage 2 - Visual training
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_image_dataset,
        packing=False,
        data_collator=collate_image_fn,
        max_seq_length=training_args.max_seq_length,
        args=AllSFTConfig(
            per_device_train_batch_size=training_args.train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_ratio=training_args.warmup_ratio,
            logging_dir=os.path.join(training_args.output_data_dir, "image"),
            num_train_epochs=training_args.epochs,
            learning_rate=training_args.visual_lr,
            logging_steps=training_args.checkpoint_save_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=seed,
            output_dir=os.path.join(training_args.model_checkpoint_dir, "image"),
            save_strategy="steps",
            save_steps=training_args.checkpoint_save_steps,
            restore_callback_states_from_checkpoint=True,
            deepspeed=training_args.deepspeed,
            gradient_checkpointing=training_args.gradient_checkpointing,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        ),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(training_args.output_data_dir)  # Local saving
    tokenizer.save_pretrained(training_args.output_data_dir)

    try:
        FastVisionModel.for_inference(model)
    except Exception:
        pass

    print(evaluate_visual_metrics(model, processor))

    example = {
        "query": "Describe about this character Jiraiya",
    }
    answer = generate_text_from_sample(model, processor, example, max_new_tokens=512)
    print(answer)


if __name__ == "__main__":
    main()
