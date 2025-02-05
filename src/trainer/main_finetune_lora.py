import logging
import json
import os
import sys
from typing import Any, Dict, List, NewType, Optional, Tuple

from dataclasses import dataclass, field
from datasets import load_dataset, concatenate_datasets

import torch
import transformers
from peft import peft_model, LoraConfig
from unsloth import PeftModel, FastLanguageModel
from accelerate import Accelerator
from transformers import (
    TrainingArguments,
    set_seed,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)
DEFAULT_PROMPT_FORMAT = """{} {} {}"""

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
        default=16,
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
    optim: Optional[str] = field(default="adamw_torch")
    train_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": ("The batch size for training.")},
    )
    epochs: Optional[int] = field(
        default=3, metadata={"help": ("The number of epochs to train for.")}
    )
    checkpoint_save_steps: Optional[int] = field(
        default=1000, metadata={"help": ("The number of steps to save the model.")}
    )

    logging_steps: Optional[int] = field(
        default=1000, metadata={"help": ("The number of steps to log the model.")}
    )

    weight_decay: Optional[float] = field(
        default=0.01, metadata={"help": ("The weight decay to use.")}
    )

    lr: Optional[float] = field(
        default=2e-5, metadata={"help": ("The learning rate to use.")}
    )

    output_data_dir: Optional[str] = field(
        default=os.environ.get("SM_OUTPUT_DATA_DIR"),
        metadata={"help": ("The output data directory.")},
    )

    model_dir: Optional[str] = field(
        default=os.environ.get("SM_MODEL_DIR"),
        metadata={"help": ("The model directory.")},
    )

    model_checkpoint_dir: Optional[str] = field(
        default="checkpoints",
        metadata={"help": ("The model checkpoint directory.")},
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
        default="eager", metadata={"help": ("The attention implementation.")}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": ("The deepspeed config file.")}
    )
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": ("The local rank.")}
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


def formatting_prompts_func(
    tokenizer: AutoTokenizer,
    example: Dict[str, List[str]],
    eos_token: str,
    max_length: int = 2048,
):
    texts = []
    for i in range(len(example["input"])):
        instruction = example["instruction"][i] if "instruction" in example else ""
        input = example["input"][i] if "input" in example else ""
        response = example["response"][i] if "response" in example else ""
        output = json.dumps(response) if isinstance(response, dict) else response
        text = DEFAULT_PROMPT_FORMAT.format(instruction, input, output) + eos_token

        texts.append(text)

    output = tokenizer(
        texts,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    return output


def main(seed: int = 3407):
    # Set seed for reproducibility
    set_seed(seed)

    ###############
    # Setup logging
    ###############
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # if torch.mps.is_available():
    #     device = "mps"

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model_args, data_args, training_args = parse_args()

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args.model_checkpoint_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name,
    )
    # tokenizer.pad_token = (
    #     tokenizer.unk_token
    # )  # use unk rather than eos token to prevent endless generation
    # tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if getattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        torch.bfloat16 if not torch.cuda.is_bf16_supported() else torch.float16
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Train the model
    # @title Show current memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    ds = load_dataset("SteveTran/naruto-instruction-prompts")["analyze"]
    dataset = (
        ds.map(
            lambda row: formatting_prompts_func(
                tokenizer=tokenizer,
                example=row,
                eos_token=EOS_TOKEN,
                max_length=training_args.max_seq_length,
            ),
            batched=True,
        )
        .shuffle(seed=seed)
        .remove_columns(["input", "response", "instruction"])
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name,
        max_seq_length=training_args.max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=model_args.load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    FastLanguageModel.for_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model: peft_model.PeftModelForQuestionAnswering = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=target_modules,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            per_device_train_batch_size=training_args.train_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_ratio=training_args.warmup_ratio,
            logging_dir=training_args.output_data_dir,
            num_train_epochs=training_args.epochs,
            learning_rate=training_args.lr,
            # 'fp16' is set to True if bfloat16 is not supported, which means the model will use 16-bit floating point precision for training if possible.
            logging_steps=training_args.checkpoint_save_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=seed,
            output_dir=training_args.model_checkpoint_dir,
            save_strategy="steps",
            save_steps=training_args.checkpoint_save_steps,
            restore_callback_states_from_checkpoint=True,
            deepspeed=training_args.deepspeed,
            # gradient_checkpointing=True,
        ),
    )
    trainer_stats = trainer.train(resume_from_checkpoint=last_checkpoint)

    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

    trainer.save_model(training_args.output_data_dir)  # Local saving
    tokenizer.save_pretrained(training_args.output_data_dir)

    inputs = tokenizer(
        ["Describe about this character Jiraiya"],
        return_tensors="pt",
    ).to(device)

    outputs = trainer.model.generate(
        **inputs, max_new_tokens=512, use_cache=True, temperature=0.7
    )
    print("Answer: ", tokenizer.batch_decode(outputs))


if __name__ == "__main__":
    main()
