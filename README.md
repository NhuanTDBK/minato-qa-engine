# Introduction

This repository contains the code for fine-tuning LLM on Naruto manga domain. The code is based on the `transformers` library by Hugging Face.

We introduce multimodal data to the model by concatenating the text with the image features extracted from the image using a pre-trained model.

# Approaches

## Text Data
Sources:
- Fandom
- Sportseekda
- Reddit

## Image Data
Sources:
- Fandom
- Reddit

## Video data
Sources:
- Youtube

## Model


# Evaluation

- binary_choices_v2.json

| Model                                     | Accuracy           |
| ----------------------------------------- | ------------------ |
| unsloth/Qwen2-1.5B-Instruct-bnb-4bit      | 0.5925925925925926 |
| unsloth/Phi-3-medium-4k-instruct-bnb-4bit | 0.7469135802469136 |
| unsloth/Qwen2-7B-Instruct-bnb-4bit        | 0.691358024691358  |
| unsloth/Phi-3-mini-4k-instruct-bnb-4bit   | 0.5864197530864198 |
| unsloth/gemma-2-9b-it-bnb-4bit            | 0.7530864197530864 |
| unsloth/llama-3-8b-Instruct-bnb-4bit      | 0.7037037037037037 |
| unsloth/Yi-1.5-6B-bnb-4bit                | 0.6666666666666666 |
| unsloth/gemma-1.1-7b-it-bnb-4bit          | 0.691358024691358  |
| unsloth/Qwen2.5-14B-bnb-4bit              | 0.7283950617283951 |

- binary_choices_synthetic.json

| Model                                     | Accuracy            |
| ----------------------------------------- | ------------------- |
| unsloth/Qwen2-1.5B-Instruct-bnb-4bit      | 0.3767931144405483  |
| unsloth/Phi-3-medium-4k-instruct-bnb-4bit | 0.45999362448198916 |
| unsloth/Qwen2-7B-Instruct-bnb-4bit        | 0.4874083519285942  |
| unsloth/Phi-3-mini-4k-instruct-bnb-4bit   | 0.4284348103283392  |
| unsloth/gemma-2-9b-it-bnb-4bit            | 0.36914249282754225 |
| unsloth/llama-3-8b-Instruct-bnb-4bit      | 0.44819891616193813 |
| unsloth/Yi-1.5-6B-bnb-4bit                | 0.4185527574115397  |
| unsloth/gemma-1.1-7b-it-bnb-4bit          | 0.4405482945489321  |

## Qwen 7b 4bit lr 5e-5, 3 epochs
synthetic.json
| Checkpoint                           | Accuracy          |
|--------------------------------------|-------------------|
| /opt/ml/input/data/model_name/checkpoint-50   | 0.4896397832323876  |
| /opt/ml/input/data/model_name/checkpoint-100  | 0.5183296142811603  |
| /opt/ml/input/data/model_name/checkpoint-150  | 0.5361810647115078  |
| /opt/ml/input/data/model_name/checkpoint-200  | 0.559132929550526   |
| /opt/ml/input/data/model_name/checkpoint-250  | 0.5635957921581128  |
| /opt/ml/input/data/model_name/checkpoint-300  | 0.5811284666879184  |
| /opt/ml/input/data/model_name/checkpoint-350  | 0.5875039846987568  |
| /opt/ml/input/data/model_name/checkpoint-400  | 0.5964297099139305  |
| /opt/ml/input/data/model_name/checkpoint-450  | 0.5948358304112209  |
| /opt/ml/input/data/model_name/checkpoint-500  | 0.598342365317182   |
| /opt/ml/input/data/model_name/checkpoint-513  | 0.6037615556263947  |

binary_choice_v2.json
| Checkpoint                           | Accuracy          |
|--------------------------------------|-------------------|
| /opt/ml/input/data/model_name/checkpoint-50   | 0.6851851851851852 |
| /opt/ml/input/data/model_name/checkpoint-100  | 0.7160493827160493 |
| /opt/ml/input/data/model_name/checkpoint-150  | 0.6975308641975309 |
| /opt/ml/input/data/model_name/checkpoint-200  | 0.7283950617283951 |
| /opt/ml/input/data/model_name/checkpoint-250  | 0.7160493827160493 |
| /opt/ml/input/data/model_name/checkpoint-300  | 0.7407407407407407 |
| /opt/ml/input/data/model_name/checkpoint-350  | 0.7469135802469136 |
| /opt/ml/input/data/model_name/checkpoint-400  | 0.7407407407407407 |
| /opt/ml/input/data/model_name/checkpoint-450  | 0.7283950617283951 |
| /opt/ml/input/data/model_name/checkpoint-500  | 0.7345679012345679 |
| /opt/ml/input/data/model_name/checkpoint-513  | 0.7098765432098766 |

## Gemma2 9B 4bit lr 5e-5, 3 epochs
| Checkpoint                           | Accuracy          |
|--------------------------------------|-------------------|
| /opt/ml/input/data/model_name/checkpoint-50   | 0.8209876543209876 |
| /opt/ml/input/data/model_name/checkpoint-100  | 0.7901234567901234 |
| /opt/ml/input/data/model_name/checkpoint-150  | 0.7839506172839507 |
| /opt/ml/input/data/model_name/checkpoint-200  | 0.7839506172839507 |
| /opt/ml/input/data/model_name/checkpoint-250  | 0.7962962962962963 |
| /opt/ml/input/data/model_name/checkpoint-300  | 0.7839506172839507 |
| /opt/ml/input/data/model_name/checkpoint-350  | 0.7901234567901234 |
| /opt/ml/input/data/model_name/checkpoint-400  | 0.7777777777777778 |
| /opt/ml/input/data/model_name/checkpoint-450  | 0.7901234567901234 |
| /opt/ml/input/data/model_name/checkpoint-500  | 0.7716049382716049 |
| /opt/ml/input/data/model_name/checkpoint-550  | 0.7654320987654321 |
| /opt/ml/input/data/model_name/checkpoint-600  | 0.7777777777777778 |
| /opt/ml/input/data/model_name/checkpoint-650  | 0.7777777777777778 |
| /opt/ml/input/data/model_name/checkpoint-700  | 0.7839506172839507 |
| /opt/ml/input/data/model_name/checkpoint-750  | 0.7592592592592593 |
| /opt/ml/input/data/model_name/checkpoint-800  | 0.7654320987654321 |
| /opt/ml/input/data/model_name/checkpoint-850  | 0.7716049382716049 |
| /opt/ml/input/data/model_name/checkpoint-900  | 0.7654320987654321 |
| /opt/ml/input/data/model_name/checkpoint-950  | 0.7777777777777778 |
| /opt/ml/input/data/model_name/checkpoint-1000 | 0.7469135802469136 |
| /opt/ml/input/data/model_name/checkpoint-1029 | 0.7654320987654321 |