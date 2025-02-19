`bash
uv pip install deepspeed==0.15.4 "numpy<2" torchvision
DISABLE_MLFLOW_INTEGRATION=TRUE deepspeed --num_gpus=4 entrypoint.py --model_name Qwen/Qwen2.5-7B-Instruct --deepspeed deepspeed_config/config_fp16_stage_3_offload.json --train_batch_size 4 --gradient_accumulation_steps 4 --model_checkpoint_dir /Volumes/staging_boost_lakehouse/silver/model/pretrained_llm/deepspeed/qwen25-7B
python3 src/trainer/main_finetune_multistage_vl.py --model_name unsloth/Qwen2-VL-2B-Instruct --train_batch_size 16 --text_lr 2e-4 --visual_lr 1e-4 --output_data_dir qwen25_vl_3b_lora_data --model_dir modelling_qwen2_vl_3b_lora --model_checkpoint_dir checkpoint_qwen2_vl_3b_lora --gradient_accumulation_steps 8 --lora_r 16 --lora_alpha 32 --optim adamw_torch_fused --evaluate_before_training True --lora_target_modules q_proj,v_proj --attn_implementation flash_attention_2
`
