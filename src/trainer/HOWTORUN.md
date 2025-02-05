`bash
!DISABLE_MLFLOW_INTEGRATION=TRUE deepspeed --num_gpus=4 entrypoint.py --model_name Qwen/Qwen2.5-14B-Instruct --deepspeed deepspeed_config/config_fp16_stage_3_offload.json --train_batch_size 4 --gradient_accumulation_steps 4 --model_checkpoint_dir /Volumes/staging_boost_lakehouse/silver/model/pretrained_llm/deepspeed/qwen25-14B
`
