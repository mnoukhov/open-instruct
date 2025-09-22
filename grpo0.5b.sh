#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --time=1:00:00
#SBATCH -p main

source mila.sh

WANDB_ENTITY=mnoukhov
WANDB_PROJECT=open-instruct

model_name_or_path="Qwen/Qwen2.5-0.5B"
dataset_list="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"                                                                                                                                                  
LOCAL_EVALS="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"                                                                                                                                               
LOCAL_EVAL_SPLITS="test"
EXP_NAME="grpo_0.5b"


uv run --active open_instruct/grpo_fast.py \
    --exp_name $EXP_NAME \
    --output_dir $SCRATCH/open_instruct/results/ \
    --dataset_mixer_list $dataset_list \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_token_length 512 \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --num_mini_batches 1 \
    --total_episodes 32000 \
    --stop_strings "<|endoftext|>" "</answer>" \
    --model_name_or_path $model_name_or_path \
    --chat_template_name r1_simple_chat_postpend_think \
    --apply_verifiable_reward \
    --non_stop_penalty False \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching \
    --beta 0.0 \
    --seed 42 \
    --save_freq 1000 \
    --local_eval_every 50 \
    --vllm_gpu_memory_utilization 0.5 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --async_steps 0 \
    --vllm_sync_backend gloo \
    --fused_optimizer \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --with_tracking $@ 
