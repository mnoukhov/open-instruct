#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --time=4:00:00
#SBATCH -p long

source mila.sh
exp_name="grpo_1.7b_base"

model_name_or_path="Qwen/Qwen3-1.7B-Base"
dataset_list="hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2"

uv run open_instruct/grpo_fast.py \
    --exp_name $exp_name \
    --output_dir $SCRATCH/open_instruct/results/ \
    --dataset_mixer_list $dataset_list \
    --max_token_length 256 \
    --max_prompt_token_length 256 \
    --response_length 1024 \
    --pack_length 4096 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout 16 \
    --num_samples_per_prompt_rollout 8 \
    --num_mini_batches 4 \
    --total_episodes 64000 \
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
    --beta 0.001 \
    --seed 3 \
    --save_freq 1000 \
    --vllm_gpu_memory_utilization 0.4 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --async_steps 0 \
    --vllm_sync_backend gloo \
    --fused_optimizer \
    --with_tracking $@  
