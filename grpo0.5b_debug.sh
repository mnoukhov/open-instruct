#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=24G
#SBATCH -c 4
#SBATCH --time=2:00:00
#SBATCH -p main

source mila.sh

WANDB_ENTITY=mnoukhov
WANDB_PROJECT=open-instruct
VLLM_ALLOW_INSECURE_SERIALIZATION=1

model_name_or_path="Qwen/Qwen2.5-0.5B"
dataset_list="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"
LOCAL_EVALS="ai2-adapt-dev/rlvr_gsm8k_zs 1.0"
LOCAL_EVAL_SPLITS="test"
EXP_NAME="grpo_0.5b"

num_mini_batches=$1
tv_cliprange=$2
async_steps=1
seed=42
# for seed in {0..2}; do
uv run --active open_instruct/grpo_fast.py \
    --exp_name $EXP_NAME \
    --tv_cliprange $tv_cliprange \
    --output_dir $SCRATCH/open_instruct/results/ \
    --dataset_mixer_list $dataset_list \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_prompt_token_length 512 \
    --response_length 512 \
    --pack_length 8192 \
    --per_device_train_batch_size 1 \
    --num_unique_prompts_rollout $(( 32 * $num_mini_batches )) \
    --num_samples_per_prompt_rollout 8 \
    --num_mini_batches $num_mini_batches \
    --total_episodes 2560 \
    --stop_strings "<|endoftext|>" \
    --model_name_or_path $model_name_or_path \
    --chat_template_name simple_think \
    --apply_verifiable_reward \
    --non_stop_penalty False \
    --temperature 1.0 \
    --learning_rate 1e-6 \
    --num_epochs 1 \
    --num_learners_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching \
    --beta 0.0 \
    --seed $seed \
    --save_freq 1000 \
    --local_eval_every $(( 10 / $num_mini_batches )) \
    --vllm_gpu_memory_utilization 0.4 \
    --single_gpu_mode \
    --deepspeed_stage 2 \
    --eval_temperature 0. \
    --eval_top_p 0.95 \
    --vllm_sync_backend gloo \
    --fused_optimizer \
    --wandb_entity $WANDB_ENTITY \
    --wandb_project $WANDB_PROJECT \
    --async_steps $async_steps \
    --with_tracking
# done
