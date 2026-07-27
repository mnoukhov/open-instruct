for i in 1 2 3; do 
    export NAME="adaseq"

    WORKSPACE=ai2/oe-adapt-code \
    PRIORITY=high \
    BEAKER_IMAGE=michaeln/open-instruct-integration-test-michaeln-merge \
    EXP="${NAME}_seed${i}" \
    bash scripts/train/qwen/qwen2.5_0.5b_gsm8k_buckets.sh \
    --wandb_group_name $NAME \
    --vllm_num_engines 3 \
    --num_learners_per_node 1 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 64 \
    --max_grad_norm 25 \
    --active_sampling \
    --sync_sampling True \
    --async_steps 1 \
    --never_give_up 0.75 \
    --maintain_ngu_completions_downsample True \
    --maintain_pending_ngu_completions True \
    --maintain_pending_ngu_counts False \
    --maintain_pending_ngu_age 100

done

# --max_samples_multiplier -1 \
