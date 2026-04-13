#!/usr/bin/env bash

# Baseline run for comparison with `my/train_focal.sh`.
# This keeps the same data/model/training setup, but disables focal aggregation.

set -x

LOCAL_CONFIG="${LOCAL_CONFIG:-my/train_local.sh}"
if [ -f "$LOCAL_CONFIG" ]; then
    # Local-only overrides live outside git. Keep this file untracked.
    # shellcheck disable=SC1090
    source "$LOCAL_CONFIG"
fi

export REWARD_SERVER_URL="http://10.244.163.154:48001/compute_reward_v2"

: "${MODEL_PATH:?Set MODEL_PATH in $LOCAL_CONFIG}"
: "${PROJECT_NAME:=frontend_focal}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in $LOCAL_CONFIG}"
: "${WANDB_MODE:=offline}"

EXPERIENT_NAME=baseline_3_$(basename "$MODEL_PATH")

python3 -m verl.trainer.main_ppo \
    trainer.rollout_data_dir=rollouts/$EXPERIENT_NAME \
    algorithm.use_focal=False \
    algorithm.focal.epsilon=0.05 \
    algorithm.focal.temperature=10.0 \
    algorithm.focal.gamma=3.0 \
    trainer.resume_mode="auto" \
    reward.custom_reward_function.path=my/focal_reward_client_v2.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    algorithm.adv_estimator=grpo \
    data.train_files=my/data/websight_train_2k_nothink.parquet \
    data.val_files=my/data/websight_val_2k_nothink.parquet \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.use_shm=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.total_epochs=3 \
    trainer.test_freq=5 \
    trainer.val_before_train=True
