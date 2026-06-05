#!/usr/bin/env bash

set -x

LOCAL_CONFIG="${LOCAL_CONFIG:-my/train_local.sh}"
if [ -f "$LOCAL_CONFIG" ]; then
    # Local-only overrides live outside git. Keep this file untracked.
    # shellcheck disable=SC1090
    source "$LOCAL_CONFIG"
fi

export REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://10.244.143.149:48000/compute_reward_v6}"

: "${MODEL_PATH:?Set MODEL_PATH in $LOCAL_CONFIG}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in $LOCAL_CONFIG}"
: "${WANDB_MODE:?Set WANDB_MODE in $LOCAL_CONFIG}"
: "${TP_SIZE:?Set TP_SIZE in $LOCAL_CONFIG}"

: "${EXPERIMENT_PREFIX:?Set EXPERIMENT_PREFIX before running my/train_baseline.sh}"
: "${PROJECT_NAME:=frontend_focal}"
: "${VISUAL_JUDGE_MODE:=joint}"

export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_$(basename "$MODEL_PATH")"

python3 -m verl.trainer.main_ppo \
    trainer.rollout_data_dir=rollouts/$EXPERIMENT_NAME \
    algorithm.use_focal=False \
    algorithm.focal.epsilon=0.05 \
    algorithm.focal.temperature=5.0 \
    algorithm.focal.gamma=3.0 \
    algorithm.focal.weight_min=0.05 \
    algorithm.focal.weight_max=0.3 \
    trainer.resume_mode="auto" \
    reward.custom_reward_function.path=my/naive_single_reward_client_v6.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    +reward.custom_reward_function.reward_kwargs.visual_judge_mode=$VISUAL_JUDGE_MODE \
    algorithm.adv_estimator=grpo \
    data.train_files=my/data/websight_train_a11y_4k_nothink.parquet \
    data.val_files=my/data/websight_val_a11y_4k_nothink.parquet \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.n=16 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.use_shm=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-6 \
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
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$TP_SIZE \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.total_epochs=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=True
