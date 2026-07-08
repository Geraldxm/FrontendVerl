#!/usr/bin/env bash

# ============================================================
# WebBench RL 训练启动脚本
# 用法: EXPERIMENT_PREFIX=xxx bash my/train_webbench.sh
# ============================================================

# 加载本地配置文件（如 MODEL_PATH、WANDB_API_KEY 等），该文件不纳入版本控制

LOCAL_CONFIG="${LOCAL_CONFIG:-my/train_local.sh}"
if [ -f "$LOCAL_CONFIG" ]; then
    # Local-only overrides live outside git. Keep this file untracked.
    # shellcheck disable=SC1090
    source "$LOCAL_CONFIG"
fi

# WebBench 数据目录与 Reward Server 地址
WEBBENCH_ROOT="${WEBBENCH_ROOT:-/inspire/hdd/global_user/gexinmu-253108100065/Repos/web-bench}"
WEBBENCH_DATA_DIR="${WEBBENCH_DATA_DIR:-$WEBBENCH_ROOT/reward_server/artifacts/rl_model_data_selection/task1-1-v2}"

export REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://127.0.0.1:48006/compute_webbench_reward}"

# 检查本地配置中的必要环境变量是否已设置
: "${MODEL_PATH:?Set MODEL_PATH in $LOCAL_CONFIG}"
: "${WANDB_API_KEY:?Set WANDB_API_KEY in $LOCAL_CONFIG}"
: "${WANDB_MODE:?Set WANDB_MODE in $LOCAL_CONFIG}"
: "${TP_SIZE:?Set TP_SIZE in $LOCAL_CONFIG}"

# 覆盖变量
WANDB_MODE="online"
MODEL_PATH="/inspire/hdd/global_user/gexinmu-253108100065/Resources/models/LLMs/Qwen3-14B"
TP_SIZE=4


# 实验标识与训练超参数默认值（可通过环境变量覆盖）
: "${EXPERIMENT_PREFIX:?Set EXPERIMENT_PREFIX before running my/train_webbench.sh}"
: "${PROJECT_NAME:=webbench_rl}"
: "${TRAIN_FILE:=$WEBBENCH_DATA_DIR/train.parquet}"
: "${VAL_FILE:=$WEBBENCH_DATA_DIR/val.parquet}"
: "${MAX_PROMPT_LENGTH:=16384}"
: "${MAX_RESPONSE_LENGTH:=16384}"
: "${ROLLOUT_N:=32}"
: "${TRAIN_BATCH_SIZE:=40}"
: "${SAVE_FREQ:=10}"
: "${TEST_FREQ:=2}"
: "${TOTAL_EPOCHS:=20}"
: "${TOTAL_TRAINING_STEPS:=null}"
: "${VAL_BEFORE_TRAIN:=True}"
: "${ROLLOUT_MAX_MODEL_LEN:=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# 拼接实验名称：前缀_模型名
export EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_$(basename "$MODEL_PATH")"

# 调试模式开关：DEBUG_SHELL=1 时打印每条执行的命令
if [ "${DEBUG_SHELL:-0}" = "1" ]; then
    set -x
fi

# 启动 PPO 训练主流程
python3 -m verl.trainer.main_ppo \
    +data.apply_chat_template_kwargs.enable_thinking=true \
    trainer.rollout_data_dir=rollouts/$EXPERIMENT_NAME \
    algorithm.use_focal=False \
    algorithm.focal.base_weights='[1.0,0.0]' \
    trainer.resume_mode="auto" \
    reward.custom_reward_function.path=$WEBBENCH_ROOT/reward_server/client.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=False \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_MODEL_LEN \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
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
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$TP_SIZE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.test_freq=$TEST_FREQ \
    trainer.val_before_train=$VAL_BEFORE_TRAIN
