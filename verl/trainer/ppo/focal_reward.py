# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward postprocessing utilities for focal/baseline aggregation in PPO.

This module sits between reward extraction and advantage computation.
It takes the raw reward client outputs already attached to a `DataProto`,
classifies valid vs invalid reward samples, aggregates multi-signal rewards
per rollout group, and returns:

1. the final token-level reward tensor used by PPO;
2. scalar extra fields safe to store in `non_tensor_batch` and validation metrics;
3. raw nested fields reserved for rollout/validation dumps;
4. compact monitoring metrics grouped by signal/status modules for cleaner dashboards.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from verl import DataProto

# success 表示走完了整个 judger 流程; llm_generation_error 表示 LLM 生成错误: 这两种方式都是有效的样本
# 而其他情况则是 reward server 出故障, 如 render/judge/parse 失败
VALID_FOCAL_OVERALL_STATUSES = frozenset({"success", "llm_generation_error"})
RAW_REWARD_DEBUG_KEYS = ("overall_status", "error_message", "reward_signals", "render_info", "judge_info")


@dataclass
class RewardPostprocessResult:
    """Container for postprocessed reward outputs consumed by trainer and validation."""

    reward_tensor: torch.Tensor
    derived_extra_info: dict[str, np.ndarray]
    validation_extra_info: dict[str, list[Any]]
    dump_extra_info: dict[str, list[Any]]
    metrics: dict[str, float]


def slugify_reward_name(name: str) -> str:
    """
    输入: 原始 reward/signal/status 名称字符串。
    输出: 适合用作 metric key / batch key 的稳定 slug。
    意图: 统一不同来源字段的命名，避免空格和特殊字符污染 key。
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return slug or "unknown"


def make_json_serializable(obj: Any) -> Any:
    """
    输入: 可能包含 numpy 标量、数组或嵌套 dict/list 的对象。
    输出: 可直接写入 JSONL dump 的基础 Python 类型。
    意图: 保留原始 reward 调试信息，同时避免 dump 阶段因类型不兼容失败。
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {str(key): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(value) for value in obj]
    return obj


def _to_list(values: Any, *, key: str, expected_length: int) -> list[Any]:
    """
    输入: 任意序列型 reward extra info、字段名、期望 batch 长度。
    输出: 与 batch 对齐的 Python list。
    意图: 在进入 focal 聚合前统一校验长度和容器类型，避免后续索引错位。
    """
    if isinstance(values, np.ndarray):
        normalized = values.tolist()
    elif isinstance(values, list):
        normalized = values
    elif isinstance(values, tuple):
        normalized = list(values)
    else:
        raise ValueError(f"Reward extra info '{key}' must be a sequence, got {type(values).__name__}")

    if len(normalized) != expected_length:
        raise ValueError(
            f"Reward extra info '{key}' length mismatch: expected {expected_length}, got {len(normalized)}"
        )
    return normalized


def _get_focal_param(config: Any, name: str, default: Any) -> Any:
    """
    输入: dict / dataclass 风格配置对象、字段名、默认值。
    输出: 对应 focal 参数值。
    意图: 同时兼容 OmegaConf、dict 和普通对象访问方式。
    """
    if hasattr(config, name):
        return getattr(config, name)
    if hasattr(config, "get"):
        return config.get(name, default)
    return default


def _build_base_weights(signal_keys: tuple[str, ...], base_weights: Any) -> np.ndarray:
    """
    输入: 信号键顺序与用户配置的基础权重。
    输出: 与信号顺序对齐、且和为 1 的 numpy 权重向量。
    意图:
    1. 空配置时退化为平均权重；
    2. 显式配置时强校验长度和合法性，避免静默错配。
    """
    if base_weights is None:
        base_weights = []
    if hasattr(base_weights, "tolist"):
        base_weights = base_weights.tolist()
    base_weights = list(base_weights)

    if not base_weights:
        return np.full(len(signal_keys), 1.0 / len(signal_keys), dtype=np.float64)

    if len(base_weights) != len(signal_keys):
        raise ValueError(
            "algorithm.focal.base_weights length mismatch: "
            f"expected {len(signal_keys)} for {signal_keys}, got {len(base_weights)}"
        )

    weights = np.asarray(base_weights, dtype=np.float64)
    if np.any(weights < 0):
        raise ValueError("algorithm.focal.base_weights must be non-negative")

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("algorithm.focal.base_weights must sum to a positive value")
    return weights / weight_sum


def _infer_signal_keys(reward_signal_items: list[Any]) -> tuple[str, ...]:
    """
    输入: batch 内每个样本的 reward_signals 对象列表。
    输出: 稳定的信号键顺序。
    意图: 将 reward client 返回的 dict 信号矩阵化，后续 direct/focal 都依赖该顺序。
    """
    for reward_signal in reward_signal_items:
        if isinstance(reward_signal, dict):
            signal_keys = tuple(reward_signal.keys())
            break
    else:
        raise ValueError("reward_signals must contain at least one dict item")

    if not signal_keys:
        raise ValueError("reward_signals dict is empty")

    for reward_signal in reward_signal_items:
        if not isinstance(reward_signal, dict):
            raise ValueError("Each reward_signals item must be a dict")
        if tuple(reward_signal.keys()) != signal_keys:
            raise ValueError("reward_signals keys must be stable and identical across the batch")
    return signal_keys


def _build_signal_matrix(reward_signal_items: list[dict[str, Any]], signal_keys: tuple[str, ...]) -> np.ndarray:
    """
    输入: reward_signals 列表和信号键顺序。
    输出: shape = [batch_size, num_signals] 的浮点矩阵。
    意图: 把字典形式的多维 reward 变成可直接做 baseline/focal 聚合的数值矩阵。
    """
    signal_matrix = np.zeros((len(reward_signal_items), len(signal_keys)), dtype=np.float64)
    for row_idx, reward_signal in enumerate(reward_signal_items):
        for col_idx, signal_key in enumerate(signal_keys):
            signal_matrix[row_idx, col_idx] = float(reward_signal.get(signal_key, 0.0))
    return signal_matrix


def _compute_response_mask(batch: DataProto) -> torch.Tensor:
    """
    输入: 训练 batch。
    输出: 仅覆盖 response token 的 mask。
    意图: 当上游尚未写入 `response_mask` 时，给 reward tensor 回填提供兜底。
    """
    responses = batch.batch["responses"]
    response_length = responses.size(1)
    attention_mask = batch.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _build_reward_tensor(batch: DataProto, raw_reward_tensor: torch.Tensor, final_scores: np.ndarray) -> torch.Tensor:
    """
    输入: 原始 reward tensor、batch、每条样本的最终标量 reward。
    输出: 只在 response 最后一个有效 token 位置写入最终 reward 的新 tensor。
    意图: 与 verl 现有 sequence-level reward 表达保持一致，避免改动 PPO 下游接口。
    """
    response_mask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() else _compute_response_mask(batch)
    reward_tensor = torch.zeros_like(raw_reward_tensor, dtype=torch.float32)
    response_end_positions = response_mask.sum(dim=-1).to(torch.long) - 1
    batch_indices = torch.arange(reward_tensor.size(0), device=reward_tensor.device)
    reward_tensor[batch_indices, response_end_positions] = torch.as_tensor(
        final_scores,
        device=reward_tensor.device,
        dtype=torch.float32,
    )
    return reward_tensor


def _distribution_metrics(prefix: str, values: np.ndarray) -> dict[str, float]:
    """
    输入: metric 前缀和一维分数数组。
    输出: mean/var/min/max 四个摘要指标。
    意图:
    1. 统一 direct/focal/final/raw 与 reward_signals 的日志形式。
    2. 在可视化面板中同时观察均值与方差，排查奖励分布抖动。
    """
    return {
        f"{prefix}/mean": float(np.mean(values)),
        f"{prefix}/var": float(np.var(values)),
        f"{prefix}/min": float(np.min(values)),
        f"{prefix}/max": float(np.max(values)),
    }


def _compute_group_focal_scores(
    *,
    signal_matrix: np.ndarray,
    base_weights: np.ndarray,
    temperature: float,
    gamma: float,
    epsilon: float,
    max_signal_score: float = 10.0,
) -> dict[str, np.ndarray]:
    """
    输入: 单个 rollout group 的有效信号矩阵与 focal 超参数。
    输出: 该组的 direct 分数、focal 分数和归一化 focal 权重。
    意图:
    1. direct 分数用于 baseline 路径；
    2. focal 权重只由组内有效样本估计；
    3. 返回组内结果，交由上层做无效样本补偿和跨组汇总。
    """
    direct_scores = signal_matrix @ base_weights

    temperature = max(float(temperature), 1e-6)
    scaled_scores = direct_scores / temperature
    scaled_scores = scaled_scores - np.max(scaled_scores)
    sample_weights = np.exp(scaled_scores)
    sample_weights_sum = float(np.sum(sample_weights))
    if sample_weights_sum <= 0:
        raise ValueError("Focal sample weights must sum to a positive value")
    sample_weights = sample_weights / sample_weights_sum

    weighted_scores = sample_weights @ signal_matrix
    pace = np.clip(weighted_scores / max_signal_score, 0.0, 1.0)
    difficulty = 1.0 - pace + float(epsilon)

    focal_weights = base_weights * np.power(difficulty, float(gamma))
    focal_weight_sum = float(np.sum(focal_weights))
    if focal_weight_sum <= 0:
        raise ValueError("Focal weights must sum to a positive value")
    normalized_focal_weights = focal_weights / focal_weight_sum

    focal_scores = signal_matrix @ normalized_focal_weights
    return {
        "direct_scores": direct_scores,
        "focal_scores": focal_scores,
        "normalized_focal_weights": normalized_focal_weights,
    }


def postprocess_reward(
    *,
    batch: DataProto,
    raw_reward_tensor: torch.Tensor,
    reward_extra_infos_dict: dict[str, Any],
    use_focal: bool,
    focal_config: Any,
) -> RewardPostprocessResult:
    """
    输入:
    1. 当前训练/验证 batch；
    2. 原始 token-level reward tensor；
    3. reward client 透传的 extra info；
    4. 是否启用 focal；
    5. focal 超参数配置。

    输出: `RewardPostprocessResult`，包含训练用 reward tensor、batch 透传字段、dump 字段和监控指标。

    意图:
    1. 将 `reward_signals` 聚合为 direct/focal/final 三类分数；
    2. 将 `success` 和 `llm_generation_error` 当作有效样本参与聚合；
    3. 对系统失败样本做组均值补偿，避免基础设施噪声进入 PPO 更新；
    4. 把验证可聚合的标量字段和仅用于调试的原始嵌套字段分开。

    当前分支只支持 focal / baseline 两种 reward 聚合方式。
    因此 `reward_signals` 是必需输入，缺失时直接报错，而不是静默透传旧 reward。
    """
    if "reward_signals" not in reward_extra_infos_dict:
        raise ValueError(
            "reward_signals is required for focal/baseline reward postprocessing, "
            f"got keys: {sorted(reward_extra_infos_dict.keys())}"
        )

    batch_size = len(batch)
    # raw_scores 仍按旧逻辑从 token-level tensor 求和得到，便于和历史指标做同口径对比。
    raw_scores = raw_reward_tensor.sum(-1).detach().cpu().numpy().astype(np.float64)

    # 先把 reward extra 信息统一拉平为等长 list，后续所有索引都按样本维度对齐。
    overall_statuses = _to_list(
        reward_extra_infos_dict.get("overall_status", ["success"] * batch_size),
        key="overall_status",
        expected_length=batch_size,
    )
    reward_signal_items = _to_list(
        reward_extra_infos_dict["reward_signals"],
        key="reward_signals",
        expected_length=batch_size,
    )
    render_infos = _to_list(
        reward_extra_infos_dict.get("render_info", [{}] * batch_size),
        key="render_info",
        expected_length=batch_size,
    )
    judge_infos = _to_list(
        reward_extra_infos_dict.get("judge_info", [{}] * batch_size),
        key="judge_info",
        expected_length=batch_size,
    )
    error_messages = _to_list(
        reward_extra_infos_dict.get("error_message", [""] * batch_size),
        key="error_message",
        expected_length=batch_size,
    )

    # `signal_matrix` shape = [batch_size, num_signals]，是后面 direct/focal 聚合的基础。
    signal_keys = _infer_signal_keys(reward_signal_items)
    signal_matrix = _build_signal_matrix(reward_signal_items, signal_keys)
    signal_slugs = [slugify_reward_name(signal_key) for signal_key in signal_keys]
    if len(set(signal_slugs)) != len(signal_slugs):
        raise ValueError(f"reward_signals keys slugify to non-unique names: {signal_keys}")

    base_weights = _build_base_weights(signal_keys, _get_focal_param(focal_config, "base_weights", []))
    temperature = float(_get_focal_param(focal_config, "temperature", 10.0))
    gamma = float(_get_focal_param(focal_config, "gamma", 3.0))
    epsilon = float(_get_focal_param(focal_config, "epsilon", 0.05))

    # valid sample 参与 focal 权重估计；invalid sample 不参与估计，只做补偿回填。
    valid_sample_mask = np.asarray(
        [str(status) in VALID_FOCAL_OVERALL_STATUSES for status in overall_statuses],
        dtype=bool,
    )
    llm_generation_error_mask = np.asarray(
        [str(status) == "llm_generation_error" for status in overall_statuses],
        dtype=bool,
    )

    uid_values = _to_list(batch.non_tensor_batch["uid"], key="uid", expected_length=batch_size)
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, uid_value in enumerate(uid_values):
        uid_to_indices[uid_value].append(idx)

    direct_scores = np.zeros(batch_size, dtype=np.float64)
    focal_scores = np.zeros(batch_size, dtype=np.float64)
    group_weights: list[np.ndarray] = []
    invalid_group_indices: list[list[int]] = []

    # 同 uid 的样本视作同一 rollout group（例如同 prompt 多次采样）。
    for group_indices in uid_to_indices.values():
        group_valid_mask = valid_sample_mask[group_indices]
        if not np.any(group_valid_mask):
            # 整组都无效时先延后处理，等所有有效组算完后再用 batch 级均值补偿。
            invalid_group_indices.append(group_indices)
            continue

        valid_group_signals = signal_matrix[group_indices][group_valid_mask]
        group_score_result = _compute_group_focal_scores(
            signal_matrix=valid_group_signals,
            base_weights=base_weights,
            temperature=temperature,
            gamma=gamma,
            epsilon=epsilon,
        )
        group_direct_valid = group_score_result["direct_scores"]
        group_focal_valid = group_score_result["focal_scores"]

        group_direct_scores = np.full(len(group_indices), float(np.mean(group_direct_valid)), dtype=np.float64)
        group_focal_scores = np.full(len(group_indices), float(np.mean(group_focal_valid)), dtype=np.float64)
        # 组内无效样本不参与 focal 权重估计，但保留在 batch 中，并使用该组有效样本均值补偿。
        group_direct_scores[group_valid_mask] = group_direct_valid
        group_focal_scores[group_valid_mask] = group_focal_valid

        direct_scores[group_indices] = group_direct_scores
        focal_scores[group_indices] = group_focal_scores
        group_weights.append(group_score_result["normalized_focal_weights"])

    if not group_weights:
        status_counts = defaultdict(int)
        for status in overall_statuses:
            status_counts[str(status)] += 1
        raise ValueError(
            "No valid focal reward groups found. "
            f"Expected overall_status in {sorted(VALID_FOCAL_OVERALL_STATUSES)}, got {dict(status_counts)}"
        )

    batch_valid_direct_mean = float(np.mean(direct_scores[valid_sample_mask]))
    batch_valid_focal_mean = float(np.mean(focal_scores[valid_sample_mask]))
    for group_indices in invalid_group_indices:
        # 对完全无效的组，只能退化到 batch 级有效组均值，做到“不奖不罚”。
        direct_scores[group_indices] = batch_valid_direct_mean
        focal_scores[group_indices] = batch_valid_focal_mean

    final_scores = focal_scores if use_focal else direct_scores
    reward_tensor = _build_reward_tensor(batch, raw_reward_tensor, final_scores)

    # 这些字段会回写给 batch，供训练日志、验证聚合和数据导出共享。
    derived_extra_info = {
        "reward_raw_score": raw_scores.astype(np.float32),
        "reward_direct_score": direct_scores.astype(np.float32),
        "reward_focal_score": focal_scores.astype(np.float32),
        "reward_final_score": final_scores.astype(np.float32),
        "reward_valid_sample": valid_sample_mask.astype(np.int32),
        "reward_is_llm_generation_error": llm_generation_error_mask.astype(np.int32),
    }
    for signal_idx, signal_slug in enumerate(signal_slugs):
        derived_extra_info[f"reward_signal_{signal_slug}"] = signal_matrix[:, signal_idx].astype(np.float32)

    # validation_extra_info 仅保留标量 list，保证下游聚合函数可直接统计。
    validation_extra_info = {
        "reward_raw_score": raw_scores.tolist(),
        "reward_direct_score": direct_scores.tolist(),
        "reward_focal_score": focal_scores.tolist(),
        "reward_final_score": final_scores.tolist(),
        "reward_valid_sample": valid_sample_mask.astype(np.int32).tolist(),
        "reward_is_llm_generation_error": llm_generation_error_mask.astype(np.int32).tolist(),
    }
    for signal_idx, signal_slug in enumerate(signal_slugs):
        validation_extra_info[f"reward_signal_{signal_slug}"] = signal_matrix[:, signal_idx].tolist()

    # dump_extra_info 保留原始嵌套调试信息，优先服务问题定位。
    dump_extra_info = {}
    for key in RAW_REWARD_DEBUG_KEYS:
        if key not in reward_extra_infos_dict:
            continue
        values = _to_list(reward_extra_infos_dict[key], key=key, expected_length=batch_size)
        # 原始嵌套字段只进入 dump，不进入 validation metrics，避免字典类型破坏聚合逻辑。
        dump_extra_info[key] = [make_json_serializable(item) for item in values]
    dump_extra_info.update(validation_extra_info)
    dump_extra_info["reward"] = final_scores.tolist()

    # Metrics blocks:
    # 1) reward_signals/*:   各 reward signal 的分布统计与 focal 权重
    # 2) reward_status/*:    overall_status / error_message / 核心 reward 分数摘要
    # 3) render_status/*:    render 阶段状态占比
    # 4) judge_status/*:     judge 阶段状态占比
    # 离散状态类指标统一只打 rate，不再打 count，便于面板阅读。
    metrics = {}
    overall_status_counts = defaultdict(int)
    render_status_counts = defaultdict(int)
    judge_status_counts = defaultdict(int)
    error_message_counts = defaultdict(int)

    def _normalize_error_message(error_message: Any) -> str:
        if error_message is None:
            return "none"
        if isinstance(error_message, str):
            stripped = error_message.strip()
            return stripped if stripped else "none"
        if isinstance(error_message, dict):
            for key in ("message", "error", "status"):
                value = error_message.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
        try:
            serialized = json.dumps(make_json_serializable(error_message), sort_keys=True)
            return serialized if serialized else "none"
        except TypeError:
            fallback = str(error_message).strip()
            return fallback if fallback else "none"

    for overall_status, render_info, judge_info in zip(overall_statuses, render_infos, judge_infos, strict=True):
        overall_status_counts[str(overall_status)] += 1
        render_status = render_info.get("status", "unknown") if isinstance(render_info, dict) else "unknown"
        judge_status = judge_info.get("status", "unknown") if isinstance(judge_info, dict) else "unknown"
        render_status_counts[str(render_status)] += 1
        judge_status_counts[str(judge_status)] += 1
    for error_message in error_messages:
        error_message_counts[_normalize_error_message(error_message)] += 1

    for status, count in overall_status_counts.items():
        status_slug = slugify_reward_name(status)
        metrics[f"reward_status/overall_status/{status_slug}/rate"] = float(count / batch_size)
    for error_message, count in error_message_counts.items():
        error_slug = slugify_reward_name(error_message)
        metrics[f"reward_status/error_message/{error_slug}/rate"] = float(count / batch_size)
    for status, count in render_status_counts.items():
        status_slug = slugify_reward_name(status)
        metrics[f"render_status/{status_slug}/rate"] = float(count / batch_size)
    for status, count in judge_status_counts.items():
        status_slug = slugify_reward_name(status)
        metrics[f"judge_status/{status_slug}/rate"] = float(count / batch_size)

    metrics.update(_distribution_metrics("reward_status/raw_score", raw_scores))
    metrics.update(_distribution_metrics("reward_status/direct_score", direct_scores))
    metrics.update(_distribution_metrics("reward_status/focal_score", focal_scores))
    metrics.update(_distribution_metrics("reward_status/final_score", final_scores))
    metrics["reward_status/valid_sample/rate"] = float(np.mean(valid_sample_mask.astype(np.float64)))
    metrics["reward_status/llm_generation_error/rate"] = float(np.mean(llm_generation_error_mask.astype(np.float64)))
    metrics["reward_status/valid_group/rate"] = float(len(group_weights) / max(1, len(uid_to_indices)))

    valid_signal_matrix = signal_matrix[valid_sample_mask]
    for signal_idx, signal_slug in enumerate(signal_slugs):
        metrics.update(_distribution_metrics(f"reward_signals/{signal_slug}", valid_signal_matrix[:, signal_idx]))

    mean_group_weights = np.mean(np.stack(group_weights, axis=0), axis=0)
    for signal_idx, signal_slug in enumerate(signal_slugs):
        metrics[f"reward_signals/{signal_slug}/focal_weight_mean"] = float(mean_group_weights[signal_idx])

    # 为每条样本生成“归一化后的 focal 权重”视图，用于 dump 诊断。
    # 对无效 group，使用全局均值权重兜底（与分数补偿口径一致）。
    uid_to_group_weights: dict[Any, np.ndarray] = {}
    for uid_value, group_indices in uid_to_indices.items():
        group_valid_mask = valid_sample_mask[group_indices]
        if np.any(group_valid_mask):
            valid_group_signals = signal_matrix[group_indices][group_valid_mask]
            group_score_result = _compute_group_focal_scores(
                signal_matrix=valid_group_signals,
                base_weights=base_weights,
                temperature=temperature,
                gamma=gamma,
                epsilon=epsilon,
            )
            uid_to_group_weights[uid_value] = group_score_result["normalized_focal_weights"]
        else:
            uid_to_group_weights[uid_value] = mean_group_weights

    focal_weight_dicts: list[dict[str, float]] = []
    for uid_value in uid_values:
        group_weights_vec = uid_to_group_weights[uid_value]
        focal_weight_dicts.append(
            {signal_slug: float(group_weights_vec[idx]) for idx, signal_slug in enumerate(signal_slugs)}
        )

    # 结构化的 dump 字段，减少扁平 key 的混乱度。
    # 保持旧字段兼容的同时，新增 `reward_detail` 供调试直接读取。
    reward_detail: list[dict[str, Any]] = []
    for idx in range(batch_size):
        reward_detail.append(
            {
                "scores": {
                    "raw": float(raw_scores[idx]),
                    "direct": float(direct_scores[idx]),
                    "focal": float(focal_scores[idx]),
                    "final": float(final_scores[idx]),
                },
                "signals": {
                    signal_slug: float(signal_matrix[idx, signal_i])
                    for signal_i, signal_slug in enumerate(signal_slugs)
                },
                "focal_weights": focal_weight_dicts[idx],
                "status": {
                    "overall": overall_statuses[idx],
                    "error_message": error_messages[idx],
                    "render": render_infos[idx],
                    "judge": judge_infos[idx],
                },
                "valid_sample": int(valid_sample_mask[idx]),
                "is_llm_generation_error": int(llm_generation_error_mask[idx]),
            }
        )

    dump_extra_info["reward_detail"] = [make_json_serializable(item) for item in reward_detail]
    dump_extra_info["reward_focal_weights"] = focal_weight_dicts

    return RewardPostprocessResult(
        reward_tensor=reward_tensor,
        derived_extra_info=derived_extra_info,
        validation_extra_info=validation_extra_info,
        dump_extra_info=dump_extra_info,
        metrics=metrics,
    )
