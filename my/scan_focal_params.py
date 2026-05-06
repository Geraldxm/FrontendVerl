#!/usr/bin/env python3
"""基于 rollout JSONL 离线扫描 focal 参数网格并导出对比报告。

目标:
1. 从新旧两种 rollout 结构恢复样本级 rubric 信号。
2. 按 focal_reward.py 的口径重算 focal 分数并比较 direct/focal 排序差异。
3. 输出 CSV、热力图和中文摘要，辅助选择后续实验参数。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FLOAT_TOLERANCE = 1e-9
MAX_SIGNAL_SCORE = 10.0
VALID_OVERALL_STATUSES = frozenset({"success", "llm_generation_error"})

RUBRIC_KEYS = [
    "a11y_score",
    "color_harmony_and_theme_fit",
    "console_errors",
    "element_hit_rate",
    "first_view_content_messaging",
    "format_score",
    "network_violations",
    "typography_rhythm_and_readability",
    "ui_spatial_score",
]
SUBJECTIVE_ALIASES = {
    "color_harmony_and_theme_fit": (
        "Color Harmony and Theme Fit",
        "Color harmony and theme fit",
    ),
    "first_view_content_messaging": (
        "First-view Content Messaging",
        "First view content messaging",
    ),
    "typography_rhythm_and_readability": (
        "Typography Rhythm and Readability",
        "Typography rhythm and readability",
    ),
}


@dataclass
class GroupData:
    input_hash: str
    sample_matrix: np.ndarray
    direct_scores: np.ndarray
    current_focal_scores: np.ndarray
    valid_mask: np.ndarray
    stored_group_signal: np.ndarray | None
    stored_weights: np.ndarray | None

    @property
    def group_size(self) -> int:
        return int(self.sample_matrix.shape[0])

    @property
    def valid_count(self) -> int:
        return int(np.sum(self.valid_mask))


@dataclass
class StepData:
    step: int
    row_count: int
    groups: list[GroupData]
    status_counter: Counter[str]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="从 rollout JSONL 扫描 focal 参数网格。")
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("rollouts/baseline_v3_n16_0422_Qwen3-1.7B-Base"),
        help="包含按 step 存储 JSONL 的目录。",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="1",
        help="可选 step 过滤，如 '65'、'60-70'、'58,60-62'；留空表示全目录。",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.1,0.5,1,3,5,10",
        help="temperature 列表，逗号分隔，如 '5,10,15'。",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default="0:8:0.5",
        help="gamma 配置。支持列表('1,2,3')或区间('start:end:step')。",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0,0.01,0.02,0.05,0.1,0.2",
        help="epsilon 配置。支持列表('0,0.05')或区间('0:0.2:0.01')。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("my/report/focal_param_scan"),
        help="输出目录（CSV/热力图/summary）。",
    )
    parser.add_argument(
        "--focus-step",
        type=int,
        default=1,
        help="summary 里重点展示的 step。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="summary 中展示的候选参数 Top-K 数量。",
    )
    return parser.parse_args()


def numeric_step(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"rollout 文件名应为纯数字 step，实际为: {path.name}") from exc


def input_hash(input_text: str) -> str:
    return hashlib.sha256(input_text.encode("utf-8")).hexdigest()


def mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def nan_mean(values: list[float]) -> float:
    filtered = [value for value in values if not math.isnan(value)]
    return mean(filtered)


def nan_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return math.nan
    return float(np.percentile(array, percentile))


def safe_max(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(max(values))


def sign(value: float, *, tolerance: float = FLOAT_TOLERANCE) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def z_scores(values: list[float]) -> list[float]:
    value_mean = mean(values)
    variance = mean([(value - value_mean) ** 2 for value in values])
    std = math.sqrt(variance)
    if std <= FLOAT_TOLERANCE:
        return [0.0 for _ in values]
    return [(value - value_mean) / std for value in values]


def pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("相关系数计算输入长度不一致")
    if len(left) < 2:
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm <= FLOAT_TOLERANCE or right_norm <= FLOAT_TOLERANCE:
        return math.nan
    return float(numerator / (left_norm * right_norm))


def average_ranks_desc(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: (-values[idx], idx))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and math.isclose(values[order[end]], values[order[position]], abs_tol=FLOAT_TOLERANCE):
            end += 1
        average_rank = (position + end - 1) / 2.0
        for order_idx in range(position, end):
            ranks[order[order_idx]] = average_rank
        position = end
    return ranks


def normalized_kendall_distance(direct_scores: list[float], focal_scores: list[float]) -> float:
    if len(direct_scores) != len(focal_scores):
        raise ValueError("Kendall 距离计算输入长度不一致")
    pair_count = len(direct_scores) * (len(direct_scores) - 1) / 2
    if pair_count <= 0:
        return 0.0

    distance = 0.0
    for left_idx in range(len(direct_scores)):
        for right_idx in range(left_idx + 1, len(direct_scores)):
            direct_sign = sign(direct_scores[left_idx] - direct_scores[right_idx])
            focal_sign = sign(focal_scores[left_idx] - focal_scores[right_idx])
            if direct_sign == focal_sign:
                continue
            if direct_sign == 0 or focal_sign == 0:
                distance += 0.5
            else:
                distance += 1.0
    return float(distance / pair_count)


def parse_step_filter(spec: str) -> set[int] | None:
    spec = spec.strip()
    if not spec:
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"step 区间非法 '{token}': end < start")
            for value in range(start, end + 1):
                selected.add(value)
        else:
            selected.add(int(token))
    return selected


def _append_float_tokens(target: list[float], token: str) -> None:
    token = token.strip()
    if not token:
        return
    if token.count(":") == 2:
        start_text, end_text, step_text = token.split(":")
        start = float(start_text)
        end = float(end_text)
        step = float(step_text)
        if step <= 0:
            raise ValueError(f"区间步长必须为正数: '{token}'")
        value = start
        guard = 0
        max_iters = 1_000_000
        while value <= end + FLOAT_TOLERANCE:
            target.append(float(round(value, 12)))
            value += step
            guard += 1
            if guard > max_iters:
                raise ValueError(f"解析区间 '{token}' 时值数量过多")
        return
    target.append(float(token))


def parse_float_values(spec: str) -> list[float]:
    values: list[float] = []
    for token in spec.split(","):
        _append_float_tokens(values, token)
    if not values:
        raise ValueError(f"数值配置为空: '{spec}'")
    dedup = sorted({float(round(value, 12)) for value in values})
    return dedup


def canonical_vector_from_dict(
    source: dict[str, Any],
    *,
    stats_counter: Counter[str],
    stats_prefix: str,
) -> np.ndarray:
    vector: list[float] = []
    for rubric in RUBRIC_KEYS:
        if rubric not in source:
            stats_counter[f"{stats_prefix}_missing_{rubric}"] += 1
        vector.append(float(source.get(rubric, 0.0)))
    return np.asarray(vector, dtype=np.float64)


def _pick_subjective_score(subjective_scores: dict[str, Any], rubric: str) -> float:
    for alias in SUBJECTIVE_ALIASES[rubric]:
        if alias in subjective_scores:
            return float(subjective_scores[alias])
    return 0.0


def extract_sample_signal(
    row: dict[str, Any],
    *,
    stats_counter: Counter[str],
) -> np.ndarray:
    """提取单样本 rubric 信号（优先新字段，缺失时按旧字段重建）。"""
    sample_signal = row.get("sample_reward_signal")
    if isinstance(sample_signal, dict) and sample_signal:
        stats_counter["sample_signal_source/sample_reward_signal"] += 1
        return canonical_vector_from_dict(
            sample_signal,
            stats_counter=stats_counter,
            stats_prefix="sample_reward_signal",
        )

    stats_counter["sample_signal_source/reconstructed"] += 1
    render_info = row.get("render_info") if isinstance(row.get("render_info"), dict) else {}
    if not render_info:
        stats_counter["missing/render_info"] += 1
    metrics = render_info.get("metrics") if isinstance(render_info.get("metrics"), dict) else {}
    if not metrics:
        stats_counter["missing/render_info.metrics"] += 1

    judge_info = row.get("judge_info") if isinstance(row.get("judge_info"), dict) else {}
    if not judge_info:
        stats_counter["missing/judge_info"] += 1
    parsed_response = (
        judge_info.get("parsed_response") if isinstance(judge_info.get("parsed_response"), dict) else {}
    )
    if not parsed_response:
        stats_counter["missing/judge_info.parsed_response"] += 1
    subjective_scores = (
        parsed_response.get("subjective_scores")
        if isinstance(parsed_response.get("subjective_scores"), dict)
        else {}
    )
    if not subjective_scores:
        stats_counter["missing/subjective_scores"] += 1

    if "ui_issue_types" in parsed_response:
        issue_types = parsed_response.get("ui_issue_types")
        if not isinstance(issue_types, list):
            stats_counter["invalid/ui_issue_types_not_list"] += 1
            issue_types = []
        ui_spatial_score = max(0.0, 10.0 - 2.0 * len(issue_types))
    else:
        stats_counter["missing/ui_issue_types_key"] += 1
        ui_spatial_score = 0.0

    vector_map = {
        "a11y_score": float(metrics.get("a11y_score", 0.0)),
        "console_errors": float(metrics.get("console_errors", 0.0)),
        "element_hit_rate": float(metrics.get("element_hit_rate", 0.0)),
        "format_score": float(metrics.get("format_score", 0.0)),
        "network_violations": float(metrics.get("network_violations", 0.0)),
        "color_harmony_and_theme_fit": _pick_subjective_score(subjective_scores, "color_harmony_and_theme_fit"),
        "first_view_content_messaging": _pick_subjective_score(
            subjective_scores, "first_view_content_messaging"
        ),
        "typography_rhythm_and_readability": _pick_subjective_score(
            subjective_scores, "typography_rhythm_and_readability"
        ),
        "ui_spatial_score": float(ui_spatial_score),
    }

    for key in (
        "a11y_score",
        "console_errors",
        "element_hit_rate",
        "format_score",
        "network_violations",
    ):
        if key not in metrics:
            stats_counter[f"missing/render_metric/{key}"] += 1
    for rubric in (
        "color_harmony_and_theme_fit",
        "first_view_content_messaging",
        "typography_rhythm_and_readability",
    ):
        if vector_map[rubric] == 0.0:
            found_alias = False
            for alias in SUBJECTIVE_ALIASES[rubric]:
                if alias in subjective_scores:
                    found_alias = True
                    break
            if not found_alias:
                stats_counter[f"missing/subjective/{rubric}"] += 1

    return canonical_vector_from_dict(
        vector_map,
        stats_counter=stats_counter,
        stats_prefix="reconstructed_sample_signal",
    )


def extract_group_signal(
    row: dict[str, Any],
    *,
    stats_counter: Counter[str],
) -> np.ndarray | None:
    for key in ("group_mean_reward_signal", "reward_signals"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return canonical_vector_from_dict(
                value,
                stats_counter=stats_counter,
                stats_prefix=f"{key}",
            )
    stats_counter["missing/group_signal_dict"] += 1
    return None


def extract_weight_vector(
    row: dict[str, Any],
    *,
    stats_counter: Counter[str],
) -> np.ndarray | None:
    for key in ("sample_reward_weight", "group_mean_reward_weight", "reward_weights", "focal_weights"):
        value = row.get(key)
        if isinstance(value, dict) and value:
            return canonical_vector_from_dict(
                value,
                stats_counter=stats_counter,
                stats_prefix=f"{key}",
            )
    stats_counter["missing/weight_dict"] += 1
    return None


def extract_direct_and_focal(row: dict[str, Any]) -> tuple[float, float]:
    reward_scores = row.get("reward_scores")
    if isinstance(reward_scores, dict) and "direct" in reward_scores and "focal" in reward_scores:
        return float(reward_scores["direct"]), float(reward_scores["focal"])

    if "reward_direct_score" in row and "reward_focal_score" in row:
        return float(row["reward_direct_score"]), float(row["reward_focal_score"])
    raise ValueError("当前行缺少 direct/focal 分数字段")


def extract_valid_sample(row: dict[str, Any]) -> bool:
    reward_scores = row.get("reward_scores")
    if isinstance(reward_scores, dict) and "valid_sample" in reward_scores:
        return int(reward_scores["valid_sample"]) == 1
    if "reward_valid_sample" in row:
        return int(row["reward_valid_sample"]) == 1

    status = row.get("status")
    if isinstance(status, dict):
        return str(status.get("overall", "")) in VALID_OVERALL_STATUSES
    if isinstance(status, str):
        return status in VALID_OVERALL_STATUSES
    if isinstance(row.get("overall_status"), str):
        return str(row["overall_status"]) in VALID_OVERALL_STATUSES
    return False


def extract_overall_status(row: dict[str, Any]) -> str:
    status = row.get("status")
    if isinstance(status, dict):
        return str(status.get("overall", "missing"))
    if isinstance(status, str):
        return status
    if isinstance(row.get("overall_status"), str):
        return str(row["overall_status"])
    return "missing"


def analyze_group_scores(direct_scores: np.ndarray, focal_scores: np.ndarray) -> dict[str, float]:
    direct_values = [float(value) for value in direct_scores]
    focal_values = [float(value) for value in focal_scores]

    direct_z = z_scores(direct_values)
    focal_z = z_scores(focal_values)
    top_direct_idx = max(range(len(direct_values)), key=lambda idx: (direct_values[idx], -idx))
    top_focal_idx = max(range(len(focal_values)), key=lambda idx: (focal_values[idx], -idx))

    return {
        "normalized_kendall_distance": normalized_kendall_distance(direct_values, focal_values),
        "spearman_rank_corr": pearson_correlation(
            average_ranks_desc(direct_values),
            average_ranks_desc(focal_values),
        ),
        "z_advantage_corr": pearson_correlation(direct_z, focal_z),
        "z_advantage_sign_flip_rate": mean(
            [float(sign(direct) != sign(focal)) for direct, focal in zip(direct_z, focal_z, strict=True)]
        ),
        "mean_abs_z_advantage_delta": mean(
            [abs(direct - focal) for direct, focal in zip(direct_z, focal_z, strict=True)]
        ),
        "top1_same_rate": float(top_direct_idx == top_focal_idx),
    }


def _parse_rows_in_file(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON 解析失败") from exc
    if not rows:
        raise ValueError(f"{path} 不包含有效 JSON 行")
    return rows


def load_step_data(path: Path, stats_counter: Counter[str]) -> StepData:
    """读取单个 step 文件并按 `sha256(input)` 组织 group 数据。"""
    step = numeric_step(path)
    rows = _parse_rows_in_file(path)

    sample_vectors: list[np.ndarray] = []
    direct_scores: list[float] = []
    focal_scores: list[float] = []
    valid_mask: list[bool] = []
    status_counter: Counter[str] = Counter()
    group_signals: list[np.ndarray | None] = []
    weight_vectors: list[np.ndarray | None] = []
    input_hashes: list[str] = []

    for row in rows:
        sample_vectors.append(extract_sample_signal(row, stats_counter=stats_counter))
        direct_score, focal_score = extract_direct_and_focal(row)
        direct_scores.append(direct_score)
        focal_scores.append(focal_score)
        valid_mask.append(extract_valid_sample(row))
        status_counter[extract_overall_status(row)] += 1
        group_signals.append(extract_group_signal(row, stats_counter=stats_counter))
        weight_vectors.append(extract_weight_vector(row, stats_counter=stats_counter))
        input_hashes.append(input_hash(str(row.get("input", ""))))

    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for idx, hash_value in enumerate(input_hashes):
        grouped_indices[hash_value].append(idx)

    groups: list[GroupData] = []
    for hash_value in sorted(grouped_indices):
        indices = grouped_indices[hash_value]
        sample_matrix = np.stack([sample_vectors[idx] for idx in indices], axis=0)
        direct_array = np.asarray([direct_scores[idx] for idx in indices], dtype=np.float64)
        focal_array = np.asarray([focal_scores[idx] for idx in indices], dtype=np.float64)
        valid_array = np.asarray([valid_mask[idx] for idx in indices], dtype=bool)

        stored_group_signal = None
        for idx in indices:
            if group_signals[idx] is not None:
                stored_group_signal = group_signals[idx]
                break

        stored_weights = None
        has_missing_weight = any(weight_vectors[idx] is None for idx in indices)
        if not has_missing_weight:
            stored_weights = np.stack([weight_vectors[idx] for idx in indices], axis=0)

        groups.append(
            GroupData(
                input_hash=hash_value,
                sample_matrix=sample_matrix,
                direct_scores=direct_array,
                current_focal_scores=focal_array,
                valid_mask=valid_array,
                stored_group_signal=stored_group_signal,
                stored_weights=stored_weights,
            )
        )

    return StepData(
        step=step,
        row_count=len(rows),
        groups=groups,
        status_counter=status_counter,
    )


def format_param_slug(value: float) -> str:
    text = f"{value:.12g}"
    text = text.replace("-", "m").replace(".", "p")
    return text


def compute_focal_scores_for_combo(
    groups: list[GroupData],
    *,
    temperature: float,
    gamma: float,
    epsilon: float,
) -> list[np.ndarray]:
    """按给定 (temperature, gamma, epsilon) 重算每个 group 的 focal 分数。

    行为与训练口径对齐:
    - 仅 valid_sample=1 参与组内权重估计；
    - 组内无效样本回填该组有效样本均值；
    - 全无效组回填当前 step 的 batch-valid 均值。
    """
    rubric_count = len(RUBRIC_KEYS)
    base_weights = np.full(rubric_count, 1.0 / rubric_count, dtype=np.float64)

    output_scores: list[np.ndarray | None] = [None] * len(groups)
    valid_values: list[np.ndarray] = []
    invalid_group_indices: list[int] = []

    safe_temperature = max(float(temperature), 1e-6)
    safe_gamma = float(gamma)
    safe_epsilon = float(epsilon)

    for group_idx, group in enumerate(groups):
        if group.valid_count == 0:
            invalid_group_indices.append(group_idx)
            continue

        valid_signals = group.sample_matrix[group.valid_mask]
        direct_valid = valid_signals @ base_weights
        scaled_scores = direct_valid / safe_temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)
        sample_weights = np.exp(scaled_scores)
        sample_weight_sum = float(np.sum(sample_weights))
        if sample_weight_sum <= 0:
            raise ValueError("focal 样本权重和必须为正数")
        sample_weights = sample_weights / sample_weight_sum

        weighted_scores = sample_weights @ valid_signals
        pace = np.clip(weighted_scores / MAX_SIGNAL_SCORE, 0.0, 1.0)
        difficulty = 1.0 - pace + safe_epsilon
        focal_weights = base_weights * np.power(difficulty, safe_gamma)
        focal_weight_sum = float(np.sum(focal_weights))
        if focal_weight_sum <= 0:
            raise ValueError("focal rubric 权重和必须为正数")
        normalized_focal_weights = focal_weights / focal_weight_sum

        focal_valid = valid_signals @ normalized_focal_weights
        group_focal = np.full(group.group_size, float(np.mean(focal_valid)), dtype=np.float64)
        group_focal[group.valid_mask] = focal_valid
        output_scores[group_idx] = group_focal
        valid_values.append(focal_valid.astype(np.float64))

    if not valid_values:
        raise ValueError("参数扫描中没有找到有效样本组")
    batch_valid_focal_mean = float(np.mean(np.concatenate(valid_values)))
    for group_idx in invalid_group_indices:
        output_scores[group_idx] = np.full(groups[group_idx].group_size, batch_valid_focal_mean, dtype=np.float64)

    return [scores if scores is not None else np.full(groups[idx].group_size, batch_valid_focal_mean) for idx, scores in enumerate(output_scores)]


def calc_consistency_metrics(step_data: StepData) -> dict[str, float]:
    group_signal_errors: list[float] = []
    focal_replay_errors: list[float] = []
    group_sizes = Counter(group.group_size for group in step_data.groups)

    for group in step_data.groups:
        if group.valid_count > 0 and group.stored_group_signal is not None:
            recon_group_signal = np.mean(group.sample_matrix[group.valid_mask], axis=0)
            group_signal_errors.append(float(np.max(np.abs(recon_group_signal - group.stored_group_signal))))

        if group.stored_weights is not None and group.valid_count > 0:
            row_focal = np.sum(group.sample_matrix * group.stored_weights, axis=1)
            valid_mean = float(np.mean(row_focal[group.valid_mask]))
            replay = row_focal.copy()
            replay[~group.valid_mask] = valid_mean
            replay_error = np.abs(replay - group.current_focal_scores)
            focal_replay_errors.append(float(np.max(replay_error)))

    return {
        "step": step_data.step,
        "row_count": step_data.row_count,
        "group_count": len(step_data.groups),
        "group_size_counts": json.dumps(dict(sorted(group_sizes.items())), sort_keys=True),
        "status_counts": json.dumps(dict(sorted(step_data.status_counter.items())), sort_keys=True),
        "group_signal_abs_error_mean": mean(group_signal_errors),
        "group_signal_abs_error_max": safe_max(group_signal_errors),
        "focal_replay_abs_error_mean": mean(focal_replay_errors),
        "focal_replay_abs_error_max": safe_max(focal_replay_errors),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"无可写入行: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def init_grid_group_writer(path: Path) -> tuple[csv.DictWriter, Any]:
    fieldnames = [
        "step",
        "input_hash",
        "group_size",
        "valid_count",
        "temperature",
        "gamma",
        "epsilon",
        "normalized_kendall_distance",
        "spearman_rank_corr",
        "z_advantage_corr",
        "z_advantage_sign_flip_rate",
        "mean_abs_z_advantage_delta",
        "top1_same_rate",
    ]
    file = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    return writer, file


def aggregate_group_metric_rows(
    *,
    step: int,
    temperature: float,
    gamma: float,
    epsilon: float,
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_keys = [
        "normalized_kendall_distance",
        "spearman_rank_corr",
        "z_advantage_corr",
        "z_advantage_sign_flip_rate",
        "mean_abs_z_advantage_delta",
        "top1_same_rate",
    ]
    step_row: dict[str, Any] = {
        "step": step,
        "temperature": temperature,
        "gamma": gamma,
        "epsilon": epsilon,
        "group_count": len(metric_rows),
        "row_count": int(sum(int(row["group_size"]) for row in metric_rows)),
    }
    for key in metric_keys:
        values = [float(row[key]) for row in metric_rows]
        step_row[f"{key}_mean"] = nan_mean(values)
        step_row[f"{key}_p90"] = nan_percentile(values, 90.0)
    return step_row


def try_import_matplotlib() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "生成热力图依赖 matplotlib。"
            "可执行: /inspire/hdd/global_user/gexinmu-253108100065/conda/frontrl/bin/python -m pip install matplotlib"
        ) from exc
    return matplotlib, plt


def plot_heatmap(
    *,
    x_values: list[float],
    y_values: list[float],
    matrix: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    _, plt = try_import_matplotlib()
    fig, axis = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_invalid(matrix)
    im = axis.imshow(masked, aspect="auto", origin="lower", cmap="viridis")
    axis.set_xticks(range(len(x_values)))
    axis.set_xticklabels([f"{value:g}" for value in x_values], rotation=45, ha="right")
    axis.set_yticks(range(len(y_values)))
    axis.set_yticklabels([f"{value:g}" for value in y_values])
    axis.set_xlabel("epsilon")
    axis.set_ylabel("gamma")
    axis.set_title(title)
    fig.colorbar(im, ax=axis, shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _metric_row_for_group(
    *,
    step: int,
    group: GroupData,
    focal_scores: np.ndarray,
    temperature: float,
    gamma: float,
    epsilon: float,
) -> dict[str, Any]:
    metrics = analyze_group_scores(group.direct_scores, focal_scores)
    return {
        "step": step,
        "input_hash": group.input_hash,
        "group_size": group.group_size,
        "valid_count": group.valid_count,
        "temperature": temperature,
        "gamma": gamma,
        "epsilon": epsilon,
        **metrics,
    }


def _current_metric_row(step: int, group: GroupData) -> dict[str, Any]:
    metrics = analyze_group_scores(group.direct_scores, group.current_focal_scores)
    return {
        "step": step,
        "input_hash": group.input_hash,
        "group_size": group.group_size,
        "valid_count": group.valid_count,
        **metrics,
    }


def _combo_aggregate_key(step_row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(step_row["temperature"]),
        float(step_row["gamma"]),
        float(step_row["epsilon"]),
    )


def write_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    rollout_files: list[Path],
    consistency_rows: list[dict[str, Any]],
    group_metrics_current: list[dict[str, Any]],
    grid_step_rows: list[dict[str, Any]],
    stats_counter: Counter[str],
) -> None:
    """写出中文 summary，包含一致性检查和参数候选排行。"""
    lines: list[str] = []
    lines.append("# Focal 参数网格扫描总结")
    lines.append("")
    lines.append(f"- Rollout 目录: `{args.rollout_dir}`")
    lines.append(f"- 解析 step 数: {len(rollout_files)}")
    lines.append(f"- 重点 step: {args.focus_step}")
    lines.append(f"- temperatures: `{args.temperatures}`")
    lines.append(f"- gammas: `{args.gammas}`")
    lines.append(f"- epsilons: `{args.epsilons}`")
    lines.append("")

    total_rows = sum(int(item["row_count"]) for item in consistency_rows)
    total_groups = sum(int(item["group_count"]) for item in consistency_rows)
    lines.append("## 解析数据规模")
    lines.append("")
    lines.append(f"- 总行数: {total_rows}")
    lines.append(f"- 总组数: {total_groups}")
    lines.append(
        f"- 当前排序对比组数（direct vs 已存 focal）: {len(group_metrics_current)}"
    )
    lines.append(f"- 网格 step 点位数: {len(grid_step_rows)}")
    lines.append("")

    lines.append("## 一致性检查")
    lines.append("")
    lines.append(
        f"- group signal 最大绝对误差: {format_float(safe_max([float(item['group_signal_abs_error_max']) for item in consistency_rows]))}"
    )
    lines.append(
        f"- focal 回放最大绝对误差: {format_float(safe_max([float(item['focal_replay_abs_error_max']) for item in consistency_rows]))}"
    )

    focus_consistency = next((item for item in consistency_rows if int(item["step"]) == args.focus_step), None)
    if focus_consistency is not None:
        lines.append(f"- 重点 step {args.focus_step} 行数: {focus_consistency['row_count']}")
        lines.append(f"- 重点 step {args.focus_step} 组数: {focus_consistency['group_count']}")
        lines.append(
            f"- 重点 step {args.focus_step} 组大小分布: `{focus_consistency['group_size_counts']}`"
        )
        lines.append(
            f"- 重点 step {args.focus_step} group_signal 最大误差: {format_float(float(focus_consistency['group_signal_abs_error_max']))}"
        )
        lines.append(
            f"- 重点 step {args.focus_step} focal 回放最大误差: {format_float(float(focus_consistency['focal_replay_abs_error_max']))}"
        )
    lines.append("")

    lines.append("## 缺失字段统计")
    lines.append("")
    if stats_counter:
        for key, value in sorted(stats_counter.items()):
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- 无")
    lines.append("")

    lines.append("## 参数候选 Top-K")
    lines.append("")
    combo_buckets: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in grid_step_rows:
        combo_buckets[_combo_aggregate_key(row)].append(row)

    combo_rank_rows: list[dict[str, float]] = []
    for (temperature, gamma, epsilon), rows in combo_buckets.items():
        kendall_means = [float(item["normalized_kendall_distance_mean"]) for item in rows]
        signflip_means = [float(item["z_advantage_sign_flip_rate_mean"]) for item in rows]
        kendall_p90s = [float(item["normalized_kendall_distance_p90"]) for item in rows]
        signflip_p90s = [float(item["z_advantage_sign_flip_rate_p90"]) for item in rows]
        combo_rank_rows.append(
            {
                "temperature": temperature,
                "gamma": gamma,
                "epsilon": epsilon,
                "kendall_mean": nan_mean(kendall_means),
                "signflip_mean": nan_mean(signflip_means),
                "kendall_p90_mean": nan_mean(kendall_p90s),
                "signflip_p90_mean": nan_mean(signflip_p90s),
            }
        )

    combo_rank_rows.sort(
        key=lambda item: (
            item["kendall_mean"],
            item["signflip_mean"],
            item["kendall_p90_mean"],
            item["signflip_p90_mean"],
        ),
        reverse=True,
    )

    if not combo_rank_rows:
        lines.append("- 没有可用候选参数。")
    else:
        top_k = min(int(args.top_k), len(combo_rank_rows))
        lines.append("| 排名 | temp | gamma | epsilon | kendall_mean | signflip_mean | kendall_p90_mean | signflip_p90_mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for index in range(top_k):
            item = combo_rank_rows[index]
            lines.append(
                "| "
                + f"{index + 1} | {item['temperature']:g} | {item['gamma']:g} | {item['epsilon']:g} | "
                + f"{format_float(item['kendall_mean'])} | {format_float(item['signflip_mean'])} | "
                + f"{format_float(item['kendall_p90_mean'])} | {format_float(item['signflip_p90_mean'])} |"
            )
    lines.append("")

    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `group_metrics_current.csv`")
    lines.append("- `grid_group_metrics.csv`")
    lines.append("- `grid_step_metrics.csv`")
    lines.append("- `heatmaps/*.png`")
    lines.append("- `summary.md`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def main() -> None:
    """主流程: 解析参数 -> 加载 step -> 计算指标 -> 写出文件。"""
    args = parse_args()
    if not args.rollout_dir.is_dir():
        raise FileNotFoundError(f"未找到 rollout 目录: {args.rollout_dir}")

    selected_steps = parse_step_filter(args.steps)
    temperatures = parse_float_values(args.temperatures)
    gammas = parse_float_values(args.gammas)
    epsilons = parse_float_values(args.epsilons)

    rollout_files = sorted(args.rollout_dir.glob("*.jsonl"), key=numeric_step)
    if selected_steps is not None:
        rollout_files = [path for path in rollout_files if numeric_step(path) in selected_steps]
    if not rollout_files:
        raise FileNotFoundError(
            f"在 {args.rollout_dir} 下没有匹配到 step 文件。step 过滤条件: {args.steps!r}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.out_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    stats_counter: Counter[str] = Counter()
    step_data_list = [load_step_data(path, stats_counter) for path in rollout_files]

    consistency_rows = [calc_consistency_metrics(step_data) for step_data in step_data_list]
    write_csv(args.out_dir / "step_consistency.csv", consistency_rows)

    group_metrics_current: list[dict[str, Any]] = []
    for step_data in step_data_list:
        for group in step_data.groups:
            group_metrics_current.append(_current_metric_row(step_data.step, group))
    write_csv(args.out_dir / "group_metrics_current.csv", group_metrics_current)

    grid_group_writer, grid_group_file = init_grid_group_writer(args.out_dir / "grid_group_metrics.csv")
    grid_step_rows: list[dict[str, Any]] = []

    for step_data in step_data_list:
        for temperature in temperatures:
            kendall_mean_grid = np.full((len(gammas), len(epsilons)), np.nan, dtype=np.float64)
            kendall_p90_grid = np.full((len(gammas), len(epsilons)), np.nan, dtype=np.float64)
            signflip_mean_grid = np.full((len(gammas), len(epsilons)), np.nan, dtype=np.float64)
            signflip_p90_grid = np.full((len(gammas), len(epsilons)), np.nan, dtype=np.float64)

            for gamma_idx, gamma in enumerate(gammas):
                for epsilon_idx, epsilon in enumerate(epsilons):
                    focal_scores_by_group = compute_focal_scores_for_combo(
                        step_data.groups,
                        temperature=temperature,
                        gamma=gamma,
                        epsilon=epsilon,
                    )
                    group_rows_for_combo: list[dict[str, Any]] = []
                    for group, focal_scores in zip(step_data.groups, focal_scores_by_group, strict=True):
                        row = _metric_row_for_group(
                            step=step_data.step,
                            group=group,
                            focal_scores=focal_scores,
                            temperature=temperature,
                            gamma=gamma,
                            epsilon=epsilon,
                        )
                        group_rows_for_combo.append(row)
                        grid_group_writer.writerow(row)

                    step_row = aggregate_group_metric_rows(
                        step=step_data.step,
                        temperature=temperature,
                        gamma=gamma,
                        epsilon=epsilon,
                        metric_rows=group_rows_for_combo,
                    )
                    grid_step_rows.append(step_row)

                    kendall_mean_grid[gamma_idx, epsilon_idx] = float(
                        step_row["normalized_kendall_distance_mean"]
                    )
                    kendall_p90_grid[gamma_idx, epsilon_idx] = float(
                        step_row["normalized_kendall_distance_p90"]
                    )
                    signflip_mean_grid[gamma_idx, epsilon_idx] = float(
                        step_row["z_advantage_sign_flip_rate_mean"]
                    )
                    signflip_p90_grid[gamma_idx, epsilon_idx] = float(
                        step_row["z_advantage_sign_flip_rate_p90"]
                    )

            step_slug = str(step_data.step)
            temp_slug = format_param_slug(temperature)
            plot_heatmap(
                x_values=epsilons,
                y_values=gammas,
                matrix=kendall_mean_grid,
                title=f"Step {step_data.step} Temp {temperature:g} Kendall Mean",
                out_path=heatmap_dir / f"step_{step_slug}_temp_{temp_slug}_kendall_mean.png",
            )
            plot_heatmap(
                x_values=epsilons,
                y_values=gammas,
                matrix=kendall_p90_grid,
                title=f"Step {step_data.step} Temp {temperature:g} Kendall P90",
                out_path=heatmap_dir / f"step_{step_slug}_temp_{temp_slug}_kendall_p90.png",
            )
            plot_heatmap(
                x_values=epsilons,
                y_values=gammas,
                matrix=signflip_mean_grid,
                title=f"Step {step_data.step} Temp {temperature:g} SignFlip Mean",
                out_path=heatmap_dir / f"step_{step_slug}_temp_{temp_slug}_signflip_mean.png",
            )
            plot_heatmap(
                x_values=epsilons,
                y_values=gammas,
                matrix=signflip_p90_grid,
                title=f"Step {step_data.step} Temp {temperature:g} SignFlip P90",
                out_path=heatmap_dir / f"step_{step_slug}_temp_{temp_slug}_signflip_p90.png",
            )

    grid_group_file.close()
    write_csv(args.out_dir / "grid_step_metrics.csv", grid_step_rows)
    write_summary(
        args.out_dir / "summary.md",
        args=args,
        rollout_files=rollout_files,
        consistency_rows=consistency_rows,
        group_metrics_current=group_metrics_current,
        grid_step_rows=grid_step_rows,
        stats_counter=stats_counter,
    )

    print(f"已写入: {args.out_dir / 'step_consistency.csv'}")
    print(f"已写入: {args.out_dir / 'group_metrics_current.csv'}")
    print(f"已写入: {args.out_dir / 'grid_group_metrics.csv'}")
    print(f"已写入: {args.out_dir / 'grid_step_metrics.csv'}")
    print(f"已写入: {args.out_dir / 'summary.md'}")
    print(f"已写入热力图目录: {heatmap_dir}")


if __name__ == "__main__":
    main()
