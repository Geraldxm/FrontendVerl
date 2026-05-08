#!/usr/bin/env python3
"""批量分析 rollout 后期质量并生成中文报告。

用途:
1. 自动识别每个实验目录内按 step 递增且修改时间连续的 JSONL。
2. 排除 reward server 下线、request_error 暴涨等低有效率 step。
3. 用 valid_sample=1 口径统计后期 rubric、focal 权重和 direct/focal 排序扰动。
4. 输出 CSV 和中文 Markdown，辅助比较 baseline 与 focal 实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUBRIC_KEYS = [
    "format_score",
    "console_errors",
    "network_violations",
    "a11y_score",
    "element_hit_rate",
    "ui_spatial_score",
    "color_harmony_and_theme_fit",
    "typography_rhythm_and_readability",
    "first_view_content_messaging",
]
OBJECTIVE_KEYS = RUBRIC_KEYS[:6]
SUBJECTIVE_KEYS = RUBRIC_KEYS[-3:]


@dataclass
class StepSummary:
    experiment: str
    step: int
    path: Path
    mtime: float
    stale_tail: bool
    row_count: int
    group_count: int
    valid_count: int
    valid_group_count: int
    status_counts: Counter[str]
    score_means_all: dict[str, float]
    score_means_valid: dict[str, float]
    rubric_means_valid: dict[str, float]
    weight_means_valid: dict[str, float]
    top1_same_rate: float
    spearman_mean: float
    z_sign_flip_rate: float
    direct_group_var_mean: float
    focal_group_var_mean: float
    max_weight_mean: float

    @property
    def valid_rate(self) -> float:
        return self.valid_count / self.row_count if self.row_count else math.nan

    @property
    def valid_group_rate(self) -> float:
        return self.valid_group_count / self.group_count if self.group_count else math.nan

    @property
    def request_error_rate(self) -> float:
        return self.status_counts.get("request_error", 0) / self.row_count if self.row_count else math.nan


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="批量分析 rollout 后期质量。")
    parser.add_argument("--rollout-root", type=Path, default=Path("rollouts"), help="rollouts 根目录。")
    parser.add_argument(
        "--experiments",
        type=str,
        default="",
        help="逗号分隔的实验目录名；留空表示自动扫描全部含 JSONL 的目录。",
    )
    parser.add_argument("--late-window", type=int, default=5, help="每个实验纳入后期汇总的 step 数。")
    parser.add_argument("--min-valid-rate", type=float, default=0.5, help="纳入后期汇总的最低 valid_sample 比例。")
    parser.add_argument(
        "--max-request-error-rate",
        type=float,
        default=0.25,
        help="纳入后期汇总的最高 request_error 比例。",
    )
    parser.add_argument(
        "--mtime-reset-seconds",
        type=float,
        default=3600.0,
        help="按 step 递增时若修改时间倒退超过该阈值，则标记为旧运行尾巴。",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("my/report/late_rollout_quality"), help="输出目录。")
    return parser.parse_args()


def mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return float(sum(values) / len(values)) if values else math.nan


def variance(values: list[float]) -> float:
    if not values:
        return math.nan
    center = mean(values)
    return mean([(value - center) ** 2 for value in values])


def numeric_step(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def average_ranks_desc(values: list[float]) -> list[float]:
    """返回降序平均排名，用于粗略 Spearman 相关。"""
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][1] == indexed[pos][1]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in range(pos, end):
            ranks[indexed[idx][0]] = avg_rank
        pos = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var <= 0 or right_var <= 0:
        return math.nan
    cov = sum((lval - left_mean) * (rval - right_mean) for lval, rval in zip(left, right))
    return float(cov / math.sqrt(left_var * right_var))


def sign(value: float, *, tolerance: float = 1e-9) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_signal(row: dict[str, Any]) -> dict[str, float]:
    signal = row.get("sample_reward_signal")
    if isinstance(signal, dict):
        return {key: safe_float(signal.get(key), 0.0) for key in RUBRIC_KEYS}

    reward_detail = row.get("reward_detail")
    if isinstance(reward_detail, dict):
        detail_signal = reward_detail.get("sample_reward_signal")
        if isinstance(detail_signal, dict):
            return {key: safe_float(detail_signal.get(key), 0.0) for key in RUBRIC_KEYS}

    raw_signal = row.get("reward_signals")
    if isinstance(raw_signal, dict):
        return {
            "format_score": safe_float(raw_signal.get("format_score"), 0.0),
            "console_errors": safe_float(raw_signal.get("console_errors"), 0.0),
            "network_violations": safe_float(raw_signal.get("network_violations"), 0.0),
            "a11y_score": safe_float(raw_signal.get("a11y_score"), 0.0),
            "element_hit_rate": safe_float(raw_signal.get("element_hit_rate"), 0.0),
            "ui_spatial_score": safe_float(raw_signal.get("ui_spatial_score"), 0.0),
            "color_harmony_and_theme_fit": safe_float(raw_signal.get("Color Harmony and Theme Fit"), 0.0),
            "typography_rhythm_and_readability": safe_float(raw_signal.get("Typography Rhythm and Readability"), 0.0),
            "first_view_content_messaging": safe_float(raw_signal.get("First-view Content Messaging"), 0.0),
        }

    return {key: 0.0 for key in RUBRIC_KEYS}


def extract_weight(row: dict[str, Any]) -> dict[str, float]:
    for key in ("sample_reward_weight", "group_mean_reward_weight", "reward_weights", "focal_weights"):
        weight = row.get(key)
        if isinstance(weight, dict):
            return {rubric: safe_float(weight.get(rubric), 0.0) for rubric in RUBRIC_KEYS}
    return {rubric: math.nan for rubric in RUBRIC_KEYS}


def extract_scores(row: dict[str, Any]) -> dict[str, float]:
    reward_scores = row.get("reward_scores")
    if isinstance(reward_scores, dict):
        return {
            "direct": safe_float(reward_scores.get("direct")),
            "focal": safe_float(reward_scores.get("focal")),
            "final": safe_float(reward_scores.get("final")),
            "valid_sample": safe_float(reward_scores.get("valid_sample"), 0.0),
        }
    return {
        "direct": safe_float(row.get("reward_direct_score")),
        "focal": safe_float(row.get("reward_focal_score")),
        "final": safe_float(row.get("reward_final_score")),
        "valid_sample": safe_float(row.get("reward_valid_sample"), 0.0),
    }


def extract_status(row: dict[str, Any]) -> str:
    status = row.get("status")
    if isinstance(status, dict):
        return str(status.get("overall", "missing"))
    return str(row.get("overall_status", "missing"))


def summarize_step(experiment: str, path: Path, stale_tail: bool) -> StepSummary:
    """汇总单个 JSONL step。"""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    status_counts: Counter[str] = Counter()
    score_values_all: dict[str, list[float]] = defaultdict(list)
    score_values_valid: dict[str, list[float]] = defaultdict(list)
    rubric_values_valid: dict[str, list[float]] = defaultdict(list)
    weight_values_valid: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        status_counts[extract_status(row)] += 1
        scores = extract_scores(row)
        for key, value in scores.items():
            if not math.isnan(value):
                score_values_all[key].append(value)
        if scores["valid_sample"] == 1:
            signal = extract_signal(row)
            weight = extract_weight(row)
            for key, value in scores.items():
                if not math.isnan(value):
                    score_values_valid[key].append(value)
            for key in RUBRIC_KEYS:
                rubric_values_valid[key].append(signal[key])
                weight_values_valid[key].append(weight[key])
        groups[str(row.get("input", ""))].append(row)

    top1_same_values: list[float] = []
    spearman_values: list[float] = []
    sign_flip_values: list[float] = []
    direct_group_vars: list[float] = []
    focal_group_vars: list[float] = []
    valid_group_count = 0

    for group_rows in groups.values():
        valid_rows = [row for row in group_rows if extract_scores(row)["valid_sample"] == 1]
        if not valid_rows:
            continue
        valid_group_count += 1
        direct_scores = [extract_scores(row)["direct"] for row in valid_rows]
        focal_scores = [extract_scores(row)["focal"] for row in valid_rows]
        if len(direct_scores) >= 2:
            top_direct = max(range(len(direct_scores)), key=lambda idx: (direct_scores[idx], -idx))
            top_focal = max(range(len(focal_scores)), key=lambda idx: (focal_scores[idx], -idx))
            top1_same_values.append(float(top_direct == top_focal))
            spearman_values.append(pearson(average_ranks_desc(direct_scores), average_ranks_desc(focal_scores)))

            direct_var = variance(direct_scores)
            focal_var = variance(focal_scores)
            direct_group_vars.append(direct_var)
            focal_group_vars.append(focal_var)
            direct_std = math.sqrt(direct_var) if not math.isnan(direct_var) else 0.0
            focal_std = math.sqrt(focal_var) if not math.isnan(focal_var) else 0.0
            if direct_std > 1e-9 and focal_std > 1e-9:
                direct_mean = mean(direct_scores)
                focal_mean = mean(focal_scores)
                for direct_score, focal_score in zip(direct_scores, focal_scores):
                    direct_z = (direct_score - direct_mean) / direct_std
                    focal_z = (focal_score - focal_mean) / focal_std
                    sign_flip_values.append(float(sign(direct_z) != sign(focal_z)))

    max_weights = []
    for row in rows:
        if extract_scores(row)["valid_sample"] != 1:
            continue
        weights = [value for value in extract_weight(row).values() if not math.isnan(value)]
        if weights:
            max_weights.append(max(weights))

    return StepSummary(
        experiment=experiment,
        step=int(path.stem),
        path=path,
        mtime=path.stat().st_mtime,
        stale_tail=stale_tail,
        row_count=len(rows),
        group_count=len(groups),
        valid_count=len(score_values_valid["valid_sample"]),
        valid_group_count=valid_group_count,
        status_counts=status_counts,
        score_means_all={key: mean(values) for key, values in score_values_all.items()},
        score_means_valid={key: mean(values) for key, values in score_values_valid.items()},
        rubric_means_valid={key: mean(rubric_values_valid[key]) for key in RUBRIC_KEYS},
        weight_means_valid={key: mean(weight_values_valid[key]) for key in RUBRIC_KEYS},
        top1_same_rate=mean(top1_same_values),
        spearman_mean=mean(spearman_values),
        z_sign_flip_rate=mean(sign_flip_values),
        direct_group_var_mean=mean(direct_group_vars),
        focal_group_var_mean=mean(focal_group_vars),
        max_weight_mean=mean(max_weights),
    )


def list_experiment_dirs(root: Path, experiments: str) -> list[Path]:
    if experiments.strip():
        return [root / item.strip() for item in experiments.split(",") if item.strip()]
    return sorted(
        [path for path in root.iterdir() if path.is_dir() and any(path.glob("*.jsonl"))],
        key=lambda path: path.name,
    )


def mark_stale_tail(files: list[Path], *, reset_seconds: float) -> dict[Path, bool]:
    """按 step 递增检测修改时间大幅倒退的旧运行尾巴。"""
    stale: dict[Path, bool] = {path: False for path in files}
    previous_mtime: float | None = None
    in_stale_tail = False
    for path in files:
        current_mtime = path.stat().st_mtime
        if previous_mtime is not None and current_mtime + reset_seconds < previous_mtime:
            in_stale_tail = True
        stale[path] = in_stale_tail
        previous_mtime = current_mtime
    return stale


def summarize_experiment(path: Path, *, reset_seconds: float) -> list[StepSummary]:
    files = sorted(
        [file for file in path.glob("*.jsonl") if numeric_step(file) is not None],
        key=lambda file: int(file.stem),
    )
    stale_map = mark_stale_tail(files, reset_seconds=reset_seconds)
    return [summarize_step(path.name, file, stale_map[file]) for file in files]


def row_from_summary(summary: StepSummary) -> dict[str, Any]:
    subject_mean = mean([summary.rubric_means_valid[key] for key in SUBJECTIVE_KEYS])
    objective_mean = mean([summary.rubric_means_valid[key] for key in OBJECTIVE_KEYS])
    return {
        "experiment": summary.experiment,
        "step": summary.step,
        "stale_tail": int(summary.stale_tail),
        "row_count": summary.row_count,
        "group_count": summary.group_count,
        "valid_rate": summary.valid_rate,
        "valid_group_rate": summary.valid_group_rate,
        "request_error_rate": summary.request_error_rate,
        "direct_valid": summary.score_means_valid.get("direct", math.nan),
        "focal_valid": summary.score_means_valid.get("focal", math.nan),
        "final_valid": summary.score_means_valid.get("final", math.nan),
        "objective_valid": objective_mean,
        "subjective_valid": subject_mean,
        "ui_spatial_score": summary.rubric_means_valid["ui_spatial_score"],
        "color_harmony_and_theme_fit": summary.rubric_means_valid["color_harmony_and_theme_fit"],
        "typography_rhythm_and_readability": summary.rubric_means_valid["typography_rhythm_and_readability"],
        "first_view_content_messaging": summary.rubric_means_valid["first_view_content_messaging"],
        "ui_weight": summary.weight_means_valid["ui_spatial_score"],
        "subjective_weight": mean([summary.weight_means_valid[key] for key in SUBJECTIVE_KEYS]),
        "max_weight_mean": summary.max_weight_mean,
        "top1_same_rate": summary.top1_same_rate,
        "spearman_mean": summary.spearman_mean,
        "z_sign_flip_rate": summary.z_sign_flip_rate,
        "direct_group_var_mean": summary.direct_group_var_mean,
        "focal_group_var_mean": summary.focal_group_var_mean,
        "status_counts": json.dumps(dict(sorted(summary.status_counts.items())), ensure_ascii=False, sort_keys=True),
    }


def aggregate_rows(rows: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    numeric_keys = [key for key in rows[0] if key not in {"experiment", "status_counts"}]
    output = {"experiment": experiment, "steps": ",".join(str(row["step"]) for row in rows), "step_count": len(rows)}
    for key in numeric_keys:
        output[key] = mean([safe_float(row[key]) for row in rows])
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, late_rows: list[dict[str, Any]], aggregate_rows_data: list[dict[str, Any]]) -> None:
    lines = [
        "# 后期 rollout 批量质量分析",
        "",
        "## 口径",
        "",
        "- 后期窗口只纳入非旧运行尾巴、`valid_rate` 达标、`request_error_rate` 未超阈值的 step。",
        "- rubric 使用 `valid_sample=1` 口径，避免 reward server 失败行 0 分污染。",
        "- `final` 只表示该实验实际训练 reward；baseline 的 `final=direct`，focal 的 `final=focal`，两者不是同一标尺。",
        "",
        "## 后期实验汇总",
        "",
        "| 实验 | steps | direct | focal | final | subjective | ui | valid_rate | req_err | max_w | top1_same | signflip |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate_rows_data, key=lambda item: item["experiment"]):
        lines.append(
            f"| `{row['experiment']}` | {row['steps']} | {row['direct_valid']:.3f} | "
            f"{row['focal_valid']:.3f} | {row['final_valid']:.3f} | {row['subjective_valid']:.3f} | "
            f"{row['ui_spatial_score']:.3f} | {row['valid_rate']:.3f} | {row['request_error_rate']:.3f} | "
            f"{row['max_weight_mean']:.3f} | {row['top1_same_rate']:.3f} | {row['z_sign_flip_rate']:.3f} |"
        )

    lines.extend(["", "## 逐 step 后期明细", ""])
    lines.append("| 实验 | step | direct | focal | final | subjective | ui | valid_rate | req_err | status |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in sorted(late_rows, key=lambda item: (item["experiment"], item["step"])):
        lines.append(
            f"| `{row['experiment']}` | {int(row['step'])} | {row['direct_valid']:.3f} | "
            f"{row['focal_valid']:.3f} | {row['final_valid']:.3f} | {row['subjective_valid']:.3f} | "
            f"{row['ui_spatial_score']:.3f} | {row['valid_rate']:.3f} | {row['request_error_rate']:.3f} | "
            f"`{row['status_counts']}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """主流程: 读取实验、生成逐步和后期汇总。"""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    late_rows: list[dict[str, Any]] = []
    aggregate_data: list[dict[str, Any]] = []

    for experiment_dir in list_experiment_dirs(args.rollout_root, args.experiments):
        if not experiment_dir.exists():
            continue
        summaries = summarize_experiment(experiment_dir, reset_seconds=args.mtime_reset_seconds)
        rows = [row_from_summary(summary) for summary in summaries]
        all_rows.extend(rows)
        eligible = [
            row
            for row in rows
            if not row["stale_tail"]
            and row["valid_rate"] >= args.min_valid_rate
            and row["request_error_rate"] <= args.max_request_error_rate
        ]
        selected = eligible[-args.late_window :]
        late_rows.extend(selected)
        if selected:
            aggregate_data.append(aggregate_rows(selected, experiment_dir.name))

    write_csv(args.out_dir / "step_quality.csv", all_rows)
    write_csv(args.out_dir / "late_step_quality.csv", late_rows)
    write_csv(args.out_dir / "late_experiment_summary.csv", aggregate_data)
    write_report(args.out_dir / "summary.md", late_rows, aggregate_data)
    print(f"wrote {args.out_dir / 'summary.md'}")
    print(f"experiments={len(aggregate_data)} late_steps={len(late_rows)}")


if __name__ == "__main__":
    main()
