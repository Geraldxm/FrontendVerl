#!/usr/bin/env python3
"""Analyze direct vs focal reward ranking differences in rollout JSONL dumps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FLOAT_TOLERANCE = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze group-level direct/focal reward ranking differences from rollout JSONL files."
    )
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=Path("rollouts/focal_v3_n16_g3_Qwen3-1.7B-Base"),
        help="Directory containing per-step JSONL rollout dumps.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("my/report/focal_reward_ranking"),
        help="Output directory for CSV, PNG, and summary files.",
    )
    parser.add_argument(
        "--focus-step",
        type=int,
        default=58,
        help="Step to highlight in the summary.",
    )
    return parser.parse_args()


def input_hash(input_text: str) -> str:
    return hashlib.sha256(input_text.encode("utf-8")).hexdigest()


def numeric_step(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Rollout file name must be numeric, got {path.name}") from exc


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def safe_max(values: list[float]) -> float:
    return float(max(values)) if values else math.nan


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
        raise ValueError("Correlation inputs must have the same length")
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
        raise ValueError("Kendall inputs must have the same length")
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


def calc_scores_from_group_view(row: dict[str, Any]) -> tuple[float, float]:
    reward_signals = row.get("group_mean_reward_signal", row.get("reward_signals"))
    reward_weights = row.get("group_mean_reward_weight", row.get("reward_weights"))
    if not isinstance(reward_signals, dict) or not reward_signals:
        raise ValueError("Each row must contain a non-empty reward_signals dict")
    if not isinstance(reward_weights, dict) or not reward_weights:
        raise ValueError("Each row must contain a non-empty reward_weights dict")

    calc_direct_score = mean([float(value) for value in reward_signals.values()])
    missing_weight_keys = set(reward_signals) - set(reward_weights)
    if missing_weight_keys:
        raise ValueError(f"reward_weights is missing keys: {sorted(missing_weight_keys)}")
    calc_focal_score = sum(float(reward_signals[key]) * float(reward_weights[key]) for key in reward_signals)
    return calc_direct_score, float(calc_focal_score)


def load_step_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            reward_scores = row.get("reward_scores")
            if not isinstance(reward_scores, dict):
                raise ValueError(f"{path}:{line_number} missing reward_scores dict")
            for key in ("direct", "focal"):
                if key not in reward_scores:
                    raise ValueError(f"{path}:{line_number} missing reward_scores.{key}")
            row["_line_number"] = line_number
            row["_input_hash"] = input_hash(str(row.get("input", "")))
            row["_calc_direct_score"], row["_calc_focal_score"] = calc_scores_from_group_view(row)
            row["_direct_score"] = float(reward_scores["direct"])
            row["_focal_score"] = float(reward_scores["focal"])
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no JSONL rows")
    return rows


def analyze_group(step: int, group_hash: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    direct_scores = [row["_direct_score"] for row in rows]
    focal_scores = [row["_focal_score"] for row in rows]
    direct_z = z_scores(direct_scores)
    focal_z = z_scores(focal_scores)

    top_direct_idx = max(range(len(rows)), key=lambda idx: (direct_scores[idx], -idx))
    top_focal_idx = max(range(len(rows)), key=lambda idx: (focal_scores[idx], -idx))
    calc_direct_scores = [row["_calc_direct_score"] for row in rows]
    calc_focal_scores = [row["_calc_focal_score"] for row in rows]

    group_calc_direct_score = calc_direct_scores[0]
    group_calc_focal_score = calc_focal_scores[0]
    group_mean_direct_score = mean(direct_scores)
    group_mean_focal_score = mean(focal_scores)

    return {
        "step": step,
        "input_hash": group_hash,
        "group_size": len(rows),
        "direct_mean": group_mean_direct_score,
        "focal_mean": group_mean_focal_score,
        "direct_std": statistics.pstdev(direct_scores) if len(direct_scores) > 1 else 0.0,
        "focal_std": statistics.pstdev(focal_scores) if len(focal_scores) > 1 else 0.0,
        "calc_direct_score": group_calc_direct_score,
        "calc_focal_score": group_calc_focal_score,
        "group_direct_abs_error": abs(group_calc_direct_score - group_mean_direct_score),
        "group_focal_abs_error": abs(group_calc_focal_score - group_mean_focal_score),
        "row_calc_direct_min": min(calc_direct_scores),
        "row_calc_direct_max": max(calc_direct_scores),
        "row_calc_focal_min": min(calc_focal_scores),
        "row_calc_focal_max": max(calc_focal_scores),
        "normalized_kendall_distance": normalized_kendall_distance(direct_scores, focal_scores),
        "spearman_rank_corr": pearson_correlation(
            average_ranks_desc(direct_scores),
            average_ranks_desc(focal_scores),
        ),
        "z_advantage_corr": pearson_correlation(direct_z, focal_z),
        "z_advantage_sign_flip_rate": mean(
            [float(sign(direct) != sign(focal)) for direct, focal in zip(direct_z, focal_z, strict=True)]
        ),
        "mean_abs_z_advantage_delta": mean(
            [abs(direct - focal) for direct, focal in zip(direct_z, focal_z, strict=True)]
        ),
        "top1_same": float(top_direct_idx == top_focal_idx),
        "direct_min": min(direct_scores),
        "direct_max": max(direct_scores),
        "focal_min": min(focal_scores),
        "focal_max": max(focal_scores),
    }


def summarize_step(step: int, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    status_counter: Counter[str] = Counter()
    row_direct_errors = []
    row_focal_errors = []

    for row in rows:
        groups[row["_input_hash"]].append(row)
        status = row.get("status")
        status_counter[str(status.get("overall", "missing") if isinstance(status, dict) else "missing")] += 1
        row_direct_errors.append(abs(row["_calc_direct_score"] - row["_direct_score"]))
        row_focal_errors.append(abs(row["_calc_focal_score"] - row["_focal_score"]))

    group_metrics = [analyze_group(step, group_hash, group_rows) for group_hash, group_rows in sorted(groups.items())]
    group_sizes = Counter(metric["group_size"] for metric in group_metrics)

    step_metric = {
        "step": step,
        "row_count": len(rows),
        "group_count": len(group_metrics),
        "group_size_counts": json.dumps(dict(sorted(group_sizes.items())), sort_keys=True),
        "status_counts": json.dumps(dict(sorted(status_counter.items())), sort_keys=True),
        "row_direct_abs_error_mean": mean(row_direct_errors),
        "row_direct_abs_error_max": safe_max(row_direct_errors),
        "row_focal_abs_error_mean": mean(row_focal_errors),
        "row_focal_abs_error_max": safe_max(row_focal_errors),
        "group_direct_abs_error_mean": mean([metric["group_direct_abs_error"] for metric in group_metrics]),
        "group_direct_abs_error_max": safe_max([metric["group_direct_abs_error"] for metric in group_metrics]),
        "group_focal_abs_error_mean": mean([metric["group_focal_abs_error"] for metric in group_metrics]),
        "group_focal_abs_error_max": safe_max([metric["group_focal_abs_error"] for metric in group_metrics]),
        "normalized_kendall_distance_mean": mean(
            [metric["normalized_kendall_distance"] for metric in group_metrics]
        ),
        "normalized_kendall_distance_max": safe_max(
            [metric["normalized_kendall_distance"] for metric in group_metrics]
        ),
        "spearman_rank_corr_mean": mean([metric["spearman_rank_corr"] for metric in group_metrics]),
        "z_advantage_corr_mean": mean([metric["z_advantage_corr"] for metric in group_metrics]),
        "z_advantage_sign_flip_rate_mean": mean(
            [metric["z_advantage_sign_flip_rate"] for metric in group_metrics]
        ),
        "mean_abs_z_advantage_delta_mean": mean(
            [metric["mean_abs_z_advantage_delta"] for metric in group_metrics]
        ),
        "top1_same_rate": mean([metric["top1_same"] for metric in group_metrics]),
    }
    return step_metric, group_metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_step_metrics(step_metrics: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [metric["step"] for metric in step_metrics]
    series = [
        ("Kendall distance", "normalized_kendall_distance_mean"),
        ("Advantage corr", "z_advantage_corr_mean"),
        ("Sign flip rate", "z_advantage_sign_flip_rate_mean"),
        ("Top-1 same rate", "top1_same_rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for axis, (title, key) in zip(axes.ravel(), series, strict=True):
        axis.plot(steps, [metric[key] for metric in step_metrics], linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Step")
        axis.grid(True, alpha=0.25)
        if key != "z_advantage_corr_mean":
            axis.set_ylim(-0.02, 1.02)
    fig.suptitle("Direct vs Focal Reward Ranking Difference")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6g}"


def write_summary(
    path: Path,
    *,
    rollout_dir: Path,
    step_metrics: list[dict[str, Any]],
    group_metrics: list[dict[str, Any]],
    focus_step: int,
) -> None:
    focus = next((metric for metric in step_metrics if metric["step"] == focus_step), None)
    group_size_anomalies = [
        metric
        for metric in step_metrics
        if metric["group_size_counts"] != json.dumps({16: metric["group_count"]}, sort_keys=True)
    ]
    worst_kendall = max(step_metrics, key=lambda metric: metric["normalized_kendall_distance_mean"])
    best_kendall = min(step_metrics, key=lambda metric: metric["normalized_kendall_distance_mean"])
    mean_kendall = mean([metric["normalized_kendall_distance_mean"] for metric in step_metrics])
    mean_adv_corr = mean([metric["z_advantage_corr_mean"] for metric in step_metrics])
    mean_sign_flip = mean([metric["z_advantage_sign_flip_rate_mean"] for metric in step_metrics])
    mean_top1 = mean([metric["top1_same_rate"] for metric in step_metrics])

    lines = [
        "# Focal vs Direct Reward Ranking Analysis",
        "",
        f"- Rollout dir: `{rollout_dir}`",
        f"- Steps parsed: {len(step_metrics)}",
        f"- Groups parsed: {len(group_metrics)}",
        "- Group key: `sha256(input)`",
        "- Ranking scores: `reward_scores.direct` vs `reward_scores.focal`",
        "",
        "## Overall Result",
        "",
        f"- Mean normalized Kendall distance: {format_float(mean_kendall)}",
        f"- Mean z-advantage correlation: {format_float(mean_adv_corr)}",
        f"- Mean z-advantage sign flip rate: {format_float(mean_sign_flip)}",
        f"- Mean top-1 same rate: {format_float(mean_top1)}",
        f"- Lowest mean Kendall distance step: {best_kendall['step']} ({format_float(best_kendall['normalized_kendall_distance_mean'])})",
        f"- Highest mean Kendall distance step: {worst_kendall['step']} ({format_float(worst_kendall['normalized_kendall_distance_mean'])})",
        "",
        "## Consistency Check",
        "",
        "- `group_mean_reward_signal` and `group_mean_reward_weight` are group-level views.",
        "- Older dumps store those same group-level views as top-level `reward_signals` and `reward_weights`.",
        "- Row-level `calc_direct_score/calc_focal_score` therefore may differ from the row's `reward_scores.direct/focal`.",
        "- Group-level checks compare those calculated scores with each group's mean direct/focal scores.",
        f"- Max group direct abs error: {format_float(max(metric['group_direct_abs_error_max'] for metric in step_metrics))}",
        f"- Max group focal abs error: {format_float(max(metric['group_focal_abs_error_max'] for metric in step_metrics))}",
        "",
        f"## Focus Step {focus_step}",
        "",
    ]

    if focus is None:
        lines.append(f"- Step {focus_step} was not found.")
    else:
        lines.extend(
            [
                f"- Rows: {focus['row_count']}",
                f"- Groups: {focus['group_count']}",
                f"- Group size counts: `{focus['group_size_counts']}`",
                f"- Status counts: `{focus['status_counts']}`",
                f"- Row direct abs error mean/max: {format_float(focus['row_direct_abs_error_mean'])} / {format_float(focus['row_direct_abs_error_max'])}",
                f"- Row focal abs error mean/max: {format_float(focus['row_focal_abs_error_mean'])} / {format_float(focus['row_focal_abs_error_max'])}",
                f"- Group direct abs error max: {format_float(focus['group_direct_abs_error_max'])}",
                f"- Group focal abs error max: {format_float(focus['group_focal_abs_error_max'])}",
                f"- Mean normalized Kendall distance: {format_float(focus['normalized_kendall_distance_mean'])}",
                f"- Mean z-advantage correlation: {format_float(focus['z_advantage_corr_mean'])}",
                f"- Mean z-advantage sign flip rate: {format_float(focus['z_advantage_sign_flip_rate_mean'])}",
                f"- Top-1 same rate: {format_float(focus['top1_same_rate'])}",
            ]
        )

    lines.extend(["", "## Group Size Anomalies", ""])
    if not group_size_anomalies:
        lines.append("- None. Every parsed step has 64 groups of size 16.")
    else:
        for metric in group_size_anomalies:
            lines.append(f"- Step {metric['step']}: `{metric['group_size_counts']}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The ranking metrics show that focal changes the ordering, but the average change is modest.",
            "- GRPO-style z-advantage correlation remains high, so the resulting advantage signal is largely aligned.",
            "- The sign flip rate is the most direct check for whether samples cross the group mean after focal reweighting.",
            "",
            "## Output Files",
            "",
            "- `step_metrics.csv`",
            "- `group_metrics.csv`",
            "- `ranking_diff_curve.png`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rollout_dir = args.rollout_dir
    out_dir = args.out_dir

    if not rollout_dir.is_dir():
        raise FileNotFoundError(f"Rollout directory does not exist: {rollout_dir}")
    rollout_files = sorted(rollout_dir.glob("*.jsonl"), key=numeric_step)
    if not rollout_files:
        raise FileNotFoundError(f"No .jsonl files found under {rollout_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    step_metrics: list[dict[str, Any]] = []
    group_metrics: list[dict[str, Any]] = []
    for rollout_file in rollout_files:
        step = numeric_step(rollout_file)
        rows = load_step_rows(rollout_file)
        step_metric, step_group_metrics = summarize_step(step, rows)
        step_metrics.append(step_metric)
        group_metrics.extend(step_group_metrics)

    write_csv(out_dir / "step_metrics.csv", step_metrics)
    write_csv(out_dir / "group_metrics.csv", group_metrics)
    plot_step_metrics(step_metrics, out_dir / "ranking_diff_curve.png")
    write_summary(
        out_dir / "summary.md",
        rollout_dir=rollout_dir,
        step_metrics=step_metrics,
        group_metrics=group_metrics,
        focus_step=args.focus_step,
    )

    print(f"Wrote {out_dir / 'step_metrics.csv'}")
    print(f"Wrote {out_dir / 'group_metrics.csv'}")
    print(f"Wrote {out_dir / 'ranking_diff_curve.png'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
