#!/usr/bin/env python3
"""
分析四种方法在后期训练中 VLM 得分停止上涨的原因。
从 rollout JSONL 中提取训练 progression 数据。
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

def load_rollout_data(rollout_dir, max_steps=None):
    """加载 rollout 目录中的 JSONL 文件，返回按 step 排序的数据"""
    data = {}
    rollout_path = Path(rollout_dir)

    for f in sorted(rollout_path.glob("*.jsonl")):
        try:
            step_num = int(f.stem)
            if max_steps and step_num > max_steps:
                continue
        except ValueError:
            continue

        step_data = []
        with open(f, 'r') as fh:
            for line in fh:
                try:
                    step_data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        data[step_num] = step_data

    return data

def analyze_step_data(step_data):
    """分析单步数据的统计量"""
    if not step_data:
        return None

    stats = {
        'total_samples': len(step_data),
        'valid_samples': 0,
        'llm_errors': 0,
        'render_failures': 0,
        'success': 0,
        'direct_scores': [],
        'instructional_alignment': [],
        'visual_elements': [],
        'layout_and_cohesion': [],
        'format_score': [],
        'console_errors': [],
        'network_violations': [],
        'a11y_score': [],
        'element_hit_rate': [],
        'output_lengths': [],
    }

    for sample in step_data:
        reward_scores = sample.get('reward_scores', {})
        sample_signals = sample.get('sample_reward_signal', {})

        if reward_scores.get('valid_sample', 0) == 1:
            stats['valid_samples'] += 1
            stats['direct_scores'].append(reward_scores.get('direct', 0))
            stats['instructional_alignment'].append(sample_signals.get('instructional_alignment', 0))
            stats['visual_elements'].append(sample_signals.get('visual_elements', 0))
            stats['layout_and_cohesion'].append(sample_signals.get('layout_and_cohesion', 0))
            stats['format_score'].append(sample_signals.get('format_score', 0))
            stats['console_errors'].append(sample_signals.get('console_errors', 0))
            stats['network_violations'].append(sample_signals.get('network_violations', 0))
            stats['a11y_score'].append(sample_signals.get('a11y_score', 0))
            stats['element_hit_rate'].append(sample_signals.get('element_hit_rate', 0))

        if reward_scores.get('is_llm_generation_error', 0) == 1:
            stats['llm_errors'] += 1

        if sample.get('overall_status', '') == 'success':
            stats['success'] += 1

        output = sample.get('output', '')
        stats['output_lengths'].append(len(output))

    # 计算统计量
    result = {
        'total_samples': stats['total_samples'],
        'valid_rate': stats['valid_samples'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
        'success_rate': stats['success'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
        'llm_error_rate': stats['llm_errors'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
        'mean_direct': np.mean(stats['direct_scores']) if stats['direct_scores'] else 0,
        'std_direct': np.std(stats['direct_scores']) if stats['direct_scores'] else 0,
        'mean_instructional': np.mean(stats['instructional_alignment']) if stats['instructional_alignment'] else 0,
        'mean_visual': np.mean(stats['visual_elements']) if stats['visual_elements'] else 0,
        'mean_layout': np.mean(stats['layout_and_cohesion']) if stats['layout_and_cohesion'] else 0,
        'mean_format': np.mean(stats['format_score']) if stats['format_score'] else 0,
        'mean_console': np.mean(stats['console_errors']) if stats['console_errors'] else 0,
        'mean_network': np.mean(stats['network_violations']) if stats['network_violations'] else 0,
        'mean_a11y': np.mean(stats['a11y_score']) if stats['a11y_score'] else 0,
        'mean_element': np.mean(stats['element_hit_rate']) if stats['element_hit_rate'] else 0,
        'mean_output_length': np.mean(stats['output_lengths']) if stats['output_lengths'] else 0,
        'std_instructional': np.std(stats['instructional_alignment']) if stats['instructional_alignment'] else 0,
        'std_visual': np.std(stats['visual_elements']) if stats['visual_elements'] else 0,
        'std_layout': np.std(stats['layout_and_cohesion']) if stats['layout_and_cohesion'] else 0,
    }

    return result

def analyze_method(method_name, rollout_dir, max_steps=None):
    """分析单个方法的训练 progression"""
    data = load_rollout_data(rollout_dir, max_steps)

    if not data:
        print(f"  {method_name}: 没有数据")
        return

    print(f"\n{'='*60}")
    print(f"方法: {method_name}")
    print(f"数据目录: {rollout_dir}")
    print(f"总步数: {len(data)}")
    print(f"{'='*60}")

    # 提取各步统计量
    steps_analysis = []
    for step in sorted(data.keys()):
        step_stats = analyze_step_data(data[step])
        if step_stats:
            step_stats['step'] = step
            steps_analysis.append(step_stats)

    if not steps_analysis:
        print("  无有效数据")
        return

    # 输出 progression
    print(f"\n训练 progression（VLM 主观分）:")
    print(f"{'step':>6} | {'direct':>8} | {'instruct':>8} | {'visual':>8} | {'layout':>8} | {'valid_rate':>10} | {'output_len':>10}")
    print("-" * 80)

    for s in steps_analysis:
        print(f"{s['step']:6d} | {s['mean_direct']:8.3f} | {s['mean_instructional']:8.3f} | {s['mean_visual']:8.3f} | {s['mean_layout']:8.3f} | {s['valid_rate']:10.3f} | {s['mean_output_length']:10.1f}")

    # 分析后期趋势
    if len(steps_analysis) >= 10:
        mid = len(steps_analysis) // 2
        early = steps_analysis[:mid]
        late = steps_analysis[mid:]

        early_direct = np.mean([s['mean_direct'] for s in early])
        late_direct = np.mean([s['mean_direct'] for s in late])
        early_instruct = np.mean([s['mean_instructional'] for s in early])
        late_instruct = np.mean([s['mean_instructional'] for s in late])
        early_visual = np.mean([s['mean_visual'] for s in early])
        late_visual = np.mean([s['mean_visual'] for s in late])
        early_layout = np.mean([s['mean_layout'] for s in early])
        late_layout = np.mean([s['mean_layout'] for s in late])

        print(f"\n后期 vs 早期对比:")
        print(f"  direct: {early_direct:.3f} -> {late_direct:.3f} (变化: {late_direct-early_direct:+.3f})")
        print(f"  instructional: {early_instruct:.3f} -> {late_instruct:.3f} (变化: {late_instruct-early_instruct:+.3f})")
        print(f"  visual: {early_visual:.3f} -> {late_visual:.3f} (变化: {late_visual-early_visual:+.3f})")
        print(f"  layout: {early_layout:.3f} -> {late_layout:.3f} (变化: {late_layout-early_layout:+.3f})")

    # 分析后期（后 20%）的趋势
    if len(steps_analysis) >= 5:
        late_steps = steps_analysis[-int(len(steps_analysis)*0.2):]
        if late_steps:
            late_direct_values = [s['mean_direct'] for s in late_steps]
            late_instruct_values = [s['mean_instructional'] for s in late_steps]
            late_visual_values = [s['mean_visual'] for s in late_steps]
            late_layout_values = [s['mean_layout'] for s in late_steps]

            print(f"\n后期（最后 {len(late_steps)} 步）统计:")
            print(f"  direct: {np.mean(late_direct_values):.3f} ± {np.std(late_direct_values):.3f}")
            print(f"  instructional: {np.mean(late_instruct_values):.3f} ± {np.std(late_instruct_values):.3f}")
            print(f"  visual: {np.mean(late_visual_values):.3f} ± {np.std(late_visual_values):.3f}")
            print(f"  layout: {np.mean(late_layout_values):.3f} ± {np.std(late_layout_values):.3f}")

    return steps_analysis

def analyze_advantage_collapse(rollout_dir, max_steps=None):
    """分析 advantage 信号坍塌"""
    data = load_rollout_data(rollout_dir, max_steps)

    if not data:
        return

    print(f"\n  Advantage 信号分析:")
    print(f"  {'step':>6} | {'group_std_mean':>12} | {'dynamic_weight':>12} | {'adv_signal':>12}")
    print(f"  {'-'*60}")

    for step in sorted(data.keys())[-10:]:  # 只看最后 10 步
        step_data = data[step]

        # 提取 DVAO 相关字段
        group_stds = []
        dynamic_weights = []
        advantages = []

        for sample in step_data:
            # 尝试提取 group_dvao_reward_std
            group_dvao_std = sample.get('reward_others', {}).get('var', {})
            if group_dvao_std:
                std_vals = [v for v in group_dvao_std.values() if isinstance(v, (int, float))]
                if std_vals:
                    group_stds.append(np.mean(std_vals))

            # 尝试提取 group_dvao_weight
            group_dvao_weight = sample.get('group_mean_reward_weight', {})
            if group_dvao_weight:
                weight_vals = [v for v in group_dvao_weight.values() if isinstance(v, (int, float))]
                if weight_vals:
                    dynamic_weights.append(np.mean(weight_vals))

            # 尝试提取 advantage
            # advantage 通常在 reward_scores 中，但可能需要从其他字段提取
            # 这里用 direct score 的方差作为 proxy
            direct_score = sample.get('reward_scores', {}).get('direct', 0)
            if direct_score:
                advantages.append(direct_score)

        if group_stds:
            print(f"  {step:6d} | {np.mean(group_stds):12.4f} | {np.mean(dynamic_weights):12.4f} | {np.mean(advantages):12.4f}")

def analyze_entropy_loss(rollout_dir, max_steps=None):
    """分析输出多样性（通过输出长度和内容变化）"""
    data = load_rollout_data(rollout_dir, max_steps)

    if not data:
        return

    print(f"\n  输出多样性分析:")
    print(f"  {'step':>6} | {'output_len_mean':>12} | {'output_len_std':>12} | {'unique_patterns':>12}")
    print(f"  {'-'*60}")

    for step in sorted(data.keys())[-10:]:  # 只看最后 10 步
        step_data = data[step]

        output_lengths = []
        unique_patterns = set()

        for sample in step_data:
            output = sample.get('output', '')
            output_lengths.append(len(output))

            # 简单的 pattern 检测
            if output:
                # 提取前 100 个字符作为 pattern
                pattern = output[:100]
                unique_patterns.add(pattern)

        if output_lengths:
            print(f"  {step:6d} | {np.mean(output_lengths):12.1f} | {np.std(output_lengths):12.1f} | {len(unique_patterns):12d}")

def main():
    """主函数"""
    base_dir = Path("/inspire/hdd/global_user/gexinmu-253108100065/Repos/FrontendVerl/rollouts")

    # 定义四种方法的目录
    methods = {
        'baseline': '0605_v6_baseline_q3_17_n16_Qwen3-1.7B',
        'focal': '0605_v6_focal_q3_17_n16_Qwen3-1.7B',
        'dvao': '0606_v6_dvao_q3_17_n16_Qwen3-1.7B',
        'focal_dvao': '0606_v6_focal_dvao_q3_17_n16_Qwen3-1.7B',
    }

    print("=" * 80)
    print("四种方法在后期训练中 VLM 得分停止上涨的原因分析")
    print("=" * 80)

    all_results = {}

    for method_name, method_dir in methods.items():
        rollout_dir = base_dir / method_dir
        if rollout_dir.exists():
            result = analyze_method(method_name, rollout_dir, max_steps=100)
            all_results[method_name] = result
        else:
            print(f"\n{method_name}: 目录不存在 {rollout_dir}")

    # 分析 advantage 信号坍塌
    print("\n" + "=" * 80)
    print("Advantage 信号坍塌分析")
    print("=" * 80)

    for method_name, method_dir in methods.items():
        rollout_dir = base_dir / method_dir
        if rollout_dir.exists():
            print(f"\n{method_name}:")
            analyze_advantage_collapse(rollout_dir, max_steps=100)

    # 分析输出多样性
    print("\n" + "=" * 80)
    print("输出多样性分析")
    print("=" * 80)

    for method_name, method_dir in methods.items():
        rollout_dir = base_dir / method_dir
        if rollout_dir.exists():
            print(f"\n{method_name}:")
            analyze_entropy_loss(rollout_dir, max_steps=100)

    # 总结
    print("\n" + "=" * 80)
    print("综合结论")
    print("=" * 80)

    if all_results:
        print("\n各方法后期 VLM 得分趋势:")
        for method_name, result in all_results.items():
            if result and len(result) >= 10:
                late_steps = result[-int(len(result)*0.2):]
                early_steps = result[:int(len(result)*0.2)]

                early_direct = np.mean([s['mean_direct'] for s in early_steps])
                late_direct = np.mean([s['mean_direct'] for s in late_steps])
                early_instruct = np.mean([s['mean_instructional'] for s in early_steps])
                late_instruct = np.mean([s['mean_instructional'] for s in late_steps])

                print(f"  {method_name:12}: direct {early_direct:.3f} -> {late_direct:.3f} ({late_direct-early_direct:+.3f}), "
                      f"instruct {early_instruct:.3f} -> {late_instruct:.3f} ({late_instruct-early_instruct:+.3f})")

if __name__ == "__main__":
    main()