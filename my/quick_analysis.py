#!/usr/bin/env python3
"""快速分析：提取四种方法后期训练的 VLM 得分趋势"""
import json
import os
from pathlib import Path
import numpy as np

def load_step_data(rollout_dir, step_num):
    """加载单个 step 的数据"""
    f = Path(rollout_dir) / f"{step_num}.jsonl"
    if not f.exists():
        return None
    data = []
    with open(f, 'r') as fh:
        for line in fh:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
    return data

def analyze_step(data):
    """分析单步数据"""
    if not data:
        return None

    direct = []
    instr = []
    visual = []
    layout = []

    for s in data:
        rs = s.get('reward_scores', {})
        sigs = s.get('sample_reward_signal', {})
        if rs.get('valid_sample', 0) == 1:
            direct.append(rs.get('direct', 0))
            instr.append(sigs.get('instructional_alignment', 0))
            visual.append(sigs.get('visual_elements', 0))
            layout.append(sigs.get('layout_and_cohesion', 0))

    return {
        'direct': np.mean(direct) if direct else 0,
        'instr': np.mean(instr) if instr else 0,
        'visual': np.mean(visual) if visual else 0,
        'layout': np.mean(layout) if layout else 0,
        'n': len(direct)
    }

def main():
    base = Path("/inspire/hdd/global_user/gexinmu-253108100065/Repos/FrontendVerl/rollouts")

    methods = {
        'baseline': '0605_v6_baseline_q3_17_n16_Qwen3-1.7B',
        'focal': '0605_v6_focal_q3_17_n16_Qwen3-1.7B',
        'dvao': '0606_v6_dvao_q3_17_n16_Qwen3-1.7B',
        'focal_dvao': '0606_v6_focal_dvao_q3_17_n16_Qwen3-1.7B',
    }

    for name, dirname in methods.items():
        d = base / dirname
        if not d.exists():
            print(f"{name}: 目录不存在")
            continue

        # 找出所有 step
        steps = sorted([int(f.stem) for f in d.glob("*.jsonl") if f.stem.isdigit()])
        if not steps:
            print(f"{name}: 无数据")
            continue

        print(f"\n{'='*60}")
        print(f"方法: {name} | 步数: {len(steps)} | 范围: {steps[0]}-{steps[-1]}")
        print(f"{'='*60}")

        # 分析早期（前20%）、中期（中间20%）、后期（后20%）
        early_steps = steps[:len(steps)//5]
        mid_steps = steps[len(steps)*2//5:len(steps)*3//5]
        late_steps = steps[-len(steps)//5:]

        early_data = []
        mid_data = []
        late_data = []

        for s in early_steps:
            sd = load_step_data(d, s)
            if sd:
                r = analyze_step(sd)
                if r:
                    early_data.append(r)

        for s in mid_steps:
            sd = load_step_data(d, s)
            if sd:
                r = analyze_step(sd)
                if r:
                    mid_data.append(r)

        for s in late_steps:
            sd = load_step_data(d, s)
            if sd:
                r = analyze_step(sd)
                if r:
                    late_data.append(r)

        if early_data and late_data:
            early_direct = np.mean([x['direct'] for x in early_data])
            mid_direct = np.mean([x['direct'] for x in mid_data]) if mid_data else 0
            late_direct = np.mean([x['direct'] for x in late_data])

            early_instr = np.mean([x['instr'] for x in early_data])
            mid_instr = np.mean([x['instr'] for x in mid_data]) if mid_data else 0
            late_instr = np.mean([x['instr'] for x in late_data])

            early_visual = np.mean([x['visual'] for x in early_data])
            mid_visual = np.mean([x['visual'] for x in mid_data]) if mid_data else 0
            late_visual = np.mean([x['visual'] for x in late_data])

            early_layout = np.mean([x['layout'] for x in early_data])
            mid_layout = np.mean([x['layout'] for x in mid_data]) if mid_data else 0
            late_layout = np.mean([x['layout'] for x in late_data])

            print(f"  {'阶段':>6} | {'direct':>8} | {'instruct':>8} | {'visual':>8} | {'layout':>8}")
            print(f"  {'-'*50}")
            print(f"  {'早期':>6} | {early_direct:8.3f} | {early_instr:8.3f} | {early_visual:8.3f} | {early_layout:8.3f}")
            print(f"  {'中期':>6} | {mid_direct:8.3f} | {mid_instr:8.3f} | {mid_visual:8.3f} | {mid_layout:8.3f}")
            print(f"  {'后期':>6} | {late_direct:8.3f} | {late_instr:8.3f} | {late_visual:8.3f} | {late_layout:8.3f}")

            print(f"\n  变化趋势:")
            print(f"    direct:   {early_direct:.3f} -> {mid_direct:.3f} -> {late_direct:.3f} (Δ={late_direct-early_direct:+.3f})")
            print(f"    instruct: {early_instr:.3f} -> {mid_instr:.3f} -> {late_instr:.3f} (Δ={late_instr-early_instr:+.3f})")
            print(f"    visual:   {early_visual:.3f} -> {mid_visual:.3f} -> {late_visual:.3f} (Δ={late_visual-early_visual:+.3f})")
            print(f"    layout:   {early_layout:.3f} -> {mid_layout:.3f} -> {late_layout:.3f} (Δ={late_layout-early_layout:+.3f})")

        # 分析后期（后30%）的方差和稳定性
        if late_data:
            late_directs = [x['direct'] for x in late_data]
            late_instrs = [x['instr'] for x in late_data]
            late_visuals = [x['visual'] for x in late_data]
            late_layouts = [x['layout'] for x in late_data]

            print(f"\n  后期稳定性（最后 {len(late_data)} 步）:")
            print(f"    direct:   {np.mean(late_directs):.3f} ± {np.std(late_directs):.3f}")
            print(f"    instruct: {np.mean(late_instrs):.3f} ± {np.std(late_instrs):.3f}")
            print(f"    visual:   {np.mean(late_visuals):.3f} ± {np.std(late_visuals):.3f}")
            print(f"    layout:   {np.mean(late_layouts):.3f} ± {np.std(late_layouts):.3f}")

if __name__ == "__main__":
    main()