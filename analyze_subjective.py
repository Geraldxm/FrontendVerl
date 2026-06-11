#!/usr/bin/env python3
import json
import os
import glob
import statistics

base = 'rollouts'
dirs = {
    '0527_baseline': '0527_v5_baseline_q3_17_n16_Qwen3-1.7B',
    '0527_focal': '0527_v5_focal_q3_17_n16_e005_t5_g3_clip005_03_Qwen3-1.7B',
    '0603_dvao': '0603_dvao_n16_Qwen3-1.7B',
    '0604_focal_dvao': '0604_focal_dvao_n16_Qwen3-1.7B',
    '0605_baseline': '0605_v6_baseline_q3_17_n16_Qwen3-1.7B',
    '0605_focal': '0605_v6_focal_q3_17_n16_Qwen3-1.7B',
    '0606_dvao': '0606_v6_dvao_q3_17_n16_Qwen3-1.7B',
    '0606_focal_dvao': '0606_v6_focal_dvao_q3_17_n16_Qwen3-1.7B',
}

subjective_dims = ['instructional_alignment', 'visual_elements', 'layout_and_cohesion']
objective_dims = ['format_score', 'console_errors', 'network_violations', 'a11y_score', 'element_hit_rate']

results = {}

for label, dirname in dirs.items():
    path = os.path.join(base, dirname)
    if not os.path.isdir(path):
        print(f'WARN: {path} not found')
        continue

    jsonl_files = sorted(glob.glob(os.path.join(path, '*.jsonl')))

    subj_scores = {d: [] for d in subjective_dims}
    obj_scores = {d: [] for d in objective_dims}
    total_samples = 0
    valid_samples = 0

    for f in jsonl_files:
        with open(f, 'r') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except:
                    continue

                total_samples += 1

                if record.get('reward_scores', {}).get('valid_sample', 0) == 1:
                    valid_samples += 1

                signal = record.get('sample_reward_signal', {})
                if not signal:
                    continue

                for d in subjective_dims:
                    if d in signal and signal[d] is not None:
                        subj_scores[d].append(signal[d])

                for d in objective_dims:
                    if d in signal and signal[d] is not None:
                        obj_scores[d].append(signal[d])

    subj_avgs = {}
    for d in subjective_dims:
        subj_avgs[d] = statistics.mean(subj_scores[d]) if subj_scores[d] else 0

    obj_avgs = {}
    for d in objective_dims:
        obj_avgs[d] = statistics.mean(obj_scores[d]) if obj_scores[d] else 0

    all_subj = []
    for d in subjective_dims:
        all_subj.extend(subj_scores[d])
    overall_subj_avg = statistics.mean(all_subj) if all_subj else 0

    results[label] = {
        'total': total_samples,
        'valid': valid_samples,
        'valid_rate': valid_samples / total_samples * 100 if total_samples > 0 else 0,
        'subj_avgs': subj_avgs,
        'obj_avgs': obj_avgs,
        'overall_subj_avg': overall_subj_avg,
    }

# Print results
print('='*100)
print('主观分数分析（去除前5个客观分数：format_score, console_errors, network_violations, a11y_score, element_hit_rate）')
print('='*100)

print()
print(f"{'实验':<25} {'样本数':>6} {'有效数':>6} {'有效率':>7} | {'总体主观均分':>10} | {'instructional_alignment':>22} | {'visual_elements':>15} | {'layout_and_cohesion':>18}")
print('-'*140)

for label in ['0527_baseline', '0527_focal', '0603_dvao', '0604_focal_dvao', '0605_baseline', '0605_focal', '0606_dvao', '0606_focal_dvao']:
    if label in results:
        r = results[label]
        print(f"{label:<25} {r['total']:>6} {r['valid']:>6} {r['valid_rate']:>6.1f}% | {r['overall_subj_avg']:>10.3f} | {r['subj_avgs']['instructional_alignment']:>22.3f} | {r['subj_avgs']['visual_elements']:>15.3f} | {r['subj_avgs']['layout_and_cohesion']:>18.3f}")

print()
print('='*100)
print('客观分数对比（仅供参考）')
print('='*100)
print()
print(f"{'实验':<25} | {'format_score':>12} | {'console_errors':>14} | {'network_violations':>17} | {'a11y_score':>10} | {'element_hit_rate':>15}")
print('-'*110)

for label in ['0527_baseline', '0527_focal', '0603_dvao', '0604_focal_dvao', '0605_baseline', '0605_focal', '0606_dvao', '0606_focal_dvao']:
    if label in results:
        r = results[label]
        print(f"{label:<25} | {r['obj_avgs']['format_score']:>12.3f} | {r['obj_avgs']['console_errors']:>14.3f} | {r['obj_avgs']['network_violations']:>17.3f} | {r['obj_avgs']['a11y_score']:>10.3f} | {r['obj_avgs']['element_hit_rate']:>15.3f}")

# Compare focal/dvao vs baseline
print()
print('='*100)
print('Focal/DVAO vs Baseline 主观分数对比')
print('='*100)

if '0527_baseline' in results and '0527_focal' in results:
    b = results['0527_baseline']['overall_subj_avg']
    f = results['0527_focal']['overall_subj_avg']
    print(f'0527组: baseline={b:.3f}  focal={f:.3f}  diff={f-b:+.3f}')

if '0603_dvao' in results and '0604_focal_dvao' in results:
    d = results['0603_dvao']['overall_subj_avg']
    fd = results['0604_focal_dvao']['overall_subj_avg']
    print(f'0603-0604组: dvao={d:.3f}  focal_dvao={fd:.3f}  diff={fd-d:+.3f}')

if '0605_baseline' in results and '0606_focal_dvao' in results:
    b = results['0605_baseline']['overall_subj_avg']
    fd = results['0606_focal_dvao']['overall_subj_avg']
    print(f'0605-0606组: baseline={b:.3f}  focal_dvao={fd:.3f}  diff={fd-b:+.3f}')

if '0605_focal' in results and '0605_baseline' in results:
    b = results['0605_baseline']['overall_subj_avg']
    f = results['0605_focal']['overall_subj_avg']
    print(f'0605组: baseline={b:.3f}  focal={f:.3f}  diff={f-b:+.3f}')

if '0606_dvao' in results and '0605_baseline' in results:
    b = results['0605_baseline']['overall_subj_avg']
    d = results['0606_dvao']['overall_subj_avg']
    print(f'0606组: baseline={b:.3f}  dvao={d:.3f}  diff={d-b:+.3f}')

# Per-dimension comparison
print()
print('各维度对比:')
print()
for label in ['0527_focal', '0603_dvao', '0604_focal_dvao', '0605_focal', '0606_dvao', '0606_focal_dvao']:
    if label not in results:
        continue
    baseline_key = '0527_baseline' if label.startswith('0527') else '0605_baseline'
    if baseline_key not in results:
        continue
    print(f'{label} vs {baseline_key}:')
    for dim in subjective_dims:
        b = results[baseline_key]['subj_avgs'][dim]
        f = results[label]['subj_avgs'][dim]
        diff = f - b
        mark = '↑' if diff > 0 else '↓'
        print(f"  {dim:<25}: baseline={b:.3f}  {label}={f:.3f}  diff={diff:+.3f}  {mark}")
    print()