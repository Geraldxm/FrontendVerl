# focal 后期低于 baseline 的根因调查

生成时间: 2026-05-08

## 本次新增分析产物

- 脚本: `my/analyze_late_rollout_quality.py`
- 4B 汇总: `my/report/late_rollout_quality_4b/summary.md`
- 全量汇总: `my/report/late_rollout_quality_all/summary.md`

分析口径:

- 排除旧运行尾巴，例如 `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B/32.jsonl` 和 `33.jsonl`。
- 排除低有效率 step，例如 reward server 下线后的 `31.jsonl`。
- rubric 统一用 `valid_sample=1` 口径，避免 `request_error` 行的 0 分污染。
- baseline/focal 的 `final` 不直接比较，因为 baseline 的 `final=direct`，focal 的 `final=focal`。

## 最重要结论

focal “后期竟然低于 baseline”不是单一原因，而是三类现象叠加:

1. `final` 标尺不可比，导致视觉上 focal 总是低很多。
2. focal 会把 easy rubric 权重压到接近 0，造成某些客观项退化后没有足够惩罚，最明显的是 `a11y_score`。
3. 在后期 direct 方差变小后，focal 主要围绕少数困难 rubric 给优势，`ui_spatial_score` 的提升不如 baseline 的平均 direct 稳定。

## 证据 1: 4B 后期 valid-only 对比

后期窗口中，排除 reward server 下线 step 后:

| 对照 | direct 差值 | 主观三项差值 | UI spatial 差值 | 结论 |
|---|---:|---:|---:|---|
| `focal_v3_n16_e0_t1_g5_Qwen3-4B` vs `baseline_v3_n16_Qwen3-4B` | +0.320 | +0.660 | +0.964 | 早期无 3e-6 实验里 focal 更好 |
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` vs `baseline_v3_n16_3e-6_Qwen3-4B` | -0.124 | -0.207 | -0.384 | focal 明显落后 |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` vs `baseline_v3_n16_3e-6_4k_Qwen3-4B` | -0.026 | +0.014 | -0.193 | direct 接近，主观持平，主要输 UI spatial |

同 step 对比可以排除“只是 baseline 多训练了几步”的解释:

| 对照 | step | direct 差值 | UI spatial 差值 | 主观三项差值 |
|---|---:|---:|---:|---:|
| 4B `3e-6_4k` | 26-30 vs 26-30 | +0.001 | -0.112 | +0.067 |
| 4B `3e-6` | 35-39 vs 35-39 | -0.088 | -0.364 | -0.105 |

因此，最新 4k 实验里 focal 不是真的“各 rubric 都不如 baseline”，而是:

- `final` 因为换了 focal 标尺而低很多。
- `ui_spatial_score` 稳定低一些。
- 三项主观 rubric 持平或略好。

## 证据 2: a11y 被当成 easy rubric 后退化

`focal_v3_n16_e0_t1_g8_Qwen3-1.7B` vs `baseline_v3_n16_Qwen3-1.7B` 后期:

| rubric | focal | baseline | 差值 | focal 权重 |
|---|---:|---:|---:|---:|
| `a11y_score` | 9.036 | 9.977 | -0.941 | 0.011 |
| `ui_spatial_score` | 7.746 | 7.903 | -0.157 | 0.452 |
| `color_harmony_and_theme_fit` | 8.042 | 8.123 | -0.081 | 0.228 |
| `typography_rhythm_and_readability` | 8.316 | 8.440 | -0.124 | 0.118 |
| `first_view_content_messaging` | 8.151 | 8.284 | -0.133 | 0.191 |

分段曲线显示 a11y 差距不是一次性评测噪声:

| 阶段 | direct 差值 | UI 差值 | 主观差值 | a11y 差值 |
|---|---:|---:|---:|---:|
| step 1-5 | +0.026 | +0.058 | +0.068 | -0.022 |
| step 20-24 | -0.068 | +0.100 | +0.052 | -0.863 |
| step 40-44 | -0.180 | +0.025 | -0.117 | -1.249 |
| step 60-64 | -0.201 | -0.101 | -0.152 | -1.132 |
| step 78-82 | -0.164 | -0.141 | -0.103 | -0.943 |

解释:

- focal 按组内 pace 计算 difficulty。
- 一旦某个 rubric 的组均值很高，它的 focal 权重会被压到接近 0。
- 但单样本仍可能在 a11y 上失败；此时 focal reward 对这个失败几乎不敏感，模型就可能逐步放松 a11y 约束。

这是目前最像“根因”的机制。

## 证据 3: UI spatial 是 focal 最稳定的短板

跨实验配对后，除最早无 `3e-6` 的 4B 对照外，focal 的 `ui_spatial_score` 基本都低于对应 baseline:

| focal | baseline | UI 差值 | direct 差值 |
|---|---|---:|---:|
| `focal_v3_n16_e0_t1_g5_3e-6_Qwen3-4B` | `baseline_v3_n16_3e-6_Qwen3-4B` | -0.384 | -0.124 |
| `focal_a11y_Qwen3-1.7B-Base` | `baseline_a11y_Qwen3-1.7B-Base` | -0.305 | -0.228 |
| `focal_0414_Qwen3-1.7B-Base` | `baseline_0414_Qwen3-1.7B-Base` | -0.291 | -0.057 |
| `focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B` | `baseline_v3_n16_3e-6_4k_Qwen3-4B` | -0.193 | -0.026 |
| `focal_v3_n16_e0_t1_g8_Qwen3-1.7B` | `baseline_v3_n16_Qwen3-1.7B` | -0.157 | -0.168 |

分段看 4B `3e-6`:

| 阶段 | direct 差值 | UI 差值 | 主观差值 | a11y 差值 |
|---|---:|---:|---:|---:|
| step 1-5 | -0.035 | -0.007 | +0.048 | -0.378 |
| step 10-14 | -0.074 | +0.008 | -0.001 | -0.663 |
| step 20-24 | -0.054 | -0.169 | -0.066 | -0.133 |
| step 30-34 | -0.069 | -0.245 | -0.094 | -0.094 |
| step 35-39 | -0.088 | -0.364 | -0.105 | -0.109 |

解释:

- early 阶段 focal 没有明显 UI spatial 劣势。
- 后期 UI 差距逐渐扩大。
- 这更像训练目标造成的行为偏移，而不是 reward server 或随机样本噪声。

## 当前最可信原因链

1. 直接平均 reward 的 baseline 虽然“看起来随意”，但它持续保留所有 rubric 的惩罚信号。
2. focal 会动态压低高分 rubric 的权重。这个设计本意是聚焦短板，但副作用是高分 rubric 的偶发失败不再被强惩罚。
3. 后期模型输出质量接近饱和，组内 direct 方差变小；此时 focal 主要在少数低分项上放大优势，容易受 judge 噪声和 prompt 局部偏差影响。
4. 对 HTML 生成任务来说，`ui_spatial_score` 和主观三项高度耦合但不等价。focal 有时能维持主观分，却没能稳定提升空间布局。
5. 因此 baseline 的平均权重反而像一种正则化: 它不会让 a11y、format、console 等“已经学会”的项完全失声，也不会把 PPO 优势过度集中到少数 rubric。

## 后续验证建议

### 验证 A: 加权下限

下一轮 focal 不要让任何 rubric 权重接近 0。建议:

- `epsilon=0.05`
- `gamma=2`
- `temperature=5`
- 或显式设置最小权重地板，例如每个 rubric 至少 `0.03`。

判断标准:

- `a11y_score` 不再相对 baseline 下跌。
- `ui_spatial_score` 后期差距不再扩大。
- `z_sign_flip_rate` 不高于当前 focal。

### 验证 B: easy-rubric 保护项

把最终 reward 改成混合形式:

```text
reward = 0.7 * focal + 0.3 * direct
```

或:

```text
reward = focal - penalty_for_any_easy_rubric_below_threshold
```

判断标准:

- direct 不应再系统性低于 baseline。
- focal 仍应保留对 UI/主观短板的提升。

### 验证 C: UI spatial 单独监控

后续训练每 5 step 输出:

- `valid_sample=1` 的 `ui_spatial_score` 均值。
- `ui_spatial_score < 9` 比例。
- `ui_spatial_score` 权重均值和 p90。
- `direct top1 != focal top1` 的样本截图抽样。

判断标准:

- 如果 focal 权重高但 UI spatial 仍下降，说明该 rubric 的 judge 噪声或奖励形状不适合直接 focal 放大。
- 如果权重不高且 UI spatial 下降，说明 difficulty 估计没有正确识别 UI spatial 短板。

### 验证 D: 重评测而非只看训练 rollout

训练 rollout 带采样噪声和 reward server 状态。建议把同一 checkpoint 在同一 reward server 上重评测固定 validation set:

- baseline 后期 checkpoint。
- 当前 focal 后期 checkpoint。
- 温和 focal checkpoint。

判断标准:

- 如果重评测仍保持 UI spatial 差距，说明是模型行为差异。
- 如果重评测差距消失，说明主要是 rollout 采样或 reward server 批次噪声。

## 目前建议

下一轮不要继续用 `epsilon=0, temperature=1, gamma=5/8` 这种激进 focal。更稳的候选:

```bash
algorithm.use_focal=True
algorithm.focal.epsilon=0.05
algorithm.focal.temperature=5.0
algorithm.focal.gamma=2.0
```

如果代码允许，优先测试混合 reward:

```text
final = 0.7 * focal + 0.3 * direct
```

这能保留 focal 对短板 rubric 的关注，同时避免 easy rubric 完全失去训练信号。
