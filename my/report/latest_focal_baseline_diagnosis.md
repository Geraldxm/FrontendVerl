# 最新 focal vs baseline 诊断记录

生成时间: 2026-05-08

## 数据范围

按文件修改时间读取 `rollouts` 后，最新一轮实验主要是:

- baseline: `rollouts/baseline_v3_n16_3e-6_4k_Qwen3-4B`，最新文件到 `47.jsonl`。
- focal: `rollouts/focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B`，最新有效时间线到 `31.jsonl`。

注意: focal 目录内还有 `32.jsonl` 和 `33.jsonl`，但它们的修改时间早于 `1.jsonl`，像是旧运行遗留文件。按“最新实验结果”分析时不应把它们纳入当前时间线。

## 关键观察

1. `31.jsonl` 的 focal 结果被 reward server 失败严重污染。
   - `focal/31.jsonl`: 1024 行里只有 64 个 `success`，有 952 个 `request_error`。
   - 失败行的 `sample_reward_signal` 被写成 0，因此全量统计 rubric 均值会断崖式下降。
   - 但训练用的 `reward_scores.direct/focal/final` 会对无效样本做组均值或 batch 有效均值回填，不能直接和全量 `sample_reward_signal` 均值混用。

2. 过滤 `valid_sample=1` 后，focal 后期主观 rubric 并没有全面低于 baseline。
   - `F29-30 success`: 三项主观均值约 `8.45`，`ui_spatial_score` 约 `8.26`。
   - `B43-45 success`: 三项主观均值约 `8.44`，`ui_spatial_score` 约 `8.53`。
   - 所以“各 rubric 后期竟然都不如 baseline”的结论目前主要来自把失败行 0 分混入了全量均值；真正需要解释的是 focal 的 `ui_spatial_score` 偏低，以及训练最终标量 `final=focal` 明显小于 baseline 的 `final=direct`。

3. focal 的 `final` 天然低于 baseline 的 `final`，因为两者不是同一标尺。
   - baseline: `algorithm.use_focal=False`，训练用 `final=direct`，即 9 个 rubric 平均。
   - focal: `algorithm.use_focal=True`，训练用 `final=focal`，权重集中到当前组内更难的 rubric。
   - 在后期，大部分客观项接近 10，focal 权重几乎全部压到 `ui_spatial_score` 和三项主观分上，所以 `final` 会低于 direct，这本身不等于模型变差。

4. 当前 focal 参数为 `epsilon=0, temperature=1.0, gamma=5.0`，比 baseline 日志里用于重算的 `temperature=10.0, gamma=8.0` 更强调组内高 direct 样本的 pace。
   - `F29-30` 平均最大 rubric 权重约 `0.595`。
   - `B43-45` 日志重算权重最大值约 `0.667`，但 baseline 实际不使用 focal 作为训练 reward。
   - focal 实际训练中的组内 direct 方差更低，说明后期同组样本 direct 更接近，focal 可能在低方差状态下放大了少数 rubric 的噪声。

## 当前假设

### 假设 1: 观测口径污染

后期 rubric 均值低，主要是 reward server 的 `request_error` 行被记录为 0 分，而不是模型真实输出质量突然变差。

验证:
- 所有对比同时输出三种口径: `all rows`、`valid_sample=1`、`status=success`。
- 对每个 step 记录 `reward_status/valid_sample/rate` 与 `overall_status/request_error/rate`。
- 如果只在 `all rows` 口径下降，而 `valid_sample=1` 稳定，则应归因于评测基础设施而不是 reward 方法。

### 假设 2: final 标尺不可比

baseline 的 `final=direct`，focal 的 `final=focal`。focal 标量偏低可能是权重重分配造成的数值标尺变化，不代表 direct/rubric 全面下降。

验证:
- 比较模型质量时优先使用相同口径的 `direct` 和各 rubric，而不是 `final`。
- 对 focal 训练样本同时记录 `direct - focal` 的均值、方差和分位数。
- 离线重算 baseline 的 focal 分数时只用于观察，不作为 baseline 训练结果的质量标尺。

### 假设 3: focal 优化目标过窄，牺牲空间布局稳定性

过滤有效样本后，focal 的三项主观分接近 baseline，但 `ui_spatial_score` 低于 baseline。可能是 focal 权重长期集中在少数 rubric，导致模型朝“评审主观项可接受”方向走，而没有稳定提升结构布局。

验证:
- 统计每个 step 的 `group_mean_reward_weight`，看 `ui_spatial_score`、三项主观分的权重占比是否长期超过 90%。
- 对比 `ui_spatial_score` 的 group mean、min、var，以及 direct top1 与 focal top1 是否选择不同样本。
- 抽样查看 focal 低 `ui_spatial_score` 但主观分高的 HTML/screenshot，判断是否出现布局拥挤、缺少真实结构、模板化等模式。

### 假设 4: reward server 失败与实验配置/服务节点相关

两个训练脚本指向不同 reward server:

- focal: `http://10.246.101.52:48002/compute_reward_v3`
- baseline: `http://10.246.103.127:48002/compute_reward_v3`

`focal/31.jsonl` 出现 952 个 `request_error`，可能是 focal 节点的 reward server、网络或并发状态异常，而不是算法本身。

验证:
- 将 focal 和 baseline 临时指向同一个 reward server，跑短实验或重评测同一批 rollout。
- 对 `request_error` 的 `error_message` 聚合去重，确认是连接问题、超时、服务异常还是输入异常。
- 如果同一输出在另一台 reward server 上成功率恢复，则先修基础设施。

### 假设 5: 低方差后期 focal 权重估计噪声变大

后期同一 prompt 的 16 个样本 direct 已经很接近，focal 仍按组内有效样本估计 pace 和 difficulty。此时少量 judge 波动会改变 rubric 权重，进而改变 PPO 优势。

验证:
- 对每个 group 统计 direct 方差、focal 方差、top1 是否一致、z-advantage sign flip rate。
- 在离线扫描中比较更温和参数，例如 `gamma=1..3`、`epsilon=0.02..0.1`、`temperature=3..10`。
- 如果温和参数显著降低 sign flip 但保留困难 rubric 权重，则下一轮训练应优先试温和参数。

## 下一步验证命令

只读诊断:

```bash
python my/scan_focal_params.py \
  --rollout-dir rollouts/focal_v3_n16_e0_t1_g5_3e-6_4k_Qwen3-4B \
  --steps 29-31 \
  --temperatures 1,3,5,10 \
  --gammas 0:6:1 \
  --epsilons 0,0.02,0.05,0.1 \
  --focus-step 30 \
  --out-dir my/report/focal_param_scan_latest
```

建议补充一个专门的 `valid_sample` 对比脚本，输出:

- 每步 `all/valid/success` 三种口径的 rubric 均值。
- 每步 status 分布。
- 每步 group 内 direct/focal 排序扰动。
- 每步 focal 权重分布。

短实验验证:

1. 先固定同一个 reward server，避免基础设施差异。
2. 跑一个温和 focal 参数: `temperature=5.0, gamma=2.0, epsilon=0.05`。
3. 每 5 step 读取 valid-only rubric 和 status rate，先确认没有 `request_error` 峰值。
