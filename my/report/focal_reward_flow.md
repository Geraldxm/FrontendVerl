# Focal Reward 流程说明

这份文档说明当前仓库里的 focal reward 流程，重点回答四个问题：

1. 相比原版 `verl` 的 reward 路径改了什么；
2. 当前 focal 的主计算逻辑是什么；
3. 训练 / 验证 / rollout dump 分别记录了哪些字段；
4. 它和当前 DVAO advantage 实现如何衔接。

## 1. 原版 `verl` 的 reward 流程

原版 PPO 的 reward 路径比较直接：

1. reward client 往 batch 里写 `rm_scores`；
2. `extract_reward(batch)` 取出原始 token-level reward 张量和少量额外字段；
3. trainer 直接把这个 reward 张量用于 PPO；
4. 验证阶段记录 reward 值，并导出少量 reward 相关信息。

在原版实现里，reward 基本就是一个来自 reward client 的张量。当前这版 focal 逻辑在此基础上加了一层 reward postprocess。

## 2. 当前 focal 改了什么

现在 reward 流程多了一个“中间处理层”：

1. 先从 `rm_scores` 取出原始 reward 张量；
2. 再调用 `postprocess_reward(...)` 把 reward client 输出整理成：
   - `direct score`
   - `focal score`
   - `final score`
   - 结构化 metrics
   - 验证用标量列表
   - rollout JSONL 调试字段
3. trainer 不再直接使用原始 reward 张量，而是使用 postprocess 后的 `reward_tensor`。

和原版 `verl` 相比，关键区别是 reward 不再只是一个“黑盒张量”，而是一个带语义的结构化结果，里面显式保存：

- 每个 rubric 的 reward signal；
- 按 `uid` 分组后的 focal 权重；
- 有效/无效样本补偿逻辑；
- 训练、验证、落盘三套不同用途的输出。

## 3. focal 的核心计算逻辑

当前 focal 主逻辑在 `verl/trainer/ppo/focal_reward.py`。

### 3.1 输入

`postprocess_reward(...)` 需要这些字段：

- `reward_signals`
- `overall_status`
- `render_info`
- `judge_info`
- `error_message`
- `uid`

其中 `reward_signals` 是必需输入；其他字段主要用于过滤、统计和调试。

### 3.2 核心步骤

1. 把 `reward_signals` 整理成形状为 `[batch_size, num_rubrics]` 的信号矩阵。
2. 将 `success` 和 `llm_generation_error` 视作有效样本。
3. 按 `uid` 把样本分组。
4. 对每个 group：
   - 用 `base_weights` 计算 direct score；
   - 用 direct score 和 `temperature` 算 sample-level softmax 权重；
   - 用这些 sample 权重对组内 signal 做加权平均，得到 `weighted_scores`；
   - 将 `pace = clip(weighted_scores / max_signal_score, 0, 1)`；
   - 将 `difficulty = 1 - pace + epsilon`；
   - 根据 `base_weights * difficulty^gamma` 计算 focal weights；
   - 先归一化 focal weights，再投影到 `[weight_min, weight_max]` 区间；
   - 用投影后的权重计算 focal score。
5. 对 group 内无效样本，用该 group 有效样本的均值补偿。
6. 对整组都无效的情况，用 batch 级有效样本均值补偿。
7. 最终分数根据 `algorithm.use_focal` 决定使用 direct score 还是 focal score。

### 3.3 关键公式

- `direct_scores = signal_matrix @ base_weights`
- `sample_weights = softmax(direct_scores / temperature)`
- `weighted_scores = sample_weights @ signal_matrix`
- `pace = clip(weighted_scores / max_signal_score, 0, 1)`
- `difficulty = 1 - pace + epsilon`
- `focal_weights = base_weights * difficulty^gamma`
- `normalized_focal_weights = focal_weights / focal_weights.sum()`
- `bounded_focal_weights = project(normalized_focal_weights, [weight_min, weight_max])`
- `focal_scores = signal_matrix @ bounded_focal_weights`

### 3.4 当前默认超参数

默认值来自 `algorithm.focal`：

- `temperature = 10.0`
- `gamma = 3.0`
- `epsilon = 0.05`
- `weight_min = 0.05`
- `weight_max = 0.3`

如果没有显式配置 `base_weights`，当前实现会退化成各 rubric 等权。

### 3.5 相比旧文档需要补充的实现细节

当前实现里，有几处细节会直接影响理解和排障：

- focal 不是直接拿组均值算权重，而是先做一次组内 sample softmax 加权；
- focal 权重不是只做一次归一化，还会经过 `_project_weights_to_bounds(...)`，强制满足每个 rubric 的最小/最大占比；
- `raw_scores` 仍然保留为原始 token-level reward tensor 的逐样本求和，主要用于和旧指标做同口径对比；
- 如果某个 `uid` 对应的整个 group 都无效，则该 group 的 `group_mean_reward_signal`、`group_mean_reward_weight`、`reward_others` 会回退到 batch 内有效样本的总体统计；
- error message 在 metrics 中会先做规范化、截断，超长时追加哈希后缀，避免 W&B metric key 膨胀。

## 4. `postprocess_reward(...)` 会写出什么

`postprocess_reward(...)` 返回一个 `RewardPostprocessResult`，里面有四类输出。

### 4.1 `derived_extra_info`

这个会回写到 `batch.non_tensor_batch`，供训练和验证后续复用。

字段包括：

- `reward_raw_score`
- `reward_direct_score`
- `reward_focal_score`
- `reward_final_score`
- `reward_valid_sample`
- `reward_is_llm_generation_error`
- 每个 rubric 的 `reward_signal_<rubric_slug>`
- `group_focal_weight`：以 `reward_signal_<rubric_slug>` 为 key 的 group-level focal 权重，供 focal+DVAO 这类 advantage estimator 消费

### 4.2 `validation_extra_info`

这个只保留可以安全聚合的标量 list，供验证汇总使用。

字段包括：

- `reward_raw_score`
- `reward_direct_score`
- `reward_focal_score`
- `reward_final_score`
- `reward_valid_sample`
- `reward_is_llm_generation_error`
- 每个 rubric 的 `reward_signal_<rubric_slug>`

### 4.3 `dump_extra_info`

这个给 rollout / validation JSONL 调试 dump 使用，保留更完整、更结构化的信息。

字段包括：

- reward client 的原始调试字段：
  - `overall_status`
  - `error_message`
  - `render_info`
  - `judge_info`
- 上面的验证标量字段
- `reward`
- `reward_detail`
- `sample_reward_signal`
- `sample_reward_weight`
- `group_mean_reward_signal`
- `group_mean_reward_weight`
- `reward_others`

其中 `reward_detail` 是最有用的调试结构，每条样本一条记录，里面包含：

- `scores`：`raw`、`direct`、`focal`、`final`
- `sample_reward_signal`：当前样本每个 rubric 的原始分数
- `sample_reward_weight`：当前样本计算 focal score 时使用的 rubric 权重
- `group_mean_reward_signal`：当前 rollout group 每个 rubric 的均值
- `group_mean_reward_weight`：当前 rollout group 估计出的归一化 focal rubric 权重
- `reward_others`：当前 rollout group 在每个 rubric 上的 `max/min/var`
- `status`：`overall`、`error_message`、`render`、`judge`
- `valid_sample`
- `is_llm_generation_error`

### 4.4 `metrics`

这个是训练监控用的，主要用于 W&B / console metrics。

当前 metrics 分成这些模块：

- `group_reward_signal/*`
- `group_focal_weight/*`
- `group_reward_others/*`
- `reward_status/*`
- `render_status/*`
- `judge_status/*`

其中：

- `group_reward_signal/<rubric>/mean`
- `group_focal_weight/<rubric>/mean`
- `group_reward_others/{max,min,var}/<rubric>`
- 离散状态类指标只记录 `rate`
- 额外记录：
  - `reward_status/valid_sample/rate`
  - `reward_status/llm_generation_error/rate`
  - `reward_status/valid_group/rate`

同时，`reward_status/raw_score`、`direct_score`、`focal_score`、`final_score` 都会记录 `mean/var/min/max` 四个摘要指标。

## 5. W&B 与 rollout JSONL 字段结构

### 5.1 W&B step 级统计

W&B 只记录 step 级总体统计，不记录单条样本的完整字典。当前 focal 相关字段结构是：

```text
group_reward_signal/<rubric>/mean
group_focal_weight/<rubric>/mean
group_reward_others/max/<rubric>
group_reward_others/min/<rubric>
group_reward_others/var/<rubric>
```

含义：

- `group_reward_signal/<rubric>/mean`：先在每个 rollout group 内对该 rubric 的有效样本 signal 求均值，再对当前 step 的 groups 求平均。
- `group_focal_weight/<rubric>/mean`：当前 step 内各 group 的归一化 focal rubric 权重均值。
- `group_reward_others/max/<rubric>`：当前 step 内各 group 的 `group_mean_reward_signal[<rubric>]` 最大值。
- `group_reward_others/min/<rubric>`：当前 step 内各 group 的 `group_mean_reward_signal[<rubric>]` 最小值。
- `group_reward_others/var/<rubric>`：当前 step 内各 group 的 `group_mean_reward_signal[<rubric>]` 方差。

### 5.2 rollout JSONL 样本级调试字段

rollout JSONL 记录样本级调试信息。每一行对应一条 sample，核心 reward 字段结构是：

```json
{
  "reward_scores": {
    "raw": 0.0,
    "direct": 0.0,
    "focal": 0.0,
    "final": 0.0,
    "valid_sample": 1,
    "is_llm_generation_error": 0
  },
  "sample_reward_signal": {
    "<rubric>": 0.0
  },
  "sample_reward_weight": {
    "<rubric>": 0.0
  },
  "group_mean_reward_signal": {
    "<rubric>": 0.0
  },
  "group_mean_reward_weight": {
    "<rubric>": 0.0
  },
  "reward_others": {
    "max": {"<rubric>": 0.0},
    "min": {"<rubric>": 0.0},
    "var": {"<rubric>": 0.0}
  }
}
```

含义：

- `sample_reward_signal`：当前 sample 自己的原始 rubric 分数。
- `sample_reward_weight`：当前 sample 计算 focal score 时使用的 rubric 权重；这个权重由 sample 所属 group 估计，所以同 group 内通常相同。
- `group_mean_reward_signal`：当前 sample 所属 group 的有效样本 rubric 均值。
- `group_mean_reward_weight`：当前 sample 所属 group 的归一化 focal rubric 权重。
- `reward_others`：当前 sample 所属 group 的 rubric 级 `max/min/var`。

注意：rollout JSONL 顶层不再写 `reward_signals` / `reward_weights`，避免把 group 级视图误读为 sample 级原始值。

## 6. focal 与 DVAO 的衔接

当前仓库里，focal 和 DVAO/focal+DVAO 位于 PPO 链路的两个不同阶段：

- focal 位于 reward postprocess 阶段，决定“最终 reward 分数怎么聚合”；
- DVAO/focal+DVAO 位于 advantage 阶段，决定“多维 reward 信号怎么转成 advantage”。

如果 advantage estimator 选的是 `dvao` 或 `focal_dvao`，`verl/trainer/ppo/ray_trainer.py` 会在 dump 前用 advantage 阶段产出的动态权重覆盖通用权重视图，并额外补充：

- `group_dvao_weight`
- `group_dvao_reward_mean`
- `group_dvao_reward_std`

这样 rollout JSONL 里就能同时看到：

- reward postprocess 产出的 signal 视图；
- 当前组的统计量；
- advantage estimator 真正使用的动态权重。

## 7. 训练链路中的位置

验证与训练都先执行 `extract_reward(...)`，再执行 `postprocess_reward(...)`。训练阶段会把 `reward_tensor`、`derived_extra_info` 和 reward metrics 写回 batch / 日志；后续 advantage estimator 再决定如何消费这些 reward 相关字段。

因此，focal reward 的职责是统一 reward 聚合、调试字段和 W&B/rollout dump 统计；它不负责 advantage 的最终定义。
