# Focal Reward 流程说明

这份文档用来说明当前仓库里的 focal reward 流程，和原版 `verl` 的 reward 流程相比改了什么，我们记录和保存了哪些值，以及这些改动插在训练链路的什么位置。

## 1. 原版 `verl` 的 reward 流程

原版 PPO 的 reward 路径比较直接：

1. reward client 往 batch 里写 `rm_scores`。
2. `extract_reward(batch)` 取出原始 reward 张量和少量额外字段。
3. Trainer 直接把这个 reward 张量用于 PPO。
4. 验证阶段记录 reward 值，并导出 reward 相关的额外信息。

在原版实现里，reward 基本就是一个来自 reward client 的张量。当前这版 focal 逻辑在此基础上加了一层 postprocess。

## 2. focal 逻辑做了什么

现在的 reward 流程多了一个“中间处理层”：

1. 先从 `rm_scores` 取出原始 reward 张量。
2. 再调用 `postprocess_reward(...)` 把 reward client 的输出整理成：
   - direct score
   - focal score
   - final score
   - 结构化 metrics
   - 验证用的标量列表
   - JSONL dump 用的调试内容
3. Trainer 不再直接使用原始 reward 张量，而是使用 postprocess 后的 `reward_tensor`。

和原版 `verl` 相比，最大的区别是 reward 不再只是一个“黑盒张量”，而是一个带有明确语义的结构化结果，里面包含：

- 每个 rubric 的 reward signal
- 按 `uid` 分组后的 focal 权重
- 有效/无效样本补偿逻辑
- 训练、验证、落盘三套不同用途的输出

## 3. focal 的核心计算逻辑

当前 focal 逻辑在 `verl/trainer/ppo/focal_reward.py` 里。

### 3.1 输入

`postprocess_reward(...)` 需要这些字段：

- `reward_signals`
- `overall_status`
- `render_info`
- `judge_info`
- `error_message`
- `uid`

其中 `reward_signals` 是必需的，其他字段主要用于过滤、统计和调试。

### 3.2 核心步骤

1. 把 `reward_signals` 整理成形状为 `[batch_size, num_rubrics]` 的信号矩阵。
2. 将 `success` 和 `llm_generation_error` 视作有效样本。
3. 按 `uid` 把样本分组。
4. 对每个 group：
   - 用 base weights 计算 direct score
   - 根据 difficulty / pace 计算 focal weights
   - 对 focal weights 做归一化
   - 用归一化后的权重计算 focal score
5. 对 group 内无效样本，用该 group 的均值补偿。
6. 对整组都无效的情况，用 batch 级有效样本均值补偿。
7. 最终分数根据 `algorithm.use_focal` 决定使用 direct score 还是 focal score。

### 3.3 关键公式

- `direct_scores = signal_matrix @ base_weights`
- `focal_weights = base_weights * difficulty^gamma`
- `normalized_focal_weights = focal_weights / focal_weights.sum()`
- `focal_scores = signal_matrix @ normalized_focal_weights`

## 4. 我们记录了什么

`postprocess_reward(...)` 会返回一个 `RewardPostprocessResult`，里面有四类不同用途的输出。

### 4.1 `derived_extra_info`

这个会回写到 `batch.non_tensor_batch`，供训练和验证后续复用。

记录的字段有：

- `reward_raw_score`
- `reward_direct_score`
- `reward_focal_score`
- `reward_final_score`
- `reward_valid_sample`
- `reward_is_llm_generation_error`
- 每个 rubric 的 `reward_signal_<rubric_slug>`

### 4.2 `validation_extra_info`

这个是给验证聚合用的，只保留可以安全统计的标量列表。

记录的字段有：

- `reward_raw_score`
- `reward_direct_score`
- `reward_focal_score`
- `reward_final_score`
- `reward_valid_sample`
- `reward_is_llm_generation_error`
- 每个 rubric 的 `reward_signal_<rubric_slug>`

### 4.3 `dump_extra_info`

这个是给 JSONL 调试 dump 用的，保留更完整、更结构化的信息。

记录的字段有：

- reward client 的原始调试字段：
  - `overall_status`
  - `error_message`
  - `reward_signals`
  - `render_info`
  - `judge_info`
- 上面的验证标量字段
- `reward`
- `reward_detail`
- `sample_reward_signal`
- `sample_reward_weight`
- `group_mean_reward_signal`
- `group_mean_reward_weight`

其中 `reward_detail` 是最有用的调试结构，每条样本一条记录，里面包含：

- `scores`，包括 `raw`、`direct`、`focal`、`final`
- `sample_reward_signal`，当前样本每个 rubric 的原始分数
- `sample_reward_weight`，当前样本计算 focal score 时使用的 rubric 权重
- `group_mean_reward_signal`，当前 rollout group 每个 rubric 的均值
- `group_mean_reward_weight`，当前 rollout group 估计出的归一化 focal rubric 权重
- `status`，包括 `overall`、`error_message`、`render`、`judge`
- `valid_sample`
- `is_llm_generation_error`

### 4.4 `metrics`

这个是训练监控用的，主要给 W&B 看板看。

当前 metrics 分成这几个模块：

- `group_reward_signal/*`
- `group_focal_weight/*`
- `group_reward_others/*`
- `reward_status/*`
- `render_status/*`
- `judge_status/*`

记录的统计形式：

- `group_reward_signal/<rubric>/mean`
- `group_focal_weight/<rubric>/mean`
- `group_reward_others/{max,min,var}/<rubric>`
- 所有离散状态只记录 `rate`
- `reward_status/valid_sample/rate`
- `reward_status/llm_generation_error/rate`
- `reward_status/valid_group/rate`

### 4.5 W&B 与 rollout JSONL 字段结构

W&B 只记录 step 级总体统计，不记录单条样本的完整字典。当前字段结构是：

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

- `sample_reward_signal`：当前 sample 自己的原始 rubric 分数，适合排查单个 rollout 为什么得分高或低。
- `sample_reward_weight`：当前 sample 计算 focal score 时使用的 rubric 权重；这个权重由 sample 所属 group 估计，所以同 group 内通常相同。
- `group_mean_reward_signal`：当前 sample 所属 group 的有效样本 rubric 均值，用于对比 sample 分数与组内平均水平。
- `group_mean_reward_weight`：当前 sample 所属 group 的归一化 focal rubric 权重。
- `reward_others`：当前 sample 所属 group 的 rubric 级 `max/min/var`，用于判断组内该维度是否有区分度。

注意：rollout JSONL 顶层不再写 `reward_signals` / `reward_weights`，避免把 group 级视图误读为 sample 级原始值。

## 5. 流程改动点

### 5.1 验证流程

在 `verl/trainer/ppo/ray_trainer.py` 的验证阶段，流程变成：

1. `extract_reward(test_batch)`
2. `postprocess_reward(...)`
3. 用 `reward_tensor.sum(-1)` 得到每个样本的最终分数
4. 把 `validation_extra_info` 合并进验证指标
5. 把 `dump_extra_info` 合并进 JSONL 导出内容

### 5.2 训练流程

在 `verl/trainer/ppo/ray_trainer.py` 的训练阶段，流程变成：

1. `extract_reward(batch)`
2. `postprocess_reward(...)`
3. 把返回的 `reward_tensor` 写回 batch
4. `batch.batch["token_level_scores"] = reward_tensor`
5. 后续 PPO 使用的是 postprocess 后的 reward，而不是原始 `rm_scores`

这点是最关键的行为变化：PPO 优化的是 postprocess 后的 focal/baseline reward，不再直接用 reward client 的原始输出。

## 6. 和原版 `verl` 的区别

和原版 reward 流程相比，这版新增了：

- rubric 级 reward signal 矩阵
- direct / focal reward 分离
- 按 `uid` 分组
- 无效样本补偿
- 更清晰的 W&B metrics 分组
- 更结构化的 JSONL 调试输出
- 每条样本的归一化 focal 权重

也就是说，原版 `verl` 偏向“reward 是一个张量”，现在这版偏向“reward 是一个带训练、验证、调试三种视图的结构化对象”。

## 7. 本地启动脚本

当前启动脚本保持得比较薄：

- `my/train_focal.sh`
- `my/train_baseline.sh`

它们会优先读取本地配置文件 `my/train_local.sh`。

推荐本地文件：

- `my/train_local.sh`

模板文件：

- `my/train_local.sh.example`

这样可以把模型路径、reward server 地址、W&B key 这些敏感或环境相关的值留在本地，不进 git。
