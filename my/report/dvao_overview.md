# DVAO 实现简要说明

这份文档说明当前仓库里的 DVAO（Dynamic Variance-adaptive Advantage Optimization）与 focal+DVAO 实现、训练日志和 rollout dump 字段。

## 1. 这版 DVAO 涉及哪些文件

### 1.1 核心实现

- `verl/trainer/ppo/core_algos.py`
  - 定义 `AdvantageEstimator.DVAO`
  - 定义 `AdvantageEstimator.FOCAL_DVAO`
  - 实现 `compute_dvao_outcome_advantage(...)`
  - 实现 `compute_focal_dvao_outcome_advantage(...)`
  - 校验 `dvao_reward_keys` / `dvao_reward_weights`
  - 将动态权重、组均值、组标准差写回 `non_tensor_batch`

- `verl/trainer/ppo/focal_reward.py`
  - 写出 `group_focal_weight`，供 focal+DVAO 在 advantage 阶段消费

- `verl/trainer/ppo/ray_trainer.py`
  - 在 advantage 计算时为 DVAO/focal+DVAO 透传 `non_tensor_batch` 和 `uid`
  - 在 rollout JSONL dump 前补充 advantage 阶段的最终权重字段
  - 在训练 metrics 中记录每个 reward 维度的值、动态权重和组内标准差

### 1.2 配置入口

- `verl/trainer/config/algorithm.py`
  - 在算法配置里声明 `dvao_reward_keys` 和 `dvao_reward_weights`

- `verl/trainer/config/ppo_trainer.yaml`
  - 给 DVAO 配置项提供默认入口

### 1.3 测试与训练脚本

- `tests/trainer/ppo/test_core_algos_on_cpu.py`
  - 覆盖 DVAO/focal+DVAO 手工公式对齐、valid sample、边界退化、配置校验、debug 字段写回

- `my/train_dvao.sh`
  - 提供一个本地训练命令模板
  - 指定 `algorithm.adv_estimator=dvao`
  - 指定 DVAO 聚合使用的 reward 维度

- `my/train_focal_dvao.sh`
  - 指定 `algorithm.adv_estimator=focal_dvao`
  - 使用 focal 权重作为 DVAO 的 group-level prior

## 2. 每个文件的作用

### 2.1 `verl/trainer/ppo/core_algos.py`

这是 DVAO 的主计算位置。它不直接依赖 token 级 reward 的数值做聚合，而是从 `non_tensor_batch` 中读取多个样本级 reward 维度，例如：

- `reward_signal_format_score`
- `reward_signal_console_errors`
- `reward_signal_network_violations`
- `reward_signal_a11y_score`

然后按 rollout group（通常由 `uid` 标识）做组内统计，算出每个维度的组均值、组标准差、动态权重和最终 advantage。若存在 `reward_valid_sample`，只有有效样本参与均值和标准差估计。

### 2.2 `verl/trainer/ppo/ray_trainer.py`

这个文件做三件事：

- 在调用 advantage estimator 时，把 DVAO/focal+DVAO 需要的 `non_tensor_batch` 和 `uid` 传进去；
- 在导出 rollout JSONL 时，用 advantage 阶段才产生的最终权重覆盖通用权重视图；
- 在训练 step 结束时，把 DVAO/focal+DVAO 相关统计写进 W&B / console metrics。

### 2.3 `verl/trainer/config/algorithm.py` 与 `verl/trainer/config/ppo_trainer.yaml`

这两个文件负责把 DVAO 暴露成正式配置项，而不是写死在实现中。

- `dvao_reward_keys`：指定哪些 `non_tensor_batch` 字段参与 DVAO 聚合；
- `dvao_reward_weights`：指定各维度的基础权重；如果不传，就退化为等权。

注意：`focal_dvao` 不允许设置 `dvao_reward_weights`；基础权重应通过 `algorithm.focal.base_weights` 进入 focal 权重，避免重复计权。

### 2.4 `tests/trainer/ppo/test_core_algos_on_cpu.py`

这个测试文件不是辅助材料，而是实现边界的重要说明：

- DVAO 使用组内 population variance，也就是分母按组大小 `G`；
- 如果某个 group 只有 1 条样本，或者所有 reward 维度的组标准差都接近 0，则该组 advantage 直接置 0；
- 配置缺失、维度不匹配、reward key 缺失都会直接报错，而不是静默跳过。

### 2.5 `my/train_dvao.sh`

这是一个实验入口示例。它展示了这版 DVAO 在实际训练中如何接线：

- reward 仍然由 `my/naive_single_reward_client_v5.py` 产出；
- focal reward 后处理被关闭：`algorithm.use_focal=False`；
- advantage 聚合改成 DVAO：`algorithm.adv_estimator=dvao`；
- 参与 DVAO 的多维 reward 来自若干 `reward_signal_*` 字段。

`my/train_focal_dvao.sh` 同样使用 v5 reward client，但设置 `algorithm.adv_estimator=focal_dvao`。此时 reward 侧仍不使用 focal final score，focal 只提供 group-level 权重给 advantage estimator。

## 3. 主逻辑的实现方法

### 3.1 输入

DVAO 的主输入有四类：

- `token_level_rewards`
  - 主要用于确定 batch 大小和输出 shape；
  - DVAO 本身不直接拿它做多维聚合。

- `response_mask`
  - 用来把最终标量 advantage 扩展回 response token 位置。

- `index`
  - 通常就是 `non_tensor_batch["uid"]`；
  - 用来标识哪些样本属于同一个 rollout group。

- `non_tensor_batch`
  - 保存每个样本的多维 reward 分量；
  - DVAO 从这里读取 `dvao_reward_keys` 指定的字段。

### 3.2 配置校验

进入主逻辑前，代码会先做几层强校验：

- 必须有 `config`
- 必须有 `non_tensor_batch`
- 必须有 `index`
- `dvao_reward_keys` 不能为空
- `dvao_reward_weights` 若显式给出，则长度必须和 `dvao_reward_keys` 一致
- 每个 reward key 都必须存在于 `non_tensor_batch`

如果这些条件不满足，当前实现会直接抛 `ValueError`。

### 3.3 组内计算

对每个 rollout group，当前实现做下面几步：

1. 取出该组有效样本在各 reward 维度上的分数矩阵 `valid_group_rewards`。
2. 对每个 reward 维度计算有效样本组均值 `group_mean`。
3. 计算中心化结果 `centered_rewards = group_rewards - group_mean`。
4. 计算每个 reward 维度的组方差与组标准差：
   - `group_var = mean(centered_rewards^2)`
   - `group_std = sqrt(group_var)`
5. 用基础权重和组标准差构造动态权重：
   - `dynamic_weight_denominator = sum(base_weights * group_std)`
   - `dynamic_weights = base_weights * group_std / dynamic_weight_denominator`
6. 对每个 reward 维度计算标准化 advantage：
   - `component_advantages = centered_rewards / (group_std + epsilon)`
7. 用动态权重把各维度 advantage 聚合成一个标量，并将无效样本 advantage 置 0：
   - `group_advantages = sum(component_advantages * dynamic_weights, dim=1)`

最终，每个样本得到一个标量 advantage。

### 3.4 边界条件

当前实现对这些边界直接做零化处理：

- 有效样本数 `<= 1`
- `sum(base_weights * group_std) <= epsilon`
- 整个 group 都无效

这些情况下：

- `dynamic_weights` 全 0
- `group_advantages` 全 0

原因是组内没有可比较性，或者所有维度都没有方差，DVAO 就没有“variance-adaptive”的信息来源。

### 3.5 输出

主函数最后返回：

- `advantages`
- `returns`

当前 DVAO 实现里这两个值相同，都是：

- 先得到每个样本的标量 advantage；
- 再 `unsqueeze(-1)`；
- 最后乘上 `response_mask`，扩展到 response token 维度。

## 4. 当前会写出哪些调试信息

纯 DVAO 会写回：

- `sample_dvao_weight`
- `group_dvao_weight`
- `group_dvao_reward_mean`
- `group_dvao_reward_std`

focal+DVAO 也会写入上面的通用字段，便于 rollout dump 复用；同时额外写回：

- `group_focal_dvao_weight`
- `group_focal_dvao_variance_weight`
- `group_focal_dvao_reward_mean`
- `group_focal_dvao_reward_std`

其中 `group_focal_dvao_weight` 是 focal 权重与组内标准差相乘后重新归一化的最终 advantage 聚合权重；`group_focal_dvao_variance_weight` 是仅由组内标准差归一化得到的 variance factor 视图。

## 5. 在日志和 JSONL 里如何观察 DVAO

### 5.1 训练 metrics

`ray_trainer.py` 会额外记录三类 DVAO 指标：

- `dvao/<reward_key>/{mean,std,max,min}`
  - 当前 step 样本级 reward 分量的分布。

- `group_dvao_weight/<reward_key>/mean`
  - 当前 step 各 rollout group 在该 reward 维度上的平均动态权重。

- `group_dvao_reward_std/<reward_key>/mean`
  - 当前 step 各 rollout group 在该 reward 维度上的平均组内标准差。

`focal_dvao` 会额外记录：

- `focal_dvao/<reward_key>/{mean,std,max,min}`
- `group_focal_dvao_weight/<reward_key>/mean`
- `group_focal_dvao_variance_weight/<reward_key>/mean`
- `group_focal_dvao_reward_std/<reward_key>/mean`

### 5.2 rollout JSONL

如果当前训练流程同时保留了 reward dump，那么 DVAO 会在 dump 前覆盖/补充这些字段：

- `sample_reward_weight`
- `group_mean_reward_weight`
- `group_dvao_weight`
- `group_dvao_reward_mean`
- `group_dvao_reward_std`

含义是：

- focal reward 后处理负责产出统一的 reward 调试骨架；
- DVAO/focal+DVAO advantage 阶段再把真正用于 advantage 聚合的动态权重补进去；
- 最终落盘时，单条样本可以同时看到 reward signal、本组统计量和 advantage 动态权重。

## 6. Focal、DVAO 与 Focal + DVAO

当前仓库中，focal 位于 reward 聚合层，DVAO 位于 advantage 聚合层。为便于比较，定义：

- `r_{i,k}`：样本 `i` 在 rubric `k` 上的 reward；
- `x_{i,k} = r_{i,k} - mu_k`：该 reward 相对组均值的偏移；
- `sigma_k`：rubric `k` 的组内标准差；
- `b_k`：固定基础权重；
- `f_k`：focal 根据当前困难程度得到的 group-level 权重；
- `epsilon`：避免除零的微小正数。

### 6.1 三种方法

**Focal + GRPO** 先将 rubric reward 聚合为标量 reward，再计算 advantage：

```text
R_i = sum_k(f_k * r_{i,k})
A_i = sum_k(f_k * x_{i,k}) / std(R)
```

其中 `std(R)` 是聚合 reward 的组内标准差，包含 rubric 之间的协方差影响。Focal 决定“哪些困难目标应更重要”，因此会改变样本的学习偏好。

**DVAO** 先计算每个 rubric 的标准化 advantage，再用组内标准差构造动态权重：

```text
v_k = b_k * sigma_k / sum_l(b_l * sigma_l)
A_i = sum_k(v_k * x_{i,k} / (sigma_k + epsilon))
```

忽略 `epsilon` 时：

```text
A_i = sum_k(b_k * x_{i,k}) / sum_k(b_k * sigma_k)
```

较大的 `sigma_k` 表示当前 rollout group 中该 rubric 更能区分不同回答，但不一定表示 reward 更可靠；高方差也可能来自噪声或异常值。

**Focal + DVAO** 已实现为 `algorithm.adv_estimator=focal_dvao`，它将 focal 权重作为 DVAO 的动态基础权重：

```text
v_k_final = f_k * sigma_k / sum_l(f_l * sigma_l)
A_i = sum_k(v_k_final * x_{i,k} / (sigma_k + epsilon))
```

最终权重需要再次归一化，使其和为 `1`。忽略 `epsilon` 时：

```text
A_i = sum_k(f_k * x_{i,k}) / sum_k(f_k * sigma_k)
```

### 6.2 语义对比

| 方法 | 主要作用 | 决定样本偏好的权重 | 决定更新强度的分母 |
| --- | --- | --- | --- |
| Focal + GRPO | 关注困难 rubric，并按聚合 reward 排序 | 动态 focal 权重 `f` | 聚合 reward 的实际标准差 |
| DVAO | 在固定目标权重下稳定聚合多目标 advantage | 固定基础权重 `b` | 各 rubric 标准差的基础加权和 |
| Focal + DVAO | 保留困难目标偏好，并按多目标可区分性缩放更新 | 动态 focal 权重 `f` | 各 rubric 标准差的 focal 加权和 |

### 6.3 关键结论

- Focal 主要改变 rubric 学习偏好；DVAO 主要调整不同 rollout group 的更新强度。
- DVAO 的标准差同时出现在动态权重和 rubric advantage 分母中，通常会抵消，因此不能简单解释为“高方差 rubric 获得更大的最终梯度贡献”。
- 同一 group 内，如果 focal 权重固定、忽略 `epsilon` 且分母非零，Focal + GRPO 与 Focal + DVAO 的分子相同，样本排序和 advantage 正负号相同，区别主要是缩放量。
- Focal + GRPO 的分母包含 rubric 协方差；Focal + DVAO 不直接使用协方差，因此在多个 rubric 相互独立或存在冲突时通常更保守。
- 当前 `my/train_dvao.sh` 使用 `algorithm.use_focal=False` 和 `algorithm.adv_estimator=dvao`，纯 DVAO 与候选的 Focal + DVAO 仍应作为独立实验配置。
