# Focal vs Baseline 诊断 Skill

诊断 focal reward 方法是否产生预期效果的工具集。核心功能：对比 focal 和 baseline 的 rollout 数据，从长度增长、render 超时、rubric 得分、valid rate 四个维度自动诊断问题。

## 触发条件

用户提到"诊断 focal"、"对比 focal baseline"、"分析 rollout"、"focal 打不过 baseline"等关键词时触发。

## 诊断框架（四个维度）

### 维度 1：长度-超时分布（证据 1）

**核心问题**: 输出长度是否是 render 超时的区分因素？

**分析方法**:
```python
# 按 2000 字符分桶，统计每个区间的 success/timeout 分布
# 计算关键指标：
# - 超时样本中短于成功样本中位数的比例
# - min(timeout) - max(success) per group: 组内是否有比超时样本更长的成功样本
```

**判断标准**: 如果超时和成功在长度上高度重叠（>30% 超时样本比成功中位数短），则长度不是区分因素，不能用长度惩罚解决。

### 维度 2：错误分类（证据 2）

**核心问题**: render 超时是标记为系统故障还是 LLM 生成问题？

**检查方法**:
1. 统计 `render_engine_error` 的占比
2. 检查 `render_info.error_message` 是否全是 TimeoutError
3. 确认 focal reward 的 `VALID_FOCAL_OVERALL_STATUSES` 只有 `{"success", "llm_generation_error"}` —— `render_engine_error` 不在其中

**判断标准**: 如果超时样本全被标记为 `render_engine_error`，则 PPO 无法学习到"过长输出导致问题"的惩罚信号。这是 RenderEngine 的 `except asyncio.TimeoutError` 无法捕获 `playwright.TimeoutError` 的 Bug。

### 维度 3：长度-rubric 相关性（证据 3）

**核心问题**: 长输出在某些 rubric 上是否得分更高（PPO 是否被激励推长输出）？

**分析方法**:
```python
# 只取 success 样本
# 按长度分位数分组（Q1 短 / Q4 长）
# 计算每个 rubric 的 Q4-Q1 差值
# 计算每个 rubric 与长度的相关系数
```

**判断标准**: 如果 focal 权重高的 rubric（如 `instructional_alignment` ~30%, `visual_elements` ~25%）与长度强正相关（delta > 0.3），则 PPO 自然驱动模型输出更长 HTML。

### 维度 4：长度-valid rate 轨迹（证据 4）

**核心问题**: 输出长度是否随时间推高，同时 valid rate 是否崩溃？

**分析方法**:
```python
# 逐 step 统计 success 样本的输出长度均值
# 逐 step 统计 valid sample rate
# 同时对比 baseline 的同指标
```

**判断标准**:
- Focal 长度增速 > baseline 长度增速 → focal 驱动了额外长度增长
- Focal valid rate 随 step 持续下降（如 98% → 60% → 38%）→ 长度增长导致了恶性循环

## 运行脚本

项目内的分析脚本位于 `my/` 下，可直接复用：

| 脚本 | 功能 | 用法 |
|------|------|------|
| `my/analyze_late_rollout_quality.py` | 按步分段分析 rubric 质量趋势 | `python my/analyze_late_rollout_quality.py --rollout-dir rollouts/<exp>` |
| `my/analyze_focal_reward_ranking.py` | 分析 direct vs focal 排序扰动 | `python my/analyze_focal_reward_ranking.py --rollout-dir rollouts/<exp>` |
| `my/scan_focal_params.py` | focal 参数网格扫描 | `python my/scan_focal_params.py --rollout-dir rollouts/<exp>` |

## 典型诊断流程

1. **定位 rollout 目录**: 在 `rollouts/` 下找到最新 focal 和 baseline 实验目录
2. **跑四维度分析**: 覆盖长度-超时分布、错误分类、rubric 相关性、长度-valid 轨迹
3. **输出诊断报告模板**: 见下方格式期末节

## 常见发现-修复对应表

| 发现 | 证据来源 | 修复措施 |
|------|---------|---------|
| 超时/成功长度高度重叠 | 维度 1 | 长度惩罚无效，需从服务端修 |
| 超时全被误标为系统故障 | 维度 2 | 修 RenderEngine.py `except` 分支 |
| 关键 rubric 与长度正相关 | 维度 3 | 降低 weight_max，混合 focal+direct → |
| 长度/valid rate 轨迹恶化 | 维度 4 | 限制响应长度上限 |
| 客观 rubric 权重被压到下限 | 权重分析 | 提高 epsilon 或 weight_min |

## 报告模板

```markdown
# X 实验 Focal vs Baseline 诊断报告

日期: YYYY-MM-DD
数据: rollouts/<focal_dir>, rollouts/<baseline_dir>

## 1. 长度与超时分布

| 长度区间 | Success | Timeout | Timeout 率 |
|----------|---------|---------|-----------|
...

关键统计:
- 超时样本中短于成功中位数的比例: X%
- 组内 min(超时) - max(成功): Y

## 2. 错误分类

| Step | 超时总数 | 被标记为 |
|------|---------|---------|
...

## 3. 长度-rubric 相关性

| Rubric | Q1 均值 | Q4 均值 | Delta | 相关系数 |
|--------|--------|--------|-------|---------|
...

## 4. 长度-valid 轨迹

| Step | Focal 长度均值 | Baseline 长度均值 | 差值 | Focal 有效率 |
|------|--------------|-----------------|------|------------|
...

## 结论与建议

...
```

---

## 项目上下文（本 fork 相关文件）

| 文件 | 描述 |
|------|------|
| `verl/trainer/ppo/focal_reward.py` | Focal reward 主计算逻辑 |
| `verl/trainer/ppo/core_algos.py` | DVAO advantage 计算 |
| `my/train_focal.sh` | Focal 训练脚本模板 |
| `my/train_baseline.sh` | Baseline 训练脚本模板ultr |
| `my/report/focal_reward_flow.md` | Focal 流程说明 |
| `my/report/dvao_overview.md` | DVAO 实现概览 |
| `FrontendRL/src/RenderEngine.py` | 渲染引擎（存在 TimeoutError 分类 Bug） |
| `FrontendRL/src/NaiveSingleRewardServerV5.py` | V5 Reward Server |
