# Focal 打不过 Baseline 的根因：输出长度驱动的恶性循环

生成时间: 2026-06-05lish

## 分析数据范围

- Focal: `rollouts/0527_v5_focal_q3_17_n16_e005_t5_g3_clip005_03_Qwen3-1.7B`（参数：ε=0.05, T=5.0, γ=3.0, clip=[0.05, 0.3]）
- Baseline: `rollouts/0527_v5_baseline_q3_17_n16_Qwen3-1.7B`

两者均使用 V5 reward client，各自跑了 33 个 step。

## 结论总览

Focal 打不过 baseline 不是单一原因，而是**输出长度驱动的正向强化循环**导致的。核心机制：

1. Focal 把 ~55% 权重分配给 `instructional_alignment`（~30%）和 `visual_elements`（~25%）
2. 这两个 rubric 的分数与 HTML 长度正相关（delta +0.5/+0.3），PPO 学到"更长输出得更高分"
3. 长输出触发 Tailwind CDN 加载 + JIT 编译延迟 → render 超时
4. **超时样本被错误分类为系统故障**（`render_engine_error`），PPO 收不到任何惩罚
5. 无惩罚的正反馈迫使 PPO 持续推长输出，valid rate 崩溃（98.5% → 60.2% → 38.9%）

---

## 证据 1：长度不是超时的区分因素

Focal step 20 中，按 2000 字符分桶统计成功/超时分布：

| 长度区间 | Success | Timeout | Total | Timeout 率 |
|----------|---------|---------|-------|-----------|
| 6000-8000 | 14 | 8 | 22 | 36.4% |
| 8000-10000 | 61 | 22 | 83 | 26.5% |
| 10000-12000 | 113 | 59 | 172 | 34.3% |
| 12000-14000 | 125 | 82 | 207 | 39.6% |
| 14000-16000 | 120 | 75 | 195 | 38.5% |
| 16000-18000 | 71 | 72 | 143 | **50.3%** |
| 18000-20000 | 46 | 39 | 85 | 45.9% |
| 20000-22000 | 27 | 19 | 46 | 41.3% |
| 22000-24000 | 16 | 10 | 26 | 38.5% |
| 24000-26000 | 5 | 5 | 10 | 50.0% |
| 26000-28000 | 0 | 8 | 8 | **100.0%** |

关键统计：

- **40.2%** 的超时样本比成功样本长度中位数（13772）更短
- 所有 64 个 rollout group 中，都存在比"该组最短超时样本"更长的成功样本
- `min(timeout) - max(success)` 的组均值为 **-9875**，表明成功和超时长度严重重叠
- 一个长度阈值**无法**在截断超时样本的同时不误伤成功样本（用 8000 上限会截断 97.2% 的成功输出）

**结论：是否存在其他原因导致超时取决于其他因素，不可单纯用长度区分。**

---

## 证据 2：超时样本全部被标记为系统故障，PPO 无惩罚（根因 Bug）

### Bug 描述

`RenderEngine.py` 第 788 行使用 `except asyncio.TimeoutError` 捕获超时，但 Playwright 的 `TimeoutError` 继承自 `playwright.Error`，**不是** `asyncio.TimeoutError` 的子类。因此所有 Playwright 超时（`page.set_content` 5s / `page.screenshot` 30s）都落入 `except Exception` 分支 → `status="fail"` → `failure_source="engine"` → `overall_status="render_engine_error"`。

正确分类应为：`render_timeout` → `failure_source="llm"` → `overall_status="llm_generation_error"`。

### 各 step 超时样本的分类情况

| Step | 超时总数 | set_content 超时 | screenshot 超时 | 被标记为 | 被 focal 视为有效？ |
|------|----------|-----------------|----------------|--------|-------------------|
| 15 | 46 | 46 | 0 | render_engine_error | **NO** |
| 18 | 347 | 226 | 121 | render_engine_error | **NO** |
| 20 | 403 | 254 | 149 | render_engine_error | **NO** |
| 25 | 547 | 334 | 213 | render_engine_error | **NO** |
| 30 | 621 | 354 | 267 | render_engine_error | **NO** |

focal_reward.py 第 45 行声明：`VALID_FOCAL_OVERALL_STATUSES = {"success", "llm_generation_error"}`。`render_engine_error` 不在其中，意味着这些超时样本**不参与 focal 权重估计**，被当作"系统崩溃"而非"LLM 生成问题"处理。PPO 无法学习到"长输出会导致失败"这一重要信号，因此输出长度不受约束。

**同期的 baseline 也被同样 Bug 影响，** 但由于 baseline 等权聚合不会系统性驱动长度增长，该 Bug 的破坏性在 baseline 中较为有限。

---

## 证据 3：长输出在关键 rubric 上得分更高（长度被奖励的证据）

以下数据来自 focal 实验，只统计成功样本，按输出长度分组。

**Step 10**（成功样本 1004，长度中位数 6984）：

| Rubric | 短输出（≤6984） | 长输出（>6984） | Delta | 与长度相关系数 |
|--------|:--------------:|:--------------:|-------|:-------------:|
| instructional_alignment | — | — | **+0.523** | >0 |
| visual_elements | — | — | **+0.356** | >0 |
| layout_and_cohesion | — | — | **+0.346** | >0 |
| a11y_score | — | — | -0.420 | <0 |

**Step 15**（成功样本 963，长度中位数 10663）：

| Rubric | 短输出（≤10663） | 长输出（>10663） | Delta |
|--------|:---------------:|:---------------:|-------|
| instructional_alignment | — | — | **+0.378** |
| visual_elements | — | — | **+0.156** |
| layout_and_cohesion | — | — | **+0.112** |

**Step 20**（成功样本 608，长度中位数 13772）：

| Rubric | 短输出（≤13772） | 长输出（>13772） | Delta |
|--------|:---------------:|:---------------:|-------|
| instructional_alignment | — | — | **+0.513** |
| visual_elements | — | — | **+0.278** |
| layout_and_cohesion | — | — | **+0.146** |

**结论：** `instructional_alignment` 是稳健的长度驱动因素（长输出得分比短输出高 0.38~0.52），`visual_elements` 紧随其后（+0.16~0.翻）。这两个 rubric 共同占 focal 总权重的 ~55%。加上 focal 权重没有对长度增长的负反馈机制（证据 2），PPO 自然驱动模型朝"更长输出"方向优化。

---

## 证据 4：Focal 越来越长，且增速远超 Baseline

**输出长度（成功样本均值）轨迹对比：**

| Step | Focal 均值 | Baseline 均值 | 差值 | Focal 有效率 |
|------|-----------|-------------|------|------------|
| 1 | 4031 | 4124 | -93 | 97.7% |
| 5 | 4623 | 5160 | -537 | 99.1% |
| 10 | 7213 | 7158 | +55 | 98.5% |
| 15 | 11238 | 8203 | **+3035** | 94.7% |
| 20 | 14377 | 9773 | **+4604** | 60.2% |
| 25 | 17624 | 12612 | **+5013** | 45.9% |
| 30 | 15847 | 16959 | -1112 | 38.9% |
| 33 | 15556 | 16681 | -1125 | 43.0% |

**Tailwind class 数量（UI 复杂度代理指标）对比：**

| Step | Focal 均值 | Baseline 均值 | 差值 |
|------|-----------|-------------|------|
| 10 | 197 | 176 | +21 |
| 15 | 307 | 199 | +108 |
| 20 | 433 | 241 | +193 |
| 25 | 510 | 309 | +200 |

**解释：**
- Step 10 之前两者长度和 TW class 均匀增长。
- Step 10-25 期间，focal 长度暴涨（从 7k 到 17k，增长 2.4倍），TW class 量从 197 升至 510。baseline 同期从 7k 到 12k（1.7倍），TW class 从 176 到 309。
- Focal 的长度增速（Δ+200 TW class/15 steps ≈ 13.3/step）是 baseline（Δ+68 TW class/15 steps ≈ 4.5/step）的约 **3 倍**。
- Focal 的 valid rate 随长度同步崩溃：98.5% → 60.2% → 38.9%，在这个阶段越来越多的样本因 render 超时被丢弃。

---

## 关于为什么不是"简单加长度惩罚"就能解决的

长度惩罚面临三个结构性障碍：

1. **成功/超时长度高度重叠**（证据 1）：40% 的超时样本比成功中位数短，64/64 个组内都有比超时样本更长的成功样本。

2. **长输出存在真实的 rubric 质量提升**（证据 3）：`instructional_alignment 0.5+` 的长输出提升不是噪声，是真实的"更丰富的页面结构"带来的。

3. **关键 rubrics 需要通过足够的 HTML 结构实现**：`instructional_alignment` 和 `visual_elements` 很难在极短的 HTML 中拿到高分。

因此，限制长度会**系统性牺牲**最高质量输出的上限（现有最好样本将被截断），而非精准惩罚"产生超时"的机制。

---

## 完整因果图

```
[Step 1-10: 前期]
  Focal 将 ~55% 权重分配给 instructional_alignment + visual_elements
    → 这两个 rubric 与 HTML 长度正相关（证据 3: delta +0.5/+0.3）
    → PPO 学习到: "更长的 HTML → 更高的 focal score"
    → 输出长度开始偏离 baseline（证据 4: step 20 时差 +4604）

[Step 11+: 长度起飞阶段]
  更长的 HTML 包含更多 Tailwind class（focal: 433, baseline: 241 @ step 20）
    → 每个输出都包含 <script src="https://cdn.tailwindcss.com">
      这是 Tailwind JIT 编译器，同步阻塞式脚本，没有 defer/async
    → domcontentloaded 需要更长时间才能触发
    → page.set_content(timeout=5000, wait_until="domcontentloaded") 超时

  但超时在 RenderEngine 中未被正确分类:
    → Playwright.TimeoutError 不是 asyncio.TimeoutError 的子类（类层次 Bug）
    → 落入 except Exception 分支 → status="fail" → failure_source="engine"
    → 被标记为 render_engine_error（证据 2: 所有超时样本）
    → focal reward 将其视为无效系统故障，PPO 无任何惩罚

[恶性循环闭合]
  没有惩罚 + "长输出得分高"的正向梯度
    → PPO 持续推高输出长度
    → valid rate 崩溃: 98.5% → 60.2% → 38.9%（证据 4）
    → 训练数据严重偏向（只剩成功样本）
    → 模型质量相对 baseline 全面下降
```

---

## 根本解决方案

### A. 修复 RenderEngine 的 TimeoutError 分类 Bug（必须做）

`RenderEngine.py` 第 788 行添加对 `playwright.TimeoutError` 的捕获：

```python
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

except (asyncio.TimeoutError, PlaywrightTimeoutError):
    return self._format_error_result(status="render_timeout", ...)
```

这会让超时样本被正确分类为 `llm_generation_error`（有效），参与 PPO 训练。模型能够学习到"极长输出会导致问题"这一信号。此修复同样对 baseline 有益。

### B. 提高 set_content 超时时间

将 `timeout=5000` 改为 `timeout=15000`（改 NaiveSingleRewardServer 的构造参数，1 行配置）。不是治本，但提供安全边际。

### C. 替代 Tailwind CDN 方案

在 reward server 端预处理 HTML，将 `https://cdn.tailwindcss.com` 的 JIT 编译器替换为本地预构建的 Tailwind CSS 文件或 `<style>` 内联 CSS。消除 CDN 下载延迟和 JIT 编译时间。

### D. 调整 focal 权重上限

将 `algorithm.focal.weight_max` 从 0.3 降到 0.2，防止单个 rubric 独占过高权重。配合 `epsilon=0.05` 和适度 `gamma`（2.0~3.0）使用。

---

## 与之前诊断报告的差异

之前的报告（`focal_underperforms_root_cause.md`）提出了三个原因：

1. `final` 标尺不可比 → **仍有解释力**，但不是主要问题
2. a11y 权重被压到 0.05 → **仍是有效观察**，但属于结果而非根因
3. ui_spatial 后期偏低 → **成立**，对应本分析的 `visual_elements` + `layout_and_cohesion`

本报告的新发现是：

- **输出长度是驱动 focal 崩溃的直接因素**，其增速远超 baseline
- **RenderEngine 的异常分类 Bug** 是关键助推器，这之前未在诊断中被单独识别
- 长度惩罚**不是**有效解法，因为成功/超时样本长度严重重叠，截断会误伤最好的输出
