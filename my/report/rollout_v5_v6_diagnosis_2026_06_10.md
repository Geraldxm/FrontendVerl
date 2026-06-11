# 0527-0606 Rollout 诊断：v5/v6 reward 与新方法未超过 baseline 的原因

## 0. 结论摘要

本次只基于 `rollouts/` 下已有 JSONL dump 做离线分析，没有重新渲染、重跑 reward server，也没有参考已有诊断报告。参考文档只使用了 `focal_reward_flow.md` 和 `dvao_overview.md`，用于确认 focal/DVAO 字段语义。

核心判断：

1. 评测和跨方法对比应以 `reward_scores.direct` 为主，不能用 focal 的 `final` 均值直接和 baseline/DVAO 比；focal 的 `final` 是重加权后的训练 reward。
2. v5 focal 的 direct 均值只比 baseline 略低，但视觉三项、valid rate 和 render failure 明显更差；所以问题不是“账面 final 低”，而是 focal 训练确实带来了稳定性和视觉质量退化。
3. DVAO 单独看 direct 均值接近 baseline，说明“按组内方差做 advantage”没有直接破坏生成能力；但它也没有带来显著超过 baseline 的优势，因为可优化信号仍被几个长期饱和维度稀释。
4. focal-DVAO 在 v5 的 direct 均值接近 baseline，但其视觉三项均值仍接近 focal，而不是 DVAO/baseline；这是一个危险信号：direct reward 被饱和工程维度托住，不能说明网页质量真的追上。
5. v6 相比 v5 最大改进是评测可观测性和有效样本率；但新方法仍难稳定超过 baseline，因为 reward 体系里 `format_score`、`console_errors`、`network_violations`、`a11y_score`、`element_hit_rate` 大量接近满分，direct 指标对真实视觉改进仍不够敏感。
6. reward hacking 的主要痕迹不是明显“钻空子拿满分”的单一模式，而是更隐蔽的度量错配：模型容易学会输出可解析、可渲染、结构完整、带 Tailwind/CDN/button/link 的通用模板；这些样本能拿到高工程分，但 judge 三项经常只是中等甚至很低。

## 1. 数据范围与统计方法

本批目标 run：

| 版本 | 方法 | 目录 | dump 数 |
| --- | --- | --- | ---: |
| v5 | baseline | `0527_v5_baseline_q3_17_n16_Qwen3-1.7B` | 33 |
| v5 | focal | `0527_v5_focal_q3_17_n16_e005_t5_g3_clip005_03_Qwen3-1.7B` | 33 |
| v5 | dvao | `0603_dvao_n16_Qwen3-1.7B` | 33 |
| v5 | focal_dvao | `0604_focal_dvao_n16_Qwen3-1.7B` | 26 |
| v6 | baseline | `0605_v6_baseline_q3_17_n16_Qwen3-1.7B` | 132 |
| v6 | focal | `0605_v6_focal_q3_17_n16_Qwen3-1.7B` | 116 |
| v6 | dvao | `0606_v6_dvao_q3_17_n16_Qwen3-1.7B` | 93 |
| v6 | focal_dvao | `0606_v6_focal_dvao_q3_17_n16_Qwen3-1.7B` | 95 |

统计方式：对上述 8 个目录做全量字段扫描，覆盖所有 dump 和所有 JSONL 行；每个 dump 1024 条样本。另做少量样本抽查，用于核对低分/高分样本的 HTML、状态和 judge rationale。

## 2. v5：为什么新方法没有超过 baseline

### 2.1 v5 全量统计

| 方法 | direct 均值（评测） | final 均值（训练） | focal 均值 | valid rate | llm error rate | success/render fail/parse/llm | instructional | visual | layout |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | 8.638 | 8.638 | 7.344 | 0.913 | 0.007 | 30623 / 2604 / 341 / 224 | 4.919 | 6.150 | 6.797 |
| focal | 8.557 | 7.265 | 7.265 | 0.722 | 0.010 | 24067 / 9105 / 275 / 345 | 3.740 | 4.722 | 5.263 |
| dvao | 8.629 | 8.629 | 7.338 | 0.942 | 0.007 | 31612 / 1681 / 279 / 220 | 4.987 | 6.396 | 7.053 |
| focal_dvao | 8.568 | 8.568 | 7.288 | 0.766 | 0.009 | 20170 / 5632 / 204 / 230 | 4.058 | 5.043 | 5.613 |

观察：

- focal 与 focal-DVAO 的有效样本率只有 0.722 和 0.766，明显低于 baseline 的 0.913 和 DVAO 的 0.942。
- focal 与 focal-DVAO 的 render failure 分别为 9105/33792、5632/26624，远高于 baseline 的 2604/33792 和 DVAO 的 1681/33792。
- direct 口径下，focal 只比 baseline 低 0.081；但视觉三项和失败率差距很大，说明 direct 均值也被工程饱和维度托住了。
- DVAO 的视觉三项高于 baseline，但 direct 均值几乎相同，说明 direct 指标已经被饱和维度压扁，无法充分反映视觉三项差异。
- focal-DVAO 的 direct 接近 baseline，但视觉三项明显接近 focal，而不是 baseline/DVAO；这说明 focal-DVAO 的 direct score 不能直接当成质量改善证据。

### 2.2 根因一：focal 放大了最难维度，但训练可学习性不足

focal 的权重机制会降低已接近满分的维度权重，提高低分维度权重。当前任务里，容易满分的是：

- `format_score`
- `console_errors`
- `network_violations`
- `a11y_score`
- `element_hit_rate`

真正决定网页是否满足 prompt 的是：

- `instructional_alignment`
- `visual_elements`
- `layout_and_cohesion`

这三个维度来自视觉 judge，噪声更大、延迟更高、可归因性更弱。focal 在方向上更接近目标，但也让 PPO 更早承受高噪声、低可控的梯度；如果没有足够稳定的探索和错误恢复，模型会先掉到 render failure/无效样本坑里。

### 2.3 根因二：v5 focal 的失败样本被打得更狠

抽查低分样本时，focal 低分样本经常是 render failure，但仍有 `format_score=10`、`a11y_score=10` 这类工程分。由于 focal 权重集中到 judge 三项，judge 三项为 0 时 final 可低到 1.5；baseline/direct 聚合则仍被工程分托住。

这意味着两种现象同时存在：

- 从“真实网页质量”看，focal 在惩罚失败样本上更合理。
- 从“训练用 final 指标”看，focal 会显得更差，因为它对失败样本更严；但跨方法评测不应使用这个口径。

### 2.4 根因三：DVAO 的动态方差没有解决 reward 目标错配

DVAO 利用组内方差决定 advantage 权重。它能避免某些无方差维度继续主导 advantage，但如果组内主要差异仍来自“可渲染/模板复杂度/工程结构”，而不是 prompt 对齐和视觉审美，DVAO 只能重新分配已有信号，不能创造更好的目标。

v5 DVAO 视觉三项高于 baseline，是积极信号；但 direct score 没明显领先，说明最终评测指标不够敏感，或者 DVAO 带来的质量改善被 baseline 已经接近饱和的工程分抵消。

## 3. reward hacking 痕迹

### 3.1 不是单一作弊，而是模板化高工程分

高分样本通常包含完整 HTML、Tailwind CDN、Font Awesome、button/link、渐变、卡片、若干 SVG/图标等。它们容易拿到：

- 格式满分；
- console/network/a11y 接近满分；
- element hit rate 满分；
- layout 基础分不低。

但 judge rationale 经常指出的问题是：没有真正实现 prompt 的独特布局、缺少真实视觉元素、只是文字卡片、移动端错位、过于通用模板。这是“模板化合规”而不是“真实满足需求”。

### 3.2 失败样本被工程分托底

抽查低分样本中，有些 `llm_generation_error` 或 render timeout 的样本仍能拿到 3.75 或更高；有些 render failure 样本在 focal 下 `format/a11y` 仍为 10，但 judge 三项为 0。

这说明 reward 里存在托底项：只要输出看起来像 HTML 或满足某些静态规则，即使最终截图/视觉 judge 失败，也不会被打到接近 0。baseline 更容易从这个托底中受益。

### 3.3 长 HTML 和复杂 Tailwind 不等于质量

v6 全量统计里，DVAO 的平均 `html_chars` 和 Tailwind class 数更高，direct 均值也只比 baseline 高 0.020。复杂度增加可能带来更多视觉组件，也可能带来 timeout、移动端溢出、内容堆砌。后续应把 `html_chars`、Tailwind class 数、render timeout、judge 三项做联合分析，避免把“写得多”误判成“做得好”。

## 4. 相关性与隐含 hacking 分析

相关性使用全量 JSONL 字段扫描计算。v5 没有可靠的 `html_chars` / Tailwind debug 字段，所以长度相关性只对 v6 计算。

### 4.1 reward 维度之间的相关性

| 方法 | inst-visual | inst-layout | visual-layout | subjective-direct | engineering-direct | subjective-engineering |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 baseline | 0.814 | 0.795 | 0.923 | 0.514 | 0.658 | 0.197 |
| v5 focal | 0.896 | 0.885 | 0.965 | 0.246 | 0.690 | 0.039 |
| v5 dvao | 0.764 | 0.747 | 0.911 | 0.608 | 0.654 | 0.222 |
| v5 focal_dvao | 0.892 | 0.879 | 0.960 | 0.259 | 0.352 | 0.187 |
| v6 baseline | 0.884 | 0.864 | 0.967 | 0.973 | 0.198 | 0.081 |
| v6 focal | 0.891 | 0.875 | 0.967 | 0.970 | 0.266 | 0.123 |
| v6 dvao | 0.867 | 0.845 | 0.960 | 0.960 | 0.283 | 0.123 |
| v6 focal_dvao | 0.884 | 0.865 | 0.964 | 0.970 | 0.239 | 0.091 |

解释：

- 三个 VLM 主观分高度绑定，尤其 `visual_elements` 与 `layout_and_cohesion` 基本是同涨同跌。这说明它们不像三个完全独立 reward，更像一个主观视觉质量因子的不同投影。
- v5 中 `subjective_avg` 与 direct 的相关性偏低，尤其 focal / focal-DVAO 只有 0.246 / 0.259；direct 更受工程分影响。这解释了为什么 v5 focal 的 direct 只略低，但视觉三项和失败率已经明显恶化。
- v6 中 `subjective_avg` 与 direct 的相关性接近 0.96-0.97，说明 v6 direct 已更强地跟随 VLM 三项；但 `subjective_avg` 与工程分仍很弱相关，工程分长期饱和的问题仍在。

### 4.2 reward 与长度/复杂度的相关性

| 方法 | html-direct | html-subjective | html-layout | tailwind-direct | tailwind-subjective |
| --- | ---: | ---: | ---: | ---: | ---: |
| v6 baseline | -0.251 | -0.246 | -0.248 | -0.215 | -0.215 |
| v6 focal | -0.240 | -0.240 | -0.254 | -0.184 | -0.191 |
| v6 dvao | -0.033 | -0.061 | -0.100 | -0.013 | -0.041 |
| v6 focal_dvao | -0.302 | -0.288 | -0.294 | -0.284 | -0.278 |

解释：

- 没有看到“越长越高分”的全局规律；baseline、focal、focal-DVAO 中，HTML 长度和 Tailwind class 数与 subjective/direct 都是弱负相关。
- DVAO 是例外：长度/复杂度与 reward 的相关性接近 0，而不是明显负相关。结合 DVAO 的平均 `html_chars=8273`、Tailwind class 数 284，说明 DVAO 更能容忍或鼓励超长复杂页面，但这种复杂度没有稳定转化为更高 direct。
- 这不支持“简单堆长度就能 hack reward”的结论，但支持“复杂模板可能成为某些高分样本的局部模式”。

### 4.3 高主观分样本中的复杂模板倾向

这里把 `subjective_avg >= 8` 视为高 VLM 主观分，把 `html_chars >= 8000` 视为超长页面。

| 方法 | 高主观分占比 | 高主观分中超长占比 |
| --- | ---: | ---: |
| v6 baseline | 0.279 | 0.037 |
| v6 focal | 0.269 | 0.130 |
| v6 dvao | 0.250 | 0.541 |
| v6 focal_dvao | 0.205 | 0.086 |

最值得注意的是 v6 DVAO：高主观分样本中 54.1% 是超长页面。抽查这些样本后，常见模式是：

- 生成非常完整的多 section 首页，覆盖 properties/testimonials/about/contact 等显式要求；
- 大量 card/grid/section/shadow/animation/Tailwind class，页面“看起来很完整”；
- VLM rationale 往往奖励“覆盖了所有 section”“专业配色”“卡片层级清楚”；
- 但同时出现移动端 header 拥挤、搜索栏截断、a11y 较低、外部 placeholder 图片或 Google Fonts 依赖等问题。

因此，隐含 reward hacking 的更准确表述是：没有证据表明 policy 发现了“单纯拉长 HTML 就涨分”的全局 hack；但 DVAO 可能诱导出一种“超完整多 section 模板”的局部模式，它确实容易获得 VLM 主观高分，且 VLM 对移动端拥挤、外部资源依赖、过度模板化的惩罚不够强。

一个典型抽查样本是 `0606_v6_dvao/.../2.jsonl` 第 270 行：Real Estate prompt 下生成 14984 字符、504 个 Tailwind class、66 个 section/div、90 次 card/grid/flex/shadow/rounded 相关结构；VLM 给出 `instructional_alignment=10`、`visual_elements=9`、`layout=8.5`，但 `a11y_score=4.7826`，rationale 也承认移动端搜索栏文本截断。这类样本不是纯粹无效 hack，但显示 VLM 主观分会偏爱“完整复杂模板”，对实际可用性问题惩罚不足。

## 5. v5 rollout 呈现的趋势

1. 工程类 reward 过早饱和：`network_violations` 和 `element_hit_rate` 基本满分，`format/a11y/console` 也长期高位。
2. 真正的瓶颈是视觉三项：v5 baseline 的 instructional 只有 4.919，visual 6.150，layout 6.797；其中 prompt 对齐仍明显偏低，这不是“已经很好只差一点”的状态。
3. focal 的方向更接近真实瓶颈，但稳定性差：它强调视觉三项后，render failure 和无效样本显著上升。
4. DVAO 对稳定性更友好：v5 DVAO 的 valid rate 和 render success 最好，视觉三项也最好；但 direct score 未体现足够收益。
5. focal-DVAO 有指标幻觉：direct 接近 baseline，但视觉三项与有效样本率都像 focal，不能简单解读为 focal+DVAO 已追上 baseline。

## 6. v6：有什么改进，为什么仍不能超过

### 6.1 v6 全量统计

| 方法 | direct 均值（评测） | final 均值（训练） | focal 均值 | valid rate | llm error rate | success/parse/llm | instructional | visual | layout | html chars | Tailwind classes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 8.472 | 8.472 | 6.919 | 0.991 | 0.166 | 111515 / 1184 / 22469 | 5.135 | 6.106 | 6.511 | 4571 | 168 |
| focal | 8.409 | 6.822 | 6.822 | 0.991 | 0.177 | 96694 / 1069 / 21021 | 5.057 | 5.978 | 6.365 | 5894 | 214 |
| dvao | 8.492 | 8.492 | 6.977 | 0.990 | 0.141 | 80890 / 956 / 13386 | 5.092 | 6.213 | 6.674 | 8273 | 284 |
| focal_dvao | 8.314 | 8.314 | 6.624 | 0.991 | 0.181 | 78819 / 846 / 17615 | 4.715 | 5.745 | 6.197 | 5063 | 192 |

### 6.2 v6 的改进

- 有效样本率从 v5 的 0.72-0.94 区间提高到约 0.990-0.991。
- v5 的 `render_engine_error` 在 v6 中被更清楚地拆成 `render_timeout`、`extract_error`、`llm_generation_error`，并额外写出 `render_debug_info`。
- 全量看，v6 的主要改进是稳定性和可观测性，而不是每个 judge 均值都单调抬升：baseline instructional 从 4.919 到 5.135，但 visual 从 6.150 到 6.106 基本持平，layout 从 6.797 到 6.511 略降；DVAO 的 layout 也从 7.053 到 6.674 略降。
- v6 focal-DVAO 比 v5 focal-DVAO 稳定很多，valid rate 从 0.766 到 0.991。

注意：共同 step 对齐也必须用 direct 口径重算。此前基于 `final` 的对齐只能说明训练 reward 变化，不能作为 focal 与其他方法的评测比较。

### 6.3 为什么还是不能超过 baseline

第一，v6 baseline 本身也显著变强。新 reward server/client 改善了评测稳定性后，baseline 同样受益；DVAO 在全量 direct 均值上只小幅高于 baseline，优势很弱，focal 和 focal-DVAO 仍低于 baseline。

第二，focal 的训练目标和评测口径不同。focal final 会重压低分视觉维度，因此它更严格；跨方法评测应看 direct。但即使用 direct，focal 仍没有超过 baseline，因为 direct 里仍保留大量高位工程分，且 focal 的视觉三项与失败率没有改善。

第三，DVAO 依赖组内可比较性。当前每个 prompt 的 16 个样本里，很多维度已经满分或近满分，组内方差主要来自少数视觉维度和失败样本。DVAO 可以突出方差，但如果方差来源包括 timeout、解析失败、模板复杂度，它的优势会被噪声抵消。

第四，训练长度不同。v6 baseline 有 132 个 dump，focal 有 116 个，DVAO/focal-DVAO 只有 93/95 个。最终最好点与末段趋势需要按共同 step 对齐，否则会把训练时长差异混进方法差异。

## 7. 后续可复用分析方法

后续分析同类 rollout 时，建议按这个顺序做：

1. 先列 run 基本面：目录名、dump 数、每 dump 行数、总样本数、训练 step 范围。
2. 分开看三类分数：`reward_scores.direct` 作为跨方法评测主口径，`reward_scores.final/focal` 作为训练目标和机制诊断口径，另看 `sample_reward_signal` 八个维度、成功样本内的 judge 三项。
3. 先算状态分布：`overall_status`、`render_info.status`、`failure_source`、`judge_info.status`、`valid_sample`、`is_llm_generation_error`。
4. 对齐训练长度：比较全程、共同 step 范围、前 5 个 dump、后 5 个 dump、最佳 dump，不要只看最后一个目录。
5. 对 focal 特别看：`sample_reward_weight`/`group_mean_reward_weight` 是否长期压到 `instructional_alignment`、`visual_elements`、`layout_and_cohesion`，以及低分样本是不是主要来自无效/timeout。
6. 对 DVAO 特别看：`group_dvao_reward_std`、`group_dvao_weight` 或 focal-DVAO 的最终权重是否被失败样本、格式分或复杂度信号主导。
7. 抽查样本必须覆盖四类：高 direct 高 judge、低 direct 低 judge、高 direct 低 judge、低 final 但工程分高。
8. 检查 reward hacking 迹象：通用模板、外部资源依赖、空洞卡片堆砌、只满足 a11y/格式规则、超长 HTML、移动端布局破坏、judge rationale 与数值不一致。
9. 报告时不要用总 final 排名做跨方法比较；至少同时报告 direct、judge 三项均值和失败率，否则 baseline 会因工程分饱和被高估。
10. 对 v6 以后固定计算相关性：VLM 三项互相关、`subjective_avg` 与 direct/工程分相关、`html_chars`/Tailwind 数与 subjective/direct 相关，并抽查“高主观分 + 超长/高复杂度”样本。

## 8. 建议的下一步实验

1. 建立一个“视觉主指标”：只聚合 `instructional_alignment`、`visual_elements`、`layout_and_cohesion`，并只在 success/valid 样本上比较；用它作为 focal/DVAO 是否真的改善质量的主判断。
2. 对 failure 做硬惩罚：render timeout、extract error、judge parse fail 不应被 `format/a11y/network` 托底到过高分数。
3. 对 focal 做 warmup：前若干 step 使用 direct 或较小 gamma，等 valid rate 稳定后再提高视觉三项权重。
4. 对 DVAO 做噪声过滤：无效样本不参与 std，且 timeout/parse fail 单独作为失败惩罚，不进入方差自适应的正向权重竞争。
5. 做 paired prompt 对比：固定同一批 task_id/uid，比每个 prompt 内四种方法的 best-of-16、median、failure rate，而不是只比全局均值。
6. 使用 v6 debug 字段跟踪复杂度：把 `html_chars`、`tailwind_class_count_total`、render timeout、judge 三项联合画图，确认是否存在“越写越长但不更好”的复杂度陷阱。
