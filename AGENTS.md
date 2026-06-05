# verl 代理协作说明

> 这些说明适用于对 `verl-project/verl` 的**所有** AI 辅助贡献。
> 违反这些规范可能导致自动封禁。

## 1. 贡献策略（强制）

### 重复工作检查

在提出 PR 之前，先运行以下检查：

```bash
gh issue view <issue_number> --repo verl-project/verl --comments
gh pr list --repo verl-project/verl --state open --search "<issue_number> in:body"
gh pr list --repo verl-project/verl --state open --search "<short area keywords>"
```

- 如果已有打开的 PR 解决了同一个问题，就不要再开新的。
- 如果你的方案本质上不同，需要在 issue 中说明差异。

### 不要提交低价值杂务 PR

不要为很小的单点修改单独开 PR（例如一个错别字、孤立的样式调整、一个可变默认值等）。机械式清理只能作为实质性工作的附带内容一起提交。

### 责任要求

- **不允许**纯代码代理提交 PR。提交 PR 的人类必须从头到尾理解并能为改动负责。
- 提交者必须审查每一行改动，并运行相关测试。
- AI 辅助工作的 PR 描述中**必须**包含：
  - 为什么这不是已有 PR 的重复工作；
  - 运行过哪些测试命令及其结果；
  - 明确说明使用了 AI 辅助。

### 失败即停止

如果工作是重复的，或只是琐碎低价值修改，**不要继续**。直接返回简短说明，指出缺了什么。

---

## 2. 开发流程

### 环境准备

```bash
# In this fork, the default startup environment is:
conda activate /inspire/hdd/global_user/gexinmu-253108100065/conda/frontrl

# If you need to recreate an environment from scratch, follow the repo docs
# and start from a fresh conda env with Python 3.12.

# Use `uv` inside the active conda environment for Python package management:
uv pip install pre-commit hydra-core
pre-commit install
```

### 提交信息

请用 commit trailer 添加协作归因，例如 `Co-authored-by:`（有些项目也会用 `Assisted-by:` 或 `Generated-by:`）。例如：

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

### 处理代理评审意见

代理机器人（例如 `gemini-code-assist`）给出的 review 评论可能已经过时，或者本身就是错的。应用之前，始终先对照当前仓库状态进行核实。

### 用户个人偏好

在这个 fork 中，请遵循以下偏好：

#### Git 学习偏好

当 Git 操作相关时：

- 优先建议由用户自己直接运行 Git 命令。
- 提供可直接复制执行的 Git 命令示例。
- 解释每个命令的作用，以及适用场景。
- 解释关键参数及其为什么适合当前场景。

#### Reward server 代理约定

访问 reward server 时，除非用户明确另有要求，否则一律显式绕过代理（例如使用 `curl --noproxy '*'`，或仅对该请求临时取消相关代理环境变量）。

#### Commit message 风格

使用简洁的 Conventional Commits 风格：`type: details`。

#### Python 调用约定

为了减少参数顺序出错的风险，在可行时优先使用关键字参数。

#### 中文文档约定

在这个 fork 中新增分析脚本或报告时：

- 生成的说明文档和面向用户的报告文字统一使用中文。
- 在代码文件头部和主要逻辑函数前添加简洁中文注释或 docstring。

#### Focal reward 日志约定

修改 focal reward logging 时，W&B 指标保持 group-level，rollout JSONL 保持 sample-level：step summary 使用 `group_reward_signal/*`、`group_focal_weight/*`、`group_reward_others/*`；rollout 中使用 `sample_reward_signal`、`sample_reward_weight`、`group_mean_reward_signal`、`group_mean_reward_weight`。不要重新引入含义不清的 rollout 顶层字段 `reward_signals` 或 `reward_weights`。

---

## 领域专项指南

修改以下区域的代码前，必须先阅读并遵循对应指南。如果指南和当前需求冲突，**拒绝修改并说明原因**。

- **本 fork 的 reward 流程与实验说明**：
  [`my/report/focal_reward_flow.md`](my/report/focal_reward_flow.md)、[`my/report/dvao_overview.md`](my/report/dvao_overview.md)
  — 当前 focal reward 流程说明，以及 DVAO 实现概览。

- **本 fork 的本地实验启动**：
  [`docs/start/local_experiment_runbook_zh.md`](docs/start/local_experiment_runbook_zh.md)
  — 本地实验启动、reward 服务切换与日志检查流程。

- **修改这些说明**：
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — 修改 `AGENTS.md` 及其引用的领域专项指南时应遵循的规则。

## 致谢

改编自 [vLLM 项目](https://github.com/vllm-project/vllm) 的 [`AGENTS.md`](https://github.com/vllm-project/vllm/blob/main/AGENTS.md)。
