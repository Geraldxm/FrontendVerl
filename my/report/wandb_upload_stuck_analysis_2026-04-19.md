# W&B 上传卡死问题分析与处理记录（2026-04-19）

## 1. 问题背景
- 工作目录：`/inspire/hdd/global_user/gexinmu-253108100065/Repos/FrontendVerl/wandb`
- 用户现象：最新几个离线 run 在执行 `wandb sync` 时长期卡住，无法完成上传。
- 目标：定位根因，恢复两条关键 run 上传。

## 2. 异常发现过程（How we found the anomaly）

### 2.1 先看目录与体积分布
先做目录与体积扫描，发现：
- 大量 `offline-run-*` 存在，且近期有两条 run 的 `wandb-summary.json` 异常大。
- 同目录有超大的 `debug-cli.root.log`（约 320MB），说明历史重试非常频繁。

### 2.2 先抓运行态
通过进程查看，发现存在长期停留的同步进程：
- `wandb sync offline-run-20260415_080438-n1ksxdw4`
- PID: `60828`

### 2.3 从日志提取“失败签名”
在 `wandb/debug-cli.root.log` 中提取关键词后，看到稳定重复模式：
- `422 Client Error ... /file_stream`
- `Read timed out`（`api.wandb.ai`）
- `ProxyError(Remote end closed connection without response)`

关键判断：
- 真正致命信号是 `422`（服务端拒绝 payload）。
- `timeout/proxy` 是重试链路里的伴生现象，不是唯一根因。

## 3. 排查过程（How we narrowed it down）

### 3.1 定位到具体异常 run
聚焦两条失败 run：
- `offline-run-20260415_075321-scjbk920`
- `offline-run-20260415_080438-n1ksxdw4`

对 `files/wandb-summary.json` 做 key 长度统计：
- `scjbk920`: 文件大小 `1,664,342` bytes，最大 key 长度 `1,653,998`
- `n1ksxdw4`: 文件大小 `850,069` bytes，最大 key 长度 `839,509`

这已经明显异常（metric key 不应达到百万字符级）。

### 3.2 锁定异常 key 结构
异常 key 形态为：
- `reward_status/error_message/<超长错误文本>/rate`

其中 `<超长错误文本>` 内包含：
- `data:image/png;base64,...` 这类超长 payload。

### 3.3 回溯到代码
根因代码位于：
- `verl/trainer/ppo/focal_reward.py`

原逻辑路径：
1. `_normalize_error_message` 返回完整错误文本（含可能超长 payload）
2. `slugify_reward_name` 对文本做 slug
3. 直接生成 metric key：`reward_status/error_message/{error_slug}/rate`

结果：
- 超长错误消息被直接“结构化为 key 名”，进入 W&B summary 与事件流。

### 3.4 一次关键二次定位：仅改 summary 不够
首次仅清洗 `files/wandb-summary.json` 后重传，仍然 `422`。
继续排查后确认：
- `wandb sync` 会重放 `run-*.wandb` 事件流，
- 事件流内部（protobuf `nested_key`）仍保留旧的超长 key，
- 因此即使 summary 文件干净，上传依然失败。

这是本次排查里最关键的“第二层根因”。

## 4. 解决过程（How we fixed it）

### 4.1 先做止血隔离
先把两条异常 run 移入 `wandb/offline-back/`，避免影响其它 run 的批量 sync。

### 4.2 修复代码（防止未来再发生）
修改文件：
- `verl/trainer/ppo/focal_reward.py`

改动要点：
1. 新增 `compact_metric_slug(name, max_len=180)`
- 对 key 名限长；
- 超长时截断并追加 `sha1` 后缀，保留可区分性。

2. 增强 `_normalize_error_message`
- 先替换 `data:image/...` 内联负载为占位符；
- 对超长错误文本截断并加 hash。

3. error_message 指标统一走限长 key
- `error_slug = compact_metric_slug(error_message, max_len=180)`

4. 基础校验
- `python3 -m py_compile verl/trainer/ppo/focal_reward.py` 通过。

### 4.3 清洗历史坏数据（恢复这两条历史 run 上传）
对两条 run 执行两层清洗并保留备份：

1. 清洗文件层：
- `files/wandb-summary.json`
- 将超长 key 压缩为短 key + hash
- 备份：`wandb-summary.json.bak_before_clean`

2. 清洗事件流层（关键）：
- `run-scjbk920.wandb`
- `run-n1ksxdw4.wandb`
- 重写 protobuf 中 `history/summary` 的 `nested_key`（超长转短 key）
- 清洗后最长 `nested_key` 统一降到 `180`
- 备份：`run-*.wandb.bak_before_nestedkey_clean`

### 4.4 重新上传验证
执行同步结果：
- `wandb sync offline-run-20260415_075321-scjbk920` -> `done.`
- `wandb sync offline-run-20260415_080438-n1ksxdw4` -> `done.`

对应 run 链接：
- `https://wandb.ai/1979986188-jilin-university/frontend_focal/runs/scjbk920`
- `https://wandb.ai/1979986188-jilin-university/frontend_focal/runs/n1ksxdw4`

## 5. 本次结论

### 5.1 主因
主因不是纯网络，而是：
- 错误文本（包含超长 base64）被直接拼入 metric key，
- 导致 `.wandb` 事件流与 summary 中出现超长 key，
- 服务端在 `file_stream` 阶段返回 `422`，客户端持续重试表现为卡住。

### 5.2 次要现象
- `Read timed out` / `ProxyError` 是重试期间出现的伴随异常，不能单独解释问题本体。

### 5.3 已完成结果
- 两条目标 run 已成功上传。
- 代码已加防护，后续新 run 不应再因同类超长 key 问题卡在上传阶段。

## 6. 备注
- 过程中发现某些 `wandb sync` 进程在当前环境中存在“`ps` 可见、信号返回 `No such process`”现象，表现为命名空间/进程视图不一致。
- 该现象不影响本次根因判断与最终修复。
