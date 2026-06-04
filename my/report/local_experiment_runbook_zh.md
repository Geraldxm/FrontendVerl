# 本地实验启动指南

本文档记录这个 fork 中常用的本地实验启动流程，重点覆盖本地配置、启动脚本修改点、reward 服务地址调整，以及日志查看方式。

## 1. 准备本地配置

优先使用本地未跟踪配置文件保存环境相关参数，不要把敏感信息直接写进训练脚本。

- 搜索 `my/train_local.sh`
- 如果不存在，就从 `my/train_local.sh.example` 复制一份
- 保持 `my/train_local.sh` 为未跟踪状态

建议在本地配置里维护这些值：

- `MODEL_PATH`
- `WANDB_API_KEY`
- `WANDB_MODE`
- 本地注释里约定的 conda 启动方式

## 2. 选择启动脚本

当前常用脚本：

- `my/train_focal.sh`
- `my/train_dvao.sh`
- `my/train_focal_dvao.sh`
- `my/train_baseline.sh`

这两个脚本都会优先读取 `my/train_local.sh`，然后再使用脚本内定义的实验参数。

## 3. 启动前需要修改的内容

### 3.1 调整训练起始模型

一般先在本地配置中修改 base model，也就是 `MODEL_PATH`，作为训练起始模型。

如果实验需要切换到新的 checkpoint，优先改这里，而不是把模型路径硬编码到公共脚本里。

### 3.2 调整 TP size

在对应实验脚本中检查并修改 tensor parallel size。当前脚本里常见参数名是：

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

改这个值时，同时确认 GPU 数量和部署方式匹配，避免启动后才发现并行配置不一致。

### 3.3 修改实验名称

在 `my/train_local.sh` 顶部注释提到的“参考启动脚本”位置，先更新这次实验使用的实验名或前缀。

当前训练脚本通常要求先设置：

```bash
EXPERIMENT_PREFIX=...
```

然后脚本会自动拼出：

```bash
EXPERIMENT_NAME="${EXPERIMENT_PREFIX}_$(basename "$MODEL_PATH")"
```

所以实验名通常由“实验前缀 + 当前模型目录名”组成。启动前先确认这一点，避免 W&B、rollout 目录和实际实验目的对不上。

### 3.4 修改 reward 服务地址和相关参数

如果本次实验使用的 reward 服务变了，需要在对应实验脚本里同步修改：

- IP
- 端口
- 完整服务地址
- 与该服务配套的相关参数

当前脚本里使用的是这类配置：

```bash
export REWARD_SERVER_URL="http://10.244.71.6:48002/compute_reward_v3"
```

或者：

```bash
export REWARD_SERVER_URL="http://10.246.103.127:48002/compute_reward_v3"
```

如果切换了 reward server，不要只改地址；还要一起检查：

- `reward.custom_reward_function.path`
- `reward.custom_reward_function.name`
- 该 reward 服务要求的输入输出格式
- focal / baseline 脚本是否引用了同一套 reward 逻辑

## 4. 启动实验

在默认环境中启动：

```bash
conda activate /inspire/hdd/global_user/gexinmu-253108100065/conda/frontrl
```

然后按目标实验选择脚本，例如：

```bash
EXPERIMENT_PREFIX=your_exp_name bash my/train_focal.sh
```

或：

```bash
EXPERIMENT_PREFIX=your_exp_name bash my/train_baseline.sh
```

## 5. 启动后检查日志

实验拉起后，及时去 log 中看进程状态，至少确认以下几点：

- 训练进程是否真的启动
- reward 服务是否可连通
- 模型路径是否加载正确
- TP size 是否和预期一致
- W&B 是否正常记录

如果日志里出现 reward 请求失败、端口不通、模型路径错误或并行配置不匹配，优先回到第 3 节核对启动前修改项。

## 6. 最小检查清单

启动前至少确认一次：

- 已激活 `frontrl` 环境
- `MODEL_PATH` 指向正确的起始模型
- `EXPERIMENT_PREFIX` 已更新
- `actor_rollout_ref.rollout.tensor_model_parallel_size` 已核对
- `REWARD_SERVER_URL` 指向本次实验对应服务
- 相关 reward 参数已同步调整
- 准备好查看日志定位启动问题
