# cv_agent

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./pyproject.toml)

[English](./README.md)

`cv_agent` 是面向 YOLO 目标检测的迭代训练命令行工具。它会验证数据集、训练并评估模型、判断本轮表现、提出下一轮超参数，再循环执行，同时保存完整的实验状态与产物。

它适合需要可控迭代、可追溯决策、检查点管理及中断恢复的实验，而不只是一次性执行 `yolo train`。

## 功能概览

```text
验证数据 → 训练一轮 → 评估 → 决策 → 调参 → 下一轮
                           │
                 Green / Yellow / Red
```

- 以“轮”为单位运行 YOLO 训练，而不是执行一次不可见的长任务。
- 根据验证集指标为每轮打分；可额外提高某个类别的重要性。
- 使用规则化的 Green / Yellow / Red 三态决策，管理检查点、恢复策略和局部最优逃逸。
- 使用 Optuna 在设定范围内提出超参数变更。
- 提供交互式 `ask` 模式和无人值守的 `auto` 模式。
- 维护 Top-N 与手工命名检查点，支持恢复和从检查点分叉新实验。
- 在训练前验证数据；连续失败后生成数据缺口报告。
- 可写入 MLflow；HTTP MLflow 服务不可达时自动回退到本地 `./mlruns`。
- 可选接入 OpenAI 兼容的 LLM，用于 Ask 模式中的自然语言指导和数据缺口分析；核心训练流程不需要 API Key。

## 环境要求

- Python 3.10 或更高版本。
- 与目标 CPU/GPU、CUDA 环境匹配的 PyTorch。
- 可运行 Ultralytics 的环境；实际训练强烈建议使用 CUDA GPU。
- 只有使用 Docker 时才需要 Docker 和 NVIDIA Container Toolkit。

依赖均定义在 `pyproject.toml`，包括 Ultralytics、PyTorch、Optuna、MLflow 与 Rich。

## 安装

```bash
git clone <repository-url>
cd cv_agent

# 激活已安装适配本机硬件的 PyTorch 的 Python 环境。
python -m pip install -e .

# 可选：开发依赖
python -m pip install -e ".[dev]"
```

下列两个命令等价：

```bash
cv_agent --help
cvagent --help
```

## 快速开始

仓库提供两个用途不同的训练配置：

| 配置 | 用途 | 数据集 | 默认模型 |
| --- | --- | --- | --- |
| `cv_agent.quick.yaml` | 验证完整流程的快速冒烟测试 | COCO128 | `yolo26n` |
| `cv_agent.yaml` | 更长的正式实验 | COCO | `yolo26s` |

建议先运行快速配置。如果指定的 Ultralytics 内置数据集尚未下载，`cv_agent` 会请求 Ultralytics 自动准备数据；首次下载需要网络且可能耗时较长。

```bash
cv_agent --config cv_agent.quick.yaml run
```

使用自定义数据集时，先准备 YOLO 数据集 YAML。可从 `dataset.yaml.example` 开始：

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

然后启动一个限定轮数的实验：

```bash
cv_agent run \
  --data-yaml /absolute/path/to/dataset.yaml \
  --model yolo26n \
  --max-rounds 5
```

仅验证数据、不开始训练：

```bash
cv_agent validate --data-yaml /absolute/path/to/dataset.yaml
```

## 选择交互模式

`cv_agent.yaml` 默认使用 `ask`，适合有交互式终端的本地运行。每轮决策时可查看参数变更、补充指导、拒绝回滚，或保存一个命名检查点。

`auto` 适合 CI、Docker、`tmux` 或无人值守服务器。决策会在配置的倒计时后自动接受；该模式不会收集每轮的人工指导。

```bash
# 交互式审阅
cv_agent --interaction ask run --data-yaml dataset.yaml

# 无人值守
cv_agent --interaction auto run --data-yaml dataset.yaml --max-rounds 20
```

## 决策机制

第一轮完成后会建立本次实验自己的基线。之后每轮都与同一实验中的历史最佳分数比较。决策引擎按相对阈值以及可选的绝对阈值分类；开启 `dynamic_thresholds: true` 后，阈值会随训练进度逐步收紧。

| 结果 | 含义 | 常见动作 |
| --- | --- | --- |
| Green | 明显提升，或被接受的轻微提升 | 提交更优检查点；适用时由 Optuna 提出下一组参数。 |
| Yellow | 接近基线，可能进入局部最优 | 执行保守的诊断性调整，或采用配置的逃逸策略。 |
| Red | 性能下降 | 执行恢复策略；严重 Red 可能回滚至最佳检查点。 |

连续出现 `red_escalation_count` 次 Red（默认 3 次）后，工具会强制恢复，生成 `data_gap_report.md` 与 `data_gap_report.json`，在可用时应用有边界的损失权重建议，并进入数据补充处理流程。

分类完全由可复现的指标和规则决定。LLM 可以提出有边界的策略补丁、解释用户指导，但不会替代评估指标，也不能直接决定检查点。

## 恢复、检查点与输出

每次实验均写入 `runs/exp_<timestamp>/`。主要产物如下：

| 路径 | 内容 |
| --- | --- |
| `weights/` | 当前 YOLO 权重和最佳模型快照。 |
| `checkpoints/` | Top-N 排行与手工保存的检查点。 |
| `optuna_study.db` | 本次实验专属的 Optuna study。 |
| `session_state.json` | 恢复所需的轮数、参数、最佳分数、Red 连续次数和试验次数。 |
| `cv_agent.log` | 会话日志。 |
| `decision_log.json` / 策略产物 | 决策与策略审计记录。 |
| `final/best.pt` | 导出的最佳模型，及 `final/summary.json`。 |

列出已保存的检查点：

```bash
cv_agent list-checkpoints
```

恢复被中断的实验：

```bash
cv_agent resume --run-dir runs/exp_<timestamp>
# 等价的显式写法
cv_agent run --start resume --run-dir runs/exp_<timestamp>
```

从已保存的检查点启动一个新实验：

```bash
cv_agent list-checkpoints
cv_agent run --start from-checkpoint --checkpoint-id <checkpoint-id>
```

达到目标时可提前停止并导出最佳模型：

```bash
cv_agent run \
  --data-yaml dataset.yaml \
  --early-stop \
  --early-stop-metric mAP50 \
  --early-stop-target 0.75
```

支持的提前停止指标：`score`、`mAP50`、`mAP50_95`、`precision`、`recall`、`mAP50_class:<类别名或 ID>`。

## 配置

项目使用 YAML 配置，优先级从低到高为：

1. 指定配置文件（`--config`；默认 `cv_agent.yaml`）
2. 同名本地覆盖文件（存在时，例如 `cv_agent.local.yaml`）
3. 命令行参数

默认配置对应的本地覆盖文件是 `cv_agent.local.yaml`。它已被 Git 忽略，应只存放密钥或机器相关的覆盖项；不要把凭据写入受版本控制的配置。

```bash
cp cv_agent.local.yaml.example cv_agent.local.yaml
```

```yaml
# cv_agent.local.yaml
device: "0"
workers: 0
llm:
  api_key: ""
```

常用顶级配置字段：

| 字段 | 用途 |
| --- | --- |
| `model_variant`、`epochs_per_round`、`max_rounds` | 训练规模。 |
| `device`、`workers`、`model_verbose` | 硬件与运行方式。 |
| `data` | 数据集 YAML 与校验阈值。 |
| `initial_hyperparams` | YOLO 初始训练参数。 |
| `optuna` | 试验预算、Yellow 逃逸策略和搜索范围。 |
| `decision` | Green / Yellow / Red 与升级处理阈值。 |
| `strategy` | 策略规划频率、记忆和目标权重。 |
| `checkpoints`、`output_root` | 产物保存位置与保留策略。 |
| `early_stop` | 提前停止与导出目标。 |

支持的模型标识符：`yolo26{n,s,m,l,x}`、`yolov8{n,s,m,l,x}`、`yolo11{n,s,m,l,x}`。

可使用 `--device auto`、`--device 0`、`--device 0,1,2,3` 或 `--device cpu` 覆盖设备选择。Windows 上建议优先从 `workers: 0` 开始。

### 优化单个类别

`--optimize-for` 会在奖励分数中提高指定类别 mAP@0.5 的权重：

```bash
cv_agent run --data-yaml dataset.yaml --optimize-for vehicle
```

类别名必须能在数据集 YAML 中找到；无法解析时，程序会给出警告并回退到全局指标。

## LLM 集成

LLM 完全可选。内置适配器使用 OpenAI 兼容 API，提供的默认配置指向 DeepSeek 端点。推荐通过环境变量提供密钥：

```bash
export CV_AGENT_LLM_KEY="<your-key>"
# 同时兼容 DEEPSEEK_API_KEY
```

Ask 模式中，LLM 可理解“只调 lr”“保持 mosaic 不变”等自然语言约束。未配置密钥时仍保留正则和启发式回退处理。调用次数由 `llm.max_calls_per_session` 限制。

## MLflow

在所选配置中设置 `mlflow_uri` 与 `experiment_name` 即可连接跟踪服务。若 HTTP(S) 服务不可达，程序会在训练前发现并自动改用 `./mlruns` 本地文件存储；训练不会因此中断，产物也始终写入运行目录。

查看本地跟踪数据：

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Docker

仓库内 Dockerfile 基于 Ultralytics 镜像，入口即为该 CLI。

```bash
docker build -t cv_agent:latest .

docker run --rm -it --gpus '"device=0"' \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/datasets:/app/datasets" \
  -v "$(pwd)/cv_agent.quick.yaml:/app/cv_agent.quick.yaml:ro" \
  cv_agent:latest --config cv_agent.quick.yaml run
```

自定义数据集需要挂载到容器中，并确保数据集 YAML 中的路径在容器内有效：

```bash
docker run --rm -it --gpus '"device=0"' \
  -v "$(pwd)/runs:/app/runs" \
  -v "/host/path/to/dataset:/data:ro" \
  -v "$(pwd)/dataset.yaml:/app/dataset.yaml:ro" \
  cv_agent:latest --interaction auto run --data-yaml /app/dataset.yaml --device 0
```

多 GPU DDP 场景应暴露所需 GPU，并分配足够共享内存，例如使用 `--ipc=host` 或 `--shm-size=8g`。仅当宿主机确实存在 `cv_agent.local.yaml` 时才挂载它；否则 Docker 可能在该位置创建目录。

## 开发

```bash
python -m pytest
ruff check src tests
```

测试覆盖配置、决策引擎、优化流程、状态持久化、检查点、CLI 行为和终端展示。

## 项目结构

```text
src/cv_agent/
  cli/          命令行入口
  core/         配置、状态机与训练编排
  data/         数据引导、校验、补充与缺口报告
  decision/     决策、Optuna、策略与 LLM 指导
  interaction/  Ask 与 Auto 工作流
  tracking/     运行目录、检查点与 MLflow
  trainer/      YOLO 执行、评估、设备与提前停止
tests/          自动化测试
```

## 许可证

MIT。详见 [pyproject.toml](./pyproject.toml) 中的软件包元数据。
