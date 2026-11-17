# AIAA3102 SuperAgent 小组项目

**组员从这一份开始。** 这是基于 MLEvolve 的课程 SuperAgent：自主生成 ML 方案和代码，在隔离环境训练、验证、修复与改进，自动冻结最佳有效模型。课程要求、论文资料、文件职责及分工建议见本文后半部分。

小组公开仓库：[Xiaoyu-Li0912/superagent-workspace](https://github.com/Xiaoyu-Li0912/superagent-workspace)。查看和克隆无需登录；提交代码仍需用获授权的 GitHub 账号完成身份验证。`print-666`、`LiQingfeng111` 已接受协作者邀请，具有写入权限。

算法在 `third_party/MLEvolve/`，课程运行基础设施在 `src/superagent/`，完整实验记录在 `runs/`。当前保留了 MLEvolve 的主要原始机制，但不是严格论文复现；标题、图像、多分类和全部高级搜索分支尚未完成真实验收。详细集成说明、网络恢复和历史实验见给 AI 按需查阅的 [运行参考](docs/agent-runtime.md)，不要求组员先读工作日志。

## 上游来源与修改

MLEvolve 以**普通源码目录**纳入本仓库，不再是 Git 子模块。以后普通 `git clone` 即可取得我们修改后的算法源码，**无需初始化子模块或再次打补丁**。

- 上游：[InternScience/MLEvolve](https://github.com/InternScience/MLEvolve)。固定基线提交：[`f4a37b96c948bfed6564e62f3229e0b38e908d33`](https://github.com/InternScience/MLEvolve/commit/f4a37b96c948bfed6564e62f3229e0b38e908d33)。
- **保留许可证、版权声明、论文引用，并注明基于哪个上游版本及我们的修改。** 上游 [LICENSE](third_party/MLEvolve/LICENSE) 为 Apache-2.0，原版权为 `Copyright 2026 MLEvolve Team, Shanghai Artificial Intelligence Laboratory`；原始 README、引用和源码中的声明保留。上游许可不自动覆盖课程资料、数据或本组所有新增文件，不能把整个仓库笼统当成全部可自由转载。
- 引用：Du et al., *MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery*, 2026，[论文](https://arxiv.org/abs/2606.06473)。完整作者列表和 BibTeX 见[上游 Citation](third_party/MLEvolve/README.md#citation)，报告和展示应明确引用，不把上游算法写成本组原创。
- 我们的修改：新增 `engine/scheduler.py`，从上游入口抽取两阶段调度；在 `engine/agent_search.py`、`engine/execution.py` 增加显式课程集成接口；在草稿/修复/改进/审查/结果分析、分阶段 coder 和提示文件中接入课程 train/predict 协议与可信分数；`llm/__init__.py` 增加统一 transport。修改文件的具体清单见[本地修改声明](third_party/MLEvolve/LOCAL_CHANGES.md)。不在外层替换搜索器、偷偷关闭记忆或缩减算法机制。
- `src/superagent/`、外层配置与测试是课程集成实现，处理数据、API、预算、沙箱、评分、冻结和交付。`configs/mlevolve-integration.patch` 保留为此前集成改动的审阅/对照材料，**不是运行前必须应用的补丁**；以后变动以实际源码及 Git 历史为准。

## 如何运行

### 1. 环境与安装

需要 **Linux / WSL、Python 3.12、bubblewrap、util-linux**。不在 Windows 原生 Python 中直接执行候选，也不要直接启动上游 `run.py`。以下命令除特别标注外均在仓库根目录的 **Bash** 中执行。

当前机器已有 WSL Ubuntu-24.04 和 `/opt/superagent-env`。在本仓库根目录打开 PowerShell 后进入 WSL：

```powershell
wsl -d Ubuntu-24.04 --cd "$PWD"
```

然后激活已有环境：

```bash
source /opt/superagent-env/bin/activate
```

新机器可直接在 Ubuntu 24.04 / WSL 中克隆并安装，无需 GitHub 登录；推送修改时才需协作者身份验证。MLEvolve 已包含在源码中，不需要子模块命令：

```bash
git clone https://github.com/Xiaoyu-Li0912/superagent-workspace.git
cd superagent-workspace
sudo apt-get update
sudo apt-get install -y python3.12-venv bubblewrap util-linux libgomp1
sudo python3.12 -m venv /opt/superagent-env
sudo chown -R "$(id -u):$(id -g)" /opt/superagent-env
source /opt/superagent-env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

专用环境放在 `/opt/`，不要放进项目目录或直接使用 Windows Anaconda。依赖以 `requirements.txt` 为准；历史 run 的 `environment.txt` 可用于版本复核，目前不是完整锁定的跨机器环境。若 bubblewrap 因内核/user namespace 策略失败，需要排查系统权限，不要改成无沙箱执行。单元测试使用合成数据和假调用，不是模型效果证明，不启动收费搜索或最终 test。

### 2. 准备 Hugging Face 记忆模型

MLEvolve 的实验记忆使用 **`BAAI/bge-base-en-v1.5`**，将经验文本转成 768 维向量进行检索。它不是学校 API 的大语言模型，也不是优惠券预测模型；必须先下载，不能为跑通而关掉记忆。

本机已缓存的 revision 为 `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`，权重约 438 MB。当前 WSL 用户的缓存位于 `/home/xlicomputer/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/`。**其他同学、其他 WSL 用户和 Windows Python 不自动共享此缓存**，需要下载或完整迁移缓存；不要只复制带有失效软链接的 snapshots 目录。

在将要运行 SuperAgent 的同一个 Bash、用户和 Python 环境中执行一次下载。此步骤要能访问 Hugging Face，不调用收费 LLM，不读取课程数据：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="BAAI/bge-base-en-v1.5",
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    allow_patterns=["*.json", "*.txt", "model.safetensors", "1_Pooling/*"],
)
print("Memory model snapshot:", path)
PY
```

为确保上游按模型名加载时使用这一固定快照，在 `configs/runtime.yaml` 已有的 `upstream_environment` 中增加以下字段；**路径使用下载命令实际打印的 Linux 绝对路径**，不要复制别人的用户名，也不要覆盖原有的 CPU/GPU 字段：

```yaml
upstream_environment:
  agent.search.num_gpus: 0
  agent.memory_embedding_device: cpu
  agent.memory_embedding_model_path: /home/你的用户名/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots/a5beb1e3e68b9ab74eb54cfd186867f64f240e1a
```

现有机器已经能按原模型名离线加载，**无需再改这项配置**。若想换缓存位置，应在下载和搜索前都设置同一个 `HF_HOME`，并相应更新路径。完成下载后可强制只用本地缓存，先用刚才打印的路径做独立自检：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
read -r -p 'Memory model snapshot path: ' MEMORY_MODEL_PATH
export MEMORY_MODEL_PATH
python - <<'PY'
import os
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(os.environ["MEMORY_MODEL_PATH"], device="cpu")
vectors = model.encode(["Synthetic memory check"], normalize_embeddings=True)
assert vectors.shape == (1, 768) and np.isfinite(vectors).all()
print("Offline memory embedding OK:", vectors.shape)
PY
```

上述离线变量只限制 HF/Transformers 下载，不会禁用学校 API。VPN 可能帮助下载 HF，却影响学校 API 连通性，应分开确认；模型已缓存后不必为搜索持续访问 HF。启用记忆但依赖/权重不可用时程序会报错，不静默消融。HF 缓存不上传 Git；每次运行生成的历史经验另存在该 run 的 `mlevolve/global_memory/records.json`，权重缓存本身不是经验库。

### 3. 准备任务和数据

原始任务数据从获准课程渠道获取，放在仓库外。当前机器的数据位于同级 `../final_project_tasks/final_project_tasks/`，新机器自行调整路径：

| 任务 | `--task` | `--data-root` 对应目录 |
|---|---|---|
| 优惠券 | `configs/tasks/coupon.yaml` | `in_vehicle_coupon_recommendation/` |
| 标题对比较 | `configs/tasks/headlines.yaml` | `headlines_ctr_clean/` |

任务 YAML 描述文件格式、特征列、标签、模态和目标，不指定固定模型。新 CSV 分类任务可以新增 `configs/tasks/my_task.yaml`：

```yaml
format: csv
modality: tabular
description: Predict whether a customer will cancel their subscription.
metric: accuracy
train_features: train.csv
val_features: val.csv
test_features: test.csv
feature_columns: [age, monthly_fee, subscription_months]
label_column: churn
```

相应数据根目录放 `train.csv`、`val.csv`、`test.csv`，都有列名和 `churn` 标签；特征列顺序一致、标签非空、验证类别须在训练中出现。开发阶段不打开 test；仅最终评测需要它。当前支持 `csv`、`categorical_npy`、`image_csv` 三种显式格式和分类 accuracy；新的指标/数据协议需要修改数据层、接口或评分器，不能只改 YAML 名字。图像还需图像路径字段和文件，尚未完成真实图像任务验收，不能假定沙箱中已有视觉预训练权重。

### 4. 设置限制与 API，启动自主搜索

在 `configs/runtime.yaml` 修改运行限制。当前默认值：

| 限制 | 当前值 |
|---|---:|
| 上游 journal 步数 / 搜索时间窗口 | 100 / 28800 秒（8 小时） |
| 全部 Agent 与重试共享调用上限 / 本轮预算 | 400 / 20 元 |
| 单次请求超时 / 最大输出 token | 1200 秒 / 32768 |
| 候选训练与预测各自时间上限 | 各 600 秒 |
| 候选 CPU / 地址空间 | 2 / 4096 MB |

算法开关、搜索策略、记忆和融合阈值在 `third_party/MLEvolve/config/config.yaml`，不是外层 runtime 的隐藏开关。CLI 只有 `--steps`、`--budget-rmb` 可以覆盖这些对应默认值；其余限制改 runtime 文件，或用 `--runtime` 指向自己的配置副本。`--policy` 指定模型合规/价格策略，不要仅更换 API 模型名而不核对发布日期、单价和来源。

预算按**已结算 + 未知/在途预留**约束。每个新 run 独立记账，程序不会自动累加所有历史尝试；课程每任务 20 元应人工跨尝试累计，失败调用也计入。下例要求输入真实的剩余授权预算，不默认再花 20 元。配置不等于付费授权。

```bash
read -r -s -p 'API key: ' SUPERAGENT_API_KEY
printf '\n'
export SUPERAGENT_API_KEY
export SUPERAGENT_BASE_URL='https://aigc-api.hkust-gz.edu.cn/api/v2'
read -r -p 'Remaining authorized budget (RMB, <=20): ' RUN_BUDGET_RMB
DATA_ROOT="$(realpath ../final_project_tasks/final_project_tasks/in_vehicle_coupon_recommendation)"
RUN_DIR="runs/coupon-$(date +%Y%m%d-%H%M%S)"
superagent run \
  --task configs/tasks/coupon.yaml \
  --runtime configs/runtime.yaml \
  --policy configs/model_policy.json \
  --data-root "$DATA_ROOT" \
  --run-dir "$RUN_DIR" \
  --budget-rmb "$RUN_BUDGET_RMB"
unset SUPERAGENT_API_KEY
```

密钥通过隐藏输入进入监督进程，不写进命令、README 或提交文件。**CLI 不自动加载 `.env` 或 `.env.example`**；示例文件只能保留空密钥。标题任务换成 `configs/tasks/headlines.yaml` 和对应数据目录；新任务换成新 YAML 和它的数据根目录即可，不必手写预测模型。

每次必须使用**全新 run 目录**。运行中不修改代码、配置、候选，不人工指定下一实验或挑最终模型；正常搜索也可能先达到时间、预算、调用数或未知用量阈值，不能把“100 步”当成必达值。目前没有跨进程自动续跑，检查点不等于断点恢复。长运行重试、冷却、未知费用保护和故障收尾详见 [AI 运行参考](docs/agent-runtime.md#长时间运行与网络恢复)。

### 5. 一次运行怎样传递输入和输出

1. CLI 读取任务、runtime、模型价格策略；监督器建立新目录，保存配置、源码哈希和环境快照。
2. 数据层只读取 train/val，导出 `data/train/features.csv`、`labels.csv`、`schema.json` 及 `data/val/features.csv`、`schema.json`。验证标签另存在 `private/validation_labels.npy`，不挂载给候选。搜索只收到训练概况、少量训练样例及聚合验证反馈，不接触最终 test。
3. MLEvolve 自主规划、分阶段生成和合并代码、审查、调试与改进；调用经统一网关预留费用、请求 API、按 usage 结算。**一次迭代不是一次 LLM 调用**，生成/审查/分析/记忆规划都可能调用。
4. 每个候选保存为 `candidates/<id>/solution.py`。沙箱依次启动两个独立进程：训练保存完整模型和预处理；预测重新加载模型，对无标签 val 输出 CSV。候选不可联网，不获取 API 密钥、主目录、项目源码、验证标签或 test。
5. 可信评分器核对格式并计算 accuracy，交回上游分析、写记忆与继续搜索；候选自报分数不作依据。上游自动选出 `best_node`，外层将其代码和模型复制到 `best/`，用 `freeze.json` 固定文件哈希。

候选遵守的接口由系统传给 LLM，不需要人手工运行每个候选：

```bash
python solution.py train --data /data --model /model
python solution.py predict --data /data --model /model --output /output/predictions.csv
```

预测必须使用已经保存的预处理、标签映射和模型，不能在 val 上重新拟合。CSV 为 `row_id,prediction`，`row_id` 从 0 开始按输入顺序，`prediction` 使用原类别标签。

### 6. 看结果、独立预测与最终评测

首先查看 `$RUN_DIR/summary.json`：有有效方案时应为 `status: frozen`，同时有 `best/` 和 `freeze.json`；若初始化或搜索失败，可能没有这些文件，先看 `events.jsonl`、`mlevolve.log`，不能直接做最终 test。

| 想看什么 | 文件 |
|---|---|
| 状态、停止原因、最佳验证 accuracy | `summary.json` |
| 费用、token、已结算与未知预留 | `cost_summary.json`、原始 `cost.sqlite` |
| 每次请求与响应 | `calls/` |
| 每个候选、错误和模型 | `candidates/`、`candidates.json`、`decisions.json` |
| 上游搜索、记忆、当前进度 | `mlevolve/`、`checkpoint.json` |
| 最终冻结代码和完整模型 | `best/solution.py`、`best/model/`、`freeze.json` |

对自己新产生的 run 可执行 `superagent audit --run-dir "$RUN_DIR"` 从账本重新导出费用；**共享的历史 run 保持原样，分析时不要用会重写文件的命令覆盖证据**。预留金额不是学校平台确认扣款，未知费用要去学校账单核对。

对新的无标签输入，准备一个目录，包含 `features.csv`、`schema.json`；按该 run 的 `data/train/schema.json` 保持列名、顺序、模态和类别协议一致，图像还要提供对应相对路径文件。输出目录用新位置：

```bash
superagent predict --run-dir "$RUN_DIR" \
  --features /你的无标签标准化输入目录 --output /你的新预测输出目录
```

`predict` 核验冻结哈希，在沙箱加载已有模型输出 `predictions.csv` 和日志，不调用 LLM、不重新训练、不计算有标签分数。

**只在成功冻结且决定结束开发、明确授权使用最终 test 时**，才单独运行以下命令，不要把它接在每次搜索后自动执行：

```bash
superagent evaluate --run-dir "$RUN_DIR" --data-root "$DATA_ROOT"
```

此命令无需 API 密钥，不调用 LLM。先核验冻结产物并创建一次性 `final_evaluation.lock`，再加载原始 test、预测与评分，输出 `final_output/predictions.csv`、`final_result.json`；后者才是 test accuracy。已评测过的 run 不删锁、不重测；测试结果不回流调参或搜索。

### 7. 完整实验记录与 GitHub 分享

按本组要求，**直接纳入整个现有 `runs/`**，不是节选、汇总导出或仅成功实验：包括 8 次尝试、原始请求响应、账本、候选代码、全部候选模型、日志、导出数据、验证标签，以及早先已生成的最终评测文件。当前共 926 个文件，约 462.88 MiB，没有单文件超过 100 MiB；后续实验增加后需重新核查 GitHub 限制。下载与克隆会比较大。

历史 run 混有早期简化适配与后来的机制恢复版，不能把旧成绩当成当前实现成绩。逐次配置、失败原因和费用见 [实验历程](docs/agent-runtime.md#本次实际运行)；输出逐文件归属见下方[文件结构](#文件结构)。保留原始字节和已有评测锁，分析时不要手改历史记录，也不要加载不可信来源的 pickle/joblib 模型。

这些完整记录包含课程数据、训练样例、验证标签和先前最终测试产物。**仓库于 2026-11-17 按用户要求由私有改为公开，任何人都能下载；公开可见不等于课程材料和数据具有自由转载许可。** 分析优化只能用允许的开发数据与验证反馈，不能以历史 test 成绩/标签指导新搜索。真实 `.env`、HF 权重缓存、虚拟环境及仓库外原始数据不上传。

共享目标为上述小组公开仓库，完整 `runs/` 随 Git 提交交付，不另外裁剪或改写历史实验。公开读取不授予写入权限；后续新增日志和数据前仍须检查凭据、隐私及分享权限。

## 课程要做什么

开发一个 **General Algorithm SuperAgent**：给它分类任务、数据和评价指标，它自主分析、设计方法、生成代码、运行实验、根据反馈改进，最终自动选出并保存可运行预测方案。不是手工训练两个分类器后挑最好的提交。

同一个系统要能处理表格、文本、图像的二分类或多分类；除了公开任务，还要用冻结版本完成老师之后发布的未见任务。

| 必须满足 | 含义 |
|---|---|
| 通用性与自主性 | 不能按任务名调用人工预制特解；正式运行中不能人工改代码、决定下一实验或挑最终模型 |
| 模型限制 | 仅限 2026-10-21 当天或之前正式发布的 Qwen/DeepSeek 模型 |
| 每个评估任务最多 RMB 20 | 按该日官方标准输入/输出 token 价计费，覆盖所有 Agent；不扣缓存/促销/免费额度，本地部署也计费 |
| 可审计 | 记录候选代码、决策、实验结果、全部模型调用、token 和费用 |
| 测试隔离 | test 只用于最终评估，不用于训练、调参、选模，也不把测试结果反馈给搜索 |
| 版本一致 | 报告结果来自同一最终提交版本；未见任务使用冻结版本，无人工干预 |

**评分：** 公开任务性能 20%，未见任务性能 10%，通用系统设计与自主开发 30%，执行/复现/报告 10%，展示 30%。难点不只是准确率，还在自主性、低预算、通用性和复现。

**交付：** 源码与统一入口、依赖和 README、最终模型或可运行预测方案（含预处理）及结果文件、`report.pdf`、完整日志、AI 使用声明，以及最终展示。

**时间：** 本班最后一周第一节课前一晚 23:59 提交；具体年份、日期与课程时区待按课表确认。两班展示结束后发布未见任务，24 小时内交预测与日志，无需额外报告。

**开源与 AI：** 材料未明确禁止复用开源，但不等于认可原封不动提交。需核查许可证、保留引用、说明自己的改动，疑问向老师确认。允许 AI 辅助设计、编码、调试和测试，但须如实披露，三人都要理解并能解释提交内容。

## 两个公开任务

| 任务 | 输入和标签 | train / val / test | 指标 |
|---|---|---|---|
| 优惠券接受预测 | 21 个类别特征；`C_*.npy` 与 `y_*.npy` 行对齐，1 接受、0 拒绝 | 8117 / 2030 / 2537 | accuracy |
| 新闻标题 CTR 比较 | CSV 中 `headline_A`、`headline_B`；`label_pairwise` 为 1 表示 A 的观测 CTR 更高，0 表示 B 更高 | 8301 / 940 / 3232 | accuracy |

标题任务是**文本对比较**，不是单文本分类。优惠券任务要处理类别字段、缺失值，并保持训练/预测预处理一致。可信课程特征按说明用 `np.load(..., allow_pickle=True)`，不对未知来源文件启用 pickle。两个测试集都有标签，但不能用于开发。

目前没有公开图像任务，不代表系统可以不考虑图像或多分类；可先用合成或获准开发样例验证，不能提前挪用未见任务。

## 相关论文与代码

这类系统通常称为 Machine Learning Engineering Agent（MLE Agent）。已有完整框架，可以参考或适配，不必全部从零编写。下表沿用已有资料和部分源码核查，**不是本组已部署或复现的结果**。

| 系统 | 论文与代码 | 可以参考什么 |
|---|---|---|
| AIDE，2025 | [论文](https://arxiv.org/abs/2502.13138) · [代码](https://github.com/WecoAI/aideml) | 把 ML 代码作为搜索树节点，自动生成、调试、改进；相对轻量，便于理解基本闭环 |
| MLZero，2025 | [论文](https://arxiv.org/abs/2505.13941) · [代码](https://github.com/autogluon/autogluon-assistant) | 多 Agent 从原始多模态数据完成端到端 ML/DL，可参考跨模态处理 |
| ML-Master，2025 / 2.0，2026 | [1.0](https://arxiv.org/abs/2506.16499) · [2.0](https://arxiv.org/abs/2601.10402) · [代码](https://github.com/sjtu-sai-agents/ML-Master) · [2.0 实现](https://github.com/sjtu-sai-agents/EvoMaster/tree/main/playground/ml_master_2) | 探索、推理、实验记忆；2.0 强调提炼长期经验 |
| R&D-Agent，2025 | [论文](https://arxiv.org/abs/2505.14738) · [代码](https://github.com/microsoft/RD-Agent) | Research 提思路、Development 实现实验、反馈迭代 |
| AutoMind，2025 | [论文](https://arxiv.org/abs/2506.10974) · [代码](https://github.com/ant-research/automind) | 专家知识库、知识树搜索和代码调试 |
| AIRA-Dojo，2025 | [论文](https://arxiv.org/abs/2507.02554) · [代码](https://github.com/facebookresearch/aira-dojo) | 任务、搜索策略与执行环境解耦；CC BY-NC 4.0 非商业许可，不是宽松开源许可 |
| MLEvolve，2026 | [论文](https://arxiv.org/abs/2606.06473) · [代码](https://github.com/InternScience/MLEvolve) | 多 Agent 蒙特卡洛图搜索、实验记忆、增量修改、停滞检测和跨分支融合 |

**当前实现基于 MLEvolve，后续方案仍可由小组讨论迭代。** 官方说明支持包括 Qwen/DeepSeek 的 OpenAI-compatible API，采用 Apache 2.0 许可，功能范围接近课程需求。当前已接入课程数据、计费、测试隔离与预测交付；论文的大预算结果不能直接推算成本组表现。依赖或预算不合适时，可参考 AIDE 等轻量实现。

**源码直接收录：** [third_party/MLEvolve](third_party/MLEvolve/README.md)，基线为 `f4a37b96c948bfed6564e62f3229e0b38e908d33`，包含我们的显式集成修改。许可证、版权、论文及修改说明见本文开头；普通克隆即可获取，不需要子模块操作。

## 完整 Pipeline

```text
任务与数据 -> 分析 -> 生成方案/代码 -> 执行 -> 可信验证
                         ^                    |
                         +---- 修复/改进 -----+
                                              |
                                              v
                              自动选优 -> 保存模型与预测入口
```

建议第一版就形成完整自主闭环，之后优化效果与效率；这是开发建议，不是已确认决定，也不要求首版达到最高分。

| 环节 | 首版应具备的能力 |
|---|---|
| 统一入口与数据分析 | 识别模态、字段、标签、类别数、划分和文本对结构，不按数据集名走特解 |
| 方案与实验 | 自主选择方法、生成代码并训练，由可信代码计算验证指标，不信 LLM 自报分数 |
| 修复与反馈迭代 | 对错误有界重试，根据真实实验结果决定下一步，而非仅执行人工固定模型列表 |
| 选优与交付 | 自动选择最佳有效候选，保存模型及预处理，新进程可加载预测 |
| 运行控制 | 全量日志、预算预留、超时/失败控制，到边界停止并保留已有有效产物 |
| 通用性 | 验证表格、文本/文本对、基础图像，以及二分类和多分类 |

之后可优化准确率、每元有效实验数、训练时间、token 消耗、修复成功率和跨任务稳定性。持续优化不代表单次运行可突破 20 元，也不能绕过最终冻结要求。

## 分工与工作量建议

| 方向 | 主要负责 |
|---|---|
| A：自主核心 | 基线适配，规划、生成、修复、反馈选优；交可调用闭环与候选记录 |
| B：数据与预测 | 数据协议、可信指标、测试隔离、训练/预测一致性及跨模态验证 |
| C：运行与集成 | API、预算、日志、超时/失败、统一启动与打包复跑 |

先共同约定接口，数据层和控制层可并行，再集成完整闭环、验证改进、冻结交付，最后运行未见任务。报告和展示三人共同完成，不把某人只分配去做 PPT，也不按两个公开数据集拆成两套系统。认领与复核人尚未确定。

**初始工作量粗估为 20–36 总人日，不是每人 20–36 天。** 这是实施前的粗估，不是剩余工作承诺，且不含训练等待与其他课程安排：可行性 2–4、基础层 5–9、闭环 6–10、验证 4–8、提交展示 3–5 人日。整体难度中高，主要风险是低预算自主运行、接口集成和通用性；应按现有实现、实际机器、API、可用时间重估。

建议提交前至少留 3 个日历日做最终版本正式重跑、互相复现和展示。工期紧时先减昂贵搜索、融合等可选机制，不删自主性、计费和测试隔离。下一次讨论只需先确认基线、三人认领与工时、机器/API、班级截止时间。

## 原文与中文版

| 按需查阅 | 中文 | 英文原件 |
|---|---|---|
| 正式要求 | [中文版](docs/course/assignment.zh-CN.md) | [原文](docs/course/AIAA3102%20-%20Final%20Project%20-%20Full%20Assignment%20Requirements.md) |
| 优惠券任务 | [中文版](docs/tasks/coupon/README.zh-CN.md) | [原文](docs/tasks/coupon/Readme.md) · [元信息](docs/tasks/coupon/info.json) |
| 标题比较任务 | [中文版](docs/tasks/headlines/README.zh-CN.md) | [原文](docs/tasks/headlines/README.md) |
| 历史 Preview | [中文版](docs/course/preview.zh-CN.md) | [原文](docs/course/AIAA3102_Final_Project_Preview.md) |

## 工作区怎么用

**人看 README，AI 看 AGENTS，关键过程留在 journal。** 本页统一维护当前计划、分工和待办，不再另学 JSON 状态、网页生成或认领协议。课程资料按需查，不要求组员读 AI 工作记录。

### 当前状态与下一步

- 已修正外层简化，算法默认值与主要机制回归 third_party；真实记忆模型已缓存并完成组件验证，恢复版已完成一次冻结。各次配置、指标和费用统一见 [AI 运行参考](docs/agent-runtime.md)，旧简化版成绩不代表当前版本。
- 当前新配置为 100 步、8 小时、400 次派发、20 元，并加入有界网络重试、冷却、累计 unknown 阈值、检查点与受控收尾；45 项离线测试通过，尚未用新配置收费运行。不修改历史账本，不把预留当扣款，不自动汇总跨 run 预算，也不自动启动新实验。
- 三人最终选型、认领和可投入工时尚未确认。优先接手现有实现、约定接口；数据/预测和运行控制可并行，完成后集成验证。
- 待开发/复核：上游分析 schema 的 number/null 矛盾、生成的语法错误、空正文/截断和资源失败，以及新可靠性策略的真实长运行效果；跨进程自动续跑仍未实现。正式模型 build 和课程价格口径待确认，未知用量不得清零。
- 待验收：标题任务自主运行、获准开发样例上的图像/多分类、关键改进对照和另一位组员复跑。验收以可信指标、可重载模型、可重算费用及运行证据为准。
- 最终交付需同一最终版本的双任务自主结果、模型/预处理、依赖、报告、日志、AI 声明和展示；正式提交、系统最终冻结和未见任务均未完成。至少留出复现与打包缓冲，准确截止日期/时区待确认。

### 文件结构

先记住三层：**`third_party/MLEvolve/` 决定怎么搜索和生成；`src/superagent/` 负责在课程环境安全、可计费地执行；`runs/` 保存一次实际运行的证据和交付物。** 后两层不是额外的算法版本，`src/` 也不只是启动脚本，还承担数据、评分、隔离和交付逻辑。

```text
<仓库根目录>/
|-- README.md                         人类阅读入口、当前状态和运行说明
|-- docs/                             课程与任务原文、中文对照、AI 运行参考
|   `-- agent-runtime.md              集成边界、网络恢复与逐次实验历程
|-- third_party/MLEvolve/              MLEvolve 算法源码和上游配套设施
|   |-- LICENSE / LOCAL_CHANGES.md     上游许可、版权与本地修改声明
|   |-- engine/                       搜索图、选择、调度、回传、选优
|   |-- agents/                       草稿、修复、改进、进化、融合等 Agent
|   |   |-- coder/                    分阶段代码生成与 diff 应用
|   |   |-- planner/                  方案规划与记忆指导
|   |   |-- memory/                   实验记忆、嵌入和混合检索
|   |   `-- prompts/                  环境、实现和审查提示
|   |-- config/                       算法参数与上游日志保存
|   |-- llm/                          上游模型调用接口及模型参数
|   `-- utils/                        指标、解析、序列化等工具
|-- src/superagent/                   我们的课程运行基础设施
|-- configs/                          运行预算、任务格式、价格与集成补丁
|-- tests/                            运行层测试、上游离线集成测试和旧实验审计
|-- runs/                             完整实验记录，随仓库收录；历史内容不手改
|-- pyproject.toml                    包元数据、安装方式和 superagent 命令入口
|-- requirements.txt                  课程系统依赖
|-- .env.example                      不含真实密钥的变量模板
|-- .env                              本机私密凭据（被忽略，CLI 不自动加载）
|-- AGENTS.md / CLAUDE.md              给 AI 的项目规则/同规则引用
|-- journal.md / human.md              工作历史/人类专属可选笔记
|-- .github/pull_request_template.md   可选 PR 交接模板
`-- .gitignore / .gitattributes        忽略秘密和缓存；保持 runs 文件原始字节
```

#### 1. 算法核心：third_party/MLEvolve

下表路径相对于 `third_party/MLEvolve/`。这是已下载的第三方项目，本来就有多 Agent、记忆和执行配套；不是我们凭空新增了这么多课程文件。除特别注明外，它们都是上游已有文件。**保留文件不代表当前课程入口会执行每条上游路径。**

| 搜索与调度文件 | 作用与修改场景 |
|---|---|
| `engine/agent_search.py` | 搜索总控：初始化任务/记忆，协调选节点、生成、执行、解析与更新。改每轮算法流程先看这里；已有显式课程 integration 接口 |
| `engine/scheduler.py` | 本地新增：将原 `run.py` 的两阶段调度抽为课程可调用函数，先生成初始草稿，再并行执行和续搜；计步、错峰、停机排空 worker 在这里 |
| `engine/search_node.py` | `SearchNode` 和 `Journal` 数据结构：代码、父子关系、分数、访问统计、轨迹及节点级历史；不是磁盘日志文件本身 |
| `engine/node_selection.py` | UCT/探索常数、全局 Top-K、探索与利用软切换；决定下次从哪里继续搜索 |
| `engine/evaluation.py` | 节点奖励、改进判断和搜索结果回传；**不是课程 accuracy 评分器** |
| `engine/conditions.py` | 分支/全局停滞和融合触发条件；调整何时使用高级搜索动作看这里 |
| `engine/execution.py` | 执行后的提交存在性、有效性和分支登记；不是沙箱。课程集成允许合法零分结果 |
| `engine/solution_manager.py` | 更新上游最佳节点、Top-K、各分支代表，保存上游最佳方案/预测；外层冻结使用上游选出的 `best_node` |
| `engine/executor.py` | 上游原生解释器和 `ExecutionResult`。课程入口复用结果类型，但实际候选执行交给外层沙箱，不直接使用其原生解释器执行候选 |
| `engine/validation/format_client.py`、`format_server.py` | 上游 Kaggle/基准提交格式校验的客户端/服务端。课程入口改用显式课程校验，不要求启动该服务 |
| `engine/validation/quality_check.py` | 上游提交内容与格式检查、LLM 修复辅助；课程格式有效性由适配接口及可信评分器处理，不等于原基准评分服务仍在使用 |
| `engine/coldstart/knowledge.py` | 加载任务标签、模型建议和可选初始解路径；课程通用 ID 无匹配时无指导，不按优惠券名字塞特解 |
| `engine/coldstart/competition_tag_classified.json` | 上游已有任务到类别的映射，知识库输入，不是我们的实验输出 |
| `engine/coldstart/models_guidance_classified.json` | 上游类别到方法建议的知识库，输入资料，不是本次生成的模型 |
| `engine/coldstart/classify_tasks.py` | 离线构建任务分类映射的上游脚本，可能调用模型；不是每轮搜索必跑步骤，不要当免费检查随手执行 |

| Agent 与代码生成文件 | 作用与修改场景 |
|---|---|
| `agents/draft_agent.py` | 从任务描述和指导生成初始草稿，开出不同搜索分支 |
| `agents/debug_agent.py` | 根据父节点报错和相关修复记忆生成修复候选 |
| `agents/improve_agent.py` | 根据有效方案、实验反馈和记忆提出下一步改进 |
| `agents/evolution_agent.py` | 利用分支历史反思/重构方法，不只是常规局部改进 |
| `agents/fusion_agent.py` | 借鉴其他节点/分支，将多个方案的思路融合成新候选 |
| `agents/aggregation_agent.py` | 收集多个分支代表并作全局聚合；是否触发仍取决于原始时间/停滞阈值 |
| `agents/code_review_agent.py` | 生成代码执行前的 LLM 审查及修订；已有课程接口与允许 diff 响应的集成修改 |
| `agents/result_parse_agent.py` | 执行后 LLM 结果分析、故障判断、方法摘要、校验与记忆保存。课程真实分数经 integration 注入，不信 LLM 自报 accuracy |
| `agents/data_leakage_agent.py` | 触发时检查候选代码是否泄漏数据；这不替代外层 test 隔离 |
| `agents/triggers.py` | 泄漏检查触发、耐心/停滞计数和节点登记等共享逻辑 |
| `agents/planner/base_planner.py` | 构造规划提示、解析并规范化结构化方案 |
| `agents/planner/planner_with_memory.py` | 检索成功/失败经验，生成初始计划并精炼为结构化方案 |
| `agents/coder/base_coder.py` | 通用“规划 + 代码生成”调用入口 |
| `agents/coder/stepwise_coder.py` | 三类专职生成步骤及 MetaAgent 合并代码；改分阶段生成职责、合并协议看这里 |
| `agents/coder/diff_coder/diff_generate.py` | 将规划变成 SEARCH/REPLACE 增量修改并组织应用 |
| `agents/coder/diff_coder/apply.py` | 应用 diff 及失败后的受控重生成流程 |
| `agents/coder/diff_coder/patcher.py` | 具体匹配并替换代码块，处理缩进与近似匹配 |
| `agents/coder/diff_coder/prompts.py` | diff 输出格式及增量修改提示 |
| `agents/prompts/environment.py` | 给 LLM 描述真实可用环境；课程接口不再声称有不存在的 GPU/网络 |
| `agents/prompts/impl_guideline.py` | 代码实现约束、时间资源和课程 I/O 要求的提示拼装 |
| `agents/prompts/shared.py` | 通用输出格式、防泄漏、网络/预训练资源说明 |
| `agents/prompts/validation_template_prompts.py` | 代码审查提示、规则和响应格式 |

| 记忆、模型与配置文件 | 作用与修改场景 |
|---|---|
| `agents/memory/global_memory.py` | 实验经验的筛选、写入、读取和指导提示，连接搜索节点与检索器 |
| `agents/memory/record.py` | 单条实验记忆的数据结构和序列化 |
| `agents/memory/embedding_models.py` | 加载嵌入模型并编码文本；本机已缓存并验证真实权重，新机器按“如何运行”准备 |
| `agents/memory/retriever.py` | 向量/FAISS 与 BM25 混合检索、索引及排序融合 |
| `config/config.yaml` | 算法开关、搜索深度/概率、停滞/融合阈值、记忆和分阶段生成等默认设置；算法调参优先改它 |
| `config/__init__.py` | 上游配置结构、路径准备与 `save_run`（journal、配置、最佳代码保存），不只是空包标记 |
| `llm/__init__.py` | 上游统一生成/结构化调用接口；本地增加显式 transport，把课程调用交统一网关，并传播停机信号 |
| `llm/model_profiles.py` | 不同模型的采样、thinking 和结构化输出兼容参数；课程入口仍沿用这里的模型 profile |
| `llm/openai.py` | 上游 OpenAI-compatible 提供方实现；课程 transport 接管实际 HTTP 调用，不绕过账本直接走此文件 |
| `llm/gemini.py` | 上游 Gemini 实现，也提供共享提示/函数规格工具；保留和导入工具不代表课程调用了 Gemini |

| 上游工具与其他文件 | 作用 |
|---|---|
| `utils/data_preview.py` | 原生文件预览与 LLM 任务描述清理；课程数据预览由训练概况接口提供，仍保留描述清理 |
| `utils/metric.py` | 指标值、最差值和大小比较方向，用于搜索排序 |
| `utils/response.py` | 从模型响应提取代码、计划、JSON，处理格式与长度 |
| `utils/serialize.py` | 搜索 journal 等对象的 JSON 保存/恢复辅助 |
| `utils/seed.py` | 统一随机种子 |
| `utils/logging_config.py` | 上游日志配置和输出过滤 |
| `utils/visualization.py` | 将搜索结构转成文本/Rich 树，辅助检查搜索过程 |
| `utils/submission_fusion_utils.py` | 上游对已有预测文件做投票/平均等集成的独立工具；不是 `agents/fusion_agent.py` 的思路级融合，也不是课程必跑步骤 |
| `utils/__init__.py` | 上游数据复制、解压与预处理工具；不是空文件 |
| `run.py` | 上游原生运行入口，**仍保留自身调度实现**；课程不是直接运行它，而是调用新增 `engine/scheduler.py` |
| `run_single_task.sh` | 上游单任务启动脚本，不是当前课程统一启动方式 |
| `launch_server.sh` | 启动上游格式校验服务，课程入口不需要 |
| `requirements_base.txt`、`requirements_ml.txt`、`requirements_domain.txt` | 上游分层依赖清单；当前课程环境以外层 `requirements.txt` 和实际环境快照为准 |
| `README.md`、`LICENSE` | 上游使用说明、论文引用和 Apache 2.0 许可证，保留不冒充我们的原创 |
| `assets/logo.svg`、`docs/index.html`、`docs/assets/logo.svg` | 上游项目展示页面和素材，不是课程工作区运行页面或算法组件 |
| 其他各级 `__init__.py` | 包入口或导出接口，如导出 coder/planner/coldstart 工具；通常不用逐个改，不将其中的重导出误当作另一套算法 |

#### 2. 我们的运行层：src/superagent

以下均是课程适配实现，不是 MLEvolve 原仓库自带的文件。**算法优化一般不从这里改，但这些文件承担真实功能，不能删成一个启动脚本。**

| 文件 | 作用 |
|---|---|
| `__init__.py` | 标记课程 Python 包，简述用途 |
| `cli.py` | 解析 `superagent run/predict/evaluate/audit` 四类命令和配置覆盖，转交对应实现 |
| `contracts.py` | 生成代码必须遵守的 train/predict 协议、预测格式，以及 `CandidateResult` 结果结构；改变交付协议才需要改 |
| `data.py` | 读课程 NPY/CSV/图像路径数据，导出统一输入，生成训练概况，单独保管 val 标签；开发准备不读 test |
| `mlevolve_adapter.py` | 装配上游配置、检查依赖、注册 API transport，提供环境/I/O/可信指标的显式接口；不再 monkey patch 算法，不在此隐藏关机制 |
| `orchestrator.py` | 管理一次 run 生命周期，给上游提供候选执行回调、检查点与停止条件；根据上游 `best_node` 冻结，提供预测与一次性最终评测流程，不另写搜索策略 |
| `runner.py` | 通过 bubblewrap 禁网/最小挂载，配合资源和时间限制，在独立进程执行候选训练/预测 |
| `evaluator.py` | 检查预测行 ID、数量、标签范围，可信计算 accuracy；生成代码和 LLM 不决定这个分数 |
| `llm_gateway.py` | 实际模型 HTTP 请求、上游参数传递、schema 校验、原始调用留档、账本结算与 deadline/停机；恢复版不写死关闭 thinking |
| `budget.py` | SQLite 并发原子预算预留、真实 usage 结算、调用数和累计未知用量阈值；未知费用不清零 |
| `artifacts.py` | 将自动选中的代码与完整模型复制到 `best/`、生成冻结哈希，加载前后核验 |
| `audit.py` | 原子写 JSON、线程安全事件日志、SHA256 和文件清单；支撑审计，不改变算法 |

实际调用关系可以按下面顺序阅读：

```text
cli.py -> orchestrator.py -> mlevolve_adapter.py -> MLEvolve AgentSearch
                                      |
                  MLEvolve engine/scheduler.py 调度搜索
                     |                         |
        agents/planner/coder/memory      候选执行回调
                     |                         |
        llm -> llm_gateway -> budget    runner -> evaluator
                     |                         |
                     +---- 结果回到 MLEvolve ---+
                                  |
                  MLEvolve 选 best_node -> artifacts 冻结
```

下面逐个说明当前的 **12 个 Python 文件**。它们是我们编写的课程运行层，不是 12 个 Agent；LLM 生成的候选程序保存在 `runs/<运行名>/candidates/`，不会写回这个源码目录。以下输出路径均相对于一次运行的 `runs/<运行名>/`。

##### 2.1 `__init__.py`：Python 包标记

- **当前内容：** 只有一句包用途说明，让 `superagent` 作为 Python 包被导入。
- **输入/输出：** 没有运行参数，不生成文件，不创建 Agent，也不执行搜索。
- **什么时候改：** 通常不用改。它不是运行配置入口，也没有需要实现的算法。

##### 2.2 `cli.py`：把终端命令交给对应功能

- **主要入口：** `main()`，解析 `superagent` 的四个子命令。
- **`run`：** 读取任务 YAML、运行 YAML、模型价格策略 JSON，接收数据根目录和新运行目录，然后调用 `orchestrator.run_search()`。`--budget-rmb`、`--steps` 可以覆盖本轮预算和步数；其他运行参数主要通过 `--runtime` 指定的 YAML 修改。
- **`predict`：** 接收已冻结运行、统一格式的无标签特征目录、新输出目录，调用 `orchestrator.predict()`。`--features` 不是任意原始 CSV 文件，而是包含 `features.csv`、`schema.json` 等的目录。
- **`evaluate`：** 调用 `orchestrator.final_evaluate()`，对冻结结果执行一次性最终测试；不是继续搜索或调参命令。
- **`audit`：** 根据运行时保存的配置打开 `cost.sqlite`，重新导出 `cost_summary.json`。这是本地账本核对，不访问学校 API 管理平台。
- **输入/输出：** 输入来自命令参数、配置和环境变量；命令结果以 JSON 打印。搜索、预测、评测的具体磁盘产物由下层模块生成。`--key-stdin` 可从私有标准输入读取密钥；CLI **不会自动加载 `.env`**。
- **什么时候改：** 新增命令、命令参数或配置覆盖方式时改。当前参数检查要求步数等计数为正整数、预算 `0 < budget <= 20`；没有把步数硬编码限制为 100。该预算约束只覆盖本轮，仍须人工累计各次尝试以满足课程每任务 20 元上限。

##### 2.3 `contracts.py`：约定候选程序怎么交付、结果怎么表示

- **`CANDIDATE_CONTRACT`：** 给上游/LLM 使用的课程协议文本，约束最终组装的 `solution.py` 支持以下接口：

  ```text
  python solution.py train --data /data --model /model
  python solution.py predict --data /data --model /model --output /output/predictions.csv
  ```

- **训练要求：** 从 `features.csv`、单列 `labels.csv`、`schema.json` 读取训练数据；把学到的预处理、模型、标签映射完整保存到 `/model`。
- **预测要求：** 在新进程中加载只读模型，输入没有标签，不重新拟合；输出严格为 `row_id,prediction`，行号按输入顺序从 0 开始，预测值使用原始标签。中间分模块生成或 diff 响应仍遵循上游所要求的格式，不必每次都输出整个脚本。
- **`CandidateResult`：** 不可变结果对象，字段为候选 ID、`valid/failed` 状态、可信指标、耗时、诊断、产物目录；`to_dict()` 供 JSON 落盘。
- **输入/输出：** 本文件主要提供协议常量和数据结构，本身不读数据、不写产物。`orchestrator.py` 创建结果对象，适配器和网关传递协议。
- **什么时候改：** 课程训练/预测接口、模型交付要求或结果字段变化时改。**协议文本是提示约束，不是安全执行器**；能实际检查的部分还需要 `runner.py`、`evaluator.py` 和产物检查共同落实。

##### 2.4 `data.py`：把不同任务的数据整理成统一输入

- **`load_split(spec, root, split)`：** 按任务配置读取指定的 train/val/test 分割，支持 `categorical_npy`、`csv`、`image_csv`；检查路径范围、特征/标签对齐、空数据和缺失标签等，返回特征 DataFrame 与标签数组。NPY 的 pickle 读取必须由任务配置显式声明可信。
- **`export_features()`：** 导出 `features.csv`、`schema.json`；图像任务会校验原图路径、复制图片到导出目录的 `images/`，把路径替换成候选可访问的相对路径。
- **`prepare_development()`：** 只读取 train 和 val，不打开 test；检查两者字段一致、验证标签属于训练标签域，生成训练概况供上游理解任务。
- **直接输出：** `data/train/features.csv`、`data/train/labels.csv`、`data/train/schema.json`；`data/val/features.csv`、`data/val/schema.json`；以及 `private/validation_labels.npy`、`profile.json`、`data_hashes.json`。图像任务另有相应的 `images/`。
- **数据可见性：** 候选训练只挂载导出的 train，候选验证预测只挂载无标签 val。`profile.json` 包含训练统计和前 3 行训练样例，不包含验证标签；验证标签由监督进程持有。监督进程本身不是与全部原数据完全隔离的独立系统。
- **什么时候改：** 接新文件格式、字段布局或图像路径规则时改。它做的是**数据格式适配**，不负责选择模型，也不在外层预先拟合特征工程。

##### 2.5 `mlevolve_adapter.py`：将课程环境接到上游显式接口

- **`load_search_config()`：** 从 `third_party/MLEvolve/config/config.yaml` 加载算法配置，装配本轮输出目录、步数、时间、随机种子、执行资源和模型名称，解析 coldstart 文件路径。算法默认值仍来自上游。
- **环境覆盖的范围：** `runtime.upstream_environment` 只允许嵌入模型设备/路径、GPU 数量、预训练缓存路径等白名单项；试图通过它覆盖其他算法设置会报错。搜索/记忆/触发策略及超参数应在 `third_party/MLEvolve/` 修改。
- **`preflight()`：** 全局记忆启用时检查 `torch`、`sentence_transformers`、`faiss`、`rank_bm25` 等依赖。缺失就失败，不偷偷关掉记忆；依赖可导入也不代表嵌入权重已成功下载或加载。
- **`CourseIntegration`：** `instructions()` 提供课程协议和真实资源；`environment()` 报告可用软件与当前 CPU/禁网环境；`data_preview()` 提供训练概况；`validate()` 把真实执行状态交给上游；`authoritative_review()` 保留 LLM 的总结和错误判断，但用监督器真实分数替换 LLM 自报指标，并强制标记执行失败。
- **`create_search()`：** 注册统一 API transport、设置随机种子、按上游配置准备 coldstart，创建原 `AgentSearch` 和 `Journal`，由上游初始化记忆；返回搜索对象供调度器使用。
- **直接输出：** `mlevolve-effective.yaml` 是本轮实际生效的上游配置；`integration.json` 记录接入方式和记忆初始化情况。后者在搜索对象成功创建后才写出，初始化失败时不保证存在。
- **必要差异：** 关闭的是上游原生 grading server，因为课程改用沙箱执行和可信验证评分；不是取消所有验证。课程协议还替代了原 benchmark 的输入/交付约定。当前没有在此 monkey patch 搜索方法，也没有在此重新实现节点选择。
- **什么时候改：** 上游接入接口、宿主环境描述、配置映射发生变化时改；不是日常改进搜索算法的入口。

##### 2.6 `orchestrator.py`：管理一次运行的完整生命周期

这是运行层的中心文件，负责把各基础设施接起来，**不是另一套 AgentSearch**。主要有三个入口：

- **`run_search()` 的启动阶段：** 创建全新运行目录，保存配置、外层/上游源码哈希和 `pip freeze` 环境快照；准备 train/val，创建沙箱、账本、网关，再通过适配器创建上游搜索对象。已有运行目录会被拒绝，不是断点续跑。
- **`run_search()` 的搜索阶段：** 调用 `third_party/MLEvolve/engine/scheduler.py`，向它传入候选执行 `execute`、检查点 `checkpoint`、停止条件 `should_stop` 三个回调。选哪个节点、生成哪种改进、如何利用记忆，仍由上游决定。
- **`execute` 回调：** 把上游生成代码写为 `candidates/<id>/solution.py`，做语法检查；调用沙箱训练；检查模型目录非空且无符号链接；再开新进程加载模型预测 val；读取私有验证标签并可信评分。成功/失败均记录为 `CandidateResult`，将诊断返回上游。它不自行写一个模型替代 LLM 候选。
- **`checkpoint` 回调：** 让上游保存 Journal，并导出 `candidates.json`、`decisions.json`、`checkpoint.json`（进度、当前最佳节点和费用）。父节点、阶段、计划、分析来自上游 Journal，不是运行层重新规划；检查点不等于已实现跨进程续跑。
- **停止与选优：** 观察时间、预算/未知用量停机信号。结束时检查上游 `best_node` 同时满足上游有效性判断与真实执行成功，再交给 `artifacts.freeze()`。若上游没有选出有效最佳节点，就失败，**不会绕过上游自行按分数重选**。
- **`predict()`：** 检查冻结哈希，在独立沙箱加载 `best/` 并预测，结束后再次核验哈希。不调用 LLM、不重新搜索。它本身不对任意预测结果计算 accuracy。
- **`final_evaluate()`：** 冻结核验后，先排他创建 `final_evaluation.lock`，再读取 test、导出无标签 `final_features/`，调用预测和评分，写 `final_result.json`。测试分数不回流上游搜索；锁在开始访问 test 前创建，即使中途失败也不会自动删锁重试。
- **主要输出：** 配置/源码/环境快照，候选源码和 `result.json`，检查点汇总，`cost_summary.json`、`summary.json`；最终评测另有锁、`final_features/`、`final_output/`、`final_result.json`。模型由候选程序写出，预测 CSV 也由候选程序写出，不是这个文件训练或手填的。
- **失败与资源：** 网关/搜索对象初始化异常会保存失败 `summary.json`、`cost_summary.json`；更早的数据准备或落盘失败不保证具备完整摘要。时间 deadline 在数据准备之后、上游初始化之前设置，不是严格的整进程 8 小时硬截止；受控停机与强杀进程不同，不能保证任意中断都能冻结。独立 `predict/evaluate` 从该 run 的 `runtime.json` 读取执行资源；无快照的旧目录兼容 300 秒、4096 MB、2 CPU 默认值。
- **什么时候改：** 新增宿主执行阶段、运行快照、停机/交付流程时改。演化、融合、停滞检测、记忆检索和搜索回传不应搬到这里。

##### 2.7 `runner.py`：实际执行候选的隔离环境

- **核心对象：** `Sandbox(timeout=300, memory_mb=4096, cpus=2)`；`command()` 构造隔离命令，`run()` 启动并等待进程。
- **输入：** 候选 `solution.py`、本阶段的数据目录、模型目录、输出目录、`train/predict` 模式、日志路径，以及资源限制。搜索阶段的资源由运行配置传入。
- **隔离方式：** 要求 Linux/WSL、bubblewrap 和 `/opt/` 下的专用虚拟环境；清空继承环境、禁网、最小挂载，不把宿主项目、home、API 密钥或 test 目录交给候选。代码与数据只读；训练时模型目录可写，预测时模型目录只读；输出目录可写。
- **执行限制：** 用 `prlimit` 限制地址空间、CPU 时间、文件大小和打开文件数，用 `taskset` 绑定可用 CPU；墙钟超时会终止进程组。训练与预测分别启动进程，候选必须真的保存并重新加载模型。
- **输出：** stdout/stderr 写入调用者指定的 `train.log`、`predict.log` 等；成功返回耗时和日志尾部，失败/超时抛异常。`model/` 和 `predictions.csv` 是沙箱中的**候选程序输出**，runner 只是执行和挂载它们。
- **什么时候改：** 更换执行平台、增加 GPU/依赖挂载、调整隔离或资源规则时改。地址空间限制不是 cgroup 总内存限制，提示协议也不能替代隔离；这里不是“所有恶意代码绝对安全”的保证。

##### 2.8 `evaluator.py`：独立检查预测并计算真实分数

- **唯一主要函数：** `evaluate_predictions(path, labels, classes)`；由监督进程传入候选预测路径、真实标签和合法标签域。
- **检查顺序：** 文件必须存在、不是符号链接、大小不超过 20,000,000 字节；列名严格为 `row_id,prediction`；行数匹配；行号完整且按 `0..N-1` 排序；预测无缺失且属于原始标签域。
- **输出：** 返回 `float` accuracy，异常表示预测不合格。本文件本身不写结果 JSON；`orchestrator.py` 保存验证或最终测试结果。
- **与上游区别：** 这里负责“这份预测实际对了多少”；上游 `engine/evaluation.py` 等负责其搜索评价/回传机制。两个文件不应因名字相似而混为一谈。
- **什么时候改：** 正式评价指标或预测文件协议变化时改；不是让 LLM 解释一个分数，更不是根据候选 stdout 中自报的 accuracy 打分。

##### 2.9 `llm_gateway.py`：所有真实模型请求的统一出口

- **核心对象：** `Gateway`；`generate()`、`query()` 适配上游调用形式，统一进入 `call()`。`normalize_prompt()`、`stringify()` 负责把不同提示结构转换成消息格式。
- **输入：** 上游 messages、temperature、输出 token 限制、schema、stop 和 provider options，加上模型策略、账本、审计对象。密钥与 API 地址来自监督进程环境变量，不传给候选。
- **请求前：** 检查停止信号/deadline 和模型策略声明，加入课程协议，限制输出长度，按完整配置上下文上界加输出上限预留预算；余额或调用额度不足就不发请求。预留值不等于已经花掉的费用。
- **实际请求：** 向 `/chat/completions` 发送非流式 HTTP 请求，单次网络 timeout 由 `request_timeout_seconds` 控制，当前配置 1200 秒；网络重试次数、退避与冷却均由 runtime 显式配置。当前不再写死关闭 thinking，相关参数沿上游 model profile 传递。
- **请求后：** 保存原始响应；检查输入/输出/总 token 一致，按 usage 结算；检查返回模型别名，再解析正文。需要 schema 时，将 schema 作为文本要求附加并用 `jsonschema` 校验，不是原生工具调用。
- **异常规则：** 临时传输错误保留完整预留，并按明确配置有界重试；每次实际派发单独记账，以 `logical_id`、`retry_of` 关联。恢复额度耗尽则停止新请求；鉴权/usage/模型策略等非临时错误立即停止。截断、空正文、正文 JSON/schema 错误保留已结算费用，不将其重标为 unknown；上游仍可按原生成/解析异常路径另行调用。
- **直接输出：** `calls/<call_id>.request.json`、`.response.json`，以及账本和事件日志记录；成功向上游返回文本或结构化结果。请求文件不含 Authorization 密钥，但可能含提示、代码和训练概况，原始日志仍需谨慎分享。
- **什么时候改：** API 协议、模型参数映射、usage/返回结构或错误处理变化时改，不在此替换 Agent 的推理流程。`verified`、发布日期与别名检查依赖本地策略声明，不能独立证明学校底层实际模型 build；deadline 也不会即时取消已经在途的网络请求。

##### 2.10 `budget.py`：独立、持久化的调用账本

- **核心对象：** `Ledger`，底层是 `cost.sqlite` 的 `calls`、`policy`、`recovery_policy` 表；`BudgetStop` 表示应停止继续派发请求，并带可区分的停止原因。旧账本无恢复策略表时按严格模式解释，不修改历史费用。
- **`reserve()`：** 用 SQLite 写事务串行检查并发调用的预算、调用数和累计未知用量阈值，将已结算和仍占用的预留一起计入；为下一次请求生成独立 ID。`max_unknown_calls: 0` 保持旧版首次 unknown 后停止；当前配置 3 允许阈值以内恢复，成功不清零 unknown。
- **`settle()`：** 根据实际输入/输出 token 和单价结算，正常释放多余预留；如果服务端用量超过预留，记为 unknown 并停止，不掩盖超量事实。`unknown()` 将未能确认的调用保留为未决费用。
- **`summary()`：** 返回预算上限、已结算金额、unknown 预留、在途预留、已知 token、未决笔数和逐笔明细；由调用者写入 `cost_summary.json`。`charged_or_reserved_rmb` **不是全部已经确认扣款的金额**。
- **金额单位：** `cost_fen_units()` 名字是历史遗留，实际返回整数纳元，即 `1e-9` 元，不是“分”。公式本质是 `(输入 token × 输入每百万单价 + 输出 token × 输出每百万单价) / 1,000,000` 元，内部用 Decimal/整数避免浮点比较预算。
- **输入/输出：** 输入为本轮预算、单价、调用上限、未知用量阈值、预留上界及真实 usage；直接维护本轮 `cost.sqlite`。重新打开已有账本时不能改其预算/单价/调用上限/未知用量策略。
- **什么时候改：** 正式计价规则、账本状态或并发预算策略变化时改。它不查询学校账单，不自动计算缓存优惠，不跨 run 汇总课程预算；账本可在崩溃后保留，不代表搜索支持自动续跑。

##### 2.11 `artifacts.py`：把上游选出的结果固定下来

- **`freeze(candidate, run, result)`：** 接收已经由上游自动选定且经宿主核验的候选，复制其 `solution.py` 和完整 `model/` 到 `best/`，生成 `freeze.json`。
- **冻结记录：** 包含候选结果、`best/` 文件的 SHA256 清单、`locked` 标记。复制模型前检查符号链接；已有 `best/` 时拒绝覆盖。
- **`verify_frozen(run)`：** 重新计算 `best/` 文件清单，与 `freeze.json` 比较，不一致则拒绝；预测/最终评测会调用它。
- **输入/输出：** 输入为候选目录和 `CandidateResult` 字典；直接输出 `best/solution.py`、`best/model/`、`freeze.json`。模型原内容来自候选训练，freeze 只复制和登记，不再次训练。
- **什么时候改：** 提交打包格式、冻结文件范围、完整性检查变化时改。它不选最佳模型；哈希比较用于发现相对已保存清单的变化，不是文件系统只读锁，也不是防止清单本身被改写的数字签名。

##### 2.12 `audit.py`：通用留痕和文件完整性工具

- **`write_json()`：** 创建父目录，写临时 JSON 后用 `os.replace` 替换目标，避免读到半份文件；禁止 NaN 等非标准 JSON 数值。临时名固定为目标加 `.tmp`，不应把它当作任意多线程同时写同一路径的锁。
- **`sha256()` / `file_manifest()`：** 分块计算文件哈希、递归生成相对路径到哈希的清单，拒绝清单中的符号链接；用于源码快照、模型检查和冻结验证。
- **`Audit.event()`：** 在同一 Audit 实例的线程锁下追加事件到 `events.jsonl`，记录 Unix 时间和事件字段，并 flush/fsync。
- **输入/输出：** 接收调用者提供的对象、文件路径和事件细节；输出 JSON、哈希字典或 `events.jsonl`。它不自行判定一次搜索应做什么。
- **什么时候改：** 事件格式、通用 JSON 落盘、文件清单规则变化时改。这里的 `Audit` 是事件记录器；CLI 的 `audit` 子命令主要汇总 `budget.py` 账本，两者不是同一个功能，更不是学校后台账单。

##### 2.13 一次迭代中，这些文件如何配合

以“上游生成一个新候选”为例：

1. `cli.py` 读取参数，`orchestrator.py` 准备本轮目录和数据，`mlevolve_adapter.py` 创建上游搜索对象。
2. 上游调度器和 Agent 决定下一步实验，按需经 `llm_gateway.py` 调模型；每次请求都先经 `budget.py` 预留，再结算，调用证据由网关与 `audit.py` 留档。一次候选可能涉及多次 LLM 调用，候选数不等于调用数。
3. 上游把组装后的代码传给外层执行回调；`orchestrator.py` 保存候选源码，`runner.py` 执行训练及新进程预测。
4. 候选自己产出模型和预测 CSV，`evaluator.py` 独立算分；`contracts.py` 的 `CandidateResult` 承载结果，适配器把可信指标送回上游分析。
5. 上游继续负责判断、记忆、搜索与自动选优；运行层只保存检查点、执行停止约束，最终由 `artifacts.py` 冻结上游选中的结果。
6. 搜索结束后，`predict` 使用冻结模型；明确执行 `evaluate` 才访问最终 test。正常开发不靠反复测 test 来迭代。

**推荐阅读顺序：** 先看 `contracts.py` 理解交付接口，再看 `cli.py -> orchestrator.py` 理解主流程；随后分别读 `mlevolve_adapter.py`、`runner.py/evaluator.py`、`llm_gateway.py/budget.py`，最后看 `data.py`、`artifacts.py`、`audit.py` 的实现细节。要研究论文的搜索算法，应继续读上一节的 `third_party/MLEvolve/`，不是在这 12 个文件中寻找被隐藏的第二套搜索器。

#### 3. 配置、测试与阅读文件

| 路径 | 用途；什么时候需要操作 |
|---|---|
| `configs/runtime.yaml` | 100 步、28800 秒、400 调用、20 元、1200 秒请求超时、32768 输出 token、候选资源和有界网络恢复策略；CLI 能覆盖部分项目。不是算法开关总表 |
| `configs/tasks/coupon.yaml` | 优惠券文件布局、字段/标签和任务描述，属于数据格式适配，不是固定 CatBoost 解法 |
| `configs/tasks/headlines.yaml` | 新闻标题文本对的 CSV 字段、标签和描述，不是人工指定的算法方案 |
| `configs/model_policy.json` | 模型标识、发布日期、token 单价、来源和证明限制；影响合规检查与账本 |
| `configs/pricing-evidence/release-20260910.html`、`release-20261017.html` | 价格/模型相关历史公告快照，供查证，不是执行代码 |
| `configs/pricing-evidence/pricing-20261115.html`、`flash-20260910.jpeg` | 价格页面/价格表快照，不是我们的 token 消耗或账单 |
| `configs/mlevolve-integration.patch` | 此前集成修改的补丁快照，供对照和审阅，不是运行时注入文件；源码已直接收录，克隆后不要再应用 |
| `tests/test_runtime.py` | 数据协议、评分、沙箱、冻结、计费和调用协议等基础设施测试 |
| `tests/test_mlevolve.py` | 用假模型/嵌入检查上游机制、diff、调度、停机和合成数据集成路径；不是论文效果评测 |
| `tests/verify_experiment.py` | 早期实验的费用/哈希/冻结顺序复核脚本，不是通用回放器；源码变更后其旧哈希断言可能不通过，不回写历史记录 |
| `pyproject.toml`、`requirements.txt` | 安装包、注册 CLI、声明 Python 和依赖；新增算法依赖可能需要同步这里并更新环境 |
| `.env.example`、本地 `.env` | 前者可分享且为空密钥模板，后者是本机秘密；CLI 不自动加载，两者都不是算法配置 |
| `README.md` | 组员主要阅读入口，当前版本、架构、运行方式和待办 |
| `AGENTS.md`、`CLAUDE.md` | 给 AI 的约束/同规则入口，组员一般不需要操作；不是实际运行的 Agent 提示词 |
| `journal.md` | 按需追溯重要改动与 AI 协助，不是 MLEvolve 的搜索 `journal.json` |
| `human.md` | 人类专属可选笔记，AI 不代写，不是系统必须填写的表单 |
| `.github/pull_request_template.md` | 可选的代码审查交接格式，运行系统不读取它 |
| `.gitignore`、`.gitattributes` | 排除真实密钥、缓存等，明确收录完整 runs；runs 禁止 Git 文本换行转换，以保留原始证据字节 |
| `docs/agent-runtime.md` | 给 AI 按需查阅的算法集成边界、长运行/网络恢复细节、历史实验与完整 run 索引 |
| `third_party/MLEvolve/LOCAL_CHANGES.md` | 上游固定提交、版权和本地修改文件清单；不是另一套算法配置 |
| `docs/course/AIAA3102 - Final Project - Full Assignment Requirements.md` | 正式要求英文原件；`assignment.zh-CN.md` 是非官方中文对照 |
| `docs/course/AIAA3102_Final_Project_Preview.md`、`preview.zh-CN.md` | 历史预告及中文对照，不覆盖正式要求 |
| `docs/tasks/coupon/Readme.md`、`README.zh-CN.md`、`info.json` | 优惠券原任务说明、中文对照、数据集元信息 |
| `docs/tasks/headlines/README.md`、`README.zh-CN.md` | 标题比较任务原说明和中文对照 |

**人主要操作：** 阅读 README/课程材料，按目标改算法或运行配置，运行命令，查看 `summary.json`、账本和日志。**AI 管理文件不是待实现功能，`runs/` 不是手改候选的地方。** Git 内部元数据、Python 缓存和安装生成的 egg-info 不在上述树中，通常也无需手动操作。

#### 4. 一次运行的输出：runs/<运行名>

下面描述**当前恢复版代码的输出约定**，不是宣称现在已有这样一次真实运行。失败/初始化中断会缺少后续文件；旧简化版历史 run 也可能缺少恢复版新增文件，不能补写来伪装成新版证据。

“课程运行层”指 `src/superagent/`；“上游”指 `third_party/MLEvolve/`；“候选程序”指 MLEvolve 让 LLM 生成的代码。**文件由谁写入，与内容由谁产生，是两回事。**

| 文件/目录 | 谁生成、保存什么 |
|---|---|
| `runtime.json`、`task.json`、`model_policy.json` | 课程运行层保存本轮生效的运行限制、任务和价格策略快照；不是需要修改的全局配置 |
| `source_hashes.json` | 课程运行层记录外层 `src/` 源码哈希；旧运行只有哈希不等于保存了可完整恢复的旧源码 |
| `upstream_source_hashes.json` | 恢复版课程运行层新增，记录上游文件哈希；同一个 HEAD 下也能区分本地修改，不代表已经生成完整源码归档 |
| `environment.txt` | 课程运行层调用 `pip freeze` 保存实际依赖版本 |
| `profile.json`、`data_hashes.json` | 课程数据层生成训练概况和导出数据哈希；不是模型训练结果 |
| `data/train/features.csv`、`labels.csv`、`schema.json` | 课程数据层导出的统一训练输入，候选 train 阶段可读；图像任务还会导出所需图像 |
| `data/val/features.csv`、`schema.json` | 课程数据层导出的**无标签**验证特征，候选 predict 阶段可读 |
| `private/validation_labels.npy` | 课程数据层给可信评分器保管的验证标签，候选不挂载、不允许读取 |
| `mlevolve-effective.yaml` | 适配器保存装配后的上游配置，区分默认算法配置与本轮模型/资源/路径覆盖 |
| `integration.json` | 适配器记录课程接口方式、记忆是否初始化、冷启动是否找到指导；初始化失败时可能尚未生成 |
| `calls/*.request.json`、`*.response.json` | 课程网关保存所有 Agent 的请求/响应/usage。内容包含上游生成过程，不是上游提供方另记的一份账本；可能含数据样例，不能随便公开 |
| `cost.sqlite` | 课程 `budget.py` 的原始费用账本，包含预留与结算；学校平台实际扣款需另查，不把未知预留当已确认消费 |
| `cost_summary.json` | 课程运行层或 `audit` 从账本汇总的费用、token、调用数；不是 MLEvolve 自报成本 |
| `events.jsonl` | 课程审计层写入开始/候选结束/检查点/冻结等事件 |
| `mlevolve.log` | 课程运行层配置日志文件，上游与集成过程向其写入日志；不是另一份算法源码 |
| `candidates/<id>/solution.py` | **代码内容由上游 LLM 生成**，课程执行回调负责落盘；不要当作人维护的项目文件 |
| `candidates/<id>/model/` | **候选程序训练时写出的模型、预处理等**，课程运行层检查非空并测试重载；具体文件名由候选代码决定 |
| `candidates/<id>/train_output/`、`val_output/` | 候选程序的输出目录；`val_output/predictions.csv` 是新进程加载模型后的无标签验证预测 |
| `candidates/<id>/train.log`、`predict.log` | 课程沙箱捕获候选程序 stdout/stderr 的执行日志 |
| `candidates/<id>/result.json` | 课程运行层保存执行状态、可信 accuracy、耗时和诊断；不是 LLM 自报成绩 |
| `candidates.json` | 课程运行层将各候选执行结果汇总，便于审计 |
| `decisions.json` | 课程运行层从**上游 Journal** 导出节点、父节点、阶段、计划、分析、指标；算法决策来自上游，不是外层重新规划 |
| `checkpoint.json` | 课程运行层在调度检查点保存进度、上游当前最佳节点和费用，成功冻结时写入最终摘要；用于追溯，不是可直接恢复进程的完整快照 |
| `mlevolve/submission/submission_<id>.csv` | 候选程序产生验证预测，课程执行回调复制到上游预期位置；是验证预测，不是最终 test 提交 |
| `mlevolve/journal.json`、`filtered_journal.json` | 上游 `config.save_run` 保存完整/筛选搜索记录，含节点和轨迹；与人类工作日志 `journal.md` 无关 |
| `mlevolve/config.yaml`、`best_solution.py` | 上游 `save_run` 保存配置和从 journal 取出的最佳代码；可能与其他上游最佳保存路径并存，不作为课程最终交付锁 |
| `mlevolve/best_solution/` | 上游选优器保存 `solution.py`、`node_id.txt`、`metric.txt` 等当前最佳记录，不含课程完整冻结模型包 |
| `mlevolve/best_submission/submission.csv` | 上游保存当前最佳验证预测 |
| `mlevolve/top_solution/top*/` | 上游保存 Top-K 方案、指标和相关预测，目录取决于实际候选状态 |
| `mlevolve/global_memory/records.json` | 上游记忆模块保存实验经验；当前正常路径只落盘记录，重新加载时重建检索索引。检索器虽提供索引保存接口，但正常记忆路径没有调用它，不能声称一定有向量/FAISS 索引文件 |
| `best/solution.py`、`best/model/` | 课程 `artifacts.py` 按**上游 best_node**复制的最终代码和完整模型包，用于后续实际预测；外层不按原分数另挑赢家 |
| `freeze.json` | 课程交付层记录冻结候选、文件哈希等，证明后续预测使用冻结产物 |
| `summary.json` | 课程运行层写的本轮状态、停止原因、候选数量、最佳验证 accuracy 和费用；失败时内容更少 |
| `final_evaluation.lock` | 只有显式最终评测才创建的一次性 test 访问标记；不应删除后重测 |
| `final_features/` | 最终评测获准后，课程数据层导出的无标签 test 特征和 schema |
| `final_output/predictions.csv`、`predict.log` | 最终评测中候选程序生成 test 预测，沙箱保存预测日志；搜索完成本身不会自动产生 |
| `final_result.json` | 课程可信评测器保存 test accuracy；不会反馈给搜索 |

`superagent predict --output <目录>` 的输出在你指定的目录，不一定是 run 内的 `final_output/`。候选训练可能额外保存文件，不能在 README 中预先穷举每种模型内部文件。

所有现有尝试（包括失败记录）的索引与结果统一见 [AI 运行参考](docs/agent-runtime.md#完整记录索引)。`runs/experiment-audit.json` 仅是早期跨尝试复核结果，不自动代表后来所有运行。

#### 5. 后续改哪里

| 目标 | 主要位置 |
|---|---|
| 优化搜索/选择/回传/停滞触发 | `third_party/MLEvolve/engine/`、`config/config.yaml` |
| 优化规划、生成、debug、evolution/fusion、提示词 | `third_party/MLEvolve/agents/` |
| 优化记忆写入、检索与使用 | `third_party/MLEvolve/agents/memory/`、`agents/planner/` |
| 改轮数、金额、时间、调用数、CPU/内存 | 外层 `configs/runtime.yaml` 或支持的 CLI 参数 |
| 改数据格式、API、计费、沙箱或模型交付协议 | 外层 `configs/`、`src/superagent/` 的对应模块 |
| 新增库、预训练资源或 GPU 能力 | 上游算法 + 外层依赖/资源/沙箱配套；不能只改提示词声称环境已有 |

算法优化后补对应测试、更新本地修改声明，并通过本仓库 Git 历史记录实际源码变更；如另导出补丁，应标明基线与覆盖范围。保留许可证、版权和论文引用，不手改 `runs/` 历史候选/分数来“优化结果”。

GitHub 协作目标为本文开头的小组公开仓库，默认分支为 `main`；没有配置 Slack、GitHub Pages 或强制分支保护。组员不需要额外的网页或管理工具；课程资料、应用测试与完整 runs 保留，AI 规则和历史运行细节按需查阅。

<details>
<summary>开始开发时：数据、检查与交接</summary>

仓库外的原始任务数据目录不随仓库分发，每位同学从获准课程渠道获取；完整 runs 已包含历史导出的数据，不能据此假定有完整原始任务根目录。当前机器原始数据在仓库旁 `../final_project_tasks/final_project_tasks/`，其中含 `in_vehicle_coupon_recommendation/` 和 `headlines_ctr_clean/`；其他机器不保证相同路径。个人配置用本机环境变量或被 Git 忽略的文件。

已实现的候选隔离方式：训练候选只挂载 train，预测候选只挂载无标签特征；搜索提示只含训练概况和聚合验证反馈。可信监督器可读取原始数据根目录，但开发路径不读取 test；独立最终评测命令在冻结核验之后才加载 test。这里不声称监督器本身被 OS 权限隔离。本组明确要求将完整 runs（含模型、日志和导出数据）直接收录；真实密钥、HF 缓存、环境及仓库外原始数据仍排除。

使用本文开头的 `git clone https://github.com/Xiaoyu-Li0912/superagent-workspace.git`，MLEvolve 修改后的源码与完整 runs 一起交付。应用环境和测试命令见本文开头；编辑协作文档不需要网页检查，也不应触发付费调用。

确认负责模块后按需建短分支；交接说清“改了什么、怎么验证、还剩什么”，共同接口先沟通，建议另一人复核。重要决定直接更新本页，AI 将重大工作追加到 journal；无需步骤 ID、JSON 或另建进度表。

</details>

<details>
<summary>资料来源与翻译说明</summary>

正式要求和 Preview 从上一级 `project/` 同名文件复制；任务原说明和优惠券 `info.json` 从上一级 `final_project_tasks/final_project_tasks/` 对应任务目录复制，五份原件保持不变。本页介绍与论文表沿用 2026-11-13 的《团队项目简要说明》，合并原工作区的分工、数据、来源和阅读索引，不是老师新增要求。

中文对照于 2026-11-15 由 AI 辅助整理，非教师官方译文；有歧义以英文原文和老师澄清为准。Preview 仅供历史背景，不替代正式要求。优惠券中文说明中的元信息对照和安全提示是已标注的补充；外部论文、工具和代码未全文翻译。源文更新时同步核对译文，不维护另一套要求。

本地副本不等于获准转载。发布 GitHub 前须核实目标、可见性和课程材料分享权限，private 也不自动代表可上传所有材料；不获准的原文应保留本地，改用许可的来源说明。本实验备份已开始用户授权的实际 API 与自主运行验证，具体结果见本文及原始运行证据，不等于复现论文成绩或完成课程提交。三人认领与最终资源安排仍待小组确认。

</details>
