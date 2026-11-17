# Local Modifications / 本地修改声明

本目录由 AIAA3102 SuperAgent 课程项目基于以下上游代码修改并直接收录，非未经修改的上游发布版：

- Repository: https://github.com/InternScience/MLEvolve
- Base commit: `f4a37b96c948bfed6564e62f3229e0b38e908d33`
- Copyright 2026 MLEvolve Team, Shanghai Artificial Intelligence Laboratory
- License: Apache License 2.0，完整原文见 `LICENSE`，原有声明保持不变。
- Paper: Du et al., *MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery*, arXiv:2606.06473 (2026)。完整作者及 BibTeX 见 `README.md` 的 Citation。

## 已修改的上游文件

| 文件 | 课程集成修改 |
|---|---|
| `engine/agent_search.py` | 显式 integration 接口、课程数据概况与执行回调、记忆初始化失败明确报错 |
| `engine/execution.py` | 接入可信执行结果，允许合法的零 accuracy 结果 |
| `llm/__init__.py` | 显式 transport 接入课程统一 API/计费网关，传播受控停止信号 |
| `agents/draft_agent.py` | 为原草稿生成路径传递课程环境与实现协议 |
| `agents/debug_agent.py` | 为原调试路径传递课程环境与实现协议 |
| `agents/improve_agent.py` | 为原改进路径传递课程环境与实现协议 |
| `agents/code_review_agent.py` | 课程审查协议与 diff 响应集成 |
| `agents/result_parse_agent.py` | 将可信评分接入原结果分析与记忆路径，不采用 LLM 自报分数 |
| `agents/coder/stepwise_coder.py` | 原分阶段生成/合并路径传递课程实现说明 |
| `agents/prompts/environment.py` | 描述实际课程 CPU、沙箱和可用资源 |
| `agents/prompts/impl_guideline.py` | 课程 train/predict、持久化、数据及时间约束 |
| `agents/prompts/shared.py` | 输出、数据泄漏和可用资源提示与课程环境对齐 |
| `agents/prompts/validation_template_prompts.py` | 课程接口下的代码审查提示 |

## 新增文件

- `engine/scheduler.py`：从上游 `run.py` 抽取两阶段调度，供课程入口调用；保留顺序草稿、并行执行与续搜、journal 计步及排空逻辑。
- 本文件：来源、许可证与修改声明。修改文件顶部附本地修改提示，不改变算法行为。

原算法默认配置、节点选择、回传及停滞触发阈值继续由上游代码控制。必要差异包括课程数据协议、学校 API、结构化输出兼容、CPU 沙箱、可信 accuracy、预算和交付；不能据此宣称严格论文复现。详见仓库根 README 和 `docs/agent-runtime.md`。

`configs/mlevolve-integration.patch` 是此前集成补丁快照；普通克隆已经包含修改，不重复应用。后续算法改动直接在本目录开发，更新此声明及测试，并通过课程仓库 Git 历史记录；原版权与引用须继续保留。
