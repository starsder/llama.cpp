# 02 预测与缓存：静态表/XT → Fate → SMoE → 在线缓存与双门控

本章覆盖 MoE/SMoE 专家预测与设备侧专家缓存的全部已尝试路线：从静态热表与 XT 转移清单，到 Fate 在线跨层 gate、共享专家 SMoE 预测，再到在线缓存工程化（预取量、排名截止线、去重、淘汰分数、热区回填、双门控自适应传输阈值），最后到共享池/位置权重与 `LOCAL_EVIDENCE/moe-cache/optimization400` 的生命周期统计与两个未提交候选。

主线的编年与因果由 `00-research-chronology.md` 统一叙述；本章只提供该主线上**归属预测与缓存**的可追溯实验档案。PLE 缓存是研发顺序中的第一项，属 `01-host-and-devpart.md`，本章只在阶段一小结处引用，不重复其证据。

## 0. 阅读约定

### 0.1 证据路径标记

路径标记见[证据索引](evidence/README.md)：`SOURCE_TREE`是实验源码工作区，`LOCAL_EVIDENCE`是临时证据根（相关报告在其`moe-cache/`下），`LOCAL_MODELS`是模型根，`OTHER_SOURCE_TREES`是其他检出。本章来源文档使用相对链接。

- `sources/handoff.md`（源工作区交接备忘全量快照）：按 §编号 + 快照行范围引用。
- `sources/smoe-nk-degradation-plan.md`（SMoE N+k 退化计划快照）。
- `sources/moe-cache-score-aware-prd.md`（当前 PRD 快照：真实频率、共享池与位置权重；其末尾 §历史 v1 保留冻结设计原文）。
- `sources/moe-decode-perf-plan.md`、`sources/rebuild-spec.md`（旁证：预测器位置与旧结论否定清单）。
- 仓库内 `../moe-cache-score-aware-prd.md` 是**较早的冻结 v1 副本**（仅 Problem/Solution/Decisions 等章节）；引用当前策略与共享池内容时一律使用 `sources/` 快照。

### 0.2 状态词表（本章只用这五类，且与“代码在发布树里存在”分开）

| 标签 | 含义 |
|---|---|
| 已发布 | 代码在本发布树的提交（最新为 `7e01451b2`）中，且该开关有明确默认值；**不代表其性能结论已被验收** |
| 仅 WIP | 只存在于 `SOURCE_TREE` 的**未提交**修改，或只存在于旁支检出；发布树无此代码 |
| 仅设计 | 只写在文档里，未落地为可运行代码 |
| 已撤回 | 曾作为结论使用后被明确撤回（如路由错位修复前的 host 20.3 / 32 tps 速度、旧容量判据） |
| 仍开放 | 现象已记录、原因未闭案，或验证条件未具备 |

**纪律（两条，全文遵守）**：
1. **源码存在 ≠ 实现已发布，文档被归档 ≠ 实现已发布**。上表“已发布”只表示该开关/代码出现在本发布树的提交里；是否被采纳为推荐、是否有可用证据，必须另看“保留/放弃/暂停理由”字段。采集侧的旁支检出（如 SMoE trace 插桩）即使其文档进了快照，实现本身仍是旁支，不算发布。
2. 本轮没有任何优化候选被采纳进发布树；`SOURCE_TREE/build-ple-trace-mrs/bin/` 下的 DLL 在运行当时可能已被替换为 WIP 候选，引用运行结果必须给出**运行当时**的二进制哈希，且不得声称该目录“当前”仍是同一文件（见 §11 第 9 条）。

### 0.3 每条路线的统一字段

路线ID、状态/版本、为何尝试、技术机制、实验条件与证据、观察与结论边界、保留/放弃/暂停理由、遗留问题与重新开启条件。

### 0.4 硬件与验收口径（用户提供的工作点）

本机：AMD Ryzen 9 5950X（16 核）、DDR4-2666 128GB、RTX A5000 Laptop 16GB、PCIe 4.0 x8。当前 400-token 工作点：`--no-mmap --lazy-mode off`、权重完整驻留主存、`CACHE_MIB=6144`（400-token 验收基线固定 6144；auto 只出现在 128-token 系列）、16 线程、单实例锁 + 至少 90000 MiB 空闲内存闸门；当前内部筛选下限 **19 token/s**（由 20 下调，见 `sources/moe-cache-score-aware-prd.md` §当前性能验收基线）。**旧 20.3 / 32 tps 等成绩已撤回，不得作为正确性能证据引用**；历史数字必须带上下文（上下文长度、KV、预算、步数、是否带 bug 的构建）。

## 1. 路线总览

| 路线ID | 主题 | 状态 | 源（快照 §） | 关键原始日志/证据 |
|---|---|---|---|---|
| PC-01 | 静态热表（manifest.hot，pinned） | 已发布（默认 `PREDICT_STATIC=32`；pin 默认 0） | `sources/moe-cache-score-aware-prd.md` §历史 v1 §Solution | `LOCAL_MODELS/qwen38/traces/moe-predict-v1.bin`、`-static96.bin`；`SOURCE_TREE/cpunode-run.log`、`fused-final.log` |
| PC-02 | XT 跨 token 转移表（MOEPRED2） | 已发布（`PREDICT_XT`） | 同上 §Solution 1、§Implementation Decisions | `moe-predict-xt.bin`；`ablation-lazy-moe2-xt.log`；`moe-xt-12288.csv` |
| PC-03 | CrossLayer 转移表（MOEPRED1）+ 离线对照 | 已发布（`LLAMA_MOE_PREDICT`） | 同上 §Implementation Decisions | `build_expert_manifest.py`、`analyze_cross_token.py`、`moe-predict-v1.bin` |
| PC-04 | 静态/XT 阶段性能与命中（含 PLE 主线的交叉点） | 已撤回（判据）/仍开放（结论适用域） | `sources/handoff.md` §6.1 附近、`sources/moe-decode-perf-plan.md` §1–2 | `ablation-lazy-moe{4,6,8}.log`、`moe-static90-*.csv` |
| PC-05 | Fate 在线跨层 gate | 已发布但默认关；负面结论为用户回述 | `sources/moe-cache-score-aware-prd.md` §历史 v1 §Further Notes | 代码 `moe_cache_predict_fate`；**未找到 `fate=1` 运行日志** |
| PC-06 | SMoE teacher 测试（用户回述 99%） | 用户回述，协议未找到对应 | `00-research-chronology.md` 主线；本章 §4.1 | 未找到与该 99% 对应的协议/原始日志 |
| PC-07 | SMoE N+k 离线退化（recall@10） | 计划文档已归档（结论有效域受限） | `sources/smoe-nk-degradation-plan.md` §7（快照 173–238 行） | `LOCAL_MODELS/qwen38/traces/simulate_smoe_nk.py`、`standard-smoe-8-20260911` |
| PC-08 | 早期 FIFO 池模拟器与其自述局限 | 计划文档已归档（已被 PC-07 取代） | 同上 §2（65–83 行）、§5（130–162 行） | `simulate_smoe_from_standard_trace.py`、`standard-smoe-sim-20-20260910b` |
| PC-09 | `ffn_input` 缺口与两条补法 | 仅 WIP（采集侧改动在旁支检出，未进发布） | 同上 §3（84–99 行） | `SOURCE_TREE/qwen4exp-smoe-trace`（专支检出） |
| PC-10 | SMoE 在线实现与其固定开销 | 已发布（`PREDICT_SMOE=1`，需 split） | `sources/moe-cache-score-aware-prd.md` §历史 v1 §Implementation Decisions | `L2048-err.txt`、`L6144-err.txt`、`a400a-err.txt` |
| PC-11 | 静态热集与 SMoE 准确率口径混淆（98–99%/73%） | 已澄清（口径），数值未找到对应 | 同上 §6（163–172 行） | 代码 `prefetch_predicted/required`；`global-pressure128-v2.summary.json` |
| PC-12 | 预取量的边际成本 = PCIe 争用 | 已发布（开关）/结论有效 | `sources/handoff.md` §6.6（209–258 行） | 仅历史报告（无原始日志名） |
| PC-13 | 排名截止线 `TAKE_MAX`（隐式→显式） | 已发布（默认 2） | 同上 §6.6③ | 同上 |
| PC-14 | 去重（resident/pending/list/admit） | 已发布 | 同上 §6.6④ | 同上 + 400-token 生命周期（PC-30） |
| PC-15 | 容量 × 截止线交互 | 已发布（参数）/关键点未重复确认 | 同上 §6.6⑤ | 同上 |
| PC-16 | 预取 worker 数无益 + `INSERT_WORKERS` 空转更正 | 已发布（更正） | 同上 §6.6①、§6.7 未解决项 | 同上 |
| PC-17 | 淘汰分数：gate-softmax MRS → 真值路由频次 | 已发布（`EVICT_SCORE=0` 为频次） | 同上 §6.7（259–343 行） | 仅历史报告 + `a400a-err.txt` 的 oracle/每排名行 |
| PC-18 | 热区回填 + 冷启动种子 + 空闲填充 | 已发布（默认关） | 同上 §6.7 | 仅历史报告 |
| PC-19 | 每排名准确率 / 每字节效率 | 已发布（诊断） | 同上 §6.7 | `a400a-err.txt`、`a400b-err.txt`（per-rank 行） |
| PC-20 | 自适应截断三尝试（准确率/实测 yield/字节预算） | 已发布（默认关）；三法均败于固定 cut | 同上 §6.7 | 仅历史报告 |
| PC-21 | 占用率观测：价值门槛与速率控制分离 | 已发布（结论）/旧天花板估计作废 | 同上 §6.7 | 用户实测（历史报告） |
| PC-22 | YIELD_AUTO 极值搜索门槛 | 已发布（默认关） | 同上 §6.8（344–372 行） | 仅历史报告 |
| PC-23 | TREND_AUTO 模型驱动门槛与预算 | 已发布（默认关） | 同上 §6.9（373–404 行） | 仅历史报告 |
| PC-24 | 双门控自适应传输阈值（源码判据） | 已发布（`PREFETCH_GATE` 默认 1） | 代码 + §6.8/§6.9 | `SOURCE_TREE/ggml/src/ggml-backend.cpp` |
| PC-25 | `SPLIT=1` 静默算错修复 + `AHEAD` 默认 3 | 已发布（修复随 7e01451b2） | `sources/handoff.md` §6.32（1124–1173 行） | 仅历史报告（Eiffel 用例） |
| PC-26 | `AHEAD_AUTO` 极值搜索 | 已发布（默认关，实测略逊固定 3） | 同上 §6.32④ | 同上 |
| PC-27 | 非阻塞读回“晚一拍” | 已发布（默认不改） | 同上 §6.33（1174–1200 行） | 同上 |
| PC-28 | 共享池 `GLOBAL_POOL` | 已发布（默认 0）；**128 结果跑的是较晚 WIP 修过的 pool，不能为发布树旧 pool 背书** | `sources/moe-cache-score-aware-prd.md` §当前策略 | 128 系列 summary（§9.4） |
| PC-29 | 位置权重 `LAYER_AWARE`/`LFU_POS` | 仅 WIP（未提交） | 同上 §当前策略 | `global-pressure128-v2.summary.json` |
| PC-30 | 共享池内存安全边界（零槽/reading/int64 stride/图 UID） | 仅 WIP（未提交）；事故与修复有证据 | 同上 §共享池的内存安全边界 | `stride-memcheck.log`、`final-verification.json` |
| PC-31 | 128 压力：命中 +20.77% 但 DMA 几乎不变 | 仍开放 | 同上 §此前短回放诊断、§待排查问题 | `global-pressure128-v2.summary.json` |
| PC-32 | 驻留期统计 `CACHE_LIFETIME` + 400-token 基线 | 仅 WIP（未提交）/统计口径已写入当前 PRD | 同上 §400-token 驻留期观测 | `optimization400/lifetime-before.json` |
| PC-33 | 全来源频率门候选（被否） | 仅 WIP/已否（未提交、暂停） | 同上 §准入与归因（不改变默认） | `optimization400/lifetime-after.json`、`fixed400/experiment.json` |
| PC-34 | 仅热回填修复候选（轮转/空槽/eligible victim） | 仅 WIP/暂停（未提交） | 同上 §准入与归因 | `optimization400/logic-final.log`、`fixed400-final/experiment.json`、`final-r1.json` |
| PC-35 | 固定历史 400 步对照方法与基线自身波动 | 保留（方法学） | — | `fixed400*/experiment.json` |
| PC-36 | 运行闸门与记录完整性（锁/超时/误删日志） | 保留（方法学） | — | `lock-smoke.json`、`timeout-smoke.json`、`deleted-logs-incident.json` |

## 2. 阶段一：静态热表 + XT 转移清单（最早形态）

### PC-01 静态热表（每层频率前 N 的专家表）

- **状态/版本**：已发布。`LLAMA_MOE_PREDICT_STATIC`（默认 32，见 `SOURCE_TREE/ggml/src/ggml-backend.cpp:1708`）把静态热专家并入每次预测；`LLAMA_MOE_PIN_STATIC` 默认 0（当前 PRD §当前策略明确“静态热专家默认不 pin”，历史早期实现曾默认 pin）。
- **为何尝试**：单个 token 每层只激活 10/512 个专家，但层间存在稳定的高频专家；先把“永远热”的少数专家钉在显存里，不必依赖任何预测器即可获得命中底座。注意 pin **不是零成本**：它占用有限槽位（推高显存占用与加载/驻留时间），并使这部分容量无法用于自适应。
- **技术机制**：MANIFEST 文件（`MOEPRED1`/`MOEPRED2`）里的 `static` 段是“每层按激活频率降序的 N 个专家 id”（`build_expert_manifest.py` 注释：`static: n_layers * n_static * u16 expert ids (frequency desc)`）。早期实现直接按该表 pin 到槽位（日志 `pinned 3840 / 9216 / 12960 static hot expert copies`）。
- **实验条件与证据**：2026-09-03/04 采集的 `experts-20260903-082346.csv` 为训练集（hold-out 按 graph_id，train ≤ 720）；清单 `moe-predict-v1.bin`（layers=48 experts=512 trans=32 static=64）、`moe-predict-static96.bin`（static=96）。`LOCAL_EVIDENCE` 之外的原文日志在 `SOURCE_TREE`：`cpunode-run.log`、`long-run.log`、`fused-final.log`、`memprobe-run*.log`，均打印 `loaded prediction manifest ... static=64/96/128`。统计证据：`LOCAL_MODELS/qwen38/traces/moe-static90-8192.csv`（1682 条 (graph,layer) 记录）求和 hits=31890、misses=18540 → 缓存命中率 63.2%。**`pred_hits=0` 只说明该文件里没有“路由 id 命中预测位图”的记录，不能据此把全部命中归给 pin**：命中计数是驻留集合（含 pin 与需求准入共同形成的集合）上的读命中，仅凭该 CSV 无法拆分两部分。
- **观察与结论边界**：静态表能给出**不依赖预测器**的命中底座；但它对分布漂移不敏感，且 pin 占用槽位无法自适应。上表命中率只对“当时构建、当时每层/每 tensor 缓存容量与当时的权重驻留规模”成立，不可外推。
- **保留/放弃理由**：保留为**冷启动种子与预测并集的一部分**（`PREDICT_STATIC` 默认 32；`PIN_STATIC` 作为可选实验项），放弃“默认 pin 大表”（占用容量、冻结热集）。
- **遗留/重开条件**：`PIN_STATIC` 的端到端收益未在工作点验收；重开需在 400-token / 6144 MiB 下与 `PREDICT_STATIC` 组合做配对实验。

### PC-02 XT 跨 token 转移表（MOEPRED2，`PREDICT_XT`）

- **状态/版本**：已发布。`LLAMA_MOE_PREDICT_XT=1`；代码注释（`ggml-backend.cpp` 预测段）明确：“cross-token manifest (predict_xt): predict THIS layer of the next token”。
- **为何尝试**：层间（L→L+1）预测的提前量只有一层；而相同层在 token t→t+1 之间存在“同层跨 token”的复用结构，可提供整整一圈（~48 层）的提前量。
- **技术机制**：`build_xt_manifest.py` 输出 `MOEPRED2`（trans 行数 = n_layers，层内自转移）：`trans[layer][src_expert] = top-N dst experts by count`，样本对为 decode→decode（token t 层 L → token t+1 层 L），`N_TRANS=32, N_STATIC=128`。
- **实验条件与证据**：清单 `moe-predict-xt.bin`（layers=48 experts=512 trans=32 static=128）。运行证据：`ablation-lazy-moe2-xt.log`（2026-09-10 12:27，budget=2048 MiB、`predictor=1 prefetch=0 topk=26 static=0`、`pinned 3243 static hot expert copies`）、`ablation-lazy-moe4/6/8.log`（同清单，budget 4096/6144/8192）。预测覆盖统计：`moe-xt-12288.csv` 求和 `pred_hits=6031 / pred_total=16340 = 36.9%`，同文件缓存 hits=32988 / misses=17442 = 65.4%。列语义来自发射代码：`pred_hits` = 被路由的专家 id 落在该层预测位图上的数量，`pred_total` = 参与计分的路由槽数（`k = ids_tensor->ne[0] = 10`），**是覆盖口径，不是准入数也不是缓存命中**。
- **观察与结论边界**：XT 能给出 36.9% 的同层跨 token 覆盖；`moe-xt-8192.csv` 行数（3026）与其他文件（1682）不同（疑似包含额外的图/阶段），其 10.0% 覆盖率**不可与 12288 直接比较**。
- **保留/放弃理由**：清单路径保留为显式 A/B 对照（PRD §Implementation Decisions 明确 “offline XT manifest remains an explicit A/B control”）；不作为默认（覆盖不足，且预测器被在线 SMoE 取代）。
- **遗留/重开条件**：无独立重开条件；若未来重做，需要把 CSV 行数与阶段标记写进文件名/表头，避免再次出现不可比文件。

### PC-03 CrossLayer 转移表（MOEPRED1）与离线对照方法

- **状态/版本**：已发布。`LLAMA_MOE_PREDICT=<path>` 为 `MOEPRED1` 清单；代码注释指出 MOEPRED1 在 XT 模式下没有最后一层的一行（`trans_rows = n_layers - (MOEPRED1 ? 1 : 0)`）。
- **为何尝试**：把“相邻层 gate 可预测”从论文搬到本模型，需要先离线量化“上一层专家的转移表能覆盖下一层多少路由”。
- **技术机制**：`build_expert_manifest.py` 用 `(n_layers-1) * n_experts * n_trans` 的 `(cand_id, count)` 表 + 每层 static 列表，并**在脚本内用 hold-out 回放报告期望命中率**；`analyze_cross_token.py` 并列三种对照：persistence baseline（token t 自身集合，即 LRU 已能拿到的部分）、cross-token 转移表、转移表 ∪ 静态热并集（全部按 graph_id hold-out，train ≤ 720）。
- **实验条件与证据**：`SOURCE_TREE`（数据根 `LOCAL_MODELS/qwen38/traces/experts-20260903-082346.csv`）；`moe-predict-v1.bin`（trans=32 static=64）。运行态使用见 `long-run.log`/`phase-run.log`（`predictor=1`、`pinned 9216 static hot expert copies`）。
- **观察与结论边界**：**脚本自带 hold-out 回放**是本阶段最有价值的方法学资产（先离线判预测器上限，再上机）；但回放基于“同一次采集的相邻层/相邻 token”假设，未纳入 CPU/GPU 分工改变后的数值分歧（见 PC-31）。
- **保留/放弃理由**：保留为对照；默认预测器改为在线 SMoE。
- **遗留/重开条件**：重开需给出“预测器上限 → 运行态覆盖 → 命中 → 吞吐”四段链条在同一工作点的配对数据。

### PC-04 静态/XT 阶段的性能与命中（含 PLE 主线交叉点）

- **状态/版本**：**已撤回（作为判据）**；数字本身保留为历史。
- **为何尝试**：在 PLE 缓存（研发主线第一项，见 `01-host-and-devpart.md`）同期，验证“静态表 + 转移表 + 每层 bundle 缓存”是否值得继续。
- **技术机制**：`lazy-mode` 下每层 expert bundle 缓存 + 静态 pin + XT 清单载入（`prefetch=0`，即当时只做驻留不做预取）。
- **实验条件与证据**：2026-09-10 ablation 系列（`LOCAL_MODELS/qwen38/traces/ablation-lazy-moe4.log|moe6|moe8`），同一清单 `moe-predict-xt.bin`，budget 4096/6144/8192 MiB，Generation **15.2 / 16.2 / 16.7 t/s**；`ablation-lazy-moe4.log` 每 tensor 命中率 21.9%–56.8%（随层升高）。PLE-only / MoE-only / 组合的对照 CSV 为 `ablation-ple-only.csv`、`ablation-moe-only.csv`、`ablation-lazy-ple.csv`。
- **观察与结论边界**：这些 t/s 来自 lazy 模式、prefetch 关闭、旧构建，**不能**当作当前 host 路径（19 t/s 量级、SPLIT=1、prefetch+SMoE）的对照；`sources/moe-decode-perf-plan.md` §7 专门登记了被本文档体系否定掉的旧结论。
- **保留/放弃理由**：保留为“静态表 + XT 只能到这一档”的历史锚点；判据已撤回（后续改为 SPLIT=1 + direct-read + SMoE 预取 + 频次淘汰 + 回填）。
- **遗留/重开条件**：若要复用该阶段数字，必须先在同一 400-token / 6144 MiB / 全内存工作点重测。

## 3. 阶段二：Fate 在线跨层 gate，及其受挫

### PC-05 Fate 在线跨层 gate（`PREDICT_FATE`）

- **状态/版本**：已发布（默认**关**）。`LLAMA_MOE_PREDICT_FATE=1` 启用；`tools-run.py` 的受控基线显式写死 `"LLAMA_MOE_PREDICT_FATE": "0"`。
- **为何尝试**：XT/静态表的覆盖不足（PC-02），且转移表是离线产物。Fate 论文的关键可迁移结论是“相邻层 gate 输入足以支撑低开销的下一层专家预取”，因此改为**在线**：用当前层 gate 输入乘以下一层 gate 权重，得到下一层候选。
- **技术机制**（PRD §历史 v1 §Implementation Decisions，快照 223–248 行）：Fate 模式下，命名隐藏张量 `ffn_moe_gate_input-i` 在层 `i` 路由可用时拷到 host，与层 `i+1` 的 gate 权重在 **CPU** 上相乘，再压缩为字节受限的候选集；原生 router 与模型图不变。计数器：`fate_predictions`、`fate_gate_inputs`、`fate_gate_ms`（门计算耗时单列，不与预取覆盖混算）。
- **实验条件与证据**：代码路径 `moe_cache_predict_fate`（引入于 `2f1a363c8`，2026-09-11；在 `d78c8bd42` 细化，2026-09-13）。**证据缺口**：在 `SOURCE_TREE` 全树检索 `fate=1` **零命中**，在 `LOCAL_MODELS/qwen38/traces/*.log|*.txt`（即 XT/静态阶段与 SMoE 阶段的全部日志文本）检索 `fate=1`/`fate_predictions=[1-9]`/`fate_gate_inputs=[1-9]` **同样零命中**：所有可检索到的 `policy=` 行都是 `fate=0 fate_predictions=0 fate_gate_inputs=0 fate_gate_ms=0.00`。因此“Fate 在本模型上从隐藏层里榨不出有用的预测信息”目前**只有用户回述，未找到与其对应的协议/原始运行日志**（[用户回述，待原始记录对应]）。本报告不对“为什么没有记录”做归因。
- **观察与结论边界**：可以确证的是：(a) Fate 在线路径**代码存在**且在本发布树提交中可编译；(b) 它在**当前工作点被显式关闭**（`fate=0`，且 §历史 v1 §Further Notes 只保留“相邻层 gate 输入可支撑低开销预取”这一可迁移结论，并把 shallow-favoring 行为标注为“模型与预算相关”）；(c) 现存文档没有把 Fate 的命中率写成结论。不能确证的是它的实测命中率与放弃时点的量化依据。
- **保留/放弃理由**：保留代码与开关（可替换预测器的设计目标）；默认关闭、不作为推荐路径。
- **遗留/重开条件**：重开前必须先补一份 `fate=1` 的原始日志（同一 400-token 工作点、含 `fate_gate_inputs/fate_predictions/fate_gate_ms`），否则任何关于 Fate 精度的表述都只能是回述。

## 4. 阶段三：SMoE —— 用共享专家构造近似 hidden

### PC-06 SMoE teacher 测试（用户回述 99%）

- **状态/版本**：**用户回述的实测事实**；方法未知，未找到与其对应的协议/原始日志。
- **为何尝试**：Fate 受挫后，改用“本层输入 + 共享专家输出”构造下一层 gate 的近似输入（不需要等待 routed MoE 的输出），在 teacher-forced 条件下先看命中率上限。
- **技术机制**：`ffn_smoe_hidden-i = ffn_input-i + cached_gpu_routed-i + shared_expert-i`，再乘下一层原生 gate 矩阵，取候选（PRD §历史 v1 §Implementation Decisions）。
- **实验条件与证据**：**用户回述：teacher 测试命中率 99%**。在 `sources/handoff.md`、`sources/smoe-nk-degradation-plan.md`、`sources/moe-cache-score-aware-prd.md` 三份快照与 `SOURCE_TREE` 日志中**均未找到**与该 99% 对应的协议/原始日志**（采样层数、候选数、分母是 top-10 槽位还是专家集合、是否含静态热并集均未知）。本节按“用户报告事实”保留，并标注[协议未确认]。
- **观察与结论边界**：99% 与 `sources/smoe-nk-degradation-plan.md` §7 的离线 `recall@10 = 68.53%`（k=1，teacher-forced 单步）**是否同协议未确认**（分母、候选集、是否含静态热并集都未知），因此**不得**用后者否定前者，也**不得**把 99% 与运行态 `hits/misses` 之类的在线命中率混同（见 PC-11）。
- **保留/放弃理由**：保留为主线事实（SMoE 路线由此启动）。
- **遗留/重开条件**：找到与该 99% 对应的原始协议即可升格为可引用测量；否则只能作为“动机”引用，不能作为精度基线。

### PC-07 SMoE N+k 离线退化（recall@10）

- **状态/版本**：计划文档已归档；结论有效域受该文档 §保留意见限制（文档归档不代表实现已发布）。
- **为何尝试**：在线实现前需要回答两个问题——SMoE 单独的 N+1 准确率是多少；退化的速度允许把预取窗口开几层。
- **技术机制**：`simulate_smoe_nk.py`：对层 L 取 `h = ffn_input(L)`（与 L+1 同图同 token），`block = (ffn_moe_weighted(L) * resident_mask).sum(expert) + ffn_shexp_gated(L)`，`smoe_hidden = h + block`，再用 **L+k 的真实 gate** 算 logits、取 top-10，与 `ffn_moe_topk(L+k)` 比对 recall@10。
- **实验条件与证据**：数据 `LOCAL_MODELS/qwen38/traces/standard-smoe-8-20260911`（8 个 prompt，白名单加一行 `"ffn_moe_input-"`，`qwen4exp.cpp` 不需改）。表（`sources/smoe-nk-degradation-plan.md` §7，快照 181–192 行）：

| 变体 | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| oracle（用 L+k 真实输入） | 100.00% | 100.00% | 100.00% | 100.00% |
| full（h + 全部路由 + 共享） | **68.53%** | 59.94% | 55.70% | 53.77% |
| fifo（带 22 slots 掩码） | 68.29% | 59.72% | 55.47% | 53.65% |
| shared_only（h + 共享） | 68.07% | 59.52% | 55.29% | 53.47% |
| input_only（只有 h） | 67.90% | 59.34% | 55.22% | 53.44% |

  样本量 56400/55200/54000/52800。oracle 在 k=1..4 全为 100%，说明在该数据、该步上“用真实输入 + 真实 gate + topk/取数”的匹配是自洽的；**这只覆盖该匹配环节，不证明整条在线管线正确，也不证明累积 rollout 下的结论**。
- **观察与结论边界**：N+1 = 68.53%，退化平缓（k=2 仍 59.94%）；在该数据上**只用 `ffn_input` 得 67.90%，与 full 差 0.63pt**。边界（与 PC-08 一致，不与之对立）：**这是 teacher-forced 单步测量**（每层 `ffn_input` 取自真实轨迹），因此它描述的是“在真实输入下逐层预测器的单步质量”，**不是累积 rollout 结论**；“预测只作提示、误差不跨层累积”是计划文档给出的论证（[该论证为条件性推导，未经累积 rollout 验证]），本章不把它当作已证事实。0.63pt 的差距只在本数据集上成立，**不足以据此断言可以在实现里直接删掉 routed/shared 两项**。
- **保留/放弃理由**：离线结果支持继续研究提前两层，而不是证明在线可用。线上没有据此删除 routed/shared 依赖，仍为三项相加（PC-10）。
- **遗留/重开条件**：若要替换为在线口径，需在同一数据上补 `PREDICT_STATIC=0` 的运行态覆盖，并与离线 recall 对齐分母。

### PC-08 早期 FIFO 池模拟器与其自述局限

- **状态/版本**：计划文档已归档（文档归档不代表实现已发布）/已被 PC-07 取代。
- **为何尝试**：先回答“给定层内 FIFO 池容量，命中与重建误差如何”。
- **技术机制**：`simulate_smoe_from_standard_trace.py` 复现每层独立因果 FIFO 池（slots ∈ {16,22,32,44,64}），把未驻留专家贡献置零，重建该层 `l_last` 并与真实比对 cosine / relative L2，统计 `fifo_pool_hit8`、`fifo_ranked_recall8/16`。
- **实验条件与证据**：`sources/smoe-nk-degradation-plan.md` §2（快照 65–83 行）自述模式为 `teacher_forced_local_smoe`，“Each layer uses the standard run's true pre-FFN residual; this is not a cumulative counterfactual rollout”，并明确“causal SMoE rollout 需要把近似残差喂进下一层重跑图”。数据目录 `standard-smoe-sim-20-20260910b`、聚合 `standard-smoe-20-20260910b-summary/report.md`。
- **观察与结论边界**：该模拟器能测**层内**放置质量，**不能**证明累积 rollout 下后续层路由保持；其结论只在“层内损失”意义上成立。
- **保留/放弃理由**：被 PC-07 的 N+k 曲线取代（后者正面回答提前量）。保留其局限声明作为方法学引用。
- **遗留/重开条件**：若需要累积口径，需真正把近似残差喂回图（即实现并跑累积 rollout），当前未做。

### PC-09 `ffn_input` 缺口与两条补法

- **状态/版本**：**仅 WIP**：采集侧改动只存在于旁支检出 `OTHER_SOURCE_TREES/qwen4exp-smoe-trace`，不是发布树的一部分；其计划文档进了快照，不等于实现已发布。
- **为何尝试**：SMoE 近似 gate 输入需要 router 的输入（`build_hc_mix(hc_after_attn, hc_ffn_*)` 的输出），而原始白名单只 dump 了 logits/topk/weighted/shexp 等。
- **技术机制**：两条路。(a) 补 dump：白名单加 `"ffn_moe_input-"`（`ggml/src/ggml-backend.cpp:1607` 快照行号）即可，`qwen4exp.cpp` 完全不用改，因为 `build_moe_ffn` 早就把 router 输入命名为 `ffn_moe_input`（`llama-graph.cpp:1973`）；(b) 从 `hc_after_attn` 离线按 `build_hc_mix` 公式重建。
- **实验条件与证据**：`sources/smoe-nk-degradation-plan.md` §3（快照 84–99 行）记录了 (b) 的结果：**试过、未验证通过，量级差约 20 倍**（该检出 hc 实现与当前工作区可能有差异）。最终采用 (a)。
- **观察与结论边界**：补 dump 的代价是重跑 20 请求 × ~1 分钟；重建法失败的原因未定案（[推断，未证实]：可能与检出的 hc 实现差异有关）。
- **保留/放弃理由**：保留 (a)；(b) 放弃。
- **遗留/重开条件**：无。

### PC-10 SMoE 在线实现与其固定开销

- **状态/版本**：已发布。`LLAMA_MOE_PREDICT_SMOE=1`，要求 split 模式，仅限单 token decode 图。
- **为何尝试**：把 PC-07 的离线结论搬到在线：只做一次 `W_gate(L+1) @ smoe_hidden(L)` 的极小 matmul，候选直接喂预取队列。
- **技术机制**：在线 SMoE 的近似 hidden **仍是三项相加** `ffn_smoe_hidden-i = ffn_input-i + cached_gpu_routed-i + shared_expert-i`（PRD §历史 v1 §Implementation Decisions）。**PC-07 里“只用 `ffn_input` 也够”的结论没有被实现**：代码没有改成 input-only 路径。side graph 在设备侧算 gate matmul + topk；运行侧 `moe_cache_smoe_enqueue/drain` 按层做 `event_synchronize` 回读。计数器 `smoe_predictions`、`smoe_logits`。
- **实验条件与证据**：`SOURCE_TREE/L2048-err.txt`、`L6144-err.txt`（每 tensor 21/64 槽）与 `T2048_*`、`T6144_*` 组，日志行 `[MOE-CACHE] smoe per DECODE graph: total=24.64/26.09/27.67/26.80 ms (event_wait=0.85/0.94/1.16/0.89 ms process=1.25/1.44/1.42/1.44 ms) nonblock=1 deferred=1603–1608 ahead=2`。400-token 工作点：`smoe_predictions=18354`、`smoe_logits=9397248`（`lifetime-before.json`/`final-r1.json`）。历史一次更差的形态（`a400a/b-err.txt`）为 `total=284.98/306.46 ms (event_wait=8.70/11.44 ms process=20.64/21.62 ms) deferred≈18343`。
- **观察与结论边界**：SMoE 的**固定开销大头是回读/同步而不是 matmul**：`sources/smoe-nk-degradation-plan.md` §214–231（快照）记录“逐层 `event_synchronize` 实测 17 ms/token，而 CPU 处理部分只有 2 ms”。这正是 PC-27（非阻塞读回晚一拍）与 PC-25（AHEAD 默认 3）存在的原因。
- **保留/放弃理由**：保留（`PREDICT_SMOE=1` 是当前推荐配置的一部分）。
- **遗留/重开条件**：`event_wait` 随 ahead/分布变化，未做端到端分桶优化；重开条件见 PC-27。

### PC-11 静态热集与 SMoE 准确率的口径混淆（98–99% / 73%）

- **状态/版本**：口径已澄清；具体数值待与原始记录对应。
- **为何尝试**：需要判断“SMoE 预测到底准不准”，而运行态只有一个合并计数器。
- **技术机制/口径**：`prefetch_predicted/prefetch_required` 中，`prefetch_required` 是参与计分的路由槽数、`prefetch_predicted` 是路由 id 落在预测位图上的数量；而**位图本身是 SMoE 候选 ∪ 静态热集（`PREDICT_STATIC`，默认 32）的并集**。因此该比值既不是“SMoE 准确率”，也不是准入数（`prefetch_experts` 才是公共写入路径准入总数，且包含非预测填充）。
- **实验条件与证据**：`sources/smoe-nk-degradation-plan.md` §6（快照 163–172 行）原文指出：早期把这个比值（73%，更早到 98–99%）当成了 SMoE 准确率，“**它不是**”，并给出隔离方法（离线模拟，或运行态 `PREDICT_STATIC=0`）。400-token 工作点实测：`prefetch_predicted=24971 = prefetch_ready=24971`，`prefetch_required=183540` → **13.6% 覆盖**（`optimization400/lifetime-before.json`）；128 压力/auto 系列的计数器语义清单见 `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json` 的 `counter_semantics` 字段。
- **观察与结论边界**：13.6% 与 73% 是**不同配置下同一口径**的两次取值（候选集合、`TAKE_MAX`、预测器组合、工作点都不同），**不能互相换算**；更不能拿它去否定 PC-06 的 99%（那是 teacher-forced 口径）。
- **保留/放弃理由**：保留为口径规范；所有预测器质量结论必须写清“分母是什么”。
- **遗留/重开条件**：重开需在固定配置下同时打印“SMoE-only 位图覆盖”和“∪静态热后覆盖”两个计数器（当前只打印合并值）。

## 5. 阶段四：在线缓存的工程化 —— 量与截止线

### PC-12 预取量增加、等待上升与争用假说

- **状态/版本**：已发布（开关）；结论“量是成本不是收益”。
- **为何尝试**：命中率随预取量单调上升，直觉上“多传多命中就该更快”。
- **技术机制**：同一二进制、`tools-run.py` 配置 + `NONBLOCK=1 AHEAD=2`，改预取量（MB/token）。
- **实验条件与证据**（`sources/handoff.md` §6.6①，快照 209–232 行；**仅历史报告，未记原始日志名**）：

| 预取量 | 命中率 | total | t/s | 主机侧 `flag_input` |
|---|---|---|---|---|
| 161 MB/tok | 40.2% | 70.0 ms | **14.3** | 15.8 ms |
| 254 MB/tok | 46.9% | 77.7 ms | 12.9 | 23.3 ms |
| 464 MB/tok | 58.2% | 94.1 ms | 10.6 | 37.5 ms |

  边际 ≈ **0.073 ms/MB**；把该边际时间取倒数得 **≈ 13.8 GB/s**——**这是“多传 1 MB 就多花 0.073 ms”的换算，不是硬件 PCIe 带宽实测**（快照内没有带宽 profiling/计数器证据）。快照给出的**解释**是：增长项不是拷贝调用本身（80 字节），而是 `moe_cache_prefetch_layer` 上游的 `ggml_backend_event_synchronize`，即预取 DMA 与 GPU 自身访存争用 PCIe → GPU 变慢 → 主机 `event wait` 变长（[推断：争用机制未被 profiling 直接证实]）。
- **观察与结论边界**：命中率与吞吐在超量区间**反向**；“命中率要靠更准，不是更多”。数字来自早期（非 SPLIT/非 auto 预算）配置，绝对值不可直接搬到当前工作点；快照未给出 `event wait` 的分桶或带宽计数，故“PCIe 饱和/争用”仍是解释性结论。
- **保留/放弃理由**：保留为机理证据（也是双门控的动机）。
- **遗留/重开条件**：当前工作点未重做预取量扫描。

### PC-13 排名截止线 `TAKE_MAX`（隐式 → 显式）

- **状态/版本**：已发布，默认 2。
- **为何尝试**：预测排名越深越不准，但每名耗字节相同，必须显式设“取到第几名”。
- **技术机制**：`moe_cache_smoe_process` 里 `take = min(n_slots, n_topk, take_max)`；旧实现 `take_max = n_used + 2`（=12）隐式充当“只取前 N 名”，而 `n_used` 只在分区钩子里赋值，**首 token / 未过钩子的层 `n_used==0` → 截止线静默消失（bug，已修）**；现在 `rank_cut = LLAMA_MOE_TAKE_MAX`（默认 2），与 `n_used` 无关。
- **实验条件与证据**：§6.6③（快照 233–245 行）另给出一条顺序结论：“先截断再跳过已驻留”**实测优于**“跳过已驻留后继续往下填预算”（后者把准入集合从秩 1–4 挪到秩 5–10，同字节命中更低）。按名次衰减：cutoff 1 → 4.69 hits/MB，cutoff 2 → 5.25（另测 3.05），cutoff 3 → 2.21，cutoff 4 → 1.78（§6.6②）。
- **观察与结论边界**：`cutoff 2 → 5.25/3.05` 两次测量差近 1.7 倍，说明该指标噪声大，只有 1/2 与 3/4 的单调关系是稳的。
- **保留/放弃理由**：保留；默认 2 由后续 400-token 工作点继续使用（`fixed400*/experiment.json` 的 `policy_env`）。
- **遗留/重开条件**：无。

### PC-14 去重（resident / pending / list / admit / readmit）

- **状态/版本**：已发布。
- **为何尝试**：热点命中数极高，需证明“反复命中”不会变成“反复搬运”。
- **技术机制**：`expert_slot[e] >= 0` 是**去重权威**：槽位在准入时、拷贝下达**之前**就已赋值，故同一 token/同一窗口内的重复请求必被拒；计数器 `dup_resident`（候选已在驻留集）、`dup_list`（候选表内重复）、`dup_admit`（同槽自我拷贝）、`dup_pending`（在飞重复请求）、`readmit`（淘汰后重装）。
- **实验条件与证据**：§6.6④（快照 246–258 行）硬数字：候选 2852 = `dup_resident` 1496（52.5% 的候选不花字节）+ 传输 1356；`dup_list=0`、`dup_admit=0`、`dup_pending=0`；`readmit=86`（淘汰抖动 6.3%）。400-token 工作点：`dup_resident=29682/28941/30221`、`dup_pending=0`、`dup_admit=0`、`readmit=4734/4475/164`（分别见 `lifetime-before.json`、`final-r1.json`、`lifetime-after.json`）。
- **观察与结论边界**：去重是**请求级**保证，不等于“传输量已最优”——`readmit` 说明仍有相当比例字节来自淘汰后重装（见 §10.1）。
- **保留/放弃理由**：保留（与 PC-31 的“命中↑但 DMA 不降”直接相关，是该现象的排除依据之一）。
- **遗留/重开条件**：仍开放项见 PC-31（命中/去重/重装/实际传输的对应关系尚未闭案）。

### PC-15 容量 × 截止线交互

- **状态/版本**：已发布（参数结论）；关键点未重复确认。
- **为何尝试**：最优截止线是否随缓存容量移动。
- **技术机制**：同一配置扫 2048 与 6144 MiB、cutoff 1/2/4/6/8。
- **实验条件与证据**：§6.6⑤（快照 246–258 行）：30 token 下 2048 与 6144 无差别（**大缓存没被填满**：6 GB ≈ 67 槽/层，每 token 每层只准入 ~2 个，30 token 填不满；`readmit=0` 证明从未淘汰）。150 token（会填满）：2048 最优在 cutoff 1（14.6 t/s）；6144 平台在 2–4（14.8–14.9）、6 → 14.5、8 → 14.1 → **最优截止线随容量上移**（组内噪声 ±0.6 t/s，关键点尚未重复确认）。显存峰值：2048 → **9132 MiB**；6144 → **13278 MiB**（limit 15360、guard 1024）。
- **观察与结论边界**：这是早期工作点（14–15 t/s 时代）；当前 400-token / 6144 MiB / 19 t/s 量级下未重做容量扫描。
- **保留/放弃理由**：保留为“容量决定可接受的深度”的证据。
- **遗留/重开条件**：重开需在同一工作点做 ≥3 次重复的容量×cutoff 网格。

### PC-16 预取 worker 数无益 + `INSERT_WORKERS` 空转更正

- **状态/版本**：已发布（更正）。
- **为何尝试**：预取慢是否因为缺乏并行提交。
- **技术机制**：改 `LLAMA_MOE_INSERT_WORKERS`（1/3/9）。
- **实验条件与证据**：§6.6①：低量 1→3 worker = 13.3→13.6 t/s；高量 = 9.9→10.5（+6%）；pinned 环上限仅几 ms。§6.7 未解决项进一步更正：**`INSERT_WORKERS` 在 `PREFETCH=1` 下是空转**（worker 的 spawn 点都要求 `!prefetch`）——预取拷贝是主机线程**内联提交**的，故此前“worker 不是瓶颈”的结论**作废**。
- **观察与结论边界**：当前提交内联在主机线程，worker数不是该配置的有效变量。采样显示等待；具体PCIe／GPU争用机制没有隔离（PC-12）。
- **保留/放弃理由**：保留为更正记录，避免后人再扫 worker 数。
- **遗留/重开条件**：无（除非改回 worker 提交模型）。

## 6. 阶段五：淘汰分数、热区与自适应

### PC-17 淘汰分数：gate-softmax MRS → 真值路由频次

- **状态/版本**：已发布。`LLAMA_MOE_EVICT_SCORE`（0 = 真值频次，1 = 旧 mrs）、`LLAMA_MOE_HOT_HALFLIFE`（默认 512 token 基期折半）。
- **为何尝试**：命中的上限由“缓存里装的是不是真热专家”决定，而淘汰分数是唯一决定装谁的输入。
- **技术机制**：旧分数 `mrs_score` = gate 全 softmax 前 20 名 EMA；新分数 = 分区钩子里从 `part.ids` 零成本累计的 `use_count`，带滑动窗基期折半防“热集冻结”。
- **实验条件与证据**：§6.7（快照 259–343 行）。用真值路由量旧分数：`mrs_topC = 0.3%`（同容量下按 mrs 排序取前 C 名只覆盖真实路由的 0.3%）。退出诊断三行示例（同节）：
  `[MOE-CACHE] hot-set oracle: routed=… layers=48 slots/layer=C | oracle_topC=66.9% resident_set=53.0% mrs_topC=0.3% actual_hit=48.9%`。
  400-token（`--ignore-eos`，统计只取后 150 图=稳态）：

| 配置 | 稳态命中率 | 稳态 ms/token | 稳态 t/s | 报告 t/s |
|---|---|---|---|---|
| 2GB + mrs 分数（旧） | 30.5% | ~119 | 8.4 | 13.5 |
| 2GB + 频次分数 | 53.1% | 51.5 | 19.4 | 17.7 |
| 6GB + 频次分数 | — | — | — | 17.8 |
| **6GB + 频次分数 + 回填 8** | **80.8%** | **43.0** | **23.3** | 19.9 |

- **观察与结论边界**：`oracle_topC` 是**内容相关**的（策略改变 → GPU/CPU 分工改变 → 数值微差 → 贪心长跑发散 → **不同 run 的 oracle 不可直接互比**，只能各自对照自己的 oracle）。原始日志未在快照中命名（[仅历史报告]）；同机理的运行态 per-rank/oracle 行可在 `SOURCE_TREE/a400a-err.txt`、`a400b-err.txt` 看到（`oracle_topC=83.4%/77.9%`、`mrs_topC=0.3%/0.2%`、`actual_hit=67.5%/61.8%`），但那是**另一次运行**，数字与上表不同。
- **保留/放弃理由**：保留（本路线最大收益）。
- **遗留/重开条件**：`EVICT_SCORE` A/B 的时序敏感性；重开需在同一 oracle 内部比较。

### PC-18 热区回填 + 冷启动种子 + 空闲填充

- **状态/版本**：已发布，默认全关（`HOT_BACKFILL=0`、`HOT_FILL_BOOT=0`、`HOT_IDLE=0`、`PIN_STATIC=0`）。
- **为何尝试**：预取受截止期与预算限制，填不满大缓存；而“比当前最冷驻留更热”的专家本可零风险补进来。
- **技术机制**：`moe_cache_hot_backfill` 在**图末尾**（所有 split 之后、capture 之外）用候选表补入更热专家，**自终止**（收敛后零开销）；拷贝走 side stream，完成由 `slot_events` **轮询**回收；两段式预算 `HOT_FILL_BOOT`（启动期）→ `HOT_BACKFILL`（稳态）；`HOT_IDLE=1` 时后台线程在模型空闲（150 ms 无计算活动判定）继续填，所有路径检查 `graph_active`，互斥只包记账段，传输永不被等待。种子：`manifest.hot` → `PIN_STATIC=N`（此前是死代码），finalize 时每层种入 N 个且**不给淘汰保护**。
- **实验条件与证据**：§6.7。正面：上表 6GB+回填 8 → 稳态命中 80.8%、23.3 t/s。反面：**2GB 下回填有害**（thrash，`readmit` 3383→10743，稳态命中 56% 且更慢）→ 回填必须配足够容量。400-token 运行态：`hot_fill=3190`（基线）、`hot_idle=4`；`admit_reject_freq` 出现于后续候选（PC-33）。
- **观察与结论边界**：回填**不受**截止期门与准入预算限制（数值预算只约束 `prediction` 路径），这是设计选择而非疏漏。
- **保留/放弃理由**：保留（推荐配置含 `HOT_BACKFILL=8`）。
- **遗留/重开条件**：`HOT_IDLE` 未做端到端验证（需“生成→idle→再生成”交互式会话）；重开条件即该会话可跑。

### PC-19 每排名准确率 / 每字节效率

- **状态/版本**：已发布（诊断输出）。
- **为何尝试**：给截止线提供数据（哪一名还值得）。
- **技术机制**：按完整候选表的排名统计“命中该名的路由比例”，并同时记录该名的实测 hits/MiB（`y`）与估计值（`ye`，无字节时用“准确率 × 单次准入命中数”估计），过单调包络。
- **实验条件与证据**：§6.7 给出（400 token，固定 cut=2，样本 = 48 层×400）：
  `r1 77.9% y17.93  r2 66.5% y16.65  r3 58.5%  r4 52.1%  r5 46.7%  r6 42.4%  r7 38.0%  r8 35.9%  r9 31.5% r10 28.2% r11 26.4% r12 23.3% r13 21.5% r14 19.7% r15 18.2% r16 16.6%`（y = 实测 hits/MiB；未准入的排名无字节 → 不可测）。
  运行态同格式原文可在 `SOURCE_TREE/a400a-err.txt`（`r1=83.5%/y27.55 …`）、`a400b-err.txt`（`r1=79.4%/y20.72 …`）看到，含 `ye` 与样本数 18354。
- **观察与结论边界**：**逐排名 yield 不单调**（代码注释记录 r3 实测 7.9 而 r4 实测 17.0；`a400b` 中 r3=10.41、r4=11.76 同向），所以任何“按 yield 逐名放行”的实现都必须用单调包络，否则噪声会被当信号。
- **保留/放弃理由**：保留为诊断基线。
- **遗留/重开条件**：无。

### PC-20 自适应截断三尝试（均败于固定 cut）

- **状态/版本**：已发布，默认全关（`RANK_ADAPTIVE`、`YIELD_MIN`、`ADMIT_BUDGET_MIB`）。
- **为何尝试**：把 PC-19 的表变成自动截止线。
- **技术机制与失败模式**（§6.7）：
  1. 准确率门槛（`RANK_ADAPTIVE`）：鸡生蛋——窄 cutoff 只 offer 少数排名，统计饿死 → 修法是把准确率按**完整候选表**测，与准入解耦；
  2. 实测 yield 门槛（`YIELD_MIN=4`，用户提议“每字节效率 > 4 放行”）：r1/r2 实测 17.9/16.7 ≫ 4，但更深排名无字节 → yield=0 判 0 → cut 卡在 1。**意外收获**：`cut=1` + 省下的传输 = **20.6 t/s** > `cut=2` 的 19.3；
  3. “准确率 × 实测单次准入命中数”估 yield + 字节预算（`ADMIT_BUDGET_MIB`）：64 MiB → **12 t/s**（早层吃光额度、后层饿死；64/48 = 1.33 MB < 1 个 bundle 1.9 MB），128 MiB → 19.0 t/s；**都不如 1/2**。
- **观察与结论边界**：三条路径的失败点各不相同（统计饥饿 / 无字节 / 分片过小），因此“自适应截断”不能只靠换目标函数，必须同时解决“估计的可用性”和“预算的分片下限”。
- **保留/放弃理由**：三法均暂停；默认回到固定 `TAKE_MAX`。
- **遗留/重开条件**：重开需先保证每层预算下限 ≥1 个 bundle，并让准确率统计与准入解耦。

### PC-21 占用率观测：价值门槛与速率控制分离

- **状态/版本**：已发布（结论）。
- **为何尝试**：解释“预算为什么没有直接换成加速”。
- **技术机制/证据**：§6.7 的用户实测（2026-09-12）：bf8/bf16 把 decode 占用率从 40–45% 抬到 60%；此前说的“100%”其实是预填充满负载——**decode 一直没满过**，所以“GPU 40 ms 硬底 / 25 t/s 天花板”的旧估计**作废**，真实天花板更高。而 `ADMIT_BUDGET_MIB` 会把占用率**钉在中平台**（它限速的是预取准入 → GPU 侧工作集增长被限速 → 既不下探冷谷也不上到热峰）。
- **观察与结论边界**：结论是设计分工：“价值门槛（哪些排名值得）与速率控制（每 token 多少）要分开——速率交给回填，门槛只管价值”。
- **保留/放弃理由**：保留；旧天花板估计已撤回。
- **遗留/重开条件**：未在 400-token 工作点重测占用率曲线。

### PC-22 YIELD_AUTO 极值搜索门槛

- **状态/版本**：已发布，默认关。
- **为何尝试**：常量门槛不可能对——命中的边际价值**状态相关**：CPU 半边有活时一次命中 ≈ 0.074 ms，命中率到 ~80% 后 CPU 半边空了 → 边际价值 ≈ 0；而传输的边际成本始终存在（≈ 0.08 ms/MB）。最优深度在“边际价值穿过边际成本”处，且随分布/提示词/缓存状态/阶段移动。（三个数字——0.074 ms/命中、0.08 ms/MB、以及“80% 后归零”——是 §6.8 在当时的 400-token 工作点上给出的**局部拟合/观测值**，不是模型常量；TREND_AUTO 的 V/P 就是它们的在线估计。）
- **技术机制**：决策变量 = 每字节效率门槛（hits/MiB）；目标 = **中位每 token 耗时**（抗单点噪声）；每 `YIELD_AUTO_PERIOD`（默认 16）图比较前后两窗口**中位数**：变快沿同方向、变慢反向，步长乘性 ±10%，钳 [0.5,32]，每次探针打日志（`[MOE-CACHE] yield-auto: probe …`）。硬约束仍是时限门（PC-24），且只作用于预取。
- **实验条件与证据**：§6.8（快照 344–372 行）：`prose: 67.8 → 38.7 ms（门槛 4.4 → 10.9），收敛 cut=1，yield_min=11.82，18.8 t/s`；`code: 收敛在 ~4.3–4.8 ms 门槛`。同一算法在两个分布上自算出不同门槛（11.8 vs 4.5）——**人工常数 4 在 prose 上偏松 3 倍**。
- **观察与结论边界**：收益全部来自“省下不该传的字节”；控制器本身有来回探测代价（见 PC-26 的同类代价）。
- **保留/放弃理由**：保留为可选项；与 PC-23 二选一，默认都关。
- **遗留/重开条件**：当前工作点未开 YIELD_AUTO 做验收。

### PC-23 TREND_AUTO 模型驱动门槛与预算

- **状态/版本**：已发布，默认关；`BUDGET_FRAC` 默认 0（不自动设预算）。
- **为何尝试**：极值搜索是黑箱爬山，收敛慢且有探测代价；若能把“命中价值 V”和“传输代价 P”直接估出来，门槛与预算都能一步算出。
- **技术机制**：每个图采集 `(ms, 命中次数, 准入 MB)`，在 `TREND_WINDOW`（默认 32）滑窗上对 `ms ≈ a − V·hits + P·MB` 做**中心化最小二乘**（3×3，Cramer 解，~50 flops/图 → 对速度无可测影响），EWMA(0.75/0.25) 平滑（代码 `moe_cache_trend_fit` + `trend_V/trend_P` 更新）。守卫（数据不可信时保持旧模型，绝不因此变慢）：窗口 <16 点、hits 散布 <15% 或 MB 散布 <8%、行列式过小、解越界（V ∉ [0.005,2] ms 或 P ∉ [0.001,0.5] ms/MB）→ 拒绝该次拟合（计数 `rej=`）。
- **实验条件与证据**：§6.9（快照 373–393 行）：`trendV=0.0158 ms/hit  trendP=0.0936 ms/MB` → 门槛 `P/V = 5.92 hits/MB`（开 cut=4）→ 预算 `frac × ms_hat / P = 116.5 MB`（frac=0.25；与手推的 128 MiB 保险丝几乎相同）；`fits=191 rej=212`；速度 19.6 t/s（同配置历史 18.8–19.9，无回归）。**独立交叉验证**：P=0.0936 与 §6.6 由容量扫描手算的 0.08 ms/MB 相差 17%——两条独立路径给出同一代价。
- **观察与结论边界**：`fits/rej` 近乎 1:1 说明守卫频繁拒绝（数据散布不足时），因此该模型在短窗口/低波动时段会长期停留在旧值；这是设计（宁可不更新）。
- **保留/放弃理由**：保留为“自适应版”推荐组合（`TREND_AUTO=1 + BUDGET_FRAC=0.25`），但**未进入发布默认**。
- **遗留/重开条件**：与 PC-24 的 `ms_hat` 同源，预算随工作点漂移的行为未做长跑验证。

## 7. 双门控的自适应传输阈值（源码判据）

本节回答：**“双门控的自适应传输阈值”在源码里到底是哪两个判据、哪两个自适应量**。注意**双门控与本轮被否的“全来源频率门”不是同一个东西**：频率门是在准入时**另加的一条准入条件**（把候选的保留价值与**实际可淘汰 victim** 比较，见 PC-33），它既不是价值门的一种取值来源，也不改变两个门的判据本身。

### PC-24 门 1（价值门）与门 2（时限门）

- **状态/版本**：已发布。`LLAMA_MOE_PREFETCH_GATE` 默认 **1**（`prefetch_gate = true`，初始化日志打印 `gate=1`）；模型常数 `LLAMA_MOE_GATE_COPY_US` 默认 70 µs、`LLAMA_MOE_GATE_BW_GBPS` 默认 20 GB/s。
- **为何尝试**：命中率越高越慢（PC-12/PC-21）说明“多传”本身就是代价；需要两个互相独立的判据：**值不值得传**（价值）与**来不来得及**（时限）。只看其一都会退化成常量策略。
- **技术机制（判据 1：价值门，决定准入深度）**：`moe_cache_effective_rank_cut()` 逐排名检查
  `yield_est(r) ≥ yield_min`，其中 `yield_est` 来自 `moe_cache_rank_yield_est()`：有实测字节时用实测（rank 命中 / rank 字节，单位 hits/MiB）；没有字节时用 `准确率` × `单次准入命中数` ÷ `单次准入 MiB` 估计；逐排名取**单调包络**（`env = min(env, yield_est(r))`），一旦包络跌破门槛即停，返回 `cut ≥ 1`。**自适应量 = `yield_min`（hits/MiB 门槛）**，取值优先级：
  1. `TREND_AUTO` 且 V、P 有效 → `trend_P / trend_V`（盈亏平衡点），带 10% 迟滞（只有变化 >10% 才更新，防回归噪声抖动），钳 [0.5, 64]；
  2. `YIELD_AUTO` → 极值搜索探针 `yield_auto_cur`（乘性 ±10%，钳 [0.5, 32]）；
  3. 常量：`RANK_YIELD_MIN` / `RANK_ADAPTIVE` 的准确率阈值 / 固定 `TAKE_MAX`（默认 2）。
  **这个门回答**：“再深一名的每字节命中效率，是否还高于让 token 变快的盈亏平衡点”。
- **技术机制（判据 2：时限门，决定这批拷贝是否发出）**：`moe_prefetch_feasible(bytes, n_copies, layers_until_visit)`：
  ```
  eta_us    = (dma_inflight_copies + n_copies) * gate_copy_us
            + (dma_inflight_bytes  + bytes) / gate_bw_bps * 1e6
  deadline  = max(1, layers_until_visit) * layer_us_ewma
  可行  ⟺  eta_us ≤ deadline_us
  ```
  不可行 → **丢弃该预测传输**，让专家走 CPU 路径，避免把 side stream 队列撑爆。**自适应量 = `layer_us_ewma`（每层调度线程墙钟，0.75/0.25 EWMA，在线更新）与在飞量 `dma_inflight_{copies,bytes}`（提交/完成时加减）**；`gate_copy_us`/`gate_bw_bps` 是模型常数（运行时不自适应，只由 env 覆盖）。**这个门回答**：“在需要它之前能不能到达”。源码注释（函数上方）明确否决了“用实测完成率估容量”的方案：那会 **death-spiral**（测的是需求而不是带宽：传输变少 → 估计变低 → 丢得更多）。
- **两个门的适用范围不同（关键设计决定）**：时限门与字节预算都**只作用于预取（`prediction`）路径**；热回填（`hot_backfill`）、显式 warm、静态种子**不受**时限门约束（回填无时限、可在图末尾批量做）。第三个自适应量是**速率预算**（不是门）：`admit_budget_bytes = frac × ms_hat / P`（`TREND_AUTO`，钳 [16,512] MB）或手工 `ADMIT_BUDGET_MIB`。
- **时限门旁路**：调用处还要求 `!queue_on_worker`；没有有效层历史的冷启动直接放行（发布基线 `ggml-backend.cpp:4015–4024,4105–4110`）。不能因日志写 `gate=1` 就认为所有写入都经过了 deadline 检查。
- **实验条件与证据**：源码 `SOURCE_TREE/ggml/src/ggml-backend.cpp`：`moe_state` 字段 `prefetch_gate/layer_us_ewma/dma_inflight_bytes/dma_inflight_copies/gate_copy_us/gate_bw_bps`（约 1910–1925 行）；`moe_prefetch_feasible`（约 4414–4425 行）；准入门条件（约 4512 行 `if (prediction && s.prefetch_gate && !queue_on_worker)`）；在飞量加减（约 4646/4658、6595 行）；`layer_us_ewma` 更新（约 6149 行）；`moe_cache_rank_yield_est` / `moe_cache_effective_rank_cut`（约 5432–5480 行）；`moe_cache_yield_auto_step`（约 5537–5565 行）；TREND 拟合与预算（约 7700–7719 行）。行为证据（历史报告）：§6.8 的 `yield-auto: probe …` 收敛、§6.9 的 `trendV/trendP/budget` 与 `fits/rej` 计数、§6.7 的“预算限速把占用率钉在中平台”。
- **观察与结论边界**：价值门槛按源码优先级采用 TREND 的 **P/V**、YIELD 探针或固定阈值；时限门使用每层耗时 EWMA 与在飞量，拷贝常数不在线学习。两个判据作用不同，但会共同影响后续缓存状态与测量样本，不能把它们说成统计上独立。
- **保留/放弃理由**：保留，默认开启时限门（`gate=1`）；价值门默认退化为固定 `TAKE_MAX=2`。
- **遗留/重开条件**：`gate_copy_us=70 µs` / `gate_bw_bps=20 GB/s` 是手设常量，未在本机标定；若要在 PCIe 4.0 x8 之外复现，应重标定并用 §6.6 的 0.073 ms/MB 交叉验证。

### 为什么“命中更高”会“更慢”（观测、解释与反例）

1. **传输量与等待一起上升**（PC-12）：161→464 MB/token、命中40.2→58.2%、速度14.3→10.6；上游等待15.8→37.5 ms。PCIe／显存争用是当时解释，未由硬件 profiling 隔离；0.073 ms/MB 的倒数不是带宽实测。
2. **边际价值可能随状态变化**（PC-22）：约0.074 ms/hit、较高命中后收益减小、约0.08 ms/MB，都是局部估计，不是普遍阈值或恒定成本。价值门因此需要对运行状态敏感，而不只追命中率。
3. **命中计数 ≠ 传输计数**（PC-14、§10.1）：重复命中驻留专家**不产生**新 DMA（去重权威 `expert_slot[e] ≥ 0`，`dup_pending=0`）；DMA 只在准入/淘汰后重装时发生。
4. **反证**：把字节砍掉 53% 也不足以提速（PC-33：26.9→12.6 GB，耗时无改善），说明成本结构里还有等待与关键路径成分，不能只盯 DMA 字节。

## 8. 阶段六：split 时序与“晚一拍”

### PC-25 `SPLIT=1` 静默算错修复 + `AHEAD` 默认 3

- **状态/版本**：已发布（修复随本发布树提交；此前 `SPLIT=1` 曾被临时禁用为安全锁）。
- **为何尝试**：`SPLIT=1` 能把 MoE CPU 半边与 GPU 半边重叠，但要先证明它算得对。
- **技术机制（一行级根因）**：host 路径里 GPU 半边的 `ids_gpu`/`wgt_gpu` 是**本 split 的输入**（`ggml_set_input` 的 host leaf），而它们的值由分区钩子在**输入循环中途**写入——调度器可能**已经先排了这两个 leaf 的拷贝**，于是 MoE GEMM 拿到**上一层的路由**，静默算错（部分性、与图序相关）。修复：钩子写完这两个 leaf 后**立刻重发一次拷贝**（`tensor_copy` + `ggml_backend_tensor_copy`，几百字节）。
- **实验条件与证据**：§6.32①（快照 1124–1147 行）：Eiffel 用例修复前 **3/3 必现**（只出 27 字符），修复后 **4/4 通过**（答案落在 443–455 字符区间、开头与 `SPLIT=0` 参考一致、给出同一段正确答案——**并非与参考逐字相同**）。速度（8k/q8_0，auto=97 槽）：缓存关 9.2 t/s；`SPLIT=1 + auto + AHEAD=3` 命中 **90.9%**、**20.8 t/s**；`AHEAD=2` 83.5% / 18.4；`AHEAD=4` 87.8% / 20.2。快照给出的“相对关缓存 **+126%**”是**该页历史表内的算术**（单次、非严格 A/B、不是当前 6144 MiB / 400-token 工作点），**不能当作本工作点的验收结论**。`SMOE_AHEAD` 默认由 1 改为 3。
- **观察与结论边界**：**此前的 20.3 t/s 是带 bug 的假速度，已撤回**；§6.32 同时列出仍未做项：`AHEAD=1` 的 r1 异常（见 PC-27）、0 槽（如 `CACHE_MIB=64`）时 split 不产出（本该退化成全 CPU）、256k+视觉下 auto 只给 37–40 槽（显存约束而非 bug）、退出期偶发 `0xC0000005`。
- **保留/放弃理由**：保留（`SPLIT=1` 恢复为默认路径）。
- **遗留/重开条件**：0 槽退化与退出期崩溃仍是开放项。

### PC-26 `AHEAD_AUTO` 极值搜索

- **状态/版本**：已发布，默认关（固定 3 略优）。
- **为何尝试**：把 PC-22 的同族控制器套到前瞻距离上。
- **技术机制**：`AHEAD_AUTO=1`，`AHEAD_AUTO_PERIOD` 默认 32 图/窗口；**目标函数必须是窗口增量命中率**——用累计命中数/累计命中率会被缓存预热带偏。
- **实验条件与证据**：§6.32④：未修正目标时控制器会一路爬到钳位值 6 再反向走到 1；修正后实测在 **3↔4 之间来回探测**（访问分布 4×19、3×18、5×6、6×4、2×1）=找到了最优区，但**固定默认 3 仍略优（20.8 vs 19.1 t/s）**，来回探测有代价。
- **观察与结论边界**：这是“控制器正确但不划算”的典型：找到最优区 ≠ 比固定最优值更快。
- **保留/放弃理由**：保留为选开项；不作为默认。
- **遗留/重开条件**：若分布变化剧烈（长跑、混合负载），重评。

### PC-27 非阻塞读回“晚一拍”

- **状态/版本**：已发布，默认不改（`SMOE_NONBLOCK=1 + AHEAD=3`）。
- **为何尝试**：`AHEAD=1` 时 r1（第 1 名预测排名命中率）异常低，需判断是实现 bug 还是机制后果。
- **技术机制/定位**：§6.33（快照 1174–1189 行）。现象（修 SPLIT 后仍在）：r1 在 ahead=1/2/3/4 = **61.2 / 86.0 / 79.5 / 72.3%**——2→3→4 平滑衰减，唯独 1 脱轨。排除法：归属偏移正确（`target_layer = pending.layer + s.smoe_ahead`）；交付统计四个 ahead 几乎相同（`deferred=11`、`predict≈11–12k`）⇒ 不是丢预测。**决定性实验**：把 SMoE 读回改成同步（`SMOE_NONBLOCK=0`）后，**ahead=1 的 r1 从 61.2% → 93.1%**（全场最高），而 ahead=2 基本不变（86.0 → 84.7）⇒ 结论：**非阻塞读回让预测被晚一拍消费**；`ahead=1` 的窗口只有 ~1 层，恰好整拍过期（实际用到的是 L−1 的预测），`ahead≥2` 时晚一拍消费的仍是“目标为 L”的预测，故正常。
- **取舍（背靠背各 2 次，8k/q8_0/auto=97 槽）**：

| 配置 | gen (t/s) | 命中 | r1 |
|---|---|---|---|
| **`SMOE_NONBLOCK=1` + `AHEAD=3`（现默认）** | **19.9 / 20.3** | 88.5 / 89.7% | 79.5% |
| `SMOE_NONBLOCK=0` + `AHEAD=1` | 20.0 / 18.1 | 89.5 / 83.7% | **93.1%** |
| `SMOE_NONBLOCK=0` + `AHEAD=2` | 18.1 / 18.5 | 83.5 / 83.8% | 84.7% |

- **观察与结论边界**：速度上现默认最优（均值 20.1 vs 19.1 / 18.3，且同步路径方差更大）⇒ **不改默认**；若某场景更看重**预测准确度/长期放置质量**（超长生成里放置质量会复利），可用 `SMOE_NONBLOCK=0 + AHEAD=1`。本节 20.x 属**修复后** host 路径测值，不得与已撤回的旧 20.3/32 tps 混用。
- **保留/放弃理由**：保留；不改变默认。
- **遗留/重开条件**：长上下文下“r1 高是否复利成更高命中”未验证。

## 9. 阶段七：共享池、位置权重与内存安全边界

### PC-28 共享池 `GLOBAL_POOL`

- **状态/版本**：已发布（默认 0 = 每层独立缓存）。
- **为何尝试**：每层独立缓存会让冷层/热层各自持有固定槽位；共享物理槽位可让热层多占、冷层少占。
- **技术机制**：`LLAMA_MOE_GLOBAL_POOL=1` 时所有 `(layer, expert)` 共用同一物理槽位池；`moe_cache_state` 走 `global_pool` 分支（槽位分配/淘汰/direct-read 视图/图绑定多处特化）。devpart 与共享池互斥（`s.devpart = s.devpart && ... && !s.global_pool`）。
- **实验条件与证据**：`sources/moe-cache-score-aware-prd.md` §当前策略与 §此前短回放诊断。128 步 auto：每层 MRS = 9270.1 MiB / 97 槽/层 / 中位 61.6195 ms / 命中 76245 / 淘汰 9；共享池 MRS = 9281.2 MiB / 共 3839 槽 / 62.9095 ms / 76245 / 淘汰 **0**。两者命中完全相同。**重要限定**：发布树提交里虽有 `global_pool` 代码，但这一组 128 结果跑的是**较晚的 WIP 修过的 pool**（含零槽/stride/图绑定修复等），因此它们**不能为发布树里的旧 pool 路径背书**，也不能当作“已发布共享池”的性能证据。
- **观察与结论边界**：auto 容量下共享池**从未淘汰**，因此**无法**证明其淘汰策略有效；候选（LFU_POS）相对共享池 MRS 中位耗时 +0.03%（无差异）。
- **保留/放弃理由**：保留为显式实验开关；**默认每层缓存**（当前 PRD §此前短回放诊断末条明确“不自动启用共享池或位置权重”）。
- **遗留/重开条件**：需要能产生淘汰的压力工作点（512 MiB）且能解释 PC-31。

### PC-29 位置权重 `LAYER_AWARE` / `LFU_POS`

- **状态/版本**：**仅 WIP**（`SOURCE_TREE` 未提交；发布树提交 `7e01451b2` 中检索 `LAYER_AWARE`/`lfu_pos` 均为 0 命中）。
- **为何尝试**：共享池里“下一圈即将被访问的层”比“刚过去的层”更值得保留，而纯频率对所有层等价。
- **技术机制**（快照 §当前策略）：`future = l > c`；`d = future ? l - c : l - c + N`；`weight = (future ? 1.0 : 0.5) * N / (N + d)`；`retention_score = actual_use_frequency * weight`。序号来自**有序层表**（不假定模型层号连续）；本层取 `N`（最冷）；同一类别内距离越近越优先；**位置因子不修改原始频率**，故足够热的旧层专家仍可压过冷的未来层专家；权重**只在执行游标变化时重算**。要求 `GLOBAL_POOL=1`、`MRS=1`、`FIFO=0`、`EVICT_SCORE=0`，不兼容时明确告警并忽略；日志标识 `policy=LFU_POS`。
- **实验条件与证据**：512 MiB 压力（固定 replay、128 步、每侧 3 轮、剔前 16 步）：命中中位 31527 → 38076（**+20.77%**），淘汰 8610 → 8557，解码中位 77.6790 → 76.1885 ms（**−1.92%**，小于基线轮间波动）。来源 `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json`。
- **观察与结论边界**：命中涨了 20.77%，吞吐没有稳定变快（见 PC-31）；且压力候选三轮均有 1/128 步 top-1 与参考不同（step 79），因此**也不满足数值等价**。
- **保留/放弃理由**：暂停（未提交、未切默认）。当前 PRD 的结论是“本轮确认了功能及压力下命中变化，未获得切换默认所需的性能与数值等价证据”。
- **遗留/重开条件**：见 PC-31 的关闭条件。

### PC-30 共享池内存安全边界（零槽 / reading 保护 / stride / 图 UID）

- **状态/版本**：仅 WIP（未提交），但事故与修复证据链完整。
- **为何尝试**：共享池把“按行/专家偏移寻址”从每层固定布局改成全局 pitch，任何“行大小当常量”“块大小先除再乘 channel”的地方都会**静默错位**。
- **技术机制**（快照 §共享池的内存安全边界）：
  - **零槽**：direct-read 共享池保留一个全零 padding 槽；其容量**计入物理池但不计入可驻留专家数**（512 MiB 实测 211 物理 / 210 可用；本机 auto 3840/3839）。
  - **reading 保护**：GPU 正在读取的物理槽位、尚未完成写入的槽位、固定槽位不可淘汰；**读取保护在既有 backend 同步点释放**，不把历史预测列表当永久保护集。
  - **stride 修复**：CUDA MMVQ/MMQ 的权重 channel/sample stride 全程用 **int64 字节偏移**，先定位字节基址再解释量化块；行内 stride 仍按量化块；**不能先除块大小再乘 channel**，也不通过全模型 LCM 膨胀槽位掩盖；本模型共享池 pitch 为 **2,534,400 字节**，不能被 82 字节量化块整除。
  - **图 UID**：direct-view 与 gathered-copy 的 `src[0]` 绑定切换时更新 graph UID，保留 CUDA Graph 功能，避免旧图把专家 ID 当成共享池 slot ID。
- **实验条件与证据**：旧共享池在第 **29** 个解码步骤附近出现 CUDA 非法读取；原始 memcheck 捕获到槽位容量 211 时的**专家 ID 450** 寻址。修复后**七次 128 步压力运行均完成**；独立 CUDA 冒烟覆盖 IQ2_S/IQ3_S/IQ4_NL 的 MUL_MAT_ID、4D 广播、1/4/17 token、单 token fused SwiGLU，**24 个用例全部与紧凑布局逐项一致，memcheck 0 errors**（`LOCAL_EVIDENCE/moe-cache/stride-memcheck.log`；汇总 `final-verification.json`，含 `cuda_stride_smoke: {cases:24, failures:0, memcheck_errors:0}`）。
- **观察与结论边界**：该检查**不是**完整模型 memcheck，也未覆盖多 GPU 或 NVFP4 实机执行；“合法地址里的错误专家数据”仍可能不报错（见 PC-31 的正确性风险）。
- **保留/放弃理由**：保留（修复本身有效）；共享池默认仍关。
- **遗留/重开条件**：若要默认开共享池，需要把上述冒烟扩到完整模型（含时序/图绑定）。

### PC-31 128 压力：命中 +20.77% 但 DMA 几乎不变

- **状态/版本**：**仍开放**。
- **为何尝试**：验证“位置权重提高命中”能否转化为加速。
- **技术机制/口径**：固定外部 replay、128 步解码、剔前 16 步、计时只含 `llama_decode + llama_synchronize`（不含 logits 拷贝与写文件）、每侧 3 轮取中位数的中位数。
- **实验条件与证据**（快照 §此前短回放诊断 + §待排查问题；RTX A5000 Laptop 16GB / 5950X / 16 线程 / UD-IQ3_XXS / 84-token 代码提示）：

| 配置 | 物理缓存 MiB | 可驻留槽位 | 解码中位 ms | 命中中位 | 淘汰中位 | 显存峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 每层 MRS，auto | 9270.1 | 每层 97 | 61.6195 | 76245 | 9 | 15378 |
| 共享池 MRS，auto | 9281.2 | 共 3839 | 62.9095 | 76245 | 0 | 15306 |
| 共享池 LFU_POS，auto | 9281.2 | 共 3839 | 62.9310 | 76245 | 0 | 15306 |
| 共享池 MRS，512 MiB | 510.0 | 共 210 | 77.6790 | 31527 | 8610 | 6534 |
| 共享池 LFU_POS，512 MiB | 510.0 | 共 210 | 76.1885 | 38076 | 8557 | 6534 |

  核心背离：512 MiB 下命中中位 **+20.77%**，但**预取字节中位仅从 17399519232 降到 17294964224（约 −0.60%）**，淘汰仅 8610 → 8557；耗时中位 −1.92%，**小于基线重复波动**（MRS 三轮 77.6790/80.7960/76.3055；候选 76.8140/75.6135/76.1885）⇒ 无稳定加速证据。
- **正确性风险（不能先排除时序错配）**：压力候选三轮**均**在零起算 decode step **79** 出现 top-1 翻转（基线 token 271，候选 token 248046）；候选对压力参考的 mean KL 0.002296–0.004157，最大绝对 logit 差 3.193697；prefill 一致。**MRS 自身重复也存在数值差异**（`global-pressure128-v2.summary.json` 的 `baseline_repeat_max_maxdiff = 2.887`，候选 vs 参考 worst 3.194）——所以既不能据此判定候选“无损”，也不能把它当成“候选引入”。
- **口径警告**：该 summary 里的 `noise_tolerance_used = 8.66` 与 `candidate_within_baseline_repeat_noise = true` 属**旧的放大噪声容差字段，不作为验收依据**（当前脚本已移除该判定）。
- **观察与结论边界**：“反复命中驻留专家本身不产生重复传输”（去重语义，PC-14）；此前的“反复命中但仍持续搬运”解释**混淆了计算命中与预取去重**，不能作为根因。目前**既未证明去重失效，也未证明传输带宽已达上限**。
- **保留/放弃理由**：保留为开放问题（不切默认）；关闭条件已在快照 §后续排查与关闭条件写明：按 (layer, expert) 与槽位占用代次关联“预取请求 → 驻留/pending 去重 → 实际拷贝提交与完成 → 淘汰 → 重装 → 实际命中”，核实计数字段代表请求/提交/完成哪一种，并把真实传输拆成“首次装入”与“淘汰后重装”。
- **遗留/重开条件**：先定位第一次 logit 分歧（核对专家身份、槽位所有者、写入完成状态），用强制同步/关 CUDA Graph 做诊断对照，再决定是否需要 CPU/GPU 算子级比较。

## 10. 阶段八：`optimization400` 生命周期统计与两个未提交候选

### 10.1 先固定口径：为什么“热点命中”不能等同于“重复传输”

本节给出可直接引用的定量关系（全部来自 `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json`，构建 = `ggml-base.lifetime-baseline.dll`，sha256 前缀 `ee22deb6`，400 token / 6144 MiB / 全内存）：

- **命中计数是“读驻留槽位”的计数，且一专家读三次**：`gpu_uses = 133484`，`hits = 400452 = 3 × 133484`（gate/up/down 三块权重各记一次）。所以 hits 是**权重张量读**，不是传输次数。
- **传输只在准入时发生**：`admissions = 10216`，`prefetch_bytes = 20,224,040,448`（≈ 1.98 MB/准入，与 1.9 MB bundle 量级一致）。**命中/准入 ≈ 39.2**。
- **重复命中不产生新字节**：`dup_resident = 29682`（候选已在驻留集 → 直接丢弃，不花字节）、`dup_pending = 0`（无在飞重复请求）、`dup_admit = 0`（无同槽自我拷贝）。去重权威是 `expert_slot[e] ≥ 0`：槽位在准入时、拷贝下达**之前**就已赋值。
- **真正的“重复传输”是淘汰后重装**：`readmits = 4734`（占准入 46.3%），`readmit_bytes = 9,372,176,612`（占准入字节 46.3%）；其中 `readmit_within_1_graph = 696`、`readmit_within_4_graphs = 1638`。
- **另有大量“白传”**：`evicted_unused = 4530`（占淘汰 62.3%，占准入 44.3%），`evicted_unused_bytes = 8,972,577,948`（占淘汰字节 62.3%）；`evicted_used_once = 1527`、`evicted_used_many = 1215`。
- 一致性：`admissions − evictions = 10216 − 7272 = 2944 = live`（成立）；`duplicate_admissions = 0`、`evict_untracked = 0`、`use_not_resident = 0`、`use_complete = 1`（成立）。
- 字节口径警告（当前 PRD §400-token 驻留期观测）：按权重分量 stride 汇总，**既不是共享池物理 pitch，也不是 PCIe 链路实测字节**；admission 是逻辑槽位占用，pending 仍不能被计算读取。

结论：**命中率上升可以完全来自“同一批驻留专家被反复读”，代价为零；只有准入（含淘汰后重装）才搬运字节。**因此“命中↑”既不能推出“传输↑”，也不能推出“更快”——要判断收益/成本，必须同时看准入数、准入字节、重装比例与未用淘汰比例。

### PC-32 驻留期统计 `CACHE_LIFETIME` + 400-token 基线

- **状态/版本**：仅 WIP（`LLAMA_MOE_CACHE_LIFETIME` 在发布树 `7e01451b2` 中 0 命中；统计口径已写入当前 PRD §400-token 驻留期观测）。
- **为何尝试**：需要一个“不逐次打印、只出一行”的逐 (layer, expert) 驻留期账本，把“未用淘汰/重装/实际 GPU 使用”分开，而不是只看 hits/misses。
- **技术机制**：开启后按 `(layer, expert)` 追踪占用、淘汰、重新装入与真实 GPU 使用，退出时汇总一行 `[MOE-LIFETIME]`；未开启**不分配**逐专家追踪数组。判别式：CPU 实际路由频率 ≠ GPU 驻留期使用次数；零权重 padding 不算实际使用；无法完整观测的 devpath 会明确降级（`use_observations_devpart/gathered` 计数器）。
- **实验条件与证据**：`LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json`（400 token、`--ignore-eos`、6144 MiB、`SPLIT=1 DIRECT_READ=1 MRS=1 PREFETCH=1 SMOE_NONBLOCK=1 AHEAD=2 HOT_BACKFILL=8`、`GLOBAL_POOL=0`）：Generation **19.3 t/s**（`acceptance.status=pass`，门槛 19.0），显存峰值 12458 MiB，403 图。计数：准入/淘汰/结束占用 = 10216/7272/2944；预测/非预测准入 = 7026/3190；未用淘汰 4530（预测 3559 / 非预测 971）；重装 4734（≤1 图 696、≤4 图 1638）；实际 GPU 使用 133484；`dup_resident=29682`。同工作点未加观测复测 19.2 t/s，**两个单次结果不构成提速证据**。
- **观察与结论边界**：必须同时检查 `available / use_complete / 完整性错误计数`；本文件 `use_complete=1`、`use_observations_partition=19153`、`use_observations_devpart=0`。原始输出/错误/统计 CSV 指向 `SOURCE_TREE/opt400-lifetime-before-out.txt|err.txt`、`stats-opt400-lifetime-before.csv`。
- **保留/放弃理由**：保留（是回答 PC-31 所需的仪器）；代码本身未提交。
- **遗留/重开条件**：需要 `(layer, expert) → 槽位代次 → 实际拷贝提交/完成` 的关联（目前 CACHE_LIFETIME 只到准入/淘汰/使用，没有“提交 vs 完成”的区分）。

### PC-33 全来源频率门候选（**被否**）

- **状态/版本**：仅 WIP / **已否**（发布树无此代码；二进制文件名 `ggml-base.rejected-all-frequency.dll` 即维护者标注）。
- **为何尝试**：让**所有来源**（含 hot/idle/seed/phase 填充）的准入都过“频率门”，只允许比当前可淘汰 victim 更热的候选进入，从而砍掉白传字节。
- **技术机制**：源码新增 `prefetch_reject_frequency` 计数器与“与选定 victim 比较保留价值”的准入判据（仅在 `mrs && !fifo` 时生效）；命中日志新增 `admit_reject_freq=`。
- **实验条件与证据**：
  - 400-token CLI（`lifetime-after.json`，同一受控环境）：Generation **19.9 t/s**（pass），准入 **4788**（预测 2336 / 非预测 2452）、淘汰 1845、重装 **164**、命中 402648、预取字节 **9,482,722,816（−53%）**、`admit_reject_freq=4151`、`hot_fill=2452`、`dup_resident=30221`。
  - 固定历史 400 步（`optimization400/fixed400/experiment.json`，`real-qsa-check.exe`，候选哈希 `ac46f065`，三次候选计数完全一致）：命中 **322554**（基线 340299，**−5.2%**）、未命中 254886（基线 237141）、淘汰 3425（基线 10585）、预取字节 **12,623,473,152**（基线 26,904,137,728，**−53.1%**）、`prefetch_ready` 24155（基线 29862）；稳态中位 53.850/55.884/54.123 ms vs 基线 54.053/53.853/54.637 ms。
  - 正确性（以 `decode_all` 为主述，`decode_steady` 只作补充）：候选 vs 参考 `decode_all` top1 一致 **398/400**、最大 logit 差 4.8496（worst row i=369, input_token 328）；`decode_steady` top1 382/384、逐位相同 34/384——**不作为验收口径**。基线自身重复也有 `decode_steady` 341/384 位同（见 PC-35）。
- **观察与结论边界**：**省字节有效**（−53%），**命中与耗时的结论依赖工作点**：CLI 自生成负载上 19.9 t/s，固定 400 步负载上命中 −5.2% 且耗时无改善。**不要把 19.9 与 19.3 / 18.4 当成一组受控 A/B**：`lifetime-before.json`（19.3）是带观测插桩的基线、`baseline-paired-r2.json`（18.4）是更晚的旧基线单次，三者不是同一轮交替配对运行。19 门槛“通过”也不等于候选被采纳。
- **保留/放弃理由**：**放弃**（未提交、最终暂停）。理由：作为“省字节”手段成立，作为“提速”手段在所有工作点都无稳定证据；且命中下降说明它把一部分有效准入也拒了。
- **遗留/重开条件**：若要重开，必须先证明“被拒的 4151 次准入里有多少是有效（之后真被使用）”，即把 `admit_reject_freq` 与后续实际使用关联。

### PC-34 仅热回填修复候选（轮转 / 空槽 / eligible victim，**未提交、暂停**）

- **状态/版本**：仅 WIP / 暂停。二进制 `ggml-base.hot-backfill-candidate.dll`（sha256 前缀 `e0d7a83f`）；**注意**：**运行当时** `SOURCE_TREE/build-ple-trace-mrs/bin/ggml-base.dll` 就是这个候选（我在复核时读到的哈希为 md5 `5c21782f…` / sha256 `e0d7a83f…`）；该目录由本地构建脚本写入、**随时可能被替换**，因此上述对应关系只对“运行当时”成立，不得声明它“当前”仍是该文件。发布树源码里没有这个候选。
- **为何尝试**：`moe_cache_hot_backfill` 有三处逻辑缺陷，会让回填在部分层“饿死”或在空缓存/等价值替换时行为错误。
- **技术机制**：WIP 增加 `hot_cursor` **轮转游标**（按层轮转，避免只填前几层），并新增“频率驱动的回填必须优于**实际可淘汰 victim**”判据（`eligible victim`，而非与受保护的冷驻留项比较）。另有 `position_cursor`（位置权重只在执行游标变化时重算）属于**独立的 LFU_POS/共享池工作**（PC-29），不是本 hotfix 新增。
- **实验条件与证据**：
  - **逻辑冒烟**（`optimization400/cache-logic-smoke.cpp` + `build-logic-smoke.cmd`，纯 CPU 单测，不跑模型）：
    - 修复前（`logic-empty-before.log`，**3 FAIL**）：`FAIL: actual-use backfill populates an entirely empty cache`、`FAIL: backfill uses free capacity without requiring a hotter candidate`、`FAIL: hot backfill stops when eligible replacement would only tie`；同时 `hot backfill covered 48/48 layers`（该版本轮转已好）。
    - 更早期（`logic-round-robin-before.log`，**2 FAIL**）：输出行 `hot backfill covered 28/48 layers in six eight-expert budgets`、`FAIL: round-robin backfill does not starve eligible layers`、`FAIL: backfill visits sparse actual layer ids`。**28/48 是该单测合成 fixture（六个八专家预算、48 个合成层键）的行为，不是实际模型里“28 层被饿死”的观测**。
    - 修复后（`logic-final.log`，**failures=0**，**15 项 PASS** + 1 行覆盖信息 + 1 行 `failures=0`，共 17 行）：新增 `actual-use backfill populates an entirely empty cache`、`backfill uses free capacity without requiring a hotter candidate`、`hot backfill stops when eligible replacement would only tie`、`round-robin backfill does not starve eligible layers`、`backfill visits sparse actual layer ids`；并保留 `compare against eligible victim, not protected cold resident`、`zero-frequency cold start uses empty slot`、`FIFO is not changed into frequency admission`、`imminent prediction is not blocked by historical frequency`、`global pool compares actual victim owner`、`position-weighted admission favors next layer and preserves zero slot`。
  - **400-token CLI**（`optimization400/final-r1.json`）：exit 0、Generation **18.6 t/s**（`acceptance.status=fail`，门槛 19.0）、峰值 12458 MiB；命中 371631、未命中 202959、准入 10963、淘汰 7891、重装 4475、`admit_reject_freq=405`；oracle 行 `oracle_topC=72.9% resident_set=72.9% mrs_topC=0.1% actual_hit=64.7%`。**同批基线 `baseline-paired-r2.json`（18.4 t/s）是后置的旧基线单次，与 18.6 不构成受控 A/B**。第二次运行 `final-r2.json` exit **3221225477（0xC0000005）**、无吞吐行 → `acceptance.status=unavailable`（失败运行不参与验收）。
  - **固定历史400步**（`optimization400/fixed400-final/experiment.json`，候选哈希 `e0d7a83f`）：三次命中分别 **349374/349377/349377**，并非所有计数逐项一致；基线340299，约+2.7%。预取字节27,071,999,488（基线26,904,137,728，约+0.6%），淘汰10541（基线10585）；steady中位53.383/54.297/53.365 ms，对照53.946/54.058/55.587。正确性以 `decode_all` 的 **398/400 top-1相同**为主，最大logit差5.5479；`decode_steady` 的382/384只作补充。
- **观察与结论边界**：**逻辑缺陷的修复有独立证据**（单测从 3 FAIL → 0 FAIL，且不依赖 GPU）；**但这个 CPU 夹具通过只说明缓存记账/准入逻辑自洽，不代表 native 数值全面正确**（它不跑模型、不覆盖 CUDA/MoE 算子数值）。**端到端的“稳定加速 + 数值等价”都没有**：CLI 18.6 t/s 未过 19 门槛，固定 400 步只是 +2.7% 命中 / −1.25% 中位耗时，且落在基线自身波动区间（PC-35）。
- **保留/放弃理由**：**暂停**（未提交）。保留理由：修复本身正确、单测可重复；放弃进入默认的理由：没有可验收的端到端收益。
- **遗留/重开条件**：重开需 (a) 消除 `final-r2` 的 0xC0000005（退出期/异步收尾路径，**根因未定**；属 `05-correctness-and-methodology.md`），(b) 在 ≥3 次重复下把耗时差异从基线波动中分离出来。

### PC-35 固定历史 400 步对照方法，以及“基线自身也在波动”

- **状态/版本**：保留（方法学）。
- **为何尝试**：用同一输入序列、同一 400 步、交替顺序（baseline → candidate → baseline → candidate → baseline → candidate）比较两个 WIP 候选与基线，避免“单次运行 = 结论”。
- **技术机制**：`optimization400/verify-fixed400.py` 复用 `tools/tuning/moe-cache-compare.py`：`real-qsa-check.exe`（不同 exe 目录区分 backend）、ctx 8192、`--decode 400`、16 线程、`--n-gpu-layers 49 --cpu-moe`、`CACHE_MIB=6144 GLOBAL_POOL=0 LAYER_AWARE=0 HOT_BACKFILL=8`、per-layer 64 槽 / 6116.3 MiB 物理缓存 / 峰值 12196 MiB；每轮强制校验 `exit_code==0`、artifacts 完整、`decode_steps==400`、logits 无 NaN/Inf，并用 `admit_reject_freq` 是否出现作为 **revision 标记**（候选有、基线无）；启动前检查 ≥90000 MiB 空闲内存。
- **实验条件与证据**：`fixed400/experiment.json` 记录 `backend_sha256`：baseline = `d2a4af54…`（两次实验相同）、candidate₁ = `ac46f065…`（PC-33 的频率门）、candidate₂ = `e0d7a83f…`（PC-34 的回填修复）。
  **关键方法学观察**：**同一 baseline 的重复之间也会分歧**——`fixed400` 的 baseline2 vs ref 最大 logit 差 0.29、逐位相同 357/400（top1 仍全同）；`fixed400-final` 的 baseline3 vs ref 最大差 0.533、逐位相同 66/400；而另一些重复逐位完全一致（400/400）。因此**单次 400 步的 logit 对比不足以判定“等价”**，候选的 2/400 top1 翻转（口径 = `decode_all`，本章一律以它为正确性主述；`decode_steady` 的 382/384 只作补充、不作为验收）必须在能解释基线自身波动的对照里评估。
- **观察与结论边界**：本实验是“固定输入数值/延迟检查”，**不是** CLI 的 19 t/s 验收（`experiment.json.scope` 明确分开）；不运行验证不代表候选已被接受。
- **保留/放弃理由**：保留为方法学（也是 PC-31/PC-34 结论的边界来源）。
- **遗留/重开条件**：若要给候选定“等价”，需要先量化基线重复分布（更多重复次数）并给出可接受的差异上限——当前脚本刻意**不设**这类容差。

### PC-36 运行闸门与记录完整性

- **状态/版本**：保留（方法学）。
- **为何尝试**：性能对比的前提是“失败运行不冒充数据”“并发不互相污染”“历史日志不丢”。
- **技术机制与证据**：
  - **单实例锁**：`optimization400/lock-smoke.json` 显示已有运行持锁时新进程 `exit_code=2`、`exit_code_source=lock-busy`、`run_started=false`（不启动、不产生半份数据）。
  - **超时**：`optimization400/timeout-smoke.json`（`--timeout 1s`）子进程 `exit_code=124`、`tools-run` 同样 124、summary 保存、`acceptance.status=unavailable`；“超时/子进程失败立即停止后续运行，缺失数据记为不可用，不填成零”。
  - **误删事故**：`optimization400/deleted-logs-incident.json` 记录子代理清理时**误删 30 个既有用户日志**（`zz*-out.txt` 等）；同名校本、FileHistory、Win32 影子副本均不可用（`vssadmin` 需管理员），用户决定不恢复，**未伪造替代运行**（`restored=false`、`replacement_runs_fabricated=false`）。凡涉及这些日志的历史数字，本章只按“历史报告”引用。
- **观察与结论边界**：锁与超时行为都有独立 smoke 证据；被删日志范围内的历史结论**不可复核**。
- **保留/放弃理由**：保留。
- **遗留/重开条件**：无。

## 11. 口径陷阱汇总（跨路线）

1. **热点命中 ≠ 重复传输**（§10.1）：`hits = 3 × gpu_uses`（三块权重），DMA 只在准入/重装时发生；判据是 `admissions/readmit_bytes/evicted_unused`，不是 hits。
2. **`hits/misses` 不能归因预测器质量**（当前 PRD §准入与归因）：命中同时来自预测预取、fallback、热回填、phase fill、种子与未命中后准入；`prefetch_experts` 是公共写入路径的**准入总数**且含非预测填充。
3. **`prefetch_predicted/prefetch_required` 是覆盖口径且含静态热并集**（PC-11），与 teacher-forced 的 recall@10、与在线命中率三者不可互换。
4. **`oracle_topC` 内容相关**（PC-17）：策略一改，分工与数值就变，不同 run 的 oracle 不可横比。
5. **旧的放大噪声容差字段无效**（PC-31）：`noise_tolerance_used` / `candidate_within_baseline_repeat_noise` 不作为验收依据。
6. **已撤回值**：旧 host 20.3 / 32 tps（路由错位修复前）、旧“GPU 40 ms 硬底 / 25 t/s 天花板”。19 t/s 是本机内部筛选下限，不是通过即正确。
7. **不可混比的历史工作点**：不同轮次的上下文、KV 类型、缓存预算、步数、lazy/全内存设置不同（当前 PRD §当前性能验收基线点名的 `sweep-*.csv` 问题）；`AHEAD=3` 候选必须在同一 400-token / 6144 MiB / 全内存工作点比较。
8. **teacher-forced 单步 ≠ 累积 rollout**（PC-07/PC-08）：这些试验逐层取真实输入。预测仅作预取提示、不改变真实计算时，才不会直接把近似激活反馈进模型；缓存改变CPU/GPU分工与数值路径的实际影响仍需在线验证，不能据离线曲线宣称无累积影响。
9. **二进制与源码解耦**：`SOURCE_TREE/build-ple-trace-mrs/bin/ggml-base.dll` 由本地构建写入、**会随候选替换**；只能说“**运行当时**该目录里的文件是 <哈希>”（我在复核时读到的是 md5 `5c21782f…` / sha256 `e0d7a83f…` 的回填候选），不得声明“当前”仍是它。引用运行结果必须带哈希，且该目录内容**不是**发布树。
10. **源码存在 ≠ 实现已发布；文档被归档 ≠ 实现已发布**：判定发布只看发布树提交；旁支检出（trace 插桩等）与计划文档都不构成发布。

## 12. 覆盖清单（路线ID ↔ 源章节）

| 源 | 章节/行（快照） | 对应路线 |
|---|---|---|
| `sources/handoff.md` | §6.6（209–258） | PC-12、PC-13、PC-14、PC-15、PC-16 |
| | §6.7（259–343） | PC-17、PC-18、PC-19、PC-20、PC-21 |
| | §6.8（344–372） | PC-22、PC-24 |
| | §6.9（373–404） | PC-23、PC-24 |
| | §6.32（1124–1173） | PC-25、PC-26 |
| | §6.33（1174–1200） | PC-27 |
| `sources/smoe-nk-degradation-plan.md` | §2（65–83）、§5（130–162） | PC-08 |
| | §3（84–99） | PC-09 |
| | §6（163–172） | PC-11 |
| | §7（173–238，含 181–192 表、214–231 后果） | PC-07、PC-10 |
| `sources/moe-cache-score-aware-prd.md` | §当前策略（16–34） | PC-01、PC-28、PC-29 |
| | §共享池的内存安全边界（35–44） | PC-30 |
| | §准入与归因（45–52） | PC-11、PC-33、PC-34 |
| | §当前性能验收基线（53–101） | §0.4、PC-32 |
| | §400-token 驻留期观测（74–101） | PC-32、§10.1 |
| | §此前短回放诊断（102–133） | PC-28、PC-29、PC-31 |
| | §待排查问题（134–163，含 138–144、145–151、152–163） | PC-31 |
| | §历史 v1（164–297：§Solution 183、§Implementation Decisions 223、§Testing Decisions 249、§Out of Scope 277、§Further Notes 289） | PC-01、PC-02、PC-03、PC-05、PC-06、PC-10 |
| `sources/moe-decode-perf-plan.md` | §1–2、§7 | PC-04、§0.4 |
| `sources/rebuild-spec.md` | §预测器(SMoE / Fate / XT) | PC-02、PC-05、PC-06 |
| 源码 | `SOURCE_TREE/ggml/src/ggml-backend.cpp`（1700–1725 开关清单、1910–1925 门字段、4414–4425 时限门、4512/4646/6595 在飞量、5432–5480 价值门、5537–5565 极值搜索、6149 layer_us_ewma、7700–7719 拟合与预算） | PC-24 及全部“已发布”判定 |
| | `SOURCE_TREE/common/arg.cpp`、`tools-run.py`（`PREDICT_FATE=0` 默认基线） | PC-05、PC-27 |

## 13. 建议保存的小日志（精确路径）

规则：只列**小文件**（≤ ~80 KB 为佳，必要时 ≤ 2 MB）。**不要**归档 `*.logits.bin`（129–399 MB/次）、`*stats*.csv`（400+ KB/次）与 0.5–1 MB 的 128 系列 summary 原文，除非先做字段裁剪（`本地化/裁剪建议` 列）。

### 13.1 optimization400（预测/缓存章节的核心证据，全部 ≤ 1.7 MB）

- `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json`（8 285 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-after.json`（8 284 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/final-r1.json`（8 300 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/final-r2.json`（3 595 B，0xC0000005 失败样本）
- `LOCAL_EVIDENCE/moe-cache/optimization400/baseline-paired.json`（3 541 B）、`baseline-paired-r2.json`（5 623 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/lock-smoke.json`（788 B）、`timeout-smoke.json`（3 512 B）、`deleted-logs-incident.json`（2 286 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/fixed400/experiment.json`（1 607 168 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/fixed400-final/experiment.json`（1 638 363 B）
  （裁剪建议：只保留 `runs[].{name,revision,exit_code,wall_s,vram,latency.decode_ms_steady,cache.counters}` 与 `comparisons[].{prefill,decode_all,decode_steady}`，可压到 ~50 KB/份）
- `LOCAL_EVIDENCE/moe-cache/optimization400/cache-logic-smoke.cpp`（10 403 B）、`build-logic-smoke.cmd`（732 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/logic-empty-before.log`（933 B）、`logic-round-robin-before.log`（671 B）、`logic-final.log`（967 B）
- `LOCAL_EVIDENCE/moe-cache/optimization400/verify-fixed400.py`（4 720 B）、`run-baseline.py`（646 B）
- `LOCAL_EVIDENCE/moe-cache/final-verification.json`（1 708 B，含 24 用例 CUDA 冒烟与 LFU_POS CLI 收尾）
- `LOCAL_EVIDENCE/moe-cache/baseline400-6g.summary.json`（4 503 B，历史 20 t/s 判据时期的基线摘要）

### 13.2 共享池 / 位置权重（128 系列，原文较大，建议裁剪）

- `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json`（1 059 328 B）——**必须带上的字段**：`config`、`checks`（尤其 `baseline_repeat_max_maxdiff`、`candidate_vs_reference_*`、`noise_tolerance_used` 的存在性）、`latency`、`runs[].cache.counters`；可删除 `runs[].env_effective_relevant`、`vram_samples`、`rows_meta`、`tokens`。
- `LOCAL_EVIDENCE/moe-cache/global-auto128.summary.json`（973 628 B）、`layer-auto128.summary.json`（505 563 B）、`prechange-cross-comparison.json`（1 000 981 B）——同上裁剪规则。
- `LOCAL_EVIDENCE/moe-cache/stride-memcheck.log`（CUDA stride 冒烟 24 用例 / memcheck 0 errors 的原始结论）

### 13.3 源工程内的原始运行日志（若需归档，请一并取哈希）

- `SOURCE_TREE/a400a-err.txt`（37 407 B）、`SOURCE_TREE/a400b-err.txt`（37 415 B）——含 `hot-set oracle`、`per-rank prediction accuracy`、`smoe per DECODE graph` 三类原文行，是本路线最完整的单文件证据。
- `SOURCE_TREE/L2048-err.txt` / `L6144-err.txt` / `T2048_1-err.txt` / `T6144_4-err.txt`（各 ~34.7 KB）——2048/6144 MiB × 每层/每 tensor 的对照。
- `SOURCE_TREE/8k-tbq-err.txt`（37 140 B）——SMoE `ahead=2` 在 8k/TBQ 配置下的同一格式。
- `LOCAL_MODELS/qwen38/traces/ablation-lazy-moe2-xt.log`（16 632 B）、`ablation-lazy-moe4.log`/`moe6`/`moe8`（各 32.5 KB）——静态表 + XT 阶段（含 `[ Prompt: … | Generation: 15.2/16.2/16.7 t/s ]`）。
- `LOCAL_MODELS/qwen38/traces/moe-static90-8192.csv` / `-t2-` / `-t8-`（各 ~33.7 KB）、`moe-xt-12288.csv`（33 730 B）、`moe-xt-8192.csv`（60 077 B）——静态/XT 覆盖与命中列（`pred_hits/pred_total`）。
- `LOCAL_MODELS/qwen38/traces/build_expert_manifest.py`（4 774 B）、`build_xt_manifest.py`（3 424 B）、`analyze_cross_token.py`（4 920 B）——转移清单生成与 hold-out 回放（**脚本本身即原始方法记录**）。
- 清单文件（各 ~3.1 MB，可选）：`moe-predict-v1.bin`、`moe-predict-static96.bin`、`moe-predict-xt.bin`；若只留证据不留字节，至少保存三者的头部（magic/`n_layers/n_experts/n_trans/n_static`）。
- `LOCAL_MODELS/qwen38/traces/simulate_smoe_nk.py`（8 243 B）、`direct_smoe_recall.py`（22 531 B）、`ple-prefetch-probe.txt`（3 992 B，PLE 阶段交叉引用，归 `01-host-and-devpart.md`）。

### 13.4 不建议归档（体积/可复现性）

- `optimization400/fixed400*/*.logits.bin`（每次 399 MB）、`*stats*.csv`（每次 423 KB）。
- `optimization400/ggml-base.*.dll`（900 KB × 4）：若要留证，**只需保留 `sha256` 与文件名对应表**（`baseline=d2a4af54…`、`lifetime-baseline=ee22deb6…`、`rejected-all-frequency=ac46f065…`、`hot-backfill-candidate=e0d7a83f…`）。
- `LOCAL_EVIDENCE/moe-cache/global-auto128.logits.bin` 等 129 MB 级 logits。
