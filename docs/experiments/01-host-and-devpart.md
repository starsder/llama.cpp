[中文](01-host-and-devpart.md) · [English](01-host-and-devpart.en.md)

# 01 — host 路径、split、devpart、CPU 异步、逐层回读、pinned、worker、prefill 与投机

本文件是实验档案，不是实现说明。每条路线给出同一组字段：路线 ID、状态、为何尝试、技术机制、
实验条件与证据、观察与结论边界、保留或放弃理由、遗留问题与重新开启条件。

本文件不含新实验、不含新优化、不改源码。所有数字都来自源工作区已有日志或既有报告，且必须连同
其运行上下文一起引用；被撤回的数字在 §4 集中列出。

导航：[研发主线](00-research-chronology.md) · [预测与双门控](02-prediction-and-cache.md) ·
[权重内核](03-weight-quantization-and-kernels.md) · [正确性与更正](05-correctness-and-methodology.md) ·
[已保全证据](evidence/README.md) · [历史 handoff](sources/handoff.md)。

---

## 0. 阅读约定

### 0.1 状态词表（严格区分，不把"源码存在"当"已发布"）

| 状态 | 判定依据 |
|---|---|
| **已发布（master基线）** | 该标识符／行为在文档父版本 `0862af564` 的原生代码基线内；本章源码行号以该版本为准 |
| **仅 WIP** | 只存在于后续实验分支或未提交工作区中，发布基线不可达 |
| **仅设计** | 只出现在设计文档（`moe-decode-perf-plan.md`、`rebuild-spec.md`、`moe-cache-score-aware-prd.md`）中，任何树上都没有实现 |
| **已撤回** | 曾实现并实测，随后被回退或放弃（含"实验后回退"与"决定不做"） |
| **仍开放** | 已知缺陷或未复核项，无结论 |

源工作区 HEAD 为 `17ca0de85`，另有未提交的 NXQ/TBQ、缓存与工具链研究改动。发布基线
`0862af564` = 原生代码基线 `7e01451b2` + 一个文档提交，`7e01451b2` 是源工作区 HEAD 的祖先，
因此 §0.1 的"已发布"包含 qwen4exp 全部已提交历史（`c08171aa8` … `7e01451b2`），但不包含源工作区
之上的未提交改动。涉及宿主/split/devpart 行为时，本章一律按 master 可达版本描述，并显式标注 WIP。

### 0.2 硬件与门槛

- 硬件（用户提供）：AMD Ryzen 9 5950X、DDR4-2666 128 GB、RTX A5000 Laptop 16 GB、PCIe 4.0 x8。
- 当前门槛：**400 token / 6 GiB 缓存 / full RAM（`--no-mmap`）口径下 19 t/s**。
  后续本地构建已有该口径记录（见02／05章），但不等于重新构建并验收发布基线；本章不给 master 的“达标”结论。
- 不得作为成绩引用的数字：`20.3 t/s`（封板值，见 §2.11 撤回）、`20.1–22.0 t/s` 系列（同因）、
  devpart 的 `19–23 / 28.2 / 29.7 / 32.0 t/s`（错误实现下的虚高，见 §2.9）、
  `20.7 t/s`（乱码旧构建）。

### 0.3 测量口径纪律（来自既有报告的教训）

1. 性能按 warmup、prefill、decode 分列；可另报 steady 子集，但正确性必须保留全部解码步。早期46 ms回读等账本曾混入非稳态图。
2. 检测退化类缺陷**不能**带 `--ignore-eos`，否则退化尾部被掩盖。
3. 256k 上下文下 t/s 的 run-to-run 方差约 35%（怀疑 Laptop GPU 降频），单次测量不足以定"最优点"。
4. 对比速度必须同配置：`run-cur-ref.ps1` 默认 `CACHE_MIB=2048`（命中率 ~25%、13.7 t/s），
   与 `6144 + HOT_BACKFILL=8` 档不是同一工作点。

### 0.4 引用约定

本章用"文档名 §编号"引用来源，不写跨目录链接。这些文档名对应发布树中的：
`sources/handoff.md`、`sources/moe-decode-perf-plan.md`、`sources/rebuild-spec.md`、
`sources/moe-cache-score-aware-prd.md`、`sources/smoe-nk-degradation-plan.md`。
代码引用写作 `路径:行号`（相对仓库根）。

### 0.5 研发顺序（用户权威补充）与本文件的关系

本章的路线 ID 按**机制**组织，不按研发时间组织；主线叙事由 `00-research-chronology.md` 承担。
为避免把 handoff 的后期章节误当起点，此处记录用户补充的权威顺序：

1. **PLE 缓存**（第一阶段）：为 SSD→内存读取而做；PLE 权重表 26.8 GiB，需要在不改量化的前提下
   把"用到的行"留在内存/显存里。
2. **MoE/SMoE 缓存**：最初是**静态表 + XT 转移表**（离线清单/转移概率）。
3. **Fate**：改试隐藏层预测器，结论是**从隐藏层里榨取不出有用的预测信息**（受挫）。
4. **SMoE（利用共享专家）**：teacher 测试（teacher-forced）命中率 **99%**。
5. 之后才是在线实现：工程坑、命中率升高反而变慢、最后双门控自适应传输阈值。
6. 再往后：devpart 路径的时序错误、TQ4 崩溃、NXQ 与各种修复。

**关于"99%"的口径纪律（必须遵守）**：

- 99% 是**用户报告事实**，保留；不得用其它实验的数字否定它。
- 《`smoe-nk-degradation-plan.md` §7》的 `full = 68.53%` 是**离线单步 teacher-forced 的
  recall@10（每层 top-10 重合率，8 prompt、56400 样本）**；它与"teacher 测试命中率 99%"
  **是否同协议、同分母尚未确认**，不能互相替代或否定。对 99% 保留
  “**用户回述，协议/分母待原始记录对应**”，而不擅自替它定义指标。
- 在线 hit 率（27.5% / 70.4% / 80.8% / 90.9% 等）是第三个指标：它由缓存状态、准入、交付率共同决定，
  与 teacher 测试的命中率不可混同。
- 本章 §2.3 引用 68.53% 时只用于"预测器精度对输入项的敏感度"这一个用途。

同样地，**PLE 不得被写成整体失败**：见 §1A。

---

## 1. 路线索引

| ID | 路线 | 状态 | 主源章节 | 一句话结论 |
|---|---|---|---|---|
| H01 | host 路径基线账本（86 ms/token 分解） | 已发布 | handoff §3/§6.1；perf-plan §1 | 真基线 11.3 t/s；40 ms 是逐层 host↔device 会合 |
| H02 | 逐层 router 回读会合（`ids_wait`/MRS） | 已发布（问题仍在） | handoff §6.1/§6.4/§6.11/§6.12 | 当时17–19 ms；已测 pinned 改法未建立总时长收益 |
| H03 | SMoE 侧图读回事件等待 + 前瞻距离 | 已发布 | handoff §6.1/§6.4/§6.6/§6.30 补充/§6.32/§6.33 | 15.3 ms 是等 GPU；修法是加提前量（ahead 默认 3） |
| H04 | `insert_flush` 每层硬排空 | 已发布（假设作废） | handoff §6.1；perf-plan §1/§P2.1 | 实测 0.07 ms/图，"省 20 ms"作废 |
| H05 | `split_partition` 的"host CPU 循环 22.7 ms" | 已发布（假设作废） | perf-plan §1；handoff §6.5 表 | 实为等 GPU 产出 router，非主机 CPU 耗时 |
| H06 | CPU半边异步化（worker） | 已发布（默认开） | handoff §6.2／§6.6／§6.7／§6.10／§6.11／§6.24／§6.31 | 早期中性，后来一轮缩短约1.4 ms；取决于工作点 |
| H07 | 权重钉住（`pin_weights`）与内存闸 | 已发布 | handoff §5/§6.6/§6.10 | 钉住 72.6 GiB，与 mmap 互斥；已加单实例锁与内存预检 |
| H08 | split 分段记账与合并 | 实验后回退 | handoff §6.11/§6.12/§6.13 | 13.1 ms不能全当可消除的launch税；本次合并OOM |
| H09 | devpart＋host leaf | 已发布（默认关） | handoff §6.3–§6.5/§6.17–§6.25 | 假高速已撤回；与修复后host的同口径净收益未建立 |
| H10 | devpart 与热区信号/自适应系统的兼容 | 仅设计（未实施） | handoff §6.12 ③/§6.24 | 设备侧直方图 + 每 token 一次读回，否则拿命中率换 17 ms |
| H11 | host leaf 旧路由静默错误（`SPLIT=1`） | 已发布（已修复） | handoff §6.30–§6.32；另见 05 章 | 撤回 20.1–22.0 全部速度证据；修复=写完 leaf 立刻重发拷贝 |
| H12 | prefill 读缓存（D2D 暂存） | 已撤回 | handoff §6.16 | 零收益：prefill 的 H2D 本就异步重叠 |
| H13 | prefill/decode 阶段分离（热集） | 已发布 | handoff §6.14/§6.15 | 两阶段集中度本质不同；分离的是统计与用途，不是准入 |
| **H14** | **PLE 分层缓存（CPU L2 / GPU L1）—— 研发第一阶段** | **本体已发布；进算子仅设计** | handoff §6.29/§6.34/§6.35；rebuild-spec §2.3/§7；上游 `4e1865e34`、fork `2f1a363c8` | SSD/mmap 工作点有效（用户报告 1G → 90%+ 命中，属**主机行缓存层**）；full RAM 工作点收益小（GPU L1 关掉反而 17.0 → 17.9 t/s）**档案单列于 §1A** |
| H15 | MTP/投机 × 专家缓存 | 已撤回（WIP 全回退） | handoff §6.26–§6.28 | 唯一触发条件是 `SPLIT=1`；投机前端与缓存从未共存过 |
| H16 | 自适应准入（yield 门槛/极值搜索/trend 回归/ahead 自动） | 已发布（默认关） | handoff §6.6/§6.7/§6.8/§6.9/§6.32 ④ | 门槛与预算可由模型自算；固定 ahead=3 仍略优于在线搜索 |
| H17 | 命中更高却更慢与双门控 | 已发布的实验接口 | handoff §6.6–§6.9 | 价值门＋时限门；速率预算是第三个独立控制量 |
| D01 | 独立 MoE 算子（借壳出算子） | 仅设计 | perf-plan §0.1/§0；rebuild-spec §3/§4/§7.1/§8 | 只有设备侧分区 kernel 落地；算子本体未实现 |
| D02 | 内存层级路线图（RAM+VRAM → +SSD） | 仅设计 | rebuild-spec §2.3 | PLE 已是原型；MoE 缓存侧两条硬约束均未满足 |
| D03 | 专家聚类存储 | 仅设计 | rebuild-spec §2.4 | 纯置换、不改 kernel；无工具、无置换表、无实现 |
| D04 | 共享专家前置 + SMoE 提前开火 | 仅设计（未实施） | perf-plan §P1.4；nk-plan §7 | 离线实测只用 `ffn_input` 也仅差 0.63 pt，但代码仍用三项相加 |
| D05 | 准入模型重写（`gate_copy_us` 标定 / 剩余时间 deadline） | 部分设计未采用 | perf-plan §2/§P1.1/§P1.2；rebuild-spec §4/§6 | 代码仍是 70 µs + 按层数；实测改用排名截止线与前瞻距离 |

---

## 1A. 研发第一阶段：PLE 分层缓存（H14，单列档案）

> 本节按用户补充的权威研发顺序前置：PLE 缓存是最早的一阶段，先于 MoE/SMoE 缓存
> （见 §0.5）。它不应被写成"晚期 prefill 附属项"，也不是整体失败——它在
> **SSD/mmap（lazy）工作点有效**，在 **full RAM 工作点收益小**。

### 1A.1 为何尝试（动机）

- 目标模型有一张 **26.8 GiB** 的逐层 token 嵌入表 `per_layer_token_embd.weight`（类型 `iq4_nl`；
  另有 6.35 节的类型账：全模型 `iq4_nl` 47.9 GiB / 63%，其中该表 26.8 GiB）。
  它无法整体放入 16 GB 显存；主存能容纳，但完整常驻会挤占其他工作集，因此 SSD/mmap 场景采用按需缓存。
- **上游的动机是 SSD→内存读取**（`4e1865e34`，2026-08-28，unsloth/Daniel Han，
  标题 "llama: batched readahead for lazily read gather tables"）：`TENSOR_READ_LAZY` 跳过
  大表的 eager pull-in，并把它的范围标 `MADV_RANDOM`；这同时关掉了内核 readahead，
  于是"稀疏 gather 每行一次同步缺页"。做法是把一个 batch 即将 gather 的行**先合并到整页**，
  再从 `set_input` 一次性发出 `MADV_WILLNEED`（Windows `PrefetchVirtualMemory`）提示
  （"16 gathers become a couple of hints"）。两个使用方是 qwen4exp 与 gemma4；
  **只发提示，结果不变**。
- fork 侧把它做成有界缓存（`2f1a363c8`）：
  `src/models/models.h:2285-2332` 的 `ple_row_cache`，类注释原文要点——
  *"A bounded host-RAM cache for the lazily mmap'ed PLE hash-embedding table. Entries are
  64 KiB-aligned groups of complete table rows. **The cache retains the original on-disk
  representation, so quantization is never changed.**"*
  其中 `copy_pages`（`models.h:2299`）的注释是通往更低一层的关键：
  *"Materialize complete raw pages for a lower-level cache. This keeps the CPU L2 in the path;
  callers never reach into the mmap directly."*
- 页几何：`target_page_bytes = 64 KiB`，`rows_per_page = 64 KiB / row_bytes`
  （`src/models/qwen4exp.cpp:430-450`）；`iq4_nl` 下为 728 行/页，日志打印
  `[PLE-LRU] enabled 2048 MiB: 32776 pages x 63 KiB, 728 rows/page`（63 KiB 是取整显示）。
  容量由 `LLAMA_PLE_CACHE_MIB` 控制，**不设即为关闭**（`configure` 在 env 为空时直接 return）。

### 1A.2 测量与证据

- 统计机制：`LLAMA_PLE_CACHE_STATS_FILE` 每 gather 追加一行
  `rows,resident_pages,capacity_pages,hits,misses,hit_rate,total_hits,total_misses,total_prefetch_pages`
  （`src/models/qwen4exp.cpp:184-197`，代码注释明说 *"A page is the LRU unit, so these are
  page (not individual-row) hits"*）；GPU L1 另有 `LLAMA_PLE_GPU_CACHE_STATS_FILE`、
  `LLAMA_PLE_GPU_CACHE_TIMING_FILE`、`LLAMA_PLE_GPU_PREFETCH_FILE`、`LLAMA_PLE_PREFETCH_DEBUG_FILE`。
- **用户报告（权威实测，保留原述）**：**1G** PLE 缓存下，**SSD→主存（mmap 路径）**命中率达到 **90%+**，
  因此可以释放大量 PLE 常驻内存。→ 标注："**用户回述，协议/分母/样本数待原始记录对应**"。
  **不把"1G"自动精化为"1 GiB"**（用户原述单位/写法保留）。
- **层级与分母必须分开**（两条统计出口的 schema 不同，可从文件列数判定层级）：

  | 层级 | 统计出口（源码） | schema | 命中计数的分母 |
  |---|---|---|---|
  | **主机行缓存（CPU L2，SSD/mmap→主存这一级）** | `LLAMA_PLE_CACHE_STATS_FILE`（`src/models/qwen4exp.cpp:184-197`） | 9 列，含 `total_prefetch_pages` | 本层页（LRU 单元）的命中/缺页 |
  | **GPU L1（显存局部缓存，另一级）** | `LLAMA_PLE_GPU_CACHE_STATS_FILE`（`src/models/qwen4exp.cpp:604-617`） | 8 列，**无** `total_prefetch_pages` | 本批 rows 中已在 GPU 槽中的比例 |

  **主机行缓存层（与"SSD→主存 90%+"同级）**：

  | 文件 | 缓存档 | `total_hits / total_misses` | 页命中率 | 说明 |
  |---|---|---|---|---|
  | `ablation-ple-only.csv`（09-03 12:02） | capacity 65552 页 | 14899 / 1101 | **93.1%** | `total_prefetch_pages=12774`，预取在跑；与该报告**同级、方向一致** |
  | `ple-lru-1g-no-prefetch.csv` | capacity 16388 页，**关预取** | 1505 / 13327 | **10.1%** | 容量与 gather 协议也不同，不能隔离预取开关的因果贡献 |
  | `ple-cpu-l2-with-gpu-l1.csv` | 早期组合 | 48 / 6384 | 0.75% | 未成形的早期版本 |

  **GPU L1 层（不同协议，不可用来证明 SSD→主存 90%+）**：
  `ple-gpu-overlap-stats.csv`（09-03 15:06，capacity 16388 页，14791/1049 = **93.4%**，常驻 13832 页）、
  `ple-gpu-l1-rawpages-1g.csv`（3153/18912 = 16.7%）、`ple-gpu-l1-rawpages-1g-async.csv`（2076/15840 = 13.1%）、
  `ple-gpu-lookahead-1g.csv`。**GPU 局部 90%+ 与 SSD/主存 90%+ 是不同层、不同分母、不同协议，两者不互证。**

  `ple-lru-256m-o1.csv` 按 schema（8 列）属 GPU 写入器，但文件名暗示 CPU LRU
  ⇒ 归为"schema 归属明确、缓存层级待原始运行记录确认"，**不作为任何一层的结论**。

  ⇒ **1G（用户原述）档 SSD→主存 90%+ 按维护者实测保留**。
  现存主机日志的 93.1% 与 10.1% 来自不同容量（65552 对 16388 页）及不同批次，
  不是单变量预取 A/B；不能用它们反推那次 1G 实验的依赖条件。
  GPU L1 的 93.4% 只能说明另一层缓存的观测，不能替代主机级结果。
- 仓库根目录另有四份早期消融 CSV（`abl-2g-ple2g.csv`、`abl-2g-ple512.csv`、`abl-512-ple2g.csv`、
  `abl-512-ple512.csv`，2026-09-11）**内容完全相同**：35 次 gather、合计 147/1293（10.2%）、
  `total_prefetch_pages=0`、末行常驻 1293 页 **远小于** 8194 的最小容量
  ⇒ 短生成、缓存从未填满、**无法区分容量档位**，只能证明"统计装置可用、页粒度计数在跑"。
- MoE 缓存与 PLE 并存的时期有 lazy-mode 消融记录：`ablation-lazy-moe8-gpuple.log`（2026-09-10）
  用 PLE-LRU 4096 MiB + PLE-GPU-L1 512 MiB + MoE 缓存 8 GiB，`Generation: 17.0 t/s`
  （`Prompt: 11.4 t/s`）——这是"PLE 与 MoE 缓存同存"的直接证据。
- **GPU L1（同思路的第二层，`ple_gpu_row_cache`，`models.h:2334-2394`）**：注释说明它是
  "Optional GPU L1 over raw quantized PLE pages"，行经 `page_id → GPU-slot` 表映射后由 CUDA
  就地 gather/反量化原始格式；预取用**独立 CUDA copy stream** 在当批量计算期间填充，
  `gather()` 在消费前 fence（并有 `active_pages` 保护在飞页不被 lookahead 覆写）。
  实测（handoff §6.29，256k + 视觉、256 token、full RAM）：
  **加速不高**——GPU L1 "只值 +0.3 t/s（用户结论）"且抢带宽；同缓存大小下**关掉反而 17.0 → 17.9 t/s**。
  ⇒ 对 full-RAM 驻留帮助不大；PLE 的真正用途是 lazy mode 下用 1–4 GB 内存换 ~90% 命中。

### 1A.3 何时被 MoE 缓存挤掉预算（关闭）

- 预算让位的**时点与范围**（逐项核实，勿过度概括）：
  - 2026-09-03 至 09-10 的 MoE 缓存实验期，PLE **是开着的**：多数脚本设
    `LLAMA_PLE_CACHE_MIB=2048` + `LLAMA_PLE_GPU_CACHE_MIB=1024`（仓库内 40 个脚本仍保留该设置），
    lazy-mode 消融甚至用 4096 MiB + 512 MiB（§1A.2 的 `ablation-lazy-moe8-gpuple.log`）。
  - 脚本里**首次出现 `LLAMA_PLE_CACHE_MIB='0'` 的提交 = `d78c8bd42`**
    （"qwen4exp: hot-region expert cache — true-usage eviction, backfill, adaptive admission"，
    2026-09-13 00:26 +0800），落在**二分/消融脚本**里（`bisect-garbage.ps1`、`bisect2..4.ps1`、
    `dsh-probe-bz8.ps1`）——即 MoE 热区/回填/自适应准入落地时，先在这些对照脚本里把 PLE 清零。
  - 把 PLE=0 作为**推荐配置**是在 **256k + 视觉 + TBQ4 KV** 那一轮（handoff §6.29），
    理由是 GPU L1 收益小且抢带宽、full RAM 下 PLE 无用，而 16 GB 显存要与 MoE 缓存共享。
  - 因此准确表述是：**PLE 与 MoE 缓存长期共存，随后在显存/预算紧张的配置下被 MoE 缓存挤出**，
    不是"PLE 一上线就被替换"。
- 预算竞争的规模：MoE 侧同一时期的日志
  `[MOE-CACHE] 144 weight tensors grouped into 48 layer bundles, 95123456 bytes per layer-slot
  round, 22 slots/layer, 1995.8 MiB physical cache (requested=2048 effective=2048)`
  （`abl-2g-ple.log`，早期 2 GiB 档）→ 后期 6144 MiB → 256k 下 `auto` 只给到 37–40 槽；
  16 GB 显存是 PLE-GPU-L1 与 MoE 缓存的共同上限。
- 配置结论（handoff §6.29）：推荐把 PLE 两项都设为 0；`moe_cache_apply_vram_limit` 的 auto 分支
  只有在**不手工传正数**时才会自适应长大（`requested < 0 → budget = limit − used − guard`），
  手工 cap 只会被下调——**这正是"PLE 省下 1 GB 却没进缓存"的原因**。
- **环境变量名更正**：handoff §6.29 写作 `LLAMA_MOE_PLE_CACHE_MIB` / `LLAMA_MOE_PLE_GPU_CACHE_MIB`；
  代码实际读的是 `LLAMA_PLE_CACHE_MIB`（`src/models/qwen4exp.cpp:43`）与
  `LLAMA_PLE_GPU_CACHE_MIB`（`:432`），仓库脚本也一律用后者。本档案以代码为准。

### 1A.4 与 MoE 缓存的关系，以及"PLE 进算子"的设计

- rebuild-spec §2.3 把 PLE 认作**第二阶段（SSD tier）的机制本体**：`copy_pages` 就是"通往更低一层的桥"；
  模块划分不是"一个大 MoE 算子"，而是"分层分页权重存储 + 预取引擎（PLE 与专家**共用**）
  + 预测器 + MoE 算子"，差异只有页大小（63 KiB vs ~2 MB bundle）、访问模式、预测器三项。
  该结论在 rebuild-spec 里已被定为"PLE 在范围内，不是可选项"，但**实现上仍是两份独立代码**。
- IQ4_NL repack 与 PLE 的关系需要纠正历史说法：当次加载日志没有 `CPU_REPACK` 分配，
  但注册和 buffer 选型通过泛型 extra-buffer 接口完成，不能以“无直接函数调用”断言未注册。
  现有 repack 只接 `MUL_MAT` / `MUL_MAT_ID`，而 PLE 是行 gather / `GET_ROWS`；
  **重新打开 PLE 并不会自动使用 repack**。CPU_ASYNC 下 `cpu=` 还包含 join 等待口径，
  不能把 0.7/55.4 当作全部 CPU 计算或严格收益上限。详见[03 章](03-weight-quantization-and-kernels.md)。
- 遗留设计项：`logical_id → storage_slot` 与"预取 copy 源可换层"这两条硬约束在 **PLE 侧已满足**
  （`copy_pages` + 页抽象），在 **MoE 专家缓存侧未满足**（见 D02）；两者合并成同一套引擎时，
  PLE 侧是样板而不是负担。

| 字段 | 内容 |
|---|---|
| 路线 ID | H14（研发第一阶段） |
| 状态/版本 | 缓存本体**已发布**（上游 `4e1865e34` 的 lazy readahead + fork `2f1a363c8` 的 `ple_row_cache`/env/GPU L1；master 可达）；"PLE 进算子/与专家缓存合并"**仅设计**；默认关闭（env 不设 = off） |
| 为何尝试 | 26.8 GiB PLE 表超过 16 GB 显存，但可放进 128 GB 主存；SSD/mmap 场景用有界缓存节省主存，而非容量上绝对放不下 |
| 技术机制 | 上游：`TENSOR_READ_LAZY` + `MADV_RANDOM` + 批量 `MADV_WILLNEED`/`PrefetchVirtualMemory` 提示；fork：64 KiB 对齐整行的主机 RAM LRU 页缓存（保留磁盘原始量化），可选的原始页 GPU L1（独立 copy stream + fence） |
| 实验条件与证据 | 用户报告 **1G（原述）** → SSD→主存 90%+；现存主机日志另有 93.1% 与 10.1%，但容量与协议不同，非预取开关 A/B。GPU L1 93.4% 不互证；full-RAM 下 GPU PLE 收益小，取舍见 §1A.3 |
| 观察与结论边界 | **SSD/mmap（lazy）工作点有效；full RAM（`--no-mmap`）工作点收益小**。不得写成"PLE 失败"；也不得把 1G/90%（主机行缓存层）与 GPU L1 命中率（另一协议/分母）、MoE 在线 hit 率、离线 recall@10 互相混用（§0.5、§1A.2） |
| 保留/放弃理由 | 保留实现（默认关）；配置上在 full RAM 场景让预算给 MoE 缓存（`d78c8bd42` 起脚本默认 0） |
| 遗留问题/重新开启条件 | SSD/lazy 部署，或 D01 分页引擎设计重新启动时评估；先对应主机 PLE 的容量、页命中与预取协议，不用 GPU L1 数据替代。repack 须另满足类型、buffer 与算子条件，不能由“开启 PLE”推出 |

---

## 2. 逐路径档案

### 2.1 H01 host 路径基线账本

| 字段 | 内容 |
|---|---|
| 路线 ID | H01 |
| 状态/版本 | 已发布；基线构建 `build-ple-trace-mrs`（源工作区对应 `2f1a363c8` / 其后插桩版） |
| 为何尝试 | 任何结构改造都要先有一份可信的逐段账本，否则会像早期报告那样把 warmup 图污染当瓶颈 |
| 技术机制 | 在 `ggml/src/ggml-backend.cpp` 的四个会合点与输入循环加计时器，按图写 CSV（`-1/-2/-3/-4` 行） |

**实验条件与证据**（同一二进制、同配置、31 个 decode 图；`dsh-r8.txt` / `dsh-r9.txt`；
`run-cur-ref.ps1` 同档配置，`CACHE_MIB=2048`、`SPLIT=1`、`MRS=1`、`devpart=0`）：

```
host 路径 (devpart OFF)  total 86.0 ms/token ≈ 11.3 t/s    ← 真基线
  cpu   27.7   CPU 半边算未命中专家（真活）
  gpu    7.4   GPU 入队
  pre   49.7
    ├ inputs 29.1
    │   ├ split_partition 23.2 (n=144，其中 host 侧循环 22.7)
    │   ├ flag_input       4.3 (n=108)
    │   └ generic          1.1 (n=51)
    └ drain  20.5            SMoE 事件等待 ~17 + prefetch 提交 ~4

devpart ON               total 122.9 ms/token ≈ 8.1 t/s
  cpu 3.2 | gpu 7.2 | pre 111.4 → inputs 78.8，其中 generic 78.2 (n=195, ≈400 µs/次)
```

同期补充（handoff §6.1，2026-09-12）：`split_partition` 21.3 ms/图（其中已计时仅 3.3 ms）、
`ids_wait ≈19 ms`（`MRS=0` 时测得 18.57 ms）、`drain+flush 0.07 ms`、
`activate/on_ids/body = 0.00/0.00/1.1 ms`、`act_d2h 0.43 ms`、
SMoE 事件等待 15.3 ms/图（533/35）、命中率 27.5%（12306/32364）、预测准确率 73% 但交付率低、
`MRS=0`（LRU）10.7 t/s（MRS 净收益 +0.4 t/s）、`smoe-off` 崩溃 `0xC0000005`。

已落地的口径改进（perf-plan §5）：单副本调度也建事件 →
`decode 107.9 → 85.0 ms`，190 次 `cudaStreamSynchronize` 变 `cudaStreamWaitEvent`（回退开关
`GGML_SCHED_NO_SINGLE_COPY_EVENTS=1`）；decode 级计时器（CSV `-2/-3/-4`）。

perf-plan §7 记录的被推翻旧结论（不得重踩）：`ids 回读 46 ms`（decode 图里 0 次，warmup 污染）、
`专家拷贝 3.5 ms/token`（0 次）、`host_weights` 分支每图 16 次（0 次）、
`D2H 是瓶颈`（每次 activation D2H ~10 µs，48 次共 509 µs）、
`保活 view 修复了乱码`（含保活 view 的构建仍错且不确定）、
`devpart OFF = 19–21 t/s`（实测 10–11.6 t/s，无历史输出支持）、
`20.7 → 8.4 是性能悬崖`（20.7 那批是乱码旧构建）、
`CPU 2.7 ms 说明分区失效`（高命中率下正常）。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 账本只对 `CACHE_MIB=2048`、命中率 27.5% 的工作点成立；此后命中率提到 70–90%，各段绝对值全部变化（见 §2.6） |
| 保留/放弃理由 | 保留：所有后续路线都以此为参照；它同时给出"40 ms 逐层会合"这一结构性结论 |
| 遗留问题 | 计时口径易被 warmup 污染（已用 `gid < 5` 剔除，但依赖人工）；`LLAMA_MOE_CACHE_TIMING=1` 之外无常态统计 |

### 2.2 H02 逐层 router 回读会合

| 字段 | 内容 |
|---|---|
| 路线 ID | H02 |
| 状态/版本 | 已发布；机制仍在（`moe_prefetch_feasible`、`moe_cache_d2h_begin` 等，`ggml/src/ggml-backend.cpp:4015`、`:4760`） |
| 为何尝试 | 账本里 `split_partition ≈21 ms`，早期归因到 `insert_flush`/主机循环；需要用对照实验分清"等 GPU"与"等拷贝" |
| 技术机制 | 主机要拿到本层 `ffn_moe_topk`/`weighted` 才能做 GPU/CPU 分区，因此每层 `get_async` + `ggml_backend_synchronize`；MRS 路径还要把 256 个 gate 分数读回主机，两条读回的等待位置可互相搬移 |

**实验条件与证据**（handoff §6.4，decode 30 图，82.5 ms/token）：

```
MRS 全分数读回 ON :  prologue=18.04 ms  ids_wait= 1.31 ms   → split_partition ≈ 21 ms
MRS 全分数读回 OFF:  prologue= 0.83 ms  ids_wait=17.01 ms   → split_partition ≈ 21 ms
```

等待只是从"pageable D2H 的隐藏阻塞"搬到"显式 `ggml_backend_synchronize`"，总量不变
（SMoE 的 staging 本来就是 pinned；它 15 ms 的等待是等 GPU，不是等拷贝）。

pinned 化目的地的复测（handoff §6.12 ①，把调度器 ids 读回与 MRS 分数读回的目的地改 pinned，
`moe_cache_d2h_begin/end` 三处）：

```
改前: d2h_enq=3785 µs  d2h_sync=1 µs     ids_wait=1.2 ms   total=48.7 ms
改后: d2h_enq=4 µs     d2h_sync=3787 µs  ids_wait=17.3 ms  total=48.3 ms
```

新工作点复证（handoff §6.10 ④，6 GB + 回填 8）：`MRS_FULL=0` → 19.9 t/s，与基线 19.9/20.0 无差别。
MRS 全套本身约 0.4 ms，且已确认是死重（覆盖率 0.3%，见 §2.3/H16），但单独去掉净收益 ≈0。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 结论只针对"主机必须逐层拿到 router 结果"这一流水线结构；MRS 的*内容*是否有用是另一问题（§2.3、H16）。pinned 化不改变总量，只把等待记账变诚实 |
| 保留/放弃理由 | pinned目的地保留；设备侧分区是已尝试的结构方案，不是已证明唯一的方案，也未建立修复后host对照下的净收益 |
| 遗留问题 | `SPLIT=1` 下这 17–19 ms 会合仍在（修复静默错误后仍存在）；`SPLIT=0` 虽正确但 3.4–6.5 t/s（§2.11） |

### 2.3 H03 SMoE 侧图读回事件等待与前瞻距离

| 字段 | 内容 |
|---|---|
| 路线 ID | H03 |
| 状态/版本 | 已发布（`moe_cache_smoe_process` `ggml/src/ggml-backend.cpp:5217`；前瞻默认 3） |
| 为何尝试 | 账本中 `drain ≈20.5 ms`（SMoE 事件等待 ~17 ms）在临界路径上；预测服务的是"未来"，原则上不该进关键路径 |
| 技术机制 | SMoE 侧图为下一层算候选；读回拷贝在 compute 之后入队，事件必然等到该层 GPU 跑完。可调的量是"提前多少层"（`LLAMA_MOE_SMOE_AHEAD`）与读回是否阻塞（`LLAMA_MOE_SMOE_NONBLOCK`） |

**实验条件与证据**：

- 预取量与耗时的历史联合变化（handoff §6.6①，`NONBLOCK=1 AHEAD=2`）：
  161 MB/tok → 命中 40.2%、70.0 ms、14.3 t/s；254 MB → 46.9%、77.7 ms、12.9；464 MB → 58.2%、94.1 ms、10.6。
  边际约 0.073 ms/MB，其倒数约 13.8 GB/s，**不是 PCIe 带宽实测**。
  `event_synchronize` 等待变长已记录；具体 PCIe/显存争用路径是当时解释，未由硬件 profiling 隔离。
- 排名衰减与主动截断（handoff §6.6 ②③）：`cutoff 1 → 4.69 hits/MB`、`2 → 5.25`、`3 → 2.21`、`4 → 1.78`；
  `take = min(n_slots, n_topk, take_max)`，`take_max = n_used+2`；`n_used` 只在分区钩子里赋值，
  **首 token / 未过钩子的层 `n_used==0` → 截止线静默消失（bug，已修）**；现在 `rank_cut = LLAMA_MOE_TAKE_MAX`（默认 2）。
  先截断再跳过已驻留 优于 先跳过再填预算。
- 去重硬数字（handoff §6.6 ④）：候选 2852 = `dup_resident 1496`（52.5% 不花字节）+ 传输 1356；
  `dup_list=dup_admit=dup_pending=0`；`readmit=86`（6.3%）。
- 容量交互（§6.6 ⑤）：30 token 下 2048 与 6144 MiB 无差别（6 GB ≈67 槽/层，每 token 每层只准入 ~2 个，
  `readmit=0`）；150 token 下 2048 最优在 cutoff 1（14.6），6144 平台在 2–4（14.8–14.9）。显存峰值
  2048 → 9132 MiB，6144 → 13278 MiB（limit 15360、guard 1024）。
- 前瞻曲线（handoff §6.30 补充，8k/q8_0/64 槽、`SPLIT=1` 且当时带静默错误）：

  | ahead | r1 | r2 | r3 | 命中率 | gen t/s |
  |---|---|---|---|---|---|
  | 1 | 58.5% | 49.7 | 43.6 | 68.2% | 17.2 |
  | 2（旧默认） | 84.7% | 76.5 | 70.2 | 77.4% | 18.1 |
  | 3 | 80.9% | 67.9 | 60.7 | 85.4% | 19.2 |
  | 4 | 78.6% | 68.7 | 59.5 | 85.7% | 19.9 |
  | 6 | 56.1% | 46.2 | 39.7 | 80.6% | 18.5 |

  最优点 3–4 层：命中率不是由 r1 单项决定——ahead=2 的 r1 最高却是命中最低，3–4 层 r1 略降但
  提前量足以让预取真正完成。
- 修复后复测（handoff §6.32 ②，8k/q8_0、`auto`=97 槽）：`SPLIT=1+AHEAD=3` → 命中 90.9%、**20.8 t/s**
  （已验证正确）；`AHEAD=2` → 83.5%、18.4；`AHEAD=4` → 87.8%、20.2。相对关缓存 9.2 t/s = +126%。
- `ahead=1` 的 r1 异常结案（handoff §6.33）：把读回改同步后 ahead=1 的 r1 从 61.2% → **93.1%**，
  ahead=2 基本不变（86.0 → 84.7）⇒ 非阻塞读回让预测晚一拍被消费；ahead=1 的窗口恰好整拍过期。
  取舍（背靠背各 2 次）：默认 `NONBLOCK=1 + AHEAD=3` 均值 20.1 t/s 仍最优；
  `NONBLOCK=0 + AHEAD=1` 速度相当（19.1）但 r1 93.1%，留给更看重长期放置质量的场景。
- 离线对照（`smoe-nk-degradation-plan.md` §7，8 prompt，`ffn_moe_input` 新白名单）：
  oracle k=1..4 全 100%；full 68.53/59.94/55.70/53.77%；`input_only` 67.90/59.34/55.22/53.44%
  ⇒ 预测器几乎不依赖路由/共享专家输出（差 0.63 pt），本可把开火点前移到 attention 结束（见 D04）。
  **口径提醒**：这里的 68.53% 是"离线单步 teacher-forced 的 recall@10"，与用户报告的
  “teacher 测试命中率 99%”是否同协议尚未确认，不能互相替代；在线 hit 又受驻留与交付影响（§0.5）。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | §6.30 补充在错误的 SPLIT=1 下测量，趋势与绝对值都可能失真；修复后的 §6.32/§6.33 是独立证据。ahead=1 的非阻塞晚一拍解释以同步对照为依据 |
| 保留/放弃理由 | 保留：提高提前量是消掉 15 ms 等待的现实手段；D04（更早开火）是它未走完的极端形式 |
| 遗留问题 | `SMOE_AUTO` 类在线搜索默认关闭；侧图仍在逐层同步；`AHEAD_AUTO` 目标函数必须用窗口增量命中率 |

### 2.4 H04 `insert_flush` 每层硬排空（假设作废）

| 字段 | 内容 |
|---|---|
| 路线 ID | H04 |
| 状态/版本 | 已发布；假设作废（perf-plan §P2.1 的"~23 ms"未成立） |
| 为何尝试 | 代码注释说明 `moe_insert_flush` 是故意阻塞（避免侧流拷贝越过 CUDA graph 捕获边界），每层一次硬排空，账面上很可疑 |
| 技术机制 | `s.insert_cv.wait(...)` 等 insert worker 队列清空 + 在飞归零；改成"图级一次"即可保留正确性 |

**证据**：`drain+flush` 实测 **0.07 ms/图（0.3%）**（handoff §6.1）。perf-plan §1 也指出
`split_partition` 144 次调用只有 48 次做实事，其余早退，23 ms 集中在开头四行，
但 §P0.1 的计时器落地后归因到 router 会合（§2.2），不是 flush。

| 字段 | 内容 |
|---|---|
| 保留/放弃理由 | 放弃：实测非瓶颈。handoff §6.5 的待办表把该项标为"实测 0.07 ms，作废" |
| 遗留问题 | 无（作为已排除项保留，避免重复怀疑） |

### 2.5 H05 `split_partition` 的"host CPU 循环 22.7 ms"（假设作废）

| 字段 | 内容 |
|---|---|
| 路线 ID | H05 |
| 状态/版本 | 已发布；假设作废 |
| 为何尝试 | perf-plan §1 把 `split_partition` 的 23 ms 记成"里面 host CPU 循环 22,724 µs"，暗示主机 CPU 是瓶颈，可用更快的循环/并行解决 |
| 技术机制 | 分区钩子在主机执行，但循环本身极廉价；真实成本是它前面的等待 |

**证据**：handoff §6.5 待办表第 4 行明确"`split_partition` 的 host CPU 循环 → 实测非主机 CPU 耗时，作废"；
§6.10 ⑤给出分解：`prologue 17.9 ms（其中 mrs_queue 17.0 = 等 GPU 产出本层 router）`
+ `ids_wait 1.4 + partition 1.2 + act_d2h 0.5`。MRS/ pinned 两组独立实验都只搬移等待（§2.2）。

| 字段 | 内容 |
|---|---|
| 保留/放弃理由 | 放弃该归因；保留"分区本身廉价"这一事实，用于否掉"优化分区循环"一类改法 |
| 遗留问题 | 无 |

### 2.6 H06 CPU 半边异步化（worker）

| 字段 | 内容 |
|---|---|
| 路线 ID | H06 |
| 状态/版本 | 已发布，`LLAMA_MOE_CPU_ASYNC` 默认 1 |
| 为何尝试 | CPU 半边（27.7 ms）在账本里是最大单项；把它移出调度线程、与 GPU 重叠是显而易见的方向 |
| 技术机制 | 分区时把 MoE CPU 半边派给自有 worker（`moe_cpu_half_submit`），CPU split 时 join；调度线程只做 join |

**实验条件与证据（按时间顺序，注意工作点不同）**：

| 时间/章节 | 工作点 | 结果 | 解释 |
|---|---|---|---|
| handoff §6.2（2026-09-12） | `CACHE_MIB=2048`、命中 27.5% | `CPU_ASYNC=1` → 文本逐字一致、**11.0 vs 11.1 t/s（中性）** | CPU 半边本就与 GPU 执行重叠，不在临界路径上 |
| handoff §6.6 ① | 预取量扫描 | worker 数 1→3：低量 13.3→13.6；高量 9.9→10.5（+6%） | 加 worker 救不了 PCIe 争用 |
| handoff §6.7 未解决 | 预取开 | `LLAMA_MOE_INSERT_WORKERS` 在 `PREFETCH=1` 下**空转**（worker 的 spawn 点都要求 `!prefetch`） | 之前"worker 不是瓶颈"的结论作废；预取拷贝是主机线程内联提交 |
| handoff §6.10 ④ | 6 GB + 回填 8 | `CPU_ASYNC=1` → **20.5**（基线 19.9/20.0）；host 线程 cpu 12.4→5.7 ms，join 多付 2.7 ms，净省 1.4 ms | 工作点变了：命中率与准入把调度线程变成瓶颈 |
| handoff §6.11 ① | 同上，转默认开 | **21.5 / 21.4**（此前 19.9–20.0）；total 51.0→48.7 ms；cpu 12.4→6.2；compute 17.2→13.1；gpu_queue 5.3→6.9 | 用户批准转正 |
| handoff §6.24 | devpart 400 token | staging 点派发 worker（`moe_cpu_half_submit_layer_staged`）`dispatched=19153` 生效 | 因驻留集为空、CPU 工作量本身大 3 倍，未见提速 |
| handoff §6.31 | split 静默错误期 | `CPU_ASYNC=0` **不是可用退路**：不生成且缓存空转（`hits=0`） | 只能作为已排除项 |

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | **不要写成普遍定律**：同一开关在 27.5% 命中率工作点是中性的，在 6 GB + 回填 8 的工作点净省 1.4 ms/图。判据是"CPU 半边是否仍与 GPU 充分重叠、调度线程是否成为瓶颈"，与开关本身无关 |
| 保留/放弃理由 | 保留并默认开；它同时是 devpart 异步 CPU 半边的前置基础设施（§2.9） |
| 遗留问题 | `PREFETCH=1` 下 insert worker 空转这一死支仍存在（未清理也未启用）；`CPU_ASYNC=0` 与 `SPLIT=1` 组合不可用（hits=0） |

### 2.7 H07 权重钉住（`pin_weights`）与内存闸

| 字段 | 内容 |
|---|---|
| 路线 ID | H07 |
| 状态/版本 | 已发布（`moe_cache_pin_weights` `ggml/src/ggml-backend.cpp:3979`，`LLAMA_MOE_PIN_WEIGHTS` 默认开） |
| 为何尝试 | 专家权重必须常驻主机内存并供 DMA 使用；分页内存的入队阻塞是可测量的开销（§2.2 的 `d2h_enq 3.7 ms`） |
| 技术机制 | `cudaHostRegister` 锁定整份专家权重（日志：`pinned 1 weight buffers (72.6 GiB)`），使拷贝真异步 |

**实验条件与证据**：

- 内存拓扑与风险（handoff §5）：钉住与 mmap **互斥**；两者混用会同时占住 76 GB 文件页与
  72.6 GB 锁定内存 → 顶爆 128 GB。现有脚本一律 `--no-mmap` + 默认钉住。
- 环境约束：绝不同时跑两个推理；不要 `-ngl 0` 且不带 `--cpu-moe`。
- 整机事故（handoff §6.10 ②）：后台跑测循环与另一批实例重叠 → 两个 `llama-cli` 各钉 72.6 GiB
  → 2×72.6 > 131 GiB 物理内存 → 提交量耗尽、整机卡死。**与模型/显存/本次优化无关**，是操作失误。
- 防复发闸（同节 ③，已测试）：`tools-run.py` 单实例锁（`.tools-run.lock` + PID 存活检查，陈旧锁自动回收）
  + 空闲内存预检（`--min-free-mib` 默认 90000、`--wait-mem` 默认 120 s，不足则等待后报错）。
- pinned 环的收益上限只有几 ms（handoff §6.6 ①）；SMoE staging 本就是 pinned（§6.4）。
- 调度器 ids / MRS 分数读回改 pinned：入队 3785 µs → 4 µs，但等待搬到 synchronize（§2.2），中性保留。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | "pinned 化能救会合"被明确否定（§6.4/§6.12）；钉住解决的是 DMA 可否真异步，不是流水线结构 |
| 保留/放弃理由 | 保留（默认开）；配套的内存闸与串行纪律是本项目最重要的防护之一 |
| 遗留问题 | 单实例锁只在 `tools-run.py` 内，直接跑 `llama-cli` 的路径无保护（依赖人工纪律） |

### 2.8 H08 split 分段开销与合并

| 字段 | 内容 |
|---|---|
| 路线 ID | H08 |
| 状态/版本 | 实验后回退（未采用）；现状 `SPLIT=1`、CUDA graph 生效 |
| 为何尝试 | `compute` 段 13.1 ms/图，若每层 3 个权重种类能合并到 1 个 split，可省约 2/3 |
| 技术机制 | `--cpu-moe` 的专家权重常驻 CPU（WEIGHTS + 后端不兼容）→ 调度器 pass 5 规则为每个 `MUL_MAT_ID` 切一刀（`ggml_backend_sched_split_graph`）；合并即去掉该规则或改写输入来源 |

**实验条件与证据**：

- 开销结构（handoff §6.11 ②/§6.12 ②）：钩子每图跑 **142.6 次 ≈ 48 层 × 3 个权重种类**
  （real=47.5 真正分区、early=95.1 直接早退 0.00 ms）；`ggml_backend_graph_compute_async` 13.1 ms ≈ 每次 92 µs。
  CUDA graph 已生效：`GGML_CUDA_DISABLE_GRAPHS=1` 时 total 81.8 ms、compute 62.7 ms。
- 结构转储（handoff §6.13，用 `LLAMA_MOE_DUMP_SPLITS=1`，此前"已写未跑"）：decode 每图
  **242 splits / 8978 nodes / 1601 leafs**（prefill 145；两张 decode 图完全一致 → 图不增长）；
  每层 5 个 split：`attn 块(119 节点) + gate(1) + up(2) + down(21+VIEWs) + CPU MOE_CPU(1) + router 块(136~218)`。
- 两次尝试都失败：
  1. 图构建前 patch `src[0]`→cache view：`n_splits` 仍 242（钩子命中的张量不对）并引入 fail-fast 崩；
     要成功必须在 llama 图构建期做且自己接管非 direct 层的兜底拷贝 → 风险不低于 devpart。
  2. 关掉调度器那条规则（形式上是低风险）：**VRAM 峰值 13200 → 15868 MiB（> 15360 limit）→ OOM 崩**；
     原因：合并后三个权重输入必须同时驻留，每层多 ~2× 专家包。
     **这正是该规则存在的理由：用切分换权重暂存内存。**
- 回退后状态：`21.6 t/s、VRAM 13198 MiB、文本正确`。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | "13.1 ms 是白拿的"被否定；结论只在"缓存容量 = 6 GB 档"的显存预算下成立——若主动缩小 cache，合并可能重新可行 |
| 保留/放弃理由 | 本轮决定不做。缩小cache后再合并是候选权衡，未测量；不能把13.1 ms全额列为可回收启动税 |
| 遗留问题 | 该交换从未实测；`LLAMA_MOE_DUMP_SPLITS` 诊断保留 |

### 2.9 H09 devpart（设备侧分区）+ host leaf —— 完整更正链

| 字段 | 内容 |
|---|---|
| 路线 ID | H09 |
| 状态/版本 | 已发布，`LLAMA_MOE_DEVPART` 默认关闭。已修复当时定位的数据缺陷；用户停止推进，不代表完成全面数值回归 |
| 为何尝试 | 主机逐层等待router会合，设备侧分区是当时拟采用的结构改法；“唯一解”属于旧设计判断，不是排除其他实现后的证明 |
| 技术机制 | 新增设备侧分区 kernel（`ggml/src/ggml-cuda/moe-partition.cu`，81 行，kernel 在 :8/:41，驻留表读取在 :19），GPU 直接产出 `part_ids/part_wgt/table`；CPU 半边改吃主机 leaf，由 MoE 侧用 pinned staging + 每层事件异步回填 |

**更正链（时间顺序，含每次失败与修复）**：

| # | 章节/日期 | 状态与数字 | 失败/修复内容 |
|---|---|---|---|
| 1 | handoff §3（devpart ON，设备 view） | 8.1 t/s（122.9 ms/token） | 文本正确但慢：`generic` 78.2 ms = 195 次跨后端拷贝 × ~400 µs；perf-plan §1 判"账算反了" |
| 2 | §6.3/§6.4 计划 | — | 三步：设备分区（省 19 ms）→ CPU 半边输入改自有异步 D2H + worker + 边界 join → 再把 SMoE 等待纳入 |
| 3 | §6.5（host-leaf 实验） | 报告 **19–23 t/s**（`CPU_ASYNC=2` 填充 17–19；mode 1 worker 14.9）→ **撤回** | 证明 78 ms 来源确是那 195 次拷贝；但输出从第 2 token 崩坏（`The////`）。已排除 ids 填充无效、CPU 半边没算、paltry 保活 view；未查清 `cur` 取自 `src[1]` 的语义、`wgt_cpu` 偏移、CUDA graph 捕获/复放交互、`moe_cpu_half` 是否逐层命中。仓库恢复到 host 11.2 / devpart 8.1 两个可用状态 |
| 4 | §6.17（复查） | 报告 29.7 t/s、稳态 **32.0 t/s**（vs host 20.8，+54%）→ **撤回** | 崩坏与热区机制无关（`HOT_BACKFILL=0 PREFETCH=0` 仍崩，`The划分为其职 Dess Dess…`）；`LLAMA_MOE_DUMP_CPUHALF` 显示 CPU 半边 ids 全 0（host 参考值正常）→ 定位到"设备分区输出在 CPU 半边消费时尚未写出/可见" |
| 5 | §6.18（根因+修复步骤 1） | 症状前进 | 根因：devpart 分支的 `ids_cpu/wgt_cpu` 是**设备视图且没有 `ggml_set_input`** → 只能靠调度器跨后端拷贝，而它不等设备写出 → CPU 半边永远读到 0（设备 kernel 本身正确：前 k=GPU 槽位 / 后 k=CPU 专家 id、`-1` 标记）。修 1：改主机 leaf + pinned 异步 D2H 后半段（每层 ≈160 B）→ 崩坏形态变为 `viewer viewer…`（ids/权重到位，剩余输入不对）。修 2（加 `cur_cpu` leaf）**在加载期崩**（VRAM 仅 6.9 GB，非 OOM → 断言/形状假设）。性能隐忧：每张量都 `ggml_backend_synchronize`（每层 3 次会等本层 GPU 半边），必须改成每图一次 |
| 6 | §6.19（修好输出连贯） | 26.0 t/s（host 19.6） | 图构建用主机 leaf（`ids_cpu/wgt_cpu/cur_cpu`），`ids_gpu/wgt_gpu` 仍设备视图；填充从"该层 `MUL_MAT_ID` 的 `src[1]`（设备张量）"读激活。踩坑：误读 CPU op 自己的 `src[3]`（已是主机 leaf）→ `get_async` 走 CUDA 断言 `ggml-cuda.cu:2652 unsupported buffer type`（已写进代码注释）。首个 token 与宿主一致，约第 3 token 分叉 = 浮点求和次序差异（贪心下必然），非数据错误 |
| 7 | §6.20（速度路线打通） | 报告 **28.2 t/s**（host 16.8，+70%）→ **撤回** | 三件套：分区 op 独立 split（pass 5 收口，`LLAMA_MOE_PART_SPLIT=1` 默认开）；`moe_cache_devpart_readback` 在含分区 op 的 split 后立刻发起 3 组 D2H 进每层 pinned staging 并记每层事件；CPU 半边只等该层事件（不等整条流）。devpart 下强制 `cpu_half_async=0`（worker 会在 leaf 发布前被派发）。文本仍 `The/////` |
| 8 | §6.21（收敛为单点） | 诊断 | `LLAMA_MOE_DUMP_CH=1` 下 host 行打印、**devpart 行从未打印** ⇒ `rb.ready` 恒假 ⇒ `moe_cache_devpart_readback` 在某个 `continue` 提前退出 ⇒ leaf 保持零值。列出四个 bail 点 |
| 9 | §6.22（管道修通） | 28.2 t/s（仍错误） | 补回退路径：调度器可能把 CPU 半边排在分区 split 之前（leaf 是主机张量 → 无依赖边）→ staged 未就绪时直接 `tensor_get`（保正确性）。**"读回从不提交"的真因**：读取层号用 `manifest.n_layers`（未加载清单时为 0）→ 改为查 `s.layers`。诊断链证明管道在工作，缺陷在"内核读到的驻留表内容 ≠ 主机镜像"（fallback 读到 `ids: -1 -1 … wgt: 0.000 cur: 0.0000`，而 host 参考是真实 id/权重） |
| 10 | §6.23（三个真实缺陷修复） | 修正后：60 token host 14.1 / devpart **15.0**，文本**完全相同**；240 token + SMoE 三件套：host 15.6 / devpart **13.6**，文本完全相同 | **缺陷 1 驻留表未初始化**：图内存里的 `ffn_moe_part_table` 不是 -1，内核读成"每个专家都在槽 0" ⇒ 全判驻留 ⇒ CPU 半边只拿 -1/0。修：图开头把所有 `ffn_moe_part_table*` 发布为全 -1（一次），随后仍由每层 flush 覆盖。**缺陷 2 激活取错来源**：devpart 下 CPU 半边自己的 `cur_cpu` 是主机 leaf，真正的激活在 GPU 半边的 `MUL_MAT_ID`；按名字找是错的（`ffn_moe_topk-<真实层>` 与分区家族张量编号约定不同）。修：用**指针同一性**（GPU 半边 `src[2] == moe_graph_find("ffn_moe_ids_gpu", layer)` 的 `MUL_MAT_ID`，取其 `src[1]`，即 `moe_cpu_activation()`）。**缺陷 3 staging 钩错算子**：`pids` 与 `pwgt` 会被调度器切进不同 split，在 `MOE_PARTITION_IDS` 所在 split 读 `pwgt` 读到内核产出前的垃圾（prefill 走 fallback 所以没暴露，decode 全错）。修：读回改挂 `MOE_PARTITION_WGT`，ids 取其 `src[2]`（指针）；staging 槽按图失效（`rb.graph`）+ 未就绪时阻塞式回退 |
| 11 | §6.23 性能结论 | **正确后并不比宿主快**（14–15 vs 14–16，噪声级持平）；此前 28.2 **不可采信** | `LLAMA_MOE_DEVPART` 保持默认关闭；本轮临时探针（`CH-PUB`/`CH-W`/`CH-ACT`/`CH-TBL`/`CH-RB`/`CH-CPU`）删除，保留 `LLAMA_MOE_DUMP_CH` 双路径对比 dump |
| 12 | §6.24（400 token 稳态） | devpart **16.8 t/s**（total 62.3 ms；cpu 19.0 vs host 6.7）；同节 host 20.3 t/s | 收益侧真实：`ids_wait=0.0`、`partition=0`、`split_partition n=0`（17.9 ms 会合确实消掉），但被 `cpu 19.0 ms` 吃回。**根因**：devpart 驻留集恒为空（`hits=87 / misses=300219`，host 为 404757/169833 = 70.4%）——设备分区内核只**读**驻留表，而入驻/驱逐/预取插入整段在主机分区钩子里，devpart 为省 17.9 ms 把该钩子整个跳过 ⇒ 表永远全 -1 ⇒ 切分冻结 ⇒ GPU 占用一条直线 |
| 13 | §6.24 尝试（未解决，代码保留，devpart 专用） | — | ① CPU 半边由 staging 点派发给 worker（`dispatched=19153` 生效，但 CPU 工作量本身大 3 倍）；② 回放主机侧命中/未命中计数与 `moe_cache_warm_miss`（`moe_cache_devpart_account`）；③ 图尾补调 `moe_insert_drain/flush`；④ 把真实路由 stage 回来喂 `moe_cache_on_ids` |
| 14 | §6.25 天花板实测 | `CEILING_DIV=10` 只省 6.0 ms/图（+14%），`ids_wait 17.9 ms` 纹丝不动 | 结论：逐层 router 会合是 **GPU 侧流水延迟**，与 CPU 半边工作量无关；devpart 只是把它换段记账 ⇒"预测前一拍"的收益上限 ≈ 6 ms 的一部分（+8~12%），不值得动 kernel |
| 15 | §6.31/§6.32 | 安全锁期 devpart 仍被允许（"另一条已验证正确的路径"）；split 修复后状态不变 | devpart 保持默认关闭、不再推进（用户决定） |

**状态标注**：设备侧分区 kernel 与整套 devpart 读回/回退逻辑在 master 可达
（`moe-partition.cu`、`moe_cache_devpart_readback` `:4644`、`moe_cpu_half_submit_layer_staged` `:5552`、
调用点 `:7051`、`LLAMA_MOE_PART_SPLIT` 唯一出现处）；`LLAMA_MOE_DEVPART` 默认 0。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 19–23、29.7、32.0、28.2来自错误实现；16.8是修正后的历史记录，但其host 20.3对照后来也被撤回。因此与修复后host的净收益方向仍缺同口径证据 |
| 保留/放弃理由 | 代码保留，默认关闭；当时文本探针通过不等于全面正确性保证。驻留与预取决策仍依赖主机侧，是继续推进的结构障碍 |
| 遗留问题/重新开启条件 | 重新开启条件：把入驻/预取决策也搬到设备侧可驱动的路径（见 H10），或在同口径下证明 host 修复后 devpart 仍不落后 |

### 2.10 H10 devpart 与热区信号 / 自适应系统的兼容

| 字段 | 内容 |
|---|---|
| 路线 ID | H10 |
| 状态/版本 | **仅设计**（未实施）；§6.12 ③ 提出，§6.24 给出实证反证 |
| 为何尝试 | devpart 把分区放到设备侧 → 主机不再读回 `part.ids` → 整套热区机制的输入信号消失 |
| 技术机制 | 依赖方逐个列出：真值频次淘汰分（§6.7）、回填排序、每排名准确率/yield 统计（§6.8/§6.9）全部依赖 `part.ids`。兼容做法：把使用直方图放到**设备**算（对 router ids 做 scatter-add 到每层计数缓冲，或复用 top-k kernel），再**每 token 批量读回一次**（48 层 × 256 × 4 B ≈ 49 KB ≈ 可忽略） |

**证据**：§6.12 ③ 的判断——"否则就是拿 +20pp 命中率去换 17 ms，不划算"；
§6.24 的实证反证——devpart 跳过主机钩子后驻留表恒空、命中率 87/300219，直接说明该耦合是真实的、
而不是理论担忧。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 这是 devpart 能"真正用上缓存"的前提；当前实现只做到"读回真实路由喂给 `moe_cache_on_ids`"这一半（§6.24 尝试 ④），准入决策仍不在设备侧 |
| 保留/放弃理由 | 未实施：属于独立结构性工作，性价比不足以下注 |
| 重新开启条件 | 与 D01（自持算子）合并做，或先做"设备侧直方图 + 每 token 一次读回"的最小闭环并证明命中率不降 |

### 2.11 H11 host leaf 旧路由静默错误（`SPLIT=1`）——速度证据的撤回

| 字段 | 内容 |
|---|---|
| 路线 ID | H11 |
| 状态/版本 | 已发布（缺陷已修复，安全锁已解除）；正确性方法论细节由 05 章详写，本节只记**对速度证据的撤回**与修复要点 |
| 为何尝试 | 在验收"缓存收益"时发现同一 prompt 的输出长度在三档配置下差一个数量级 |
| 技术机制 | host `SPLIT` 路径的 GPU 半边 `ids_gpu/wgt_gpu` 是**本 split 的输入**（`ggml_set_input` 主机 leaf），其值由分区钩子在**输入循环中途**写入；调度器可能已先排了这两个 leaf 的拷贝 ⇒ MoE GEMM 拿到**上一层的路由** |

**实验条件与证据**：

- 复现（handoff §6.30，单 prompt、贪心、`-n 160`、8k/q8_0、host 模式）：
  缓存 OFF → 751 字符完整正确、9.2 t/s；缓存 ON + `SPLIT=0` → 751 字符正确、15.9 t/s；
  缓存 ON + `SPLIT=1`(direct) → **27 字符退化、22.0 t/s**；`SPLIT=1`(gather) → 27 字符。
  复现要点：`-p 'Write a short factual paragraph about the Eiffel Tower.' -n 160 --temp 0`，
  **必须不要 `--ignore-eos`**，否则退化尾部被掩盖——此前所有"逐字一致"验证因此失效。
- 二分（同节）：`DIRECT_READ=0` 仍错（与 direct slot-view 无关）；`PREDICT_SMOE=0` 仍错；
  `HOT_BACKFILL=0` 仍错；`SPLIT=0` 正确；`CPU_ASYNC=0` 不生成且 `hits=0`。
- 性质判断：经典竞态（CPU 半边消费的 host leaf 与"当层/当 token 的划分结果"之间的发布/消费时序），
  或 direct 模式下 slot-view 补丁未生效时按原始权重张量专家维索引 ⇒ 读到错误专家、地址合法 ⇒ 不崩、静默算错。
- 部署性矩阵（handoff §6.31）：缓存关 9.2 （正确）；`SPLIT=1` 22.0 （错误） 静默算错；
  `SPLIT=0` **3.4–6.5 t/s** （正确） 但比不用缓存还慢（专家仍在 CPU 算，权重被搬到显存 ⇒ 每次读专家变成 PCIe 拷贝）；
  `SPLIT=1` + 0 槽（64 MiB）无输出 （错误）。⇒ **缓存的全部收益依赖 split**；
  22 t/s 是假速度，15.9 是长尾里的一次偏快样本（同配置另测 6.5/4.0/3.7/3.4，不可重复）。
- 安全锁（当轮交付，后解除）：`src/llama-graph.cpp` 让 `LLAMA_MOE_SPLIT=1` 拒绝生效并打印原因
  （调试需显式 `LLAMA_MOE_SPLIT=2`）；`tools-run.py`/`run-cur-ref.ps1` 默认改 `SPLIT=0`；
  回归用例 = 上面的 prompt（不加 `--ignore-eos`），正确 751 字符、错误 27 字符，三种长度 27/652/751 ⇒ 部分性竞态。
- 修复（handoff §6.32 ①）：钩子写完两个 leaf 后**立刻重发一次拷贝**
  （`tensor_copy(...)` + `ggml_backend_tensor_copy(ids_gpu_t, dst_ids)`，同理 wgt，几百字节）。
  验证：修复前 3/3 必现 27 字符 → 修复后 **4/4 通过**（与 `SPLIT=0` 参考给出同一段正确答案）；
  安全锁解除，脚本恢复 `SPLIT=1`。
- 修复后历史速度（handoff §6.32②，8k/q8_0、`auto`=97 槽）：缓存关 9.2；
  `AHEAD=3` 为 20.8 t/s、命中 90.9%；`AHEAD=2` 为 18.4，`AHEAD=4` 为 20.2。
  `20.8/9.2−1≈126%` 是该页数字的算术，缺少完全一致的受控 A/B，不是当前 6 GiB／400 请求 token 档的新验收。

**撤回清单（对照 05 章）**：修复前的 SPLIT=1 高值与封板结论撤回；
同一路由错误下的 ahead 曲线，趋势与绝对值都可能失真。修复后同名数值须按自己的日志判定，
不能仅凭“20.3”这个数字断定属于哪一轮。最新本地开发构建的 400 请求 token 结果见 02、05 章；
本次没有重建发布 master，也没有把这些后续结果冒充发布构建验收。SPLIT=0 曾有一次 15.9，
后续仅 3.4–6.5，亦不能当作稳定性能退路。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 本节只处理"速度证据是否成立"；缺陷的复现、二分、竞态机理与回归用例的完整叙述见 05 章 |
| 保留/放弃理由 | 保留修复（`SPLIT=1` 是当前唯一有收益的路径）；保留安全锁的教训（环境变量可以拒绝生效，好过静默出错） |
| 遗留问题 | 同类风险仍在：变长 batch/多图（投机验证）与持久化的 slot-view 补丁（§2.15）；0 槽（如 `CACHE_MIB=64`）时 split 不产出（本该退化成全 CPU） |

### 2.12 H12 prefill 读缓存（D2D 暂存）—— 已撤回

| 字段 | 内容 |
|---|---|
| 路线 ID | H12 |
| 状态/版本 | **已撤回**（master 上 `LLAMA_MOE_PREFILL_CACHED`、`moe_cache_prefill_d2d` 均 0 处，已核查） |
| 为何尝试 | prefill 把"用到的专家并集"从主机 H2D 拷进设备暂存；其中已驻留缓存的部分可改 D2D（缓存只读、不写，符合"prefill 计算期间不搬运"的约束），按 25% 驻留率估算每提示可省 ~4.8 GB ≈ 370 ms |
| 技术机制 | 在 `moe_copy_experts_grouped` 里按驻留性切段：驻留→D2D、非驻留→原 H2D，保持"连续专家合并拷贝" |

**实验条件与证据**（handoff §6.16，交互式两轮自对照，同一提示词，第 1 轮 prefill 冷、第 2 轮暖）：

```
turn 1: Prompt: 37.3 t/s | Generation: ...
turn 2: Prompt: 37.3 t/s | Generation: ...     ← 完全一致，无可测增益
```

原因：prefill 的暂存拷贝用的是 `ggml_backend_tensor_set_async`（异步）→ H2D **本来就与 GPU 计算重叠**，
换 D2D 省不出时间。这同时解释了 §2.13 的"prefill 中性"：prefill 是 GPU 计算受限，权重搬运被藏住了。
→ 已回退（不留无收益的复杂度）。**顺手保留的两项工具能力**：交互式多轮测试
（`hub start` 起 `llama-cli` 不带 `-p`，PTY 下读 stdin，每轮打印 `Prompt: X t/s | Generation: Y t/s`；
注意该构建没有 `-i`/`-cnv`）；`prefill/decode tendency` 诊断行常驻。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 结论是"prefill 侧不需要缓存感知"，不是"缓存对 prefill 有害" |
| 保留/放弃理由 | 放弃实现；保留结论（避免重复尝试） |
| 重新开启条件 | 若 prefill 变成带宽受限（例如权重不再走异步 H2D、或 Q 侧改动破坏重叠），再评估 |

### 2.13 H13 prefill/decode 阶段分离（热集）

| 字段 | 内容 |
|---|---|
| 路线 ID | H13 |
| 状态/版本 | 已发布（`LLAMA_MOE_PREFILL_WEIGHT`、prefill 路由进统计） |
| 为何尝试 | 提示词长度决定冷启动成本；两阶段的访问集中度可能本质不同，用同一套策略对待两者可能是错的 |
| 技术机制 | 新增诊断：prefill 侧计数在 `used_ids` 解析处采（**分区钩子只跑 decode**，原先 prefill 路由完全不可见）；两侧分开统计并以 Q8 定点计数（decode 行 = 256，prefill 行 = 256×`PREFILL_WEIGHT`），衰减/淘汰/回填排序尺度不变 |

**实验条件与证据**（handoff §6.14，同配置、仅提示词不同；改动后命中率 70.4% 与改前一致 ⇒ 无回归）：

| 工作负载 | decode topC | 提示词热集可覆盖 | prefill 触达/层 | decode 触达/层 |
|---|---|---|---|---|
| 6-token 短提示 | 85.8% | **17.9%** | 150.4 | 211.6 |
| ~150-token 长提示 | 72.2% | **52.0%** | 209.8 | 247.8 |

结论：prefill 并行 → 每层触达 ~82%（209.8/256）、分布近乎平坦；decode 串行 → 集中在前 64 名
（覆盖路由 72.2%）；提示词当冷启动种子的价值随长度剧增（17.9% → 52.0%，= 可达上限 72.2% 的 72%）；
落地 prefill 路由进统计 + `LLAMA_MOE_PREFILL_WEIGHT`（默认 1.0，长提示场景建议 0.1；
本工作负载下 0.1 与 1.0 命中率一致 60.3%，因为 prefill 质量本就只占 ~13%）。

SMoE/缓存对 prefill 的影响（handoff §6.15，长提示、`-n 8`）：

| 配置 | Prefill (Prompt t/s) | Decode (Gen t/s) | VRAM |
|---|---|---|---|
| 全开（SMoE+缓存+回填） | 44.5 | 11.8 | 9257 |
| SMoE 关（`PREDICT_SMOE=0`） | 44.4 | 12.1 | 9255 |
| SMoE 关 + 缓存关（`CACHE_MIB=0`） | 44.7 | 8.9 | 7223 |
| 全关（stock `--cpu-moe`） | 42.7 | 8.7 | 7223 |

⇒ prefill 不被这套机制拖累（44.5 vs stock 42.7，差值为噪声）；decode 收益来自**缓存**；
prefill **不进缓存**（走 `moe_copy_experts_grouped` 拷进调度器输入，不污染缓存）——这是正确行为，
两阶段该分离的是**统计与用途**（先验/种子），不是准入。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 表格里的 decode 数字（11.8/8.9/8.7）是 `-n 8` 短生成、受冷启动填充支配，不能当稳态速度引用 |
| 保留/放弃理由 | 保留（低风险、无回归）；"把本次 prefill 的 top-C 当 seed"留作未做项 |
| 遗留问题/重新开启条件 | 未做：用 prefill 热集替换/叠加 `LLAMA_MOE_PIN_STATIC` 的静态清单（上限已测得 52%）。重新开启条件：面向"长提示 + 短生成"场景 |

### 2.14 H14 PLE 分层缓存 —— 完整档案已前置到 §1A

本节只保留与 **256k + 视觉编码器 + TBQ4 KV** 配置相关的补充数据；机制、动机、早期测量、
被 MoE 预算挤掉的时点与状态标注见 **§1A**（第一阶段单列）。

- 256k 显存账（handoff §6.29，实测 14442/16384 MiB）：MoE 缓存 3.9 GB + PLE-GPU 1 GB（建议关）
  + 视觉编码器 0.85 GB + 非 MoE 权重/KV(256k,TBQ4)/compute ≈ 9 GB ⇒ 基座 ≈11 GB。
  **TQ4 256k 本身没有超预期**（≈9 GB 里的大头是权重与 compute buffer）。
- 预算日志的名字与实际参数不同：`auto-256k-vis` 是显式请求 8192 MiB 后 clamp，
  得 40 槽／3881 MiB／64.6%／13.2 t/s；cap2700 的两次为 28 槽／56.5–58.4%／14.7–15.3 t/s。
  槽数多 **12**，但该高预算样本速度更低；不能据此宣称 auto 速度更好或已有稳定配对收益。
- 同元素数下，`iq4_nl` 4.5 bpw 对 `iq2_s` 2.5625 bpw 的字节增幅约 **75.6%**；
  旧 47% 来自另一个分母，不能沿用。降低 bpw 是候选方向，不是已经穷尽其他布局方案。
  裸算子原日志显示 `12.62 / 11.70 / 32.00 GB/s`，工具实际按 GiB/s 算；
  不直接与 PCIe 十进制带宽或 token 吞吐比较，详见[03 章](03-weight-quantization-and-kernels.md)。
- `LLAMA_PLE_GPU_CACHE_TIMING_FILE` / `LLAMA_PLE_GPU_PREFETCH_FILE` / `LLAMA_PLE_PREFETCH_DEBUG_FILE`
  是 PLE 侧的诊断出口（`src/models/qwen4exp.cpp:589/604/719/2125`），与 §1A.2 的统计口径配套。

### 2.15 H15 MTP / 投机 × 专家缓存 —— 已撤回

| 字段 | 内容 |
|---|---|
| 路线 ID | H15 |
| 状态/版本 | **已撤回**：`ggml/src/ggml-backend.cpp` 回退到封板提交 `c08171aa8`；后加层注册、`mtp_mode`、 slot 越界诊断、临时脚本全部移除。核查：master 上 `LLAMA_MOE_LATE_LAYERS`、`moe_cache_finalize_new_layers`、 `mtp_mode` 均 0 处 |
| 为何尝试 | 本 MoE-cache 设计在 host 路径已到平台期（§2.9 #14 的天花板实测），再上台阶要换轴： fork 里已有 NextN/MTP draft head（`model : add the qwen4exp NextN/MTP draft head`、 `qwen4exp: allow loading a draft-only MTP export`） |
| 技术机制 | MTP 层 = `blk.48`（`n_layer_all=49`、`n_layer_nextn=1`），自带 512 专家 MoE，权重 ≈1750 MiB （每专家 ≈3.4 MiB；64 槽 ≈220 MiB）；其张量名与 trunk 同构（`ffn_moe_*-48`），缓存按名匹配本身兼容。 缓存布局在"第一张单 token 图"处冻结（`moe_cache_ensure` finalize 后一律 null）⇒ `blk.48` 永不入缓存， 于是实现"后加层"路径：`moe_cache_finalize_new_layers`（紧跟 `moe_cache_init`，避开 CUDA graph 捕获窗口） 在权重种类集合稳定后建 bundle（→ `alloc_persistent_layer`，槽位上限 32），建好之前 `ensure` 一律返回 nullptr |

**实验条件与证据**：

- MTP 本身可用（handoff §6.26）：
  `llama-cli` **不支持投机解码**（`tools/cli` 无 `common_speculative` 调用），`-md` 时 draft 加载失败后
  **静默退化**为普通生成（实测 13.3 t/s）——这是"加载 draft 报错"的真因。
  正确跑法 = `examples/speculative-simple`：
  `llama-speculative-simple.exe -m <IQ3_XXS 00001> -md <mtp-...-Q4_K_M.gguf> --spec-type draft-mtp
   -ngl 49 --cpu-moe --no-mmap -c 8192 -n 64 --temp 0 --spec-draft-n-max 8`，
  实测 `n_drafted=27 n_accept=27 accept=100.000%`、文本正确、**无缓存 8.55 t/s**
  （draft 头是 MoE 层且也走 CPU ⇒ 不投机反而更慢）。参数：`--spec-draft-n-max`（`--draft-max` 已废弃）；
  draft 借用 target 的 `token_embd/output`（`nextn_shared_target_tensors` + `borrow_shared_tensor`，
  需要 `ml.model_shared`）。注意： fork 内已知 bug：`common/speculative.cpp:2547` 把 draft 路径取到
  `model_path` 后，实际加载用的是 `params.model.path`（target）。
- 阻塞点（§6.26/§6.27）：`llama-speculative-simple` + 缓存，在**目标自己的图**里（约 17–19 s，draft 尚未跑到）
  必崩 `CUDA error: an illegal memory access`（`ggml_cuda_kernel_launch`）。二分：

  | 二分项 | 结果 |
  |---|---|
  | `CACHE_MIB` only（**不开 SPLIT**） | 跑通（`decoded 19 tokens in 2.154 s`） |
  | SPLIT + direct | （错误） CRASH |
  | SPLIT + 无 direct（gather） | （错误） CRASH |
  | SPLIT + `CPU_ASYNC=0` | （错误） CRASH |
  | SPLIT + 关 MRS/prefetch/SMOE | （错误） CRASH |
  | `LLAMA_MOE_LATE_LAYERS=0/1` | （错误） 都 CRASH（与后加层无关） |

  ⇒ 破点收敛到 **GPU/CPU 拆分路径本身**（`moe_split_partition` + CPU 半边 leaf/激活拷贝/ids 回读），
  与 direct 补丁、异步 CPU 半边、可选特性、MTP 层均无关。
- 诊断（§6.27 补充）：`compute-sanitizer --tool memcheck` 跑到底**没有内存报告**，
  只给 `CUDA error: unknown error`（42 s 处，`cudaStreamSynchronize`）；配合 `CUDA_LAUNCH_BLOCKING=1`
  时错误停在 `ggml_cuda_kernel_launch`（`common.cuh:1700`）⇒ 更像**内核参数/指针非法**。
  `LLAMA_MOE_DUMP_CH=1` 检查 GPU 半边 slot 索引越界 ⇒ 实测 **0 次越界**（排除）。
  `SPLIT=0` + 缓存 + MTP **能跑但只有 8.8 t/s**（无缓存 8.55）⇒ 不拆分的缓存对 MTP 几乎没收益。
- 决定与放弃理由（§6.28）：唯一触发条件是 `LLAMA_MOE_SPLIT=1`；而缓存在不拆分时对 MTP 几乎无收益；
  修它属于独立的结构性工作，性价比不足以下注。且**该缓存从未与投机前端共存过**
  （只在 `llama-cli` 固定单 token decode 图上验证过）——这是既有限制，不是本轮引入。
- 留下的可用知识：MTP 唯一可用前端 = `llama-speculative-simple`；draft 是串行瓶颈（8.55 vs 13.3）；
  draft 层规模与命名；恢复路径（先修"拆分路径 × 投机前端"的崩溃，重点怀疑 `cur_cpu` 激活 D2H 字节数、
  `ids_cpu/wgt_cpu` 主机 leaf 尺寸、`node->src[0]=input_cpy` 补丁在变长 batch 下的有效性）。
  本轮 MTP 测量**无独立原始日志文件名**，只有上述报告内摘录（handoff §6.26–§6.28 的命令与输出）。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | "MTP 不行"是错的：MTP 本身 accept 100%、文本正确；不行的是"缓存 × 投机前端"组合。也不要把 8.55 t/s 当作投机收益——draft 全程在 CPU 上跑，比目标单独 13.3 t/s 还慢 |
| 保留/放弃理由 | 已撤回（WIP 全回退）；保留调查结论备查 |
| 遗留问题/重新开启条件 | 重新开启条件：变长 batch/多图下 slot-view 补丁的失效问题先解决（与 §2.11 同类风险），或先做 D01 算子（显式持有 buffer，不再与分配器斗） |

### 2.16 H16 自适应准入（yield 门槛 / 极值搜索 / trend 回归 / ahead 自动）

> 双门控（价值门槛 + 速率预算）的动机与实测见 §2.17；本节记录控制器本身的实现与收敛证据。

| 字段 | 内容 |
|---|---|
| 路线 ID | H16 |
| 状态/版本 | 已发布，全部默认关（`LLAMA_MOE_YIELD_AUTO`、`LLAMA_MOE_TREND_AUTO`、`LLAMA_MOE_AHEAD_AUTO`、 `LLAMA_MOE_ADMIT_BUDGET_MIB` 等）；默认行为与引入前一致（已回归验证） |
| 为何尝试 | 命中的边际价值是状态相关的（CPU 半边有活时一次命中 ≈0.074 ms；命中率 ~80% 后边际价值 ≈0）， 传输的边际成本始终存在（≈0.08 ms/MB）⇒ 最优深度在"边际价值穿过边际成本"处，且随分布/提示词/缓存状态/阶段移动 |
| 技术机制 | 三条并列控制器：① 极值搜索（决策变量 = 每字节效率门槛 hits/MiB，目标 = 中位每 token 耗时， 每 16 图比较前后窗口，步长乘性 ±10%，钳 [0.5,32]）；② 模型驱动（每图采 `(ms, hits, MB)`， 在 32 图滑窗上对 `ms ≈ a − V·hits + P·MB` 做中心化最小二乘，Cramer 解，EWMA 0.75/0.25 → 门槛 = P/V、预算 = frac×ms_hat/P）；③ ahead 极值搜索（目标函数 = **窗口增量命中率**） |

**实验条件与证据**：

- 门槛/预算的实测与教训（handoff §6.7 ③/§6.8/§6.9）：
  按准确率门槛（`RANK_ADAPTIVE`）鸡生蛋（窄 cutoff 只 offer 少数排名 → 统计饿死；已修：准确率按**完整候选表**测，与准入解耦）；
  按实测 yield 门槛（`YIELD_MIN=4`）r1/r2 实测 17.9/16.7 ≫ 4，但更深排名无字节 → yield=0 判 0 → cut 卡在 1
  （**意外收获**：cut=1 + 省下的传输 = 20.6 t/s > cut=2 的 19.3）；
  用"准确率 × 实测单次准入命中数"估 yield + 字节预算：64 MiB 预算 → 12 t/s（早层吃光、后层饿死），
  128 MiB → 19.0 t/s，**不如 1/2**。
  分片小于 1 个 bundle（1.9 MB）会把准入全部拒掉（64 MiB/48 = 1.33 MB → 12 t/s）⇒ 分片下限必须 ≥1 bundle。
- 占用率观测（用户实测）：bf8/bf16 把 decode 占用率从 40–45% 抬到 60%，此前"100%"其实是预填充满负载
  ⇒ **"GPU 40 ms 硬底 / 25 t/s 天花板"的旧估计作废**；`ADMIT_BUDGET_MIB` 会把占用率钉在中平台
  （它限速预取准入 → 工作集增长被限速）。结论：**价值门槛（哪些排名值得）与速率控制（每 token 多少）
  要分开**——速率交给回填，门槛只管价值。
- 极值搜索收敛（§6.8，400 token、6 GB+回填 8）：prose `67.8 → 38.7 ms`（门槛 4.4 → 10.9），
  收敛 cut=1、yield_min=11.82、18.8 t/s；code的约4.3–4.8是yield门槛，不是毫秒 ⇒ 两个分布得到不同门槛
  （人工常数 4 在 prose 上偏松 3 倍）。
- trend 模型（§6.9，400 token、6 GB+回填 8、frac=0.25）：
  `trendV=0.0158 ms/hit`、`trendP=0.0936 ms/MB` → 门槛 = P/V = 5.92 hits/MB（开 cut=4）、
  预算 = 116.5 MB；`fits=191 rej=212`；19.6 t/s（同配置历史 18.8–19.9，无回归）。
  算术更正：0.0936 相对 0.073 高约 **28.2%**；相对 0.08 才是 17%。跨工作点近似同量级，不是独立确认硬件带宽。
  守卫：窗口 <16 点、散布不足（hits 散布 <15% 或 MB 散布 <8%）、行列式过小、解越界 → 拒绝该次拟合。
- ahead 自动（§6.32 ④）：目标函数必须是窗口增量命中率（累计命中数/累计命中率会被缓存预热带偏——
  实测控制器一路爬到钳位值 6 再反向走到 1）；修正后在 **3↔4 之间来回探测**（4×19、3×18、5×6、6×4、2×1）
  = 正确找到最优区；但**固定默认 3 仍略优**（20.8 vs 19.1 t/s）⇒ 控制器默认关闭、留作选开。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 全部信号都来自主机分区钩子 ⇒ 与 devpart 天然冲突（H10）；ahead 曲线与 yield 相关实验多在 `SPLIT=1` 静默错误期或旧工作点完成 ⇒ 控制器本身可用，但"最优点"的绝对值要按 §6.32 复测口径引用 |
| 保留/放弃理由 | 保留实现（默认关）；默认配置 = 固定 `SMOE_AHEAD=3` + `HOT_BACKFILL=8` + 默认 `rank_cut=2` |
| 遗留问题/重新开启条件 | `PREFETCH_JOIN=1` 路径崩（默认 0，暂不用）；`HOT_IDLE` 空闲线程未做端到端验证（需要"生成→idle→再生成"的交互式会话） |

### 2.17 H17 “命中更高却更慢”与双门控自适应传输阈值

| 字段 | 内容 |
|---|---|
| 路线 ID | H17 |
| 状态/版本 | 已发布的实验控制接口：时限门默认开；YIELD_AUTO / TREND_AUTO 等价值自适应默认关。完整判据见[02 章 PC-24](02-prediction-and-cache.md) |
| 为何尝试 | 不能只追最高命中；要区分每字节预取的价值和能否赶上消费 |
| 技术机制 | **价值门**按排名 yield 与门槛决定 cutoff；**时限门**比较预计在飞＋本批传输时间和层距离×每层耗时 EWMA。`P/V` 可产生价值门槛，`frac×ms_hat/P` 是独立字节预算，不是第二门 |

**历史观察**（handoff §6.6–6.9，不是新的受控验收）：

| 预取量 | 命中率 | total | t/s | flag_input |
|---|---|---|---|---|
| 161 MB/tok | 40.2% | 70.0 ms | 14.3 | 15.8 ms |
| 254 MB/tok | 46.9% | 77.7 ms | 12.9 | 23.3 ms |
| 464 MB/tok | 58.2% | 94.1 ms | 10.6 | 37.5 ms |

- 观察是“量、命中、等待一起上升而速度下降”；具体争用机制仍是解释，不是硬件 profiling 结论。
- 一次命中的 0.074 ms、较高命中后边际价值减小，是当时工作点的估计，不是固定收益或普遍的 80% 阈值。
- 回填在小容量下可产生 thrash；旧 6 GB／回填 8 的 80.8%／23.3 t/s 来自后续发现有路由错误的阶段，
  **趋势和绝对值都不能作为正确路径性能依据**。
- TREND 一次拟合得 `V=0.0158`、`P=0.0936`、`P/V=5.92`、预算 116.5 MB，
  `fits=191/rej=212`，当次 19.6 t/s。它说明控制器产生了这些估计，不证明已优于所有固定参数。
- 时限门检查有适用范围：`prediction && prefetch_gate && !queue_on_worker`；
  冷启动没有有效层耗时历史时放行。hot-backfill / warm / seed 不能被概括成全受双门控。
- `70 µs/copy`、`20 GB/s` 为模型常数；层耗时与在飞量随运行变化，不是在线实测出 20 GB/s。

| 字段 | 内容 |
|---|---|
| 保留理由 | 保留“值得传／赶得上／每轮可传多少”的分解；不同控制器的默认与单次表现分别记录，不能统称双门控默认关 |
| 遗留问题 | 价值估计会受预热、共线性和输入分布影响；字节预算可能偏向早层；小于一个 bundle 的预算不可形成一次有效准入 |
| 重新开启条件 | 先在正确且同输入的工作点确认预取是主要成本，再标定模型常数、检查旁路与比较固定阈值 |

## 3. 设计但未实现的路线（`moe-decode-perf-plan.md` 与 `rebuild-spec.md`）

### 3.1 D01 独立 MoE 算子（借壳出算子）

| 字段 | 内容 |
|---|---|
| 路线 ID | D01 |
| 状态/版本 | **仅设计**。核查：master 与源工作区 HEAD 中 `GGML_OP_MOE_QWEN4EXP`、`moe_qwen4exp` 均 **0 处**；落地的只有设备侧分区 kernel `ggml/src/ggml-cuda/moe-partition.cu`（81 行）与原语 `GGML_OP_MOE_PARTITION_IDS/WGT`（`GGML_OP_MOE_PARTITION` 在 master 11 处） |
| 为何尝试 | 撞车的只有 `ggml_backend_sched`：它同时持有 tensor 生命周期、执行顺序/边界、同步点，而这三样必须由 MoE 执行自己持有（perf-plan §0.1；rebuild-spec §2.2）。目标是把 master 上 `ggml-backend.cpp` 的 MoE 缓存/split/devpart 调度钩子整段搬出去 |
| 技术机制 | 算子签名（7 个 src，`GGML_MAX_SRC=10` 放得下）：`GGML_OP_MOE_QWEN4EXP(cur, selected_experts, weights, gate_exps, up_exps, down_exps[, candidates])` → 与 GPU 半边 + CPU 半边之和等价的 device 张量。共享专家**不进算子**（只依赖 `cur`、GPU 常驻、是 SMoE 输入之一）。引擎为**持久对象**（跨层跨 token 存活）：自有 compute/prefetch stream、专家缓存、自有 CPU 线程池、预取队列与事件；入口 `llama stream --event--> 我们的 stream`，出口反向**只记一个 event**。五类"打断源"逐一封死（split 边界拷贝 / CUDA graph 捕获显式排除 / gallocr 不接管缓存 buffer / 调度器事件只在出口一次 / 后端归属显式断言）。`ggml_backend_sched_compute_splits` 回到接近上游状态 |

**阶段划分（两套编号，均未执行）**：
perf-plan §0.1：M0 空算子（行为零变化，只证明"缝"存在）→ M1 搬缓存+分区 → M2 栅栏换型
（`synchronize` → `event wait`、自持 stream、批量 D2H，86 → ~46 ms）→ M3 搬预取/预测、重写准入。
rebuild-spec §8：S1 干净 worktree + 移植清单 → **S2 差分 oracle**（一条命令跑旧 vs 新并 diff token id）
→ S3 `M0` 朴素正确算子（`LLAMA_MOE_OP=1/0` 逐位一致）→ S4 缓存+分区搬入（删掉钩子）
→ S5 栅栏换型 + 自持 stream + 批量 D2H（decode ≤ 40 ms）→ S6 预取/预测搬入。
S1a（`GGML_OP_MOE_CPU` + `moe-partition.cu` + prefetch 后端接口）是 M0 的前置。

**为什么没有继续**：

1. **清空线已被放弃**：`OTHER_SOURCE_TREES/llama-qwen4exp-clean`（worktree @ `b76199698`）里的 `clear/` 迁移快照
   （TQ4/MoE/PLE 源码 + docs + tools + patches，35 文件）**不自洽**（0.7 s 加载崩溃），
   且 HEAD 本身跑不动这个模型（handoff §2）。快照只作为"我们的代码清单"参考。
2. **优先级被 host 路径的收益吃掉**：在不动算子的前提下，host 路径从 11.3 t/s 提到 ~20 t/s（§2.11 修复后口径），
   而算子能拿回的正是剩下的两段结构开销（17–19 ms 会合 + 13.1 ms split 启动），
   其中devpart的旧host对照已失效，split合并又遇到显存障碍；两段都不能直接计入可兑现收益
   ⇒ 算子的边际收益变得不明确。
3. 设计自身的门槛（rebuild-spec §2.2 护栏指标：`git diff -- ggml-backend.cpp` 归零、`llama-graph.cpp` 只剩构建算子的几行）
   是架构目标，不是测量；没有"测量先行"的收益证明就不值得下注。
4. **正确性优先**：S2 差分 oracle 的存在理由就是"上一版每一版都写得比验得快"（devpart 是三个版本前的东西，
   今天仍在产生乱码和崩溃）——在 oracle 建成之前推进 M0 会重复同样的失败模式。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | "借壳"能消掉的东西（devpart 生命周期 bug、每 split 跨后端拷贝、`insert_flush`、195 次 generic 拷贝）里，前两项已被 host 修复或 devpart 实验覆盖；`insert_flush` 实测 0.07 ms 不成立 |
| 重新开启条件 | 出现"host 路径无法再涨、且瓶颈明确落在算子边界"的证据（例如需要跨层批量 D2H 或 SSD tier） |

### 3.2 D02 内存层级路线图（RAM+VRAM → +SSD）与两条硬约束

| 字段 | 内容 |
|---|---|
| 路线 ID | D02 |
| 状态/版本 | **仅设计**（rebuild-spec §2.3）。PLE 侧的跨层接口已发布（`copy_pages`/`prefetch_async`），MoE 缓存侧未实现 |
| 为何尝试 | 第二阶段要加 SSD tier（更大模型 / 更小 RAM）；若第一阶段不预留抽象，第二阶段要再改一次数据流 |
| 技术机制 | 模块划分：不是"一个大 MoE 算子"，而是"分层分页权重存储 + 预取引擎（PLE 与专家**共用**）+ 预测器 + MoE 算子"。差异只有三项：页大小（63 KiB vs ~2 MB bundle）、访问模式、预测器；淘汰策略、tier 管理、异步预取 + 事件栅栏**完全同构** |

**两条硬约束（现在就要满足，否则第二阶段重来）**：

1. **预取的 copy 源必须抽象化，不得写死 host 指针**——现在 `moe_cache_copy` 直接用 `input->data`（pinned host），
   第二阶段源头是 SSD，必须是 `copy_pages` 式的"从下一层物化一页"接口。
2. **提前量必须参数化，不得写死 1 层**——`RAM→VRAM ~2 MB @10 GB/s ≈ 200 µs`（1 层够）；
   `SSD→RAM ~2 MB @3 GB/s ≈ 700 µs + 延迟`；两跳叠加需要 2–3 层甚至跨 token 的提前量。
   现有 `deadline = layers_until_visit * layer_us_ewma` 是**单跳且按层数**的模型，重建时要写成多跳、按剩余时间。

**实现状态核查（master）**：约束 1 **未满足**——`moe_copy_experts_grouped` 仍从 `input->data`（主机指针）
发起 `ggml_backend_tensor_set_async`；约束 2 **未满足**——`ggml/src/ggml-backend.cpp:4015` 的
`moe_prefetch_feasible` 仍是 `eta_us = (dma_inflight_copies + n_copies) * gate_copy_us + bytes/gate_bw_bps`
与 `deadline_us = max(1, layers_until_visit) * layer_us_ewma`（默认 `gate_copy_us = 70 µs`、`gate_bw_bps = 20 GB/s`）。
没有 SSD 相关代码；PLE 的 `copy_pages` 是唯一"可换层"接口。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | PLE 与本缓存是"同一件事的另一份独立实现"这一判断只由接口形状与单元大小支撑，没有做过合并实验 |
| 重新开启条件 | 出现 RAM 不足的部署目标（如换更大模型），或 D01 动工时 |

### 3.3 D03 专家聚类存储

| 字段 | 内容 |
|---|---|
| 路线 ID | D03 |
| 状态/版本 | **仅设计**（rebuild-spec §2.4）。无聚类工具、无置换表、无加载器映射、无按 storage_slot 键的缓存 |
| 为何尝试 | 现有准入判据的主导项是 `n_copies = 3`（gate/up/down）每次 150–400 µs 的固定开销；聚类后一次传输覆盖一整簇共激活专家 ⇒ `n_copies` 3→1，且 SSD 阶段从随机读变顺序读，直接对症 `prefetch_dropped=8092`（65%） |
| 技术机制 | 纯置换：专家按共激活关系聚类存放，**量化类型不变、张量形状不变，只是 expert 维被置换** ⇒ ggml kernel 原样可用，只需多一层 id 映射（`logical_id --置换表--> storage_slot`，置换表 48 层 × 512 专家 × 2 B ≈ 48 KB）。**不写自定义 GEMM**；变长簇 / 按簇分精度才需要自定义布局，本方案不需要。簇大小待实测定夺（2–4 专家 ~4–8 MB 命中精准但 per-copy 开销仍在；8–16 专家 ~16–32 MB 摊销开销但带进不需要的专家），初估 4–8 专家/簇 |

**第二条约束（与 D02 相关，未满足）**：逻辑 expert id 与存储位置必须分开两层抽象；现在混用
（`expert_slot[expert_id]`，master 上缓存按逻辑专家号键）⇒ 一旦混用，第二期就得改缓存核心。
诚实说明：纯置换下 llama 的 GGUF loader **仍然可用**，"逼我们单干的是调度器，不是权重格式"，
因此两期可解耦（第一期引擎、第二期聚类布局 + 自有容器）。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | "聚类正好打当前瓶颈"是基于 per-copy 固定开销（150–400 µs）的推断；本项目后来测到的边际成本是 0.073 ms/MB ≈ 13.8 GB/s（偏带宽而非纯固定开销），摊销收益的上限因此需要重新评估 |
| 重新开启条件 | 预取仍是瓶颈且需要 SSD tier 时；或先把 `n_copies` 3→1 用"同层三张权重共一次传输"的方式验证收益 |

### 3.4 D04 共享专家前置 + SMoE 提前开火

| 字段 | 内容 |
|---|---|
| 路线 ID | D04 |
| 状态/版本 | **仅设计/未实施**。核查：master 上 `src/models/qwen4exp.cpp:1993-2000` 仍是 `smoe_hidden = ffn_input + smoe_gpu_out + ffn_shexp_gated` |
| 为何尝试 | perf-plan §P1.4 判为"杠杆最大的一条"：现状共享专家 FFN 在 MoE 之后构建，SMoE 只能在 FFN 末尾开火， 提前量 ≈ 0；共享专家只依赖 `cur`，与 routed MoE 无依赖边，本可以先算。取舍：用略低的准确率换高交付率 （预测覆盖与按时交付还需合并评估；旧“73%×34%=26.8%”算术不成立，且两项分母未证明可直接相乘） |
| 技术机制 | 改序：`attention → [共享专家 FFN 先算] → SMoE 开火(input + shared) → 预取 L+1 → routed MoE`， 提前量从 ~0 变成整个 routed MoE 的时长 + 下一层 attention；连带可删 SMoE side graph 与 `moe_cache_smoe_enqueue/drain` 的逐层 `event_synchronize`（实测 17 ms/token，而 CPU 处理部分只有 2 ms） |

**离线支撑（`smoe-nk-degradation-plan.md` §7，8 prompt、`ffn_moe_input` 白名单、56400 样本）**：
oracle 全 100%；**full 68.53%**、`shared_only 68.07%`、**`input_only 67.90%`**（k=1）
⇒ 该数据上 `input_only` 与 full 差 **0.63 pt**，支持继续测试删依赖的可能。
k=2/3/4 为 59.94/55.70/53.77%，支持继续验证提前量，不证明两层窗口在线“完全可用”。
这些是单步 teacher-forced 结果；只有实际计算路径保持不变等前提成立，预测提示才不把近似激活直接反馈进模型。未验证累计在线影响。

**为什么没有继续**：这条路线要同时改三件高风险的事（SMoE 输入简化、共享专家改序、删 side graph），
而 17 ms 的等待已被"加提前量"以低得多的风险解决（§2.3：ahead 默认 3 → 命中 90.9%、20.8 t/s）。
在本项目的验收门槛下，没有证据表明还能再拿一档；且它与 D01 的边界划分（共享专家留 llama、SMoE 不进算子）
纠缠，属于应当与算子一起做的改动。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 68.53% vs 67.90% 是**离线单步**结果；线上开启前需要验证"提前量变大后实际交付率与命中率的变化" |
| 重新开启条件 | 需要进一步压低预取/等待开销时，或 D01 动工时一并做 |

### 3.5 D05 准入模型重写（`gate_copy_us` 标定 / 剩余时间 deadline）

| 字段 | 内容 |
|---|---|
| 路线 ID | D05 |
| 状态/版本 | **部分设计，未采用**。核查：master `ggml/src/ggml-backend.cpp:4015-4024` 仍是旧模型； 环境覆盖 `LLAMA_MOE_GATE_COPY_US`/`LLAMA_MOE_GATE_BW_GBPS` 存在（`:2725`/`:2728`）但**没有任何脚本标定它们** |
| 为何尝试 | 旧模型两个问题：`gate_copy_us = 70 µs` 从未标定，而实测每次小传输 150–400 µs（低估 2–5 倍）； `deadline` 按"层数"算，而真实约束是"剩余时间"，且 SMoE 在 FFN 末尾才开火、提前量 ≈ 0 |
| 技术机制 | perf-plan §P1.1/P1.2：deadline 改剩余时间模型、标定 `gate_copy_us`； rebuild-spec §4 追加"前置条件按 slot 粒度（不得用 `insert_flush` 式全空等待）"与"小传输合并" |

**实际走的路**：项目没有重写 deadline 模型，而是（a）显式化排名截止线 `rank_cut`（§6.6 ③，并修掉
`n_used==0` 让截止线静默消失的 bug）；（b）去重（52.5% 候选不花字节）；（c）热区回填（无时限、不受预算限制）；
（d）`SMOE_NONBLOCK=1 + AHEAD=3` 增加提前量；（e）用 §2.16 的两条在线控制器把"门槛/预算"变成观测量。
两种工作点给出约 0.073 与 0.0936 ms/MB 的估计；后者相对前者高约28.2%。这是成本量级参考，不是 PCIe 带宽或因果机制的独立实测。

| 字段 | 内容 |
|---|---|
| 观察与结论边界 | 旧常数与 deadline 模型有待标定；预取、等待和速度的联合变化提示需要继续调查，但尚未隔离各自的因果成本 |
| 重新开启条件 | 预取重新成为主瓶颈时（例如命中率已到 ~80%、准入已不是限制） |

---

## 4. 已撤回 / 未发布汇总（引用前必查）

| 对象 | 状态 | 依据 |
|---|---|---|
| 数字 `20.3 t/s`（封板）、`20.2 t/s`（封板复核）、`20.1–22.0 t/s` | **作废** | §2.11（`SPLIT=1` 静默错误期间的假速度） |
| 数字 devpart `19–23 / 26.0 / 28.2 / 29.7 / 32.0 t/s` | **作废** | §2.9（CPU 半边数据为零的中间实现） |
| 数字 `20.7 t/s` | 作废 | perf-plan §7（乱码旧构建） |
| 假设"`insert_flush` 每层排空 = ~20 ms" | 作废 | §2.4（实测 0.07 ms） |
| 假设"`split_partition` 的 host CPU 循环 = 23 ms" | 作废 | §2.5（实为等 GPU 产出 router） |
| 假设"pinned 化能消除会合" | 作废 | §2.2（等待只是搬家） |
| "worker 数不是瓶颈" | 作废（结论范围） | §2.6（`INSERT_WORKERS` 在 `PREFETCH=1` 下空转） |
| prefill D2D 暂存（`LLAMA_MOE_PREFILL_CACHED`、`moe_cache_prefill_d2d`） | **已撤回**（master 0 处） | §2.12 |
| MTP×缓存后加层（`LLAMA_MOE_LATE_LAYERS`、`moe_cache_finalize_new_layers`、`moe_cache_build_late_layer`、`mtp_mode`） | **已撤回**（master 0 处，代码回退到 `c08171aa8`） | §2.15 |
| split 合并（去掉调度器 pass 5 规则） | 实验后回退（VRAM OOM） | §2.8 |
| `SPLIT=1` 安全锁（拒绝生效） | 已解除（§6.32 修复后） | §2.11 |
| `GGML_OP_MOE_QWEN4EXP` / SSD tier / 专家聚类 / PLE 进算子 | **仅设计**（0 处实现） | §3.1–§3.3 |
| PLE 分层缓存本体 | **已发布**（上游 `4e1865e34` + fork `2f1a363c8`）；默认关（env 不设即关）；**full RAM 工作点收益小、lazy/mmap 工作点有效** | §1A |
| "PLE 整体失败 / PLE 一上线就被 MoE 替换" | **错误表述，禁止**：两者长期共存（09-03…09-10），PLE 在显存/预算紧张配置下才被挤出 | §1A.3 |
| "teacher 测试命中率 99% 被离线 recall@10 否定" | **错误推论，禁止**：两者指标/分母不同（详见 §0.5） | §0.5 |
| D04 共享专家前置 + SMoE 提前开火；D05 deadline 模型重写 | 仅设计/未实施 | §3.4、§3.5 |
| `rebuild-spec.md` §9 的 CSV `-5` 行（inflight 峰值） | 未实现（无 `inflight_copies_peak`；只有 `prefetch_dropped` 计数） | 本章核查 |
| 源工作区未提交的NXQ/TBQ、缓存与工具改动 | 仅WIP，不在发布基线 | §0.1 |
| 保留但未验证：`LLAMA_MOE_HOT_IDLE` 空闲线程、`PREFETCH_JOIN=1`（崩）、0 槽 split（无输出）、`AHEAD=1` 侧图 off-by-one（已定案为晚一拍） | 仍开放 | §2.3、§2.16 |

---

## 5. 与 05 章（正确性与方法论）的关系

本章只记录"`SPLIT=1` 静默错误对**速度证据**的影响"：撤回 20.1–22.0 t/s 系列与封板值 20.3/20.2，
把 400 token / 6 GiB 口径标为未复核，并强调"必须不带 `--ignore-eos`"这一测量纪律的来源。
缺陷的复现步骤、二分表、竞态机理、修复的一行级 diff 与回归用例归 05 章；
本章 2.11 只保留摘要与指针，避免两处叙述互相覆盖。

同类风险（供 05 章交叉引用）：变长 batch/多图下持久化 slot-view 补丁的失效（§2.15 的 MTP 崩溃）、
0 槽时 split 不产出（§2.11 遗留）、退出期 `0xC0000005`。非零退出不是“只丢统计”；根因和影响范围未闭案，见05章。

---

## 6. 遗留问题与重新开启条件

| # | 问题 | 现状 | 重新开启条件 |
|---|---|---|---|
| 1 | 发布基线的400-token／6GiB／full-RAM验收 | 本地候选记录不等于master重新构建验收 | 后续另立同条件对照，本次只归档 |
| 2 | devpart驻留决策依赖主机路径 | 默认关；历史16.8不能与已撤回host成绩判净负 | 若重开，先闭合驻留／预取决策链 |
| 3 | devpart 与修复后 host 的同口径对比 | 缺失（对照 20.3 已作废） | 同配置重测 |
| 4 | 逐层router会合 | 已测局部改法未建立净收益，不等于排除全部结构改法 | 需新的结构方案与受控对照 |
| 5 | split分段与合并 | 13.1 ms不是全额可消除的启动税；本次合并OOM | 缩小cache再合并的交换尚未测量 |
| 6 | 投机前端 × 缓存崩溃（`SPLIT=1` 唯一触发） | WIP 已回退；缓存从未与投机共存放 | 先做 slot-view 补丁按 (graph, shape) 失效，或 D01 算子 |
| 7 | 0 槽（`CACHE_MIB=64`）时 split 不产出 | 未修（本该退化成全 CPU） | 修 split 路径时一并处理 |
| 8 | `LLAMA_MOE_HOT_IDLE` 空闲线程 | 已发布但未端到端验证 | 交互式"生成→idle→再生成"会话 |
| 9 | `PREFETCH_JOIN=1` 崩溃 | 默认 0 | 单独排查该路径 |
| 10 | 退出期偶发 `0xC0000005`（丢统计行） | 未根治；`LLAMA_MOE_CRASH_TRACE=1` 追踪器保留 | 用半自动重试加追踪器抓栈 |
| 11 | `AHEAD=1` 侧图 off-by-one | 已定案为"非阻塞读回晚一拍"，非缺陷 | 已在 §6.33 结案（仅质量偏好场景改用同步 + ahead=1） |
| 12 | D01–D05 全部设计项 | 未实现的清单与理由见 §3 | 各自条目内已写 |
| 13 | PLE 1G（维护者原述）主机行缓存90%+的原始运行对应 | 回述保留；同层日志方向一致，但协议未逐项对应。GPU L1命中不作为该结论证据 | 必要时按 `LOCAL_MODELS/qwen38/traces/` 原始运行补齐协议 |
| 14 | PLE 与专家缓存合并成同一套分页引擎（rebuild-spec §2.3） | 仅设计；PLE 侧接口（`copy_pages`/页抽象）已就绪，专家侧仍写死主机指针 | lazy/SSD 部署，或 D01 动工时 |

---

## 7. 源章节覆盖清单

| 源 | 覆盖章节 | 本文位置 |
|---|---|---|
| `handoff.md` | §6.1 已完成的测量 | §2.1、§2.2、§2.4、H03 部分 |
| | §6.2 CPU_ASYNC 中性 | §2.6 |
| | §6.3 devpart 下一步计划 | §2.9 #2 |
| | §6.4 延迟结构最终结论（拷贝机制无关） | §2.2、§2.7、§2.9 |
| | §6.5 devpart host-leaf 实验（8.1 → 19–23） | §2.9 #3（撤回） |
| | §6.10 退出期崩溃定位 + 整机事故与闸 | §2.7、§2.6 表 |
| | §6.11 CPU_ASYNC 转正 + 开销盘点 | §2.6、§2.8、§2.2 |
| | §6.12 pinned 读回（无效）+ split 结构 + devpart 兼容性 | §2.2、§2.8、§2.10 |
| | §6.13 split 合并调查（VRAM 换来的，不做） | §2.8 |
| | §6.14 prefill/decode 倾向分离 | §2.13 |
| | §6.15 SMoE/缓存对 prefill 中性 | §2.13 |
| | §6.16 prefill 读缓存（D2D）零收益已回退 | §2.12 |
| | §6.17 devpart 复查（29.7/32.0，撤回）+ 定位 | §2.9 #4（撤回） |
| | §6.18 devpart 正确性修复（根因 + 图构建崩） | §2.9 #5 |
| | §6.19 devpart 修好（26.0 t/s，浮点次序差异） | §2.9 #6 |
| | §6.20 速度路线打通（28.2，撤回）+ CPU 半边数据待诊 | §2.9 #7（撤回） |
| | §6.21 收敛为单点（readback 未提交） | §2.9 #8 |
| | §6.22 收尾（管道已通，缺陷在驻留表内容） | §2.9 #9 |
| | §6.23 devpart 已修复（三缺陷 + 逐字一致 + 不快） | §2.9 #10/#11 |
| | §6.24 devpart 400 token 稳态与"缓存不填充"根因 | §2.9 #12/#13、§2.10 |
| | §6.25 封板记录（含天花板实测、devpart 默认关） | §2.9 #14/#15、§2.11 |
| | §6.26 MTP 兼容性调查（能跑；缓存×投机崩） | §2.15 |
| | §6.27 MTP×缓存决策与二分 | §2.15 |
| | §6.28 放弃 MTP×缓存（WIP 回退） | §2.15 |
| 邻接章节（本章为补全路线所必需） | §6.6 预取量与截止线；§6.7 热区根治；§6.8/§6.9 自适应；§6.29 PLE/256k 配置；§6.30–§6.33 静默错误与修复；§6.34/§6.35 IQ4_NL repack 与 PLE | §2.3、§2.16、§2.14、§2.11、§2.14 |
| `moe-decode-perf-plan.md` | §0.1 选定路线（借壳出算子）、§1 实测账本、§2 缓存与预取真实数字、§3 P0–P4 改动点、§4 目标可达性、§5 已落地改动、§6 复现、§7 否定掉的旧结论 | §2.1、§3.1、§4（含未采用项） |
| `smoe-nk-degradation-plan.md` | §7 N+k 退化实测 | §2.3、§3.4 |
| `rebuild-spec.md` | §1 基点与验收、§2 边界、§2.2 调度归属、§2.3 内存层级、§2.4 专家聚类、§3 算子规格、§4 执行模型与准入、§5 负面结果、§6 踩坑、§7/§7.1 移植清单与 S1、§8 分阶段、§9 度量口径 | §3.1–§3.5、§2.15、§2.11、§2.7 |
| `moe-cache-score-aware-prd.md` | 策略背景（路由不变、命中/交付、准入语义；§60/§82/§86 的实现决策与规范实验配置；§128 Fate/XT 的关系） | 仅在需要解释机制时引用（§2.3/§2.16、§0.5 的预测器谱系） |
| 用户补充的权威研发顺序（PLE → 静态表+XT → Fate → SMoE/共享专家 → 在线实现 → 双门控 → devpart/TQ4/NXQ） | 本文件 §0.5（口径与边界）与 §1A（PLE 第一阶段单列）；主线叙事见 `00-research-chronology.md` | §0.5、§1A |

**与其它章节的分工**：预测器谱系（静态表/XT → Fate → SMoE，含 teacher 测试 99% 的原始记录对应）
归 `02-prediction-and-cache.md`；权重/量化与算子归 `03-weight-quantization-and-kernels.md`；
KV/TBQ/NXQ 归 `04-kv-tbq-and-nxq.md`；`SPLIT=1` 静默错误的复现/二分/机理与回归用例归
`05-correctness-and-methodology.md`。本章只保留与 host/split/devpart 性能证据直接相关的部分。

---

## 8. 原始日志候选与保全边界

以下路径相对 `SOURCE_TREE`。`（正确）已核` 表示读过内容／编码，
未核者按名称与日期对应，具体数值仍以对应章节记录为准。

| 日志 | 内容 | 对应路线 |
|---|---|---|
| `dsh-r8.txt`、`dsh-r9.txt` （正确）已核（UTF-16LE；含 `[1-]/[2-]/[3-]/[4-]` 行） | 31 个 decode 图的逐图账本（86.0 ms/token 分解） | H01 |
| `dsh-r3-timing.txt`、`dsh-r4-timing.txt`、`dsh-r5-timing.txt`、`dsh-r6.txt`、`dsh-r6-csv.txt`、`dsh-r6-decode.txt`、`dsh-r7.txt`、`dsh-timing.txt`、`dsh-hitrate.txt` | 计时/命中率迭代 | H01、H02、H03、H16 |
| `dsh-r3-text.txt`、`dsh-r5-text.txt`、`dsh-r7-text.txt`、`dsh-r5-bd.txt`、`dsh-r5-e3bd.txt` | 文本正确性对照 | H11、H13 |
| `dsh-A-*`、`dsh-A2-run1-*`、`dsh-A2-run2-*`、`dsh-A3-*`、`dsh-A4-*`、`dsh-B-*`、`dsh-C-*`、`dsh-C2-*`、`dsh-C4-*`、`dsh-E1-*`、`dsh-E2-*`、`dsh-E3-*`、`dsh-F1-*`、`dsh-F2-*`、`dsh-G1-*`、`dsh-G2-*`、`dsh-H1-*`、`dsh-H2-*`、`dsh-I1-*`、`dsh-I2-*`（各带 `-out.txt` / `-err.txt`） | P0 阶段的会合点/预取/命中实验（`dsh-G2` 是 devpart 访问违例的日志名） | H02、H03、H04、H09 早期 |
| `L2048-err.txt`、`L6144-err.txt`、`L2048-out.txt`、`L6144-out.txt` | 缓存容量档位对照 | H03、H16 |
| `T2048_1..4`、`T6144_1..4/6/8`（各带 `-out`/`-err`） | 容量 × 排名截止线扫描 | H03、H16 |
| `a400a-err.txt`、`a400a-out.txt`、`a400b-err.txt`、`a400b-out.txt` （正确）已核（UTF-16） | 400 token 稳态、含 `hot-set oracle` 与逐层 slot/hit 行 | H16、§2.9 #12 的对照口径 |
| `ab-a-*`、`ab-b-*`、`ab-c-*` （正确）已核（1 行摘要） | SPLIT/加载失败类 A/B 记录 | H11 |
| `cur-ref-out.txt`、`cur-ref-err.txt` （正确）已核 | devpart OFF 基线（含 `[PLE-LRU]`/`[PLE-GPU-L1]` 启用行、`[MOE-CACHE] enabled: budget=2048 … devpart=0`、`ids readback` 行） | H01、H14 |
| `run-cur-ref.ps1`、`run-cur-devpart.ps1`、`tools-run.py` | 复现装置（配置与环境变量全集） | 全部 |
| `sweep-ahead.csv`、`sweep-ahead2.csv`、`sweep-c1.csv`、`sweep-c1nb.csv`、`sweep-corrupt.csv`、`sweep-fix.csv`、`sweep-hd.csv`、`sweep-t2.csv`、`sweep-v.csv`、`sweep-x.csv`、`sweep-neutral.csv`、`sweep-reg.csv`、`sweep-smoke.csv`、`sweep-u.csv`、`sweep-press.csv`、`sweep-budget.csv`、`sweep-auto.csv`、`sweep-auto8k.csv`、`sweep-limit.csv`、`sweep-tune.csv`、`sweep-ab.csv`、`sweep-2x2.log` | 前瞻、截止线、非阻塞、静默错误、修复、自适应/预算、显存上限等扫描 | H03、H11、H16 |
| `sweep-vision-cache.py`、`sweep-vision.log`、`sweep-vision256k.csv`、`cases-*.txt` | 256k + 视觉下的缓存/PLE 对照（脚本内注释明写"full-RAM 下两个 PLE 缓存都是死重"） | H14 |
| `sweep-kv.csv`、`sweep-k8q8.csv`、`sweep-k8tbq.csv`、`sweep-k256q8.csv` | KV 类型 × 缓存参数（与 04 章 KV/TBQ 交叉，本章只在 PLE/256k 处引用） | H14（交叉） |
| `dsh-probe-01..09.ps1`、`dsh-probe-bz8.ps1`、`bisect-garbage.ps1`、`bisect2..4.ps1` | P0 阶段探针与二分脚本（含 PLE 开关与 `PREFETCH_JOIN` 组合；`dsh-probe-bz8.ps1`/`bisect*.ps1` 是首批把 PLE 设 0 的脚本） | H01、H11、H14 |
| `build-cli.bat` | 构建（sccache，`-j 16`；`-j 32` 会死机） | 全部 |

**PLE（第一阶段）专项候选日志** —— 注意 **PLE 统计文件不在仓库里，主目录是
`LOCAL_MODELS/qwen38/traces/`**（`run-moe-ple.bat`／`run-nomoe-ple-cold.bat`里的
`LLAMA_PLE_CACHE_STATS_FILE` 即指向该目录）：

| 日志 | 内容 | 备注 |
|---|---|---|
| `ple-gpu-overlap-stats.csv`（09-03 15:06） | **GPU L1 层**（8 列 schema）、overlap 预取：14791/1049 = **93.4%**；**不用于证明 SSD→主存 90%+** | §1A.2 |
| `ablation-ple-only.csv`（09-03 12:02） | **主机行缓存层**（9 列 schema）：14899/1101 = **93.1%**，`prefetch_pages=12774` | §1A.2 |
| `ple-lru-256m-o1.csv`（09-03 09:10） | 8 列 schema（层级待原始记录确认）：27.6%（容量受限对照） | §1A.2 |
| `ple-lru-1g-no-prefetch.csv` | **主机行缓存层**（9列）、关预取时10.1%；容量和批次也不同，不能单独归因于预取开关 | §1A.2 |
| `ple-gpu-l1-rawpages-1g.csv`、`-1g-async.csv`、`ple-gpu-lookahead-1g.csv`、`ple-cpu-l2-with-gpu-l1.csv` | 预取/看齐到位前的早期变体（13–17% / 0.75%） | §1A.2 |
| `ple-cpu-stats.log`、`ple-cpu-stats-long.log`、`ple-cpu-stats-nopf.log`（09-04） | 长/对照运行的 PLE 页统计 | §1A.2 |
| `ple-gpu-l1-timing.csv`、`ple-gpu-l1-pinned-timing.csv`、`ple-gpu-overlap-prefetch.csv`、`ple-gpu-overlap-consumer.csv` | GPU L1 计时/重叠专项 | §1A.2 |
| `ablation-lazy-moe8-gpuple.log`、`ablation-lazy-moe8-ple.csv`、`ablation-lazy-moe8-ple-2.csv`、`ablation-lazy-moe7-cpuple{,-rerun}.log`（09-10） | **PLE + MoE 缓存并存**的 lazy-mode 消融（8 GiB / 7 GiB 档） | §1A.2、§1A.3 |
| `run-moe-ple.bat`、`run-moe-ple-nopf.bat`、`run-moe-ple-long{,-ctl}.bat`、`run-nomoe-ple-cold.bat`（仓库内） | PLE 消融的复现装置（含 `LLAMA_PLE_CACHE_MIB=4096`、`LLAMA_PLE_PREFETCH`、`LLAMA_PLE_CACHE_STATS_FILE` 路径） | §1A.2 |
| 仓库内 `abl-2g-ple2g.csv`、`abl-2g-ple512.csv`、`abl-512-ple2g.csv`、`abl-512-ple512.csv`、`abl-2g-{ple,nople,nojoin,ple2g-gpu1g}.log`、`abl-512-{ple,ple2g,stats}.log`、`layer-bundle-ab-lazy.log`、`vram-samples.txt`、`ple0-guard512-{err,out}.txt`、`ple0-guard1024-{err,out}.txt`、`ple256-guard512-{err,out}.txt` | MoE×PLE 消融与显存采样（注意：四份 CSV 为短生成、缓存未填满，不能作容量结论） | §1A.2、H14 |

**无原始日志的部分**（明确只存历史报告）：MTP/投机一轮（handoff §6.26–§6.28 只留命令与输出摘录）；
devpart 的 `LLAMA_MOE_DUMP_CH`/`DUMP_SPLITS`/`CRASH_TRACE` 转储（输出在报告内引用，未留独立文件）；
`prefill/decode tendency` 行（散落在 `sweep-vision*` 与 256k 运行日志里，未按路线单独归档）。
