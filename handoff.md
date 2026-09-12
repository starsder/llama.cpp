# Qwen4Exp（源工程）现状 — 交接备忘

更新: 2026-09-12 | 工程: `F:/src/llama.cpp-unsloth-qwen4exp` | 构建: `build-ple-trace-mrs`

> 本文件是"防遗忘"的单一事实源。上下文再长，先读这里。

---

## 1. 一句话

**回到源工程继续优化。** 当前二进制功能正确：devpart ON / OFF 文本都正常；
性能基线是 **host 路径 86.0 ms/token（11.3 t/s）**，devpart **122.9 ms/token（8.1 t/s）**，
优化靶心是 `pre` 段 49.7 ms（host 侧循环 + 每层硬排空 + SMoE 事件等待）。

---

## 2. 代码与仓库状态

| 项 | 状态 |
|---|---|
| HEAD | `2f1a363c8` "qwen4exp: add TBQ KV quantization and the runtime MoE expert cache"（19:21 提交，**含保活 view 修复**） |
| 未提交 | `ggml/src/ggml-backend.cpp` +197/−3 —— **纯插桩**（inputs/SMoE 分项计时计数器 + 打印），无功能变更 |
| `AGENTS.md` | 已被用户删除（有意）；remote 指向用户自己的 fork（`starsder/llama.cpp`），upstream = unslothai |
| 二进制 | `build-ple-trace-mrs/bin/llama-cli.exe`（20:27 构建，含上述插桩） |
| 我们新增的算子 | `ggml/src/ggml-cuda/moe-partition.cu/.cuh`（设备侧分区 kernel，已提交） |
| 其他 agent 的产出 | `docs/moe-decode-perf-plan.md`（实测账本+路线）、`docs/rebuild-spec.md`（重建规格）、`docs/moe-cache-score-aware-prd.md`、56 个 `dsh-*` 实验日志、`run-cur-ref.ps1` / `run-cur-devpart.ps1` |

**clean 线实验已放弃**：`F:/src/llama-qwen4exp-clean`（worktree @ `b76199698`）里留了一份 `clear/` 迁移快照
（TQ4/MoE/PLE 源码 + docs + tools + patches），结论是那 35 文件暂存子集**不自洽**（0.7s 加载崩溃），
HEAD 本身也跑不动这个模型。**不再走这条路**，但快照可作为"我们的代码清单"参考。

---

## 3. 性能账本（同一二进制、同配置、31 个 decode 图，来自 `dsh-r8/r9.txt`）

```
host 路径 (devpart OFF)  total 86.0 ms/token ≈ 11.3 t/s   ← 真基线
  cpu   27.7   CPU 半边算未命中专家（真活）
  gpu    7.4   GPU 入队
  pre   49.7
    ├ inputs 29.1
    │   ├ split_partition 23.2   (n=144, 其中 host CPU 循环 22.7)
    │   ├ flag_input       4.3
    │   └ generic          1.1   (n=51)
    └ drain  20.5                 SMoE 事件等待 ~17 + prefetch 提交 ~4

devpart ON              total 122.9 ms/token ≈ 8.1 t/s
  cpu 3.2 | gpu 7.2 | pre 111.4 → inputs 78.8，其中 generic 拷贝 78.2 (n=195, ≈400 µs/次)
```

要点：
- host 路径的 23 ms 集中在 `moe_cache_activate_layer → moe_insert_drain → moe_insert_flush → slot_events_drain`，
  其中 **`moe_insert_flush` 是阻塞条件变量（每层等 insert worker 队列清空）**，代码注释说明是
  "避免侧流拷贝越过 CUDA graph 捕获边界"才这么做 —— 可改成**只在捕获边界 flush 一次**。
- devpart 的 78 ms 完全来自 CPU 半边输入（设备上的 `ids/wgt/cur` view）走调度器兜底拷贝：
  每次 `ggml_backend_synchronize(input_backend)` + 同步 copy。
- **作废数字**：`20.7 t/s` 是乱码路径的假速度；`19–21 t/s` 无本配置历史输出支持。

---

## 4. 正确性状态

- devpart ON：文本正确（"The user asks ... Paris"），8.1 t/s。
- devpart OFF：文本正确，9.7–11.6 t/s。
- **devpart 曾乱码的根因**：`selected_experts`(topk) / `weights` 两个 router 输出的**分配器生命周期**
  被提前结束、内存被复用 → kernel 读到脏数据。修法（已提交）＝在 `src/llama-graph.cpp` 的 devpart 分支
  追加保活 view 节点（`ffn_moe_part_keep_ids/wgt-%d`），等价于 `ggml_set_output` 但不掉速。
- 已排除（别再查）：kernel 逻辑/`-1` 约定/A1 元数据/A2 依赖边/驱逐覆写竞态/插桩污染。
- 顺带未修的 bug：`moe_cache_finalize` 晚于首图 compute → graph 1 的驻留表全 0（0 是合法 slot）；
  建议表初值改 `-1` + finalize 提前。

---

## 5. 纪律与坑（血泪）

1. **内存**：`moe_cache_pin_weights()`（`LLAMA_MOE_PIN_WEIGHTS` **默认开**）会 `cudaHostRegister`
   锁定整份专家权重（日志：`pinned 1 weight buffers (72.6 GiB)`）。它与 mmap **互斥**——
   两者混用（mmap 读入 + 默认钉住）会同时占住 76 GB 文件页与 72.6 GB 锁定内存，**顶爆 128 GB 机器**。
   现有脚本一律 `--no-mmap` + 默认钉住（全量常驻 + 锁定），属于已知的重配置；
   **不要同时跑两个推理**，跑前先看剩余内存。
2. **绝不裸跑**：诊断跑必须带 `--cpu-moe`；不要 `-ngl 0` 且不带 `--cpu-moe`（会试图把 76 GB 权重全吃进 RAM）。
3. 构建用 `cmd /c build-cli.bat`（sccache 已配，`-j 16`；`-j 32` 会死机）。
4. 日志是 **UTF-16**：先 `iconv -f UTF-16LE -t UTF-8`，`grep` 前必须转码。
5. 工具：本工程的 `bash script.sh` 会走 WSL bash（`/mnt/f/...`），脚本内尽量用 `F:/...` 或改用 Python 驱动。

---

## 6. 优化待办（按收益/风险排序）

### 6.1 已完成的测量（2026-09-12，同一二进制、`run-cur-ref.ps1` 同级配置）

| 项 | 数值 | 说明 |
|---|---|---|
| 基线速度 | **11.1 t/s**（86 ms/token） | devpart OFF |
| `split_partition` 总计 | **21.3 ms/图** | 其中已计时仅 3.3 ms |
| ├ `ids_wait`（**MRS=0 时测得 18.57 ms**） | ≈19 ms | 每层 `get_async(topk/wgt)` + **`ggml_backend_synchronize`** 把 router 结果拉回主机做分区 → **临界路径** |
| ├ drain+flush | **0.07 ms**（0.3%） | 原先怀疑的 `moe_insert_flush` 每层硬排空**不是瓶颈** |
| ├ activate / on_ids / body | 0.00 / 0.00 / 1.1 ms | 早退路径 85 次/图 ≈ 0 ms（廉价） |
| └ act_d2h | 0.43 ms | cur 的 D2H |
| SMoE 事件等待 | **15.3 ms/图**（533/35） | `moe_cache_smoe_drain` 等侧图读回事件 |
| 命中率 | 27.5%（hits 12306 / miss 32364） | 预测准确率 73% 但交付率低 |
| `smoe-off` | **崩溃** 0xC0000005 | 关掉预测器会走未保护的路径（bug） |
| `MRS=0`（LRU） | 10.7 t/s | MRS 净收益 +0.4 t/s（命中率更高） |

**机制结论**：临界路径 = 主机每层分区（19 ms）+ SMoE 事件等待（15 ms）+ CPU 真活（27.7 ms，与 GPU 重叠）。
host 分区必须等 GPU 算完 router，reorder 省不掉 → 唯一解是**设备侧分区**。

### 6.2 已实现但中性的改动

- `LLAMA_MOE_CPU_ASYNC=1`：把 MoE CPU 半边派给自己的 worker 线程（分区时就派、CPU split 时 join），
  实测**文本与基线逐字一致、速度中性（11.0 vs 11.1）**——因为 CPU 半边本来就与 GPU 执行重叠，
  它不在临界路径上。代码保留（默认关），是后续 devpart 异步 CPU 半边的基础设施。

### 6.3 下一步（已与用户确认方向）

**devpart（设备侧分区）+ CPU 半边输入改自有异步 D2H + worker 计算 + 边界 join**：

1. devpart 让 GPU 从 router 直接进 MoE，不再等主机分区（省 19 ms/图）；
2. CPU 半边的 `ids/wgt/cur` 不再走调度器兜底拷贝（现 195 次 × 全同步 = 78 ms），
   改由我们在 GPU split 处 `get_async` 进 host leaves + 记录事件（无同步）；
3. CPU split 交给 worker（已就绪的 `moe_cpu_half_submit`），join 在消费方之前；
4. 之后再把 SMoE 的 15 ms 事件等待纳入同一套异步框架。

目标：86 ms → ~40 ms（**≈25 t/s**）。

### 6.4 2026-09-12：延迟结构的最终结论（**拷贝机制无关**）

对 host 路径逐层测量（decode 30 图，82.5 ms/token）：

| 结构项 | ms/图 | 说明 |
|---|---|---|
| **每层 router 回读同步** | **~18** | 主机必须等 GPU 算完该层 `ffn_moe_probs/ids/wgt` 才能准备该层 MoE 输入 |
| **SMoE 读回事件等待** | **~15** | 读回拷贝在 `ggml-backend.cpp:5431`（`compute` **之后**）入队，事件必然等到该层 GPU 跑完 |
| CPU 半边真活（已与 GPU 重叠） | 27 | worker 实验证明移出主机线程**中性**（11.0 vs 11.1） |
| GPU 入队 | 6 | 仅入队耗时 |

**关键实验（推翻了"pinned 化可救"的假设）**：

```
MRS 全分数读回 ON :  prologue=18.04 ms  ids_wait=1.31 ms   → split_partition ≈ 21 ms
MRS 全分数读回 OFF:  prologue= 0.83 ms  ids_wait=17.01 ms  → split_partition ≈ 21 ms
```

等待只是从"pageable D2H 的隐藏阻塞"**搬到**了"显式 `ggml_backend_synchronize`"，总量不变。
（SMoE 的 staging 本来就是 pinned；它 15 ms 的等待是等 GPU，不是等拷贝。）

**结论：这两项都是流水线结构造成的 host↔GPU 逐层同步，任何拷贝/内存手段都无法消除。**

**唯一的两条真实修法**：

1. **devpart（设备侧分区）**：主机不再需要该层 router 输出 → 消掉 ~18 ms。
   已有速度证据：CPU 半边改 host leaves 后 **8.1 → 19–23 t/s**，但正确性未解决（§6.4）。
2. **给 SMoE 预测留余量**：把预测目标从 `layer+1` 改成"下一个 token"或 `layer+2..+4`，
   使读回+预取管线有足够 slack → 消掉 ~15 ms（顺带提升命中率：现在 26%）。
   当前 1 层截止期在物理上不可能满足（侧图 + D2H + 主机处理 + 专家 H2D 需要毫秒级）。

两条都做完 ≈ 82 − 33 = **49 ms/token ≈ 20 t/s**；再叠加命中率提升（CPU 半边从 27 ms 降下来）才有望到 25 t/s。

**开关**：`LLAMA_MOE_MRS_FULL=0`（默认 1）可切换 MRS 到"用已读回的 top-k 更新"的廉价路径
（本实验用；单开它速度不变，但可省掉一次读回、为后续结构改造留余地）。

### 6.5 2026-09-12：devpart 的 host-leaf 实验（速度已达标，正确性未解决）

**已达成：速度**。把 devpart 的 CPU 半边从"设备 view（走调度器跨后端拷贝）"改成
"host leaves（由我们在 GPU split 处 `get_async` 填充）"后，devpart 从 **8.1 → 19–23 t/s**
（`LLAMA_MOE_CPU_ASYNC=2` 填充 + scheduler 计算 = 17–19 t/s；mode 1 worker = 14.9 t/s）。
这证明 **78 ms 的来源确实是那 195 次跨后端拷贝**，去掉它即可越过 host 路径（11.2）。

**未达成：正确性**。该配置下输出从第 2 个 token 起崩坏（`The////`）。已排除的：
- ids 填充有效：诊断打印显示 CPU op 的 `src[4]` 收到真实专家 id（`95 85 142 280 …`），且是同一个张量；
- CPU 半边确实在算（否则不会 19 t/s 还崩）；
- 给 `pids/pwgt` 补"保活 view"（放在 CPU 半边之后）**未能修正**，说明不只是生命周期问题。

**尚未查清的可疑点**（下一步从这里入手）：
1. `cur`（CPU 半边激活）取自 `MUL_MAT_ID` 的 `src[1]`，是否与 CPU op 期望的输入一致（布局/来源）；
2. `wgt_cpu` 从 `pwgt` 后半段拷贝的偏移/语义（前 k 个=GPU、后 k 个=CPU）是否与 CPU op 的期望一致；
3. devpart 下 CPU 半边与 CUDA graph 捕获/复放的交互（图复用后再改 `src[0]` 到 cache view 的路径）；
4. `moe_cpu_half` 检测在这条路径上是否每层都命中（日志里只见到 layer 47/0 的样本）。

**当前仓库状态（已恢复可用）**：
- `src/llama-graph.cpp` = HEAD（devpart 的 CPU 半边仍用设备 view，文本正常 @8.1 t/s）；
- `ggml/src/ggml-backend.cpp` = HEAD + 插桩 + 环境变量默认关闭的异步 CPU-half worker
  （`LLAMA_MOE_CPU_ASYNC=1`：host 路径可用，文本逐字一致、速度中性；`2` 的历史填充分支已删除）；
- host 路径 11.2 t/s 文本正常；devpart 8.1 t/s 文本正常。

| # | 项 | 预期 | 风险 |
|---|---|---|---|
| 1 | `moe_insert_flush` 每层硬排空 | ~~省 20 ms~~ **实测 0.07 ms，作废** | — |
| 2 | SMoE 事件等待 15.3 ms → 与后续 split 重叠 | 省 ~10 ms | 中 |
| 3 | devpart 的 CPU 半边输入改异步（195 次同步拷贝 78 ms） | 省最多 78 ms | 高（结构改动） |
| 4 | `split_partition` 的 host CPU 循环 | 实测非主机 CPU 耗时，作废 | — |
| 5 | 修 C（表初值 `-1` + finalize 提前） | 正确性 | 低 |
| 6 | 修稳定性：退出期偶发 0xC0000005、`SMOE=0` 崩溃 | 正确性 | 低 |

目标：**25 t/s（40 ms/token）**（`docs/moe-decode-perf-plan.md` 的路线；他们的判断是
"只拆会合到不了 21.7 t/s，必须同时把预取交付率提上去"）。

### 6.6 2026-09-12：预取量与准入截止线（本次会话结论，**量是成本不是收益**）

**① 预取量的边际成本 = PCIe 争用**（同一二进制、`tools-run.py` 配置 + `NONBLOCK=1 AHEAD=2`）：

| 预取量 | 命中率 | total | t/s | 主机侧 `flag_input` |
|---|---|---|---|---|
| 161 MB/tok | 40.2% | 70.0 ms | **14.3** | 15.8 ms |
| 254 MB/tok | 46.9% | 77.7 ms | 12.9 | 23.3 ms |
| 464 MB/tok | 58.2% | 94.1 ms | 10.6 | 37.5 ms |

边际 ≈ **0.073 ms/MB ≈ 13.8 GB/s**（≈ 有效 PCIe 带宽）。`flag_input` 随量增长的项不是拷贝
（80 字节），而是 `moe_cache_prefetch_layer` 上游那个 `ggml_backend_event_synchronize`：
预取 DMA 与 GPU 自己的访存抢 PCIe → GPU 变慢 → 主机 `event wait` 变长。
**加 worker 也没救**：低量 1→3 worker = 13.3→13.6；高量 9.9→10.5（+6%）。pinned 环上限仅几 ms。

**② 命中/字节随预测排名单调衰减**（`LLAMA_MOE_TAKE_MAX` = 准入截止线）：

```
cutoff 1 → 4.69 hits/MB     cutoff 3 → 2.21
cutoff 2 → 5.25 (另测 3.05)  cutoff 4 → 1.78
```

即"topk26 的 58% 命中"是把半个缓存塞满换来的，每字节效率只有 t4 的一半。**命中率要靠"更准"不是"更多"。**

**③ 主动截断 = 排名截止线（原为隐式，已显式化）**：`moe_cache_smoe_process` 里
`take = min(n_slots, n_topk, take_max)`，`take_max = n_used+2`（=12）实际起的是"只取前 N 名"的作用；
`n_used` 只在分区钩子里赋值，**首 token / 未过钩子的层 `n_used==0` → 截止线静默消失（bug，已修）**。
现在：`rank_cut = LLAMA_MOE_TAKE_MAX (默认 2)`，与 `n_used` 无关。
"先截断再跳过已驻留"的顺序**实测优于**"跳过已驻留后继续往下填预算"（后者把准入集合从秩1-4挪到秩5-10，同字节命中更低）。

**④ 去重已确认（硬数字）**：`policy=` 行新增计数器 ——
```
候选 2852 = dup_resident 1496（已驻留，直接丢：52.5% 的候选不花字节）+ 传输 1356
dup_list=0（候选表内无重复） dup_admit=0（无同槽自我拷贝） dup_pending=0（无在飞重复请求）
readmit=86（淘汰抖动 6.3%）
```
`expert_slot[e] >= 0` 是去重权威：slot 在准入时、拷贝下达前就已赋值，故同 token/同窗口的重复请求必被拒。

**⑤ 容量 vs 参数的交互**：30 token 下 2048 与 6144 MiB 无差别（**大缓存没被填满**：6GB ≈ 67 槽/层，
每 token 每层只准入 ~2 个，30 token 填不满；`readmit=0` 证明从未发生淘汰）。
150 token 下（会填满）：2048 最优在 cutoff 1（14.6），6144 平台在 **2–4**（14.8–14.9）、6→14.5、8→14.1
→ **最优截止线随容量上移**（组内噪声 ±0.6 t/s，关键点尚未重复确认）。
显存峰值：cache 2048 → **9132 MiB**；6144 → **13278 MiB**（limit 15360、guard 1024，装得下）。

**⑥ 当前最优配置**：`LLAMA_MOE_SMOE_NONBLOCK=1 LLAMA_MOE_SMOE_AHEAD=2`（cutoff 用默认 2）
→ **14.1–14.6 t/s @30–150 token**（旧默认 cutoff=12 约 10 t/s）。结构性的下一步仍是 §6.3/6.5 的 devpart。

**⑦ 工具**：`tools-run.py` 现在每次输出 `vram_peak` 与去重指标（`dup_*` / `readmit`）。


### 6.7 2026-09-12 晚：热区根治 —— **淘汰分数用错了**（本会话最大收益）

**病根**：决定"谁被淘汰"的是 `mrs_score`（gate 全 softmax 前 20 名 EMA）。用真值路由（`part.ids`）
量它：`mrs_topC = 0.3%`（同容量下，按 mrs 排序取前 C 名只覆盖真实路由的 0.3%）——**几乎随机**。
所以缓存里沉淀的不是"热专家"，而是被一个无意义分数洗牌的结果。

新增诊断（退出时打印）：

```
[MOE-CACHE] hot-set oracle: routed=N layers=48 slots/layer=C |
            oracle_topC=66.9% resident_set=53.0% mrs_topC=0.3% actual_hit=48.9%
```
`oracle_topC` = 同容量下装"真热"专家的覆盖率上限；`resident_set` = 当前缓存的真实覆盖；
`mrs_topC` = 旧分数前 C 名的覆盖。**这个三行对比就是"缓存装对了吗"的判据**（

**修法**：淘汰分改为**真值路由频次**（`use_count`，在分区钩子里从 `part.ids` 零成本累计），
带**基期折半的滑动窗**（`LLAMA_MOE_HOT_HALFLIFE`，默认 512 token，防分布漂移后热集冻结；
窗口内分布不变则热集自然稳定，无需churn）。`LLAMA_MOE_EVICT_SCORE=1` 可切回 mrs 做 A/B。

**效果**（400 token、`--ignore-eos`、统计只取后 150 图=稳态）：

| 配置 | 稳态命中率 | 稳态 ms/token | 稳态 t/s | 报告 t/s |
|---|---|---|---|---|
| 2GB + mrs 分数（旧） | 30.5% | ~119 | 8.4 | 13.5 |
| 2GB + 频次分数 | 53.1% | 51.5 | 19.4 | 17.7 |
| 6GB + 频次分数 | — | — | — | 17.8 |
| **6GB + 频次分数 + 回填 8** | **80.8%** | **43.0** | **23.3** | 19.9 |

**热区回填（用户提议，已实现）**：`moe_cache_hot_backfill`，在**图末尾**（所有 split 之后、
capture 之外）把"比当前最冷驻留更热"的非驻留专家补进来（自终止：收敛后零开销）。
- 无等待、无 join：拷贝走 side stream，完成由 `slot_events` **轮询**回收（`prefetch_event_query`）；
- 不干扰下次预取：它在 split 边界之后才跑，预取仍是一等公民（照样准入、照样可能塞冷专家）；
- 两段式预算：`LLAMA_MOE_HOT_FILL_BOOT`（启动期，默认 0=关）→ `LLAMA_MOE_HOT_BACKFILL`（稳态，默认 0=关）；
- `LLAMA_MOE_HOT_IDLE=1`：后台线程在**模型空闲（用户打字）时**用更大批量继续填（150ms 无计算活动判定；
  所有路径都检查 `graph_active`，互斥只包记账段，传输本身永不被等待）；
- 2GB 下回填有害（thrash，readmit 3383→10743、稳态命中 56% 且更慢）→ **回填必须配足够容量**。

**冷启动种子**：`manifest.hot`（你之前采集的每层专家激活频率，`LLAMA_MOE_PREDICT_STATIC` 用的同一份数据）
接到 `LLAMA_MOE_PIN_STATIC=N`（此前是死代码）：finalize 时按清单热度每层种入 N 个，**不给淘汰保护**，
之后被实测热集与预取自然替换（配合 halflife 衰减）。35 token 的短生成填不满 6GB（3072 槽），
所以种子 + 空闲填充是短对话冷启动的关键。

**每排名准确率 / 每字节效率**（400 token，固定 cut=2；样本 = 48层×400）：

```
r1 77.9% y17.93   r2 66.5% y16.65   r3 58.5%   r4 52.1%   r5 46.7%   r6 42.4%   r7 38.0%   r8 35.9%
r9 31.5% r10 28.2% r11 26.4% r12 23.3% r13 21.5% r14 19.7% r15 18.2% r16 16.6%
```
（y = 实测 hits/MiB；未准入的排名无字节 → 不可测）

**自适应截断的三次尝试（结论：都不如固定 cut）**：
1. 按准确率门槛（`LLAMA_MOE_RANK_ADAPTIVE`）：鸡生蛋——窄 cutoff 只 offer 少数排名，
   统计饿死 → 已修（准确率按**完整候选表**测，与准入解耦）；
2. 按**实测 yield**门槛（`LLAMA_MOE_YIELD_MIN=4`，用户提议"每字节效率>4 放行"）：
   r1/r2 实测 17.9/16.7 ≫ 4，但更深排名无字节 → yield=0 判 0 → cut 卡在 1
   （**意外的收获**：cut=1 + 省下的传输 = 20.6 t/s > cut=2 的 19.3）；
3. 用"准确率 × 实测单次准入命中数"估 yield + 字节预算（`LLAMA_MOE_ADMIT_BUDGET_MIB`）：
   64MiB 预算 → 12 t/s（早层吃光额度、后层饿死），128MiB → 19.0 t/s。**不如 1/2**。

**占用率观测（用户实测，2026-09-12）**：bf8/bf16 把 decode 占用率从 40–45% 抬到 60%
（此前说的"100%"其实是预填充满负载——decode 一直没满过，所以"GPU 40ms 硬底/25 t/s 天花板"
的旧估计作废，真实天花板更高）。而 `ADMIT_BUDGET_MIB` 会把占用率**钉在中平台**：它限速的是
预取准入 → GPU 侧工作集增长被限速 → 既不下探冷谷也不上到热峰。
**结论：价值门槛（哪些排名值得）与速率控制（每 token 多少）要分开——速率交给回填，
门槛只管价值。** 64MiB 预算实测把准入压到"早层吃光、后层饿死"（12 t/s）；
把每层预算下限抬到 1 个 bundle 后恢复正常。

**当前推荐配置**：`CACHE_MIB=6144 + SMOE_NONBLOCK=1 + SMOE_AHEAD=2 + HOT_BACKFILL=8`
（cutoff 用默认 2；yield/自适应/预算默认全关）→ 稳态 23.3 t/s、命中 80.8%。
显存峰值 13278MiB（limit 15360）。

**新增 env 一览**：`LLAMA_MOE_EVICT_SCORE`(0=频次/1=mrs)、`LLAMA_MOE_HOT_HALFLIFE`、
`LLAMA_MOE_HOT_BACKFILL`、`LLAMA_MOE_HOT_FILL_BOOT`、`LLAMA_MOE_HOT_FILL_GRAPHS`、
`LLAMA_MOE_HOT_IDLE`、`LLAMA_MOE_HOT_IDLE_GAP_MS`、`LLAMA_MOE_PIN_STATIC`、
`LLAMA_MOE_RANK_ADAPTIVE`、`LLAMA_MOE_RANK_THRESHOLD`、`LLAMA_MOE_YIELD_MIN`、`LLAMA_MOE_ADMIT_BUDGET_MIB`。

**未解决 / 未验证**：
- 退出期偶发 `0xC0000005`（本次 7 次运行崩 3 次，影响取数不影响的正确性判断）；
- `hot-set oracle` 的 `oracle_topC` 是**内容相关**的：策略改变 → GPU/CPU 分工改变 → 数值微差 →
  贪心长跑发散 → 不同 run 的 oracle 不可直接互比（只能各自对照自己的 oracle）；
- 空闲线程（`HOT_IDLE=1`）尚未做端到端验证（需要"生成→idle→再生成"的交互式会话）；
- `LLAMA_MOE_INSERT_WORKERS` 在 `PREFETCH=1` 下是空转（worker 的 spawn 点都要求 `!prefetch`）——
  之前"worker 不是瓶颈"的结论作废；预取拷贝是**主机线程内联提交**的。


### 6.8 2026-09-12 深夜：自适应门槛（**极值搜索控制器**，已收敛）

**为什么必须自适应**：命中的边际价值是**状态相关**的（CPU 半边有活时一次命中≈0.074ms，
命中率到 ~80% 后 CPU 半边空了 → 边际价值≈0），而传输的边际成本始终存在（≈0.08ms/MB PCIe 争用）。
最优深度在"边际价值穿过边际成本"处，且随**分布/提示词/缓存状态/阶段**移动 → 常量门槛不可能对。

**实现**（`LLAMA_MOE_YIELD_AUTO=1`）：决策变量 = 排名要清的**每字节效率门槛**（hits/MiB）；
目标 = **中位每 token 耗时**（抗单点噪声）；每 `LLAMA_MOE_YIELD_AUTO_PERIOD`(默认16) 个图比较
前后两窗口中位数：变快沿同方向、变慢反向，步长乘性 ±10%，钳 [0.5,32]，每次探针打日志。
门槛用于"按实测 yield 放行排名"（`moe_cache_rank_yield_est`：有字节用实测，没字节用
`准确率 × 实测单次准入命中数` 估，过**单调包络**后 break，防噪声当信号）。
硬约束仍是 `moe_prefetch_feasible`（截止期模型）——**只作用在预取上**；回填无时限、不受预算限制
（`ADMIT_BUDGET_MIB` 仅约束 `prediction` 路径）。

**收敛证据（400 token，6GB+回填8）**：
```
prose : 67.8 → 38.7 ms（门槛 4.4 → 10.9），收敛 cut=1，yield_min=11.82，18.8 t/s
code  : 收敛在 ~4.3–4.8 ms 门槛
```
**同一算法在两个分布上自算出不同门槛（11.8 vs 4.5）** —— 人工常数 4 在 prose 上偏松 3 倍。

**关键教训（本次踩坑）**：
1. 常量预算 `ADMIT_BUDGET_MIB` 若**按层分片**且分片小于 1 个 bundle（1.9MB），会把准入全部拒掉
   （64MiB/48=1.33MB<1.9MB → 实测 12 t/s）→ 分片下限必须 ≥1 bundle；
2. 纯实测 yield 有**鸡生蛋**（没准入就没字节 → yield=0 → 门槛永远打不开）→ 必须能"估"；
3. 窄 cutoff 会让统计饿死（只 offer 少数排名）→ **准确率必须按完整候选表测**，与准入解耦；
4. 预算若同时约束回填，会限速热集增长 → 占用率被钉在中平台（用户实测）。


### 6.9 2026-09-12 深夜二：模型驱动自整定（B）—— 门槛与预算都自己算出来

**做法**（`LLAMA_MOE_TREND_AUTO=1`）：每个图采集 `(ms, 命中次数, 准入MB)`，在 `trend_window`(默认32)
滑动窗上对
```
ms ≈ a − V·hits + P·MB
```
做**中心化最小二乘**（3×3，Cramer 解，~50 flops/图 → 对速度无可测影响），EWMA(0.75/0.25) 平滑：
- **V** = 一次命中省下多少 ms（价值）；**P** = 每 MB 传输让 token 慢多少 ms（PCIe 争用代价）；
- 于是两个结论同时得到：
  - **准入门槛** = `P / V`（hits per MB）→ 带 10% 迟滞，替换极值搜索（B 是它的模型版）；
  - **字节预算** = `frac × ms_hat / P`（`LLAMA_MOE_BUDGET_FRAC`，如 0.25 = DMA 最多占 token 时间 25%）。

**守卫（数据不可信时保持旧模型，绝不因此变慢）**：窗口 <16 点、两个回归量的散布不足
（hits 散布 <15% 或 MB 散布 <8）、行列式过小、解越界（V∉[0.005,2]ms 或 P∉[0.001,0.5]ms/MB）
→ 拒绝该次拟合（计数 `rej=`）。

**实测（400 token，6GB+回填8，frac=0.25）**：
```
trendV=0.0158 ms/hit   trendP=0.0936 ms/MB
→ 门槛 = P/V = 5.92 hits/MB（开 cut=4）
→ 预算 = 116.5 MB      （手推的"128MiB 保险丝"几乎相同）
fits=191 rej=212       速度 19.6 t/s（同配置历史 18.8–19.9，无回归）
```
**独立交叉验证**：P=0.0936 与 §6.6 由容量扫描手算的 0.08 ms/MB 相差 17% —— 两条独立路径给出同一代价，
模型可信。门槛 5.92 也落在极值搜索在 prose(11.8) 与 code(~4.5) 之间，符合预期。

**推荐（自适应版）**：`TREND_AUTO=1 + BUDGET_FRAC=0.25`（+ 6GB / 回填 8 / NONBLOCK+AHEAD2）。
`ADMIT_BUDGET_MIB` 仍是手工覆盖（>0 时优先生效）；`YIELD_AUTO`（极值搜索）与 `TREND_AUTO` 二选一，
默认都关（关闭时行为与本节之前完全一致，已回归验证）。


### 6.10 2026-09-12 深夜三：退出期崩溃定位 + 整机事故与防复发闸

**① 崩溃根因（是我引入的）**：我为了"退出前排空在飞拷贝"在 `atexit(moe_cache_print_summary)`
里加了 `prefetch_wait` 循环。但 **`atexit` 在 `llama_free` 之后运行** → 那时 backend 已析构，
`entry->backend->iface.prefetch_wait(...)` 就是 **use-after-free**。这解释了"最近 6 连崩"
（并且把崩溃码从 0xC0000005 变成 0xC0000409 fail-fast）。撤销后 **连续运行不再崩**
（随后 13 次运行仅 1 次崩，且那次是 `PREFETCH_JOIN=1` 这条独立路径）。
诊断工具（临时，`LLAMA_MOE_CRASH_TRACE=1`）：`SetUnhandledExceptionFilter` +
`CaptureStackBackTrace` + dbghelp `SymFromAddr` 打符号化栈；**尚在代码里，待定去留**。

**② 整机被拖死的事故（我的操作失误）**：我用 `&` 把一个跑测循环放后台，那条命令的前台部分结束后
孤儿循环继续 launch，我又启动了另一批 → **两个 llama-cli 重叠**。每个实例启动时
`moe_cache_pin_weights` 会 `cudaHostRegister` **钉住 72.6 GiB 主机内存** → 2×72.6 ≈ 145 GiB
> 131 GiB 物理内存 → 提交量耗尽 → 整机卡死。**与模型/显存/本次优化无关。**

**③ 防复发闸（已实现并测试）**：
- `tools-run.py`：**单实例锁**（`.tools-run.lock` + PID 存活检查，强杀/重启留下的陈旧锁自动回收
  —— 已测：并发启动被拒、陈旧锁被接管）+ **空闲内存预检**（`--min-free-mib` 默认 90000、
  `--wait-mem` 默认 120s，不足则等待后报错退出 —— 已测拒绝）；
- 纪律：跑测一律串行，不再用 `&` 起运行循环。

**④ 配置级可压榨点调查（只测未改）**：
| 探查 | t/s | 结论 |
|---|---|---|
| 基线（6GB+回填8） | 19.9 / 20.0 | — |
| `MRS_FULL=0`（省 17ms 全 softmax 读回） | 19.9 | 无变化：等待只是搬到 `ids_wait`，§6.4 结论在新工作点依旧成立 |
| `MRS=0` | 19.2 | 略差 |
| **`CPU_ASYNC=1`** | **20.5** | 最好：host 线程 cpu 12.4→5.7ms，join 多付 2.7ms，净省 1.4ms |
| `SMOE_AHEAD=3` | 18.7 | 更差，AHEAD=2 仍最优 |
| `PREFETCH_JOIN=1` | 崩 | 该路径有独立问题（默认 0，暂不用） |

**⑤ 52.4 ms/token 的构成（基线 c1）**：
```
split_partition 20.6ms  ├─ prologue 17.9ms（其中 mrs_queue 17.0ms = 等 GPU 产出本层 router）
                        └─ ids_wait 1.4 + partition 1.2 + act_d2h 0.5
inputs          31.9ms  （输入拷贝段，含 flag_input 的事件等待）
compute         18.1ms  （MoE 图入队）
cpu             12.4ms  （CPU 半边，部分与 GPU 重叠；CPU_ASYNC=1 时 5.7ms）
gpu_queue        5.7ms  （CPU_ASYNC=1 时 8.4ms：join 代价）
```
→ 配置旋钮已到顶（19.9–20.5 平台）。要再上一个台阶只能动**结构**：devpart（§6.3/6.5）
消除逐层 router 会合（~17ms/图），`CPU_ASYNC=1` 正好是它的前置基础设施。


### 6.11 2026-09-12 深夜四：CPU_ASYNC 转正 + 逐项开销盘点（当前 48.7ms/token）

**① `LLAMA_MOE_CPU_ASYNC` 默认改为 1**（用户批准）：MoE CPU 半边派给 worker，与 GPU 半边重叠，
调度线程只做 join。实测 **21.5 / 21.4 t/s**（此前基线 19.9–20.0），文本正确（逐字与基线一致的那段）。
```
total 51.0 → 48.7ms   cpu 12.4 → 6.2ms   compute 17.2 → 13.1ms   gpu_queue(join) 5.3 → 6.9ms
```

**② 开销盘点（每 token，均有实测）**：

| 项 | ms | 是什么 | 压缩办法 |
|---|---|---|---|
| **逐层 router 会合** | **17.0** | `prologue mrs_queue`：主机等 GPU 产出该层 router 输出才能分区 | **devpart**（结构级；§6.5 正确性未解）。已验证 `MRS_FULL=0` 净收益 0（等待搬到 `ids_wait`） |
| **split 启动开销** | **13.1** | 142 个 split/图 × 92µs 的调度+replay | 减少每层 split 数（结构级）。CUDA graph **已生效**：`GGML_CUDA_DISABLE_GRAPHS=1` 时 total 81.8ms、compute 62.7ms |
| CPU 半边 | 6.2 | （本轮已从 12.4 降下） | 已做 ✓ |
| `flag_input` 事件等待 | 5.7 | 双缓冲输入的后端 event 同步 | 更多 in-flight 拷贝（改动小但收益不确定） |
| `d2h_enq` | **3.7** | **分页 D2H 的入队阻塞**：调度器 ids 读回 + MRS 分数（256 floats/层） | 把这两个主机缓冲改成 **pinned**（SMoE staging 已是 pinned；这是调度器的 ids 缓冲与 MRS 分数缓冲）→ 小而确定 |
| `expert_copy`/`generic`/`evt` | ~1.4 | 输入拷贝本体 | — |
| MRS 全套 | ~0.4 | 已是死重（覆盖率 0.3%），但去掉净收益 ≈0（§6.4 的等待搬移，本轮 m1 复证） | 需要与 ids 读回一并重构 |

**③ 潜在上限**：48.7 → ~28ms（≈35 t/s）需要 devpart(17) + 减少 split(13) 两块结构改造；
近期低风险可拿的是 **pinned 读回缓冲（~3.7ms → 45ms ≈ 22 t/s）**。

**④ `inputs=32.7ms` 与 `split_partition=20.6ms` 是重叠计时**（partition 发生在输入段内）：
输入段真正拷贝工作 ≈ 11ms（flag 5.7 + d2h 3.7 + generic 1.1 + expert 0.3）。


### 6.12 2026-09-12 深夜五：pinned 读回（无效）+ split 结构 + devpart 与自适应系统的兼容性

**⚠️ 构建状态**：最后一次**成功**构建 = pinned 读回版（已验证：21.7 / 21.5 t/s、文本正确）。
之后为"split 结构转储"打的补丁**只编译、链接失败**（DLL 被占用）→ **磁盘上的 exe 相对源码是旧的**，
下次必须先 `cmd /c build-cli.bat` 再测量。源码里现有两处**临时诊断**（env 关闭时无副作用）：
`LLAMA_MOE_CRASH_TRACE=1`（崩溃栈）、`LLAMA_MOE_DUMP_SPLITS=1`（split 组成转储，**尚未跑过**）。

**① pinned 读回缓冲：实测无效，但保留了**
把调度器 ids 读回与 MRS 分数读回的目的地改成 pinned（`moe_cache_d2h_begin/end`，3 处）后：
```
改前: d2h_enq=3785µs  d2h_sync=1µs     ids_wait=1.2ms   total=48.7ms
改后: d2h_enq=4µs     d2h_sync=3787µs  ids_wait=17.3ms  total=48.3ms
```
入队确实变成真异步（µs 级），但**等待立刻搬到紧随的 synchronize** —— 这 3.7ms 本来就是那 17ms
router 会合的一部分（pageable 只是让它躲在入队里）。与 §6.4/§6.11 的"等待只会搬移"完全一致。
**保留**（中性；计时口径变诚实；将来会合消除后它才产生价值）。

**② split 启动开销（13.1ms）的结构**：`split_partition` 的钩子每图跑 **142.6 次 ≈ 48 层 × 3 个权重种类**
（gate/up/down；其中 real=47.5 做真正分区、early=95.1 直接早退 0.00ms）→ 每层被切成 3 个 split，
`compute`（`ggml_backend_graph_compute_async`）13.1ms ≈ 每次 92µs。**合并方向 = 让同一层的 3 个权重种类
落在同一个 split**（可省 ~2/3 的 compute 与部分 inputs 段）。判断可行性需要先看 split 组成 →
`LLAMA_MOE_DUMP_SPLITS=1` 的转储就是为此写的（**未运行**）。

**③ devpart 与自适应系统的兼容性（重要，尚未实施）**：
devpart 把分区放到**设备侧** → 主机不再读回 `part.ids` → **我们整套热区机制的输入信号会消失**：
真值频次淘汰分（§6.7）、回填排序、每排名准确率/yield 统计（§6.8/6.9）全部依赖它。
**兼容做法**：把使用直方图放到**设备**算（对 router ids 做 scatter-add 到每层计数缓冲，或复用 top-k kernel），
再**每 token 批量读回一次**（48 层 ×256×4B ≈ 49KB ≈ 可忽略）→ 既消掉 17ms 的逐层会合，
又保住热区信号。**否则就是拿 +20pp 命中率去换 17ms，不划算。**


### 6.13 2026-09-12 深夜六：split 合并调查 —— **那 13.1ms 是用 VRAM 换来的，不做**

**结构（用 `LLAMA_MOE_DUMP_SPLITS=1` 实测）**：decode 每图 **242 splits / 8978 nodes / 1601 leafs**
（prefill 是 145，拓扑不同；两张 decode 图完全一致 → **图不增长**，排除泄漏）。每层 5 个 split：
```
attn 块(119节点) + gate(1) + up(2) + down(21+VIEWs) + CPU MOE_CPU(1) + router 块(136~218)
```

**gate/up/down 被切开的原因**（`ggml_backend_sched_split_graph` pass 5）：
```cpp
// check if a weight is on a different and incompatible backend
// by starting a new split, the memory of the previously offloaded weights can be reused
if (src->buffer->usage == WEIGHTS && src_backend_id != cur_backend_id && !supported)
    need_new_split = true;
```
`--cpu-moe` 的专家权重常驻 CPU（WEIGHTS + 后端不兼容）→ 每个 `MUL_MAT_ID` 切一刀。

**关键实测（两次尝试都失败，但结论明确）**：
1. **图构建前 patch `src[0]`→cache view**：n_splits 仍 242（钩子命中的张量不对），并引入 fail-fast 崩；
   要成功必须在 llama.cpp 图构建期做，且得**自己接管非 direct 层的兜底拷贝**（调度器输入拷贝会消失）→ 风险不低于 devpart；
2. **关掉调度器那条规则（让 B/C/D 合并，兜底拷贝路径不变）**：形式上是真低风险，但
   **VRAM 峰值 13200 → 15868 MiB（> 15360 limit）→ OOM 崩** —— 因为合并后三个权重输入必须同时驻留，
   每层多 ~2×专家包。**这正解释了规则为何存在：它是用切分换权重暂存内存。**
→ **结论：不做**。这 13.1ms 不是白拿的，是 VRAM 换来的。
   若将来要做，唯一合理形式是"**缩小 cache 容量换合并**"（如 cache 4GB + 合并 → 峰值约 13.9GB 可容纳），
   本质是**命中率 vs 启动开销**的交换，需要实测比较。

**当前状态（已回退并验证）**：`back2 = 21.6 t/s、VRAM 13198MiB、文本正确` ✓
另有**偶发收尾崩 0xC0000005**（约 30–50% 运行，只丢统计不影响正确性）——我自己引入的那个确定性崩已修，
这个原有的尚未根治（`LLAMA_MOE_CRASH_TRACE=1` 的追踪器仍在，可继续抓栈）。


### 6.14 2026-09-13：prefill/decode 倾向分离（实测）

**新增诊断**（退出时打印）：`prefill/decode tendency: graphs pre=.. dec=.. | decode topC=..% seeded-by-prompt-topC=..% | touched experts/layer pre=.. dec=..`
- prefill 侧计数必须在 `used_ids` 解析处采（**分区钩子只跑 decode**，原先 prefill 的路由完全不可见）；
- decode 侧沿用分区钩子；两者分开统计并以 Q8 定点计数（decode 行 = 256，prefill 行 = 256×`LLAMA_MOE_PREFILL_WEIGHT`），
  衰减/淘汰/回填的排序全部尺度不变 ✓ 实测命中率与改动前一致（70.4%）✓ **无回归**。

**实测（同配置，仅提示词不同）**：

| 工作负载 | decode topC | 提示词热集可覆盖 | prefill 触达/层 | decode 触达/层 |
|---|---|---|---|---|
| 6-token 短提示 | 85.8% | **17.9%** | 150.4 | 211.6 |
| ~150-token 长提示 | 72.2% | **52.0%** | 209.8 | 247.8 |

**结论**：
1. 两阶段的**集中度本质不同**：prefill 并行 → 每层触达 ~82%（209.8/256）的专家、分布近乎平坦；
   decode 串行 → 集中在前 64 名（覆盖路由的 72.2%）。所以**用同一套策略对待两者是错的**。
2. **提示词当冷启动种子的价值随长度剧增**：6 token 只有 17.9%，150 token 达 **52.0%（= 可达上限 72.2% 的 72%）**。
3. 落地：prefill 路由**现在进统计**（原先完全没有）+ 权重旋钮 `LLAMA_MOE_PREFILL_WEIGHT`（默认 1.0；
   长提示场景建议 0.1，使其只作先验、不淹没 512-token 滑动窗）。本工作负载下权重 0.1 与 1.0 命中率一致（60.3%）——
   因为 prefill 质量本就只占 ~13%；旋钮留给"长提示+短生成"的场景。
4. 后续可做（未做）：把**提示词热集**直接用作 seed（现成机制是 `LLAMA_MOE_PIN_STATIC` 的静态清单；
   可换/叠加为"本次 prefill 的 top-C"，实测已给出该做法的上限 = 52%）。


### 6.15 2026-09-13：SMoE/缓存对 prefill 的影响（实测：**中性**）

同一长提示、`-n 8`（prefill 时长提示 ~150 token），只改机制开关：

| 配置 | Prefill (Prompt t/s) | Decode (Gen t/s) | VRAM |
|---|---|---|---|
| 全开（SMoE+缓存+回填） | **44.5** | **11.8** | 9257 |
| SMoE 关（`PREDICT_SMOE=0`） | 44.4 | 12.1 | 9255 |
| SMoE 关 + 缓存关（`CACHE_MIB=0`） | 44.7 | 8.9 | 7223 |
| 全关（stock `--cpu-moe`） | 42.7 | **8.7** | 7223 |

**结论**：
1. **prefill 不被这套机制拖累**（44.5 vs stock 42.7，差值噪声级）→ 不需要为 prefill 做优化，
   也说明"prefill 被 SMoE 拖慢"的担心不成立；
2. **decode 的收益来自"缓存"**（缓存关 → 8.7 ≈ stock；开 → 11.8，`-n 8` 下仍被冷启动填充段主导，
   稳态见 §6.11 的 21.5+）；
3. 两阶段策略该分离的地方是**统计与用途**，不是准入：prefill **不进缓存**（现状已如此：
   走 `moe_copy_experts_grouped` 拷进调度器输入，不污染缓存）；prefill 的正确用途是**种子**
   （长提示 top-C 可覆盖 decode 路由 52.0% = 上限的 72%，见 §6.14）；
4. 可做未做：把"本次 prefill 的热集"当 seed 灌入缓存（替换/叠加 `LLAMA_MOE_PIN_STATIC` 的静态清单）。


### 6.16 2026-09-13：prefill 读缓存（D2D 暂存）—— **实测零收益，已回退**

**想法**：prefill 现在把"用到的专家并集"从主机内存 H2D 拷进设备暂存；其中已驻留缓存的那些
可以改成 **D2D 从缓存取**（缓存只读、不写 ✓ 符合"prefill 计算期间不搬运"的约束 ✓），
按 25% 驻留率估算每提示可省 ~4.8GB H2D ≈ 370ms。

**实现**（`moe_copy_experts_grouped` + `LLAMA_MOE_PREFILL_CACHED`，保持"连续专家合并拷贝"，
按驻留性切段：驻留→D2D、非驻留→原 H2D；`prefill_d2d` 计数器）→ 构建通过。

**实测（交互式两轮自对照，同一提示词，第1轮 prefill 冷 / 第2轮 prefill 暖）**：
```
turn 1: Prompt: 37.3 t/s | Generation: ...
turn 2: Prompt: 37.3 t/s | Generation: ...     ← 完全一致，无可测增益
```
**原因**：prefill 的暂存拷贝用的是 **`ggml_backend_tensor_set_async`（异步）** → H2D **本来就与 GPU
计算重叠**了，把它换成 D2D 省不出时间 ✗。这同时解释了 §6.15 的"prefill 中性"：
**prefill 是 GPU 计算受限，权重搬运被藏住了**。
→ 已**回退**（不留无收益的复杂度）；结论：prefill 侧不需要缓存感知，两阶段分离的价值只在
"统计/先验"（§6.14）与"decode 侧边界填充"（§6.13）。

**顺手得到的两条工具能力**（保留）：
- 交互式多轮测试可用（用 `hub start` 起 `llama-cli` 不带 `-p`，PTY 下它会读 stdin；每轮都会打印
  `Prompt: X t/s | Generation: Y t/s` ✓ 注意这个构建**没有 `-i`/`-cnv`**）；
- `prefill/decode tendency` 诊断行（含 top-C 重叠度）常驻 ✓。

---

## 7. 复现命令

```powershell
cmd /c build-cli.bat                              # 构建
powershell -NoProfile -ExecutionPolicy Bypass -File run-cur-ref.ps1       # devpart OFF（真基线）
powershell -NoProfile -ExecutionPolicy Bypass -File run-cur-devpart.ps1   # devpart ON
iconv -f UTF-16LE -t UTF-8 cur-ref-out.txt | tr -d '\r' | grep -a "Prompt\|Generation"
grep -a "MOE-CACHE\] inputs breakdown\|MOE-CACHE\] smoe per" cur-ref-err.txt   # 分项计时
```

模型 `F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf`，
`-ngl 49 --cpu-moe --no-mmap -c 8192 --cache-type-k q8_0 --cache-type-v q8_0 -p 'The capital of France is' -n 32 --temp 0`。
关键 env：`LLAMA_MOE_CACHE_MIB=2048`、`LLAMA_MOE_SPLIT=1`、`LLAMA_MOE_DIRECT_READ=1`、
`LLAMA_MOE_PREDICT_SMOE=1`、`LLAMA_MOE_PREFETCH=1`、`LLAMA_MOE_MRS=1`、`LLAMA_MOE_DEVPART`(0/1)、
`LLAMA_MOE_CACHE_TIMING=1`、`LLAMA_MOE_CACHE_STATS=<csv>`、`LLAMA_TOKEN_PROF=1`。
