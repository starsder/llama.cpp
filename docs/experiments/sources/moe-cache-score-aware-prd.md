# 历史来源快照：moe-cache-score-aware-prd.md

> 本文是原始研究笔记的归档，不是当前推荐配置或验收结论。包含后来撤回的数字、未实现的计划、未发布的代码描述及尚未解决的问题。请先读[档案索引](../README.md)及分主题复盘。
> 仅将维护者本机路径替换为 SOURCE_TREE / LOCAL_EVIDENCE / LOCAL_MODELS / OTHER_SOURCE_TREES / USER_HOME；这些是路径标记，不是已上传的资源。原文的错误与前后更正不静默改写。

- 原来源：`docs/moe-cache-score-aware-prd.md`（本地工作区快照，不能假定与已发布 master 相同）。
- 原文件 SHA-256：`70f6d7d302257d276b39c0541953b533e5c8a96d2315a5390b0af8a95db19cac`。
- 以下分隔线后为历史正文；主题文档引用的 § 编号及原始行号指向未加本页说明的来源。

---

# MoE 缓存：真实频率、共享池与位置权重

状态：共享池位置权重已实现，仍为显式实验开关；默认每层缓存配置不变。

## 当前策略

- 保留实际路由产生的专家使用频率；CPU/GPU 实际路由都贡献频率，预测本身不增加频率。
- `LLAMA_MOE_GLOBAL_POOL=1` 让所有 `(layer, expert)` 共用物理槽位；默认仍是每层独立缓存。
- `LLAMA_MOE_LAYER_AWARE=1` 在共享池的频率评分上乘位置权重，日志标识为 `policy=LFU_POS`。要求 `GLOBAL_POOL=1`、`MRS=1`、`FIFO=0`、`EVICT_SCORE=0`；组合不兼容时明确警告并忽略该开关。
- `MRS` 是现有日志名称；当前 `EVICT_SCORE=0` 的淘汰依据是真实路由频率，不是历史 gate-softmax 分数。后者由 `EVICT_SCORE=1` 选择，不与本位置策略混用。
- 未发布的 Least-Stale 实验及其环境开关已移除；不叠加预测 epoch、FIFO 或预测集合的永久保护。

令 `N` 为缓存管理的 MoE 层数，`c` 为当前层序号，`l` 为候选专家所属层序号。序号来自有序层表，不假定模型层号连续：

```text
future = l > c
d = future ? l - c : l - c + N
weight = (future ? 1.0 : 0.5) * N / (N + d)
retention_score = actual_use_frequency * weight
```

淘汰保留评分最低的可用槽位，同分仍按已有 LRU tick 处理。`d` 是下次访问的循环距离，本层取 `N`；未计算的层有更高权重，同一类别内距离越近越优先。位置因子不修改原始频率，因此足够热的旧层专家仍可压过冷的未来层专家。权重只在执行游标变化时重算。

## 共享池的内存安全边界

- GPU 正在读取的物理槽位、尚未完成写入的槽位以及固定槽位不可淘汰。读取保护在现有 backend 同步点释放，不把历史预测列表当成永久保护集。
- direct-read 共享池保留一个全零槽位作为 padding；其容量必须计入物理池，但不能计入可驻留专家数。512 MiB 实测为 211 个物理槽位、210 个可用槽位；本机 auto 为 3840/3839。
- CUDA MMVQ/MMQ 的权重 channel/sample stride 全程使用 `int64` 字节偏移，先定位字节基址再解释量化块。行内 stride 仍按量化块处理；每层默认布局不变。
- 本模型共享池 pitch 为 2,534,400 字节，不能被 82 字节量化块整除。不能先除块大小再乘 channel，也不通过全模型 LCM 膨胀槽位来掩盖问题。
- direct-view 与 gathered-copy 的 `src[0]` 绑定切换时更新 graph UID，保留 CUDA Graph 功能，避免旧图把专家 ID 当成共享池 slot ID。

旧共享池在第 29 个解码步骤附近出现 CUDA 非法读取；原始 memcheck 捕获到了槽位容量为 211 时的专家 ID 450 寻址。修复后七次 128 步压力运行均完成。独立 CUDA 冒烟覆盖 IQ2_S、IQ3_S、IQ4_NL 的 MUL_MAT_ID、4D 广播、1/4/17 token，以及单 token fused SwiGLU：24 个用例全部与紧凑布局逐项一致，memcheck 报告 0 errors。该检查不是完整模型的 memcheck，也未覆盖多 GPU 或 NVFP4 实机执行。

## 准入与归因

- CPU 实际路由 miss 的回填、GPU 预测准入、hot/idle/seed/phase fill 是不同来源，本轮没有改变它们的默认启停或准入阈值。
- `prefetch_experts` 是公共写入路径的准入总数，包含非预测填充；`hits` 是缓存访问命中计数。二者都不能直接称为预测器命中率。
- 比较器同时保存准入环境值、后端有效设置、各来源计数和原始日志。修复了共享池准入时未写入既有 `slot_rank` 元数据的问题。
- 下表运行的 `INSERT_ON_MISS=0`、`FALLBACK_PREFETCH=0`，hot/idle/seed/phase 计数均为 0。因此不能把结果推广到历史 CPU 热度回填配置，也不能把历史回填收益归给预测器。
- 原有 hot-set oracle 假设每层各有 C 个槽位；共享池日志明确将其标为反事实的 per-layer 容量假设，不用于共享池容量或收益判断。

## 当前性能验收基线

用户明确要求：后续性能结论统一基于 **400 token、6144 MiB 专家缓存、权重完整驻留主存**，采用此前扫描器的有效最优参数作为对照。用户后续将当前验收下限由 20 调整为 **19 token/s**；19.3 token/s 可作暂定对照，不再仅因低于历史 20 判作回退。超过下限仍需与相同条件的有效基线比较，并满足正确性要求。其他软件可能占用资源，但未证实它是速度差异的原因。

- 加载保持 `--no-mmap --lazy-mode off`，不得用 lazy、未完整装入权重或跳过计算换速度。保持单实例锁与至少 90000 MiB 空闲内存闸门。
- 保留 CPU 实际使用回填与非阻塞预测：有效历史 6 GiB / 400-token 封板使用 `HOT_BACKFILL=8`、`SMOE_NONBLOCK=1`、`SMOE_AHEAD=2`、`CPU_ASYNC=1`，host split、direct-read、每层 MRS，`PREFETCH_JOIN=0`、`FALLBACK_PREFETCH=0`。`warm2`、`freeze`、`fin1` 原始日志分别记录约 20.3/20.4/20.2 token/s、64 槽/层、72.6 GiB 页锁定权重缓冲。
- 此前扫描器为 `sweep-vision-cache.py`，参数与结果保存在 `cases-*.txt` / `sweep-*.csv`。其不同轮次的上下文、KV、缓存预算、实际环境和步数不同，不能仅按最高 t/s 抄参数。后期 `AHEAD=3` 候选须在同一 400-token / 6 GiB / 全内存工作点比较，不能直接把 auto / 256-token 的排序当作本工作点结论。
- 自适应传输预算用于限制预取对关键路径的拖累，不代表已解决传输本身的成本。减少某个等待桶不等于缩短整步耗时：历史 pinned 回读将 `d2h_enq≈3785 µs` 移到 `d2h_sync≈3787 µs`，总耗时仅 48.7→48.3 ms；devpart 也已有等待转移及净性能无收益的记录。见 `handoff.md` §6.12、§6.23–6.25。
- 以下 128-token、回填关闭的结果仅保留为短回放／压力诊断证据，**不再作为生产性能基线，也不据此确定优化优先级**。不得把 512 MiB 压力结果与上述 6 GiB 稳态吞吐混比。

### 当前二进制的基线复测

使用带安全闸的 `tools-run.py`，恢复上述 6 GiB / 400-token / 全内存参数，比较历史 `AHEAD=2` 与后期候选 `AHEAD=3`；其他受控参数相同，PLE 两级缓存关闭，CPU-KV/QSA 关闭。两次均实际分配 64 槽/层、6116.3 MiB 物理缓存，页锁定 72.6 GiB 权重缓冲且无失败。

| 提前量 | 结果 | Generation | 验收 |
|---|---|---:|---|
| AHEAD=2 | 正常退出，单次记录 | 19.3 token/s | 通过当前 19 下限，作为暂定对照 |
| AHEAD=3 | 子进程 `0xC0000005`；逐图 CSV 到 graph 403，但没有最终吞吐输出 | 不可用 | 运行失败，不纳入性能排名 |

上述结果未证明优化或选出了新最优参数。历史 `freeze` 启用了 PLE 缓存，本次按扫描器的全内存设置关闭，因此不能仅凭这一次差值归因于某次代码改动。停止继续扫描失败分支，原始日志、逐图 CSV 与受控参数保存在 `LOCAL_EVIDENCE/moe-cache/baseline400-6g.summary.json` 及其列出的证据文件中；该历史报告保留当时的 20 token/s 判据，当前验收按上面的 19 token/s 执行。

### 400-token 驻留期观测

`tools-run.py --profile baseline-6g` 复用上述有效参数，默认 400 token 并开启 `--ignore-eos`；不指定 profile 时保留原来的 32-token 默认行为。`--json PATH` 保存实际 argv、受控环境、退出码、超时、原始日志路径、缓存及驻留计数。失败运行即使留下吞吐行也不参与性能验收；缺失数据记为不可用，不填成零。

```console
python tools-run.py --tag lifetime400 --profile baseline-6g --json LOCAL_EVIDENCE/moe-cache/lifetime400.json LLAMA_MOE_CACHE_LIFETIME=1
```

`LLAMA_MOE_CACHE_LIFETIME` 默认关闭。开启后按 `(layer, expert)` 追踪占用、淘汰、重新装入和真实 GPU 使用，在退出时汇总一行 `[MOE-LIFETIME]`，不逐次打印命中。未开启时不分配逐专家追踪数组。统计区分预测与非预测准入；CPU 实际路由频率不等同于 GPU 驻留期使用次数，零权重 padding 也不算实际使用。

需要同时检查 `available`、`use_complete` 和完整性错误计数。无法完整观测的 devpart 路径会明确降级，不能据其未用计数推断浪费。admission 表示逻辑槽位占用；pending 仍不能被计算读取。字节按各权重分量的 stride 汇总，既不是共享池物理 pitch，也不是 PCIe 链路实测字节。

同一 400-token / 6 GiB / 完整主存工作点的观测基线正常退出，Generation 为 19.3 token/s，与未加观测的 19.2 token/s 复测生成文本一致；这两个单次结果不构成提速证据。

| 驻留期指标 | 基线计数 |
|---|---:|
| 准入／淘汰／结束时占用 | 10216 / 7272 / 2944 |
| 预测／非预测准入 | 7026 / 3190 |
| 淘汰前未被 GPU 使用 | 4530，其中预测 3559、非预测 971 |
| 未用即淘汰的逻辑字节 | 8,972,577,948 |
| 淘汰后重新装入 | 4734，其中间隔不超过 1/4 个 graph 为 696/1638 |
| 实际 GPU 专家使用 | 133484 |
| 已驻留请求去重 | 29682 |

该运行满足 `admissions - evictions = live`，且 `gpu_uses × 3 = hits = 400452`；重复占用、无记录淘汰和非驻留使用均为 0，`use_complete=1`。后端预取字节计数为 20,224,040,448。证据说明本工作点存在较多未用淘汰和重新装入，**不是反复计算命中导致重复传输**；尚不能据此断言 PCIe 已饱和，或把减少字节直接等同于整步提速。

原始记录见 `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json` 及其引用的日志。

## 此前短回放诊断（非性能验收基线）

环境：RTX A5000 Laptop 16 GiB、Ryzen 5950X、16 个推理线程、Qwen3.8-Flash-Next UD-IQ3_XXS；84-token 代码提示，固定外部 replay，128 步解码，每侧三轮重复。性能剔除前 16 步，表中为三个单次中位数的中位数；正确性覆盖 prefill 与全部 128 步。计时只包含 `llama_decode + llama_synchronize`，不包含 logits 拷贝和文件写出。

采用未修改的 `tools-run.py` 基线环境，`lazy-mode=off`、`LLAMA_QSA_HOST_KV=0`，显存限配置为 15667 MiB；并非历史 v1 的严格 15 GiB 配置。所有模型运行串行、持有 `.tools-run.lock`、启动前至少 90000 MiB 空闲内存，未关闭默认专家页锁定。

| 配置 | 物理缓存 MiB | 可驻留专家槽位 | 解码中位 ms | 命中计数中位 | 淘汰次数中位 | 采样显存峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|
| 每层 MRS，auto | 9270.1 | 每层 97 | 61.6195 | 76245 | 9 | 15378 |
| 共享池 MRS，auto | 9281.2 | 总共 3839 | 62.9095 | 76245 | 0 | 15306 |
| 共享池 LFU_POS，auto | 9281.2 | 总共 3839 | 62.9310 | 76245 | 0 | 15306 |
| 共享池 MRS，512 MiB | 510.0 | 总共 210 | 77.6790 | 31527 | 8610 | 6534 |
| 共享池 LFU_POS，512 MiB | 510.0 | 总共 210 | 76.1885 | 38076 | 8557 | 6534 |

- auto 共享池没有发生淘汰，不能证明位置淘汰策略有效。其全部运行与修改前基线的 prefill/128 步 logits 逐位一致；候选相对共享池 MRS 中位耗时为 +0.03%。
- 512 MiB 下命中计数中位提高 20.77%，耗时中位下降 1.92%。MRS 三轮中位数为 77.6790/80.7960/76.3055 ms，候选为 76.8140/75.6135/76.1885 ms；耗时差异小于基线运行间的波动，不能据此宣称稳定加速。
- 压力候选三个运行均有 1/128 步 top-1 与压力 MRS 参考及修改前基线不同，发生在零起算 decode step 79。候选对压力参考的 mean KL 为 0.002296–0.004157，最大绝对 logit 差为 3.193697。所有比较均有限、prefill 一致，但压力结果不满足逐位或全部 top-1 等价。
- MRS 自身重复也存在数值差异；这不是把基线差异任意放大后判定候选“无损”的依据。比较器只报告原始差异，不设置此类正确性容差。
- **保持默认每层缓存，不自动启用共享池或位置权重。** 本轮确认了功能及压力下命中变化，未获得切换默认所需的性能与数值等价证据。

证据目录为 `LOCAL_EVIDENCE/moe-cache/`：`global-auto128.summary.json`、`layer-auto128.summary.json`、`global-pressure128-v2.summary.json`、`prechange-cross-comparison.json`，以及对应原始日志、manifest、logits 和 replay。旧运行摘要中曾生成的放大噪声容差字段不作为验收依据；当前脚本已移除该判定。`stride-memcheck.log` 保存独立 CUDA 检查结论。

比较器为 `tools/tuning/moe-cache-compare.py`。本机复现命令：

```console
python tools/tuning/moe-cache-compare.py --tag position-auto-repeat --size 128 --reps 3 --threads 16 --prompt-file LOCAL_EVIDENCE/moe-ls/prompt-code.txt --replay-tokens LOCAL_EVIDENCE/moe-ls/prechange.replay.tokens.txt
python tools/tuning/moe-cache-compare.py --tag position-pressure-repeat --size 128 --reps 3 --threads 16 --cache-mib 512 --prompt-file LOCAL_EVIDENCE/moe-ls/prompt-code.txt --replay-tokens LOCAL_EVIDENCE/moe-ls/prechange.replay.tokens.txt
python tools/tuning/moe-cache-compare.py --tag per-layer-auto-repeat --size 128 --reps 3 --threads 16 --global-pool 0 --variants mrs --prompt-file LOCAL_EVIDENCE/moe-ls/prompt-code.txt --replay-tokens LOCAL_EVIDENCE/moe-ls/prechange.replay.tokens.txt
```

超时/子进程失败立即停止后续模型运行，保存原始退出码和可用证据，缺失的 manifest、令牌、行比较标为不可用，不伪造空数组的一致性。实际 1 秒超时验证：比较器保存摘要并返回 6，子进程状态为 124；`tools-run.py` 返回 124。无法回收的存活子进程保留 PID 锁，不以释放锁允许下一次并发启动。

## 待排查问题：压力下命中收益未转化为稳定加速

状态：开放。本轮已在 400-token / 6 GiB 工作点增加上述驻留期观测；它不能直接替代原 128-token / 512 MiB 压力案例的归因，也未证明链路带宽或汇合等待是瓶颈。

### 现象与证据

- 512 MiB 共享池、固定 replay、128 步解码、每侧三轮：位置权重使命中计数中位从 31527 增至 38076（+20.77%），但预取字节中位仅从 17399519232 降至 17294964224（约 −0.60%），淘汰次数从 8610 降至 8557。
- 解码中位耗时从 77.6790 ms 降至 76.1885 ms（−1.92%），差异小于基线重复波动，尚无稳定加速证据。记录来源：`LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json`。
- 预期语义：已驻留专家的预取请求应去重；在途专家也应避免重复提交；只有被淘汰后再次装入等情况才应重新传输。**反复命中驻留专家本身不产生重复传输。**
- 此前用“反复命中但仍持续搬运”解释现象混淆了计算命中与预取去重，不能作为根因结论。目前既未证明去重失效，也未证明传输带宽已达上限。

### 后续排查与关闭条件

- 按 `(layer, expert)` 及槽位占用代次关联预取请求、resident/pending 去重、实际拷贝提交与完成、淘汰、重新装入、实际命中。核实计数字段代表请求、提交还是完成，区分统计口径问题与真实重复传输。
- 验证驻留或在途期间是否仍提交了重复拷贝；把真实传输拆分为首次装入与淘汰后的重新装入，说明新增命中为何没有明显减少传输。
- 测量 H2D 活跃期间的有效带宽、拷贝粒度、等待时间及 CPU/GPU 汇合关键路径；不能仅凭总字节和总耗时推断 PCIe 饱和。
- 关闭此问题需要解释命中、去重、重新装入与实际传输之间的对应关系；若存在错误则修复并复验，若机制正确则给出实际瓶颈证据。

### 关联正确性风险：不能先排除时序错配

- 压力候选三轮均在零起算 decode step 79 出现 top-1 翻转（基线 token 271，候选 token 248046）。关闭位置权重也存在 logit 重复波动；尚未确认是本次引入的故障、已有 CPU/GPU 算子差异，还是异步槽位/图绑定问题。
- 旧版共享池曾在约第 29 步因图绑定与 ID 语义错配发生非法读取。该崩溃已修复，但不能因此认定其他时序问题均已排除。合法地址中的错误专家数据也可能不触发 CUDA 报错。
- 已有 24 个 CUDA 用例验证的是 GPU 紧凑布局与带跨度布局，不是 CPU/GPU 算子等价，也不是完整模型的缓存时序检查。
- 已保存的 18 次正式运行文本逐字节一致，但使用了强制 token replay；尚无 MRS/位置权重的自由生成配对结果，不能将回放文本一致当作自由生成一致。
- 后续先定位第一次 logit 分歧，核对实际专家身份、槽位所有者和写入完成状态；分别用强制同步、关闭 CUDA Graph 做诊断对照，并控制实际执行分配变化，再判断是否需要 CPU/GPU 算子级比较。

以下 v1 保留为设计背景；与上文不同的默认值、数值等价目标和运行约束不代表当前配置或验证结果。

---

## 历史 v1：MoE 层间预测与分层 MRS 缓存

状态：冻结设计 v1  
目标模型：Qwen3.8-Flash MoE 量化模型  
运行约束：`--lazy-mode on`、`--cpu-moe`、峰值显存不超过 15 GiB

### Problem Statement

当前模型拥有大量 MoE 专家，单个 token 只激活少量专家，但专家权重无法全部放入 16 GiB 显存。CPU-MoE 可以保证模型运行，但专家缺页、主机到设备传输和缓存抖动成为主要瓶颈。

现有缓存路径已经具备部分预测、预取和层内 bundle 能力，但仍需要固定以下行为：

- 缓存必须以 `(layer, expert)` 为管理单位。
- 同一层的 gate、up、down 权重必须共享一个 slot 生命周期。
- 预测结果不能改变原始路由和模型输出。
- 预测缓存与实际路由分数缓存不能混为一个策略。
- pending 的异步传输不能被当前计算读取或被淘汰。
- 缓存预算必须服从 15 GiB 总显存上限。

### Solution

实现一个保持原始 MoE 路由不变的层本地专家缓存：

1. 使用 XT 转移表作为低开销的 Fate-style 下一层预测器。
2. 使用单层窗口，当前层预测下一层，不默认进行三层预取。
3. 使用 HybriMoE 的 Minus Recent Score（MRS）作为层内淘汰策略。
4. 以完整路由分数更新 MRS，以预测结果只决定预取候选。
5. 使用每层独立 bundle buffer、slot 表和 pending 状态。
6. cache hit 走设备内复制或 direct-read，cache miss 仍由 CPU-MoE 计算并异步 warm。
7. 动态收紧缓存预算，确保总目标显存不超过 15360 MiB。
8. 增加可选的 SMoE 共享专家引导在线预测：用当前层输入、缓存中 GPU routed 专家贡献和常驻 shared expert 贡献构造近似 hidden，计算下一层 gate，仅把候选用于预取。

### User Stories

1. As a model user, I want to run the large MoE model with lazy mode enabled, so that dense and inactive expert weights do not unnecessarily consume device memory.
2. As a model user, I want CPU-MoE to remain enabled for cache misses, so that an absent expert never changes the model’s routing semantics.
3. As a model user, I want to configure an explicit expert cache budget, so that the model can run on a 16 GiB GPU.
4. As a model user, I want the runtime to enforce a 15 GiB peak VRAM target, so that cache allocation cannot exhaust memory needed by the model, KV cache, or graph workspace.
5. As a cache manager, I want each `(layer, expert)` pair to have an independent residency record, so that an expert from one layer cannot evict an expert from another layer.
6. As a cache manager, I want gate, up, and down weights for one expert to share one bundle slot, so that a partially resident expert is never treated as a GPU expert.
7. As a cache manager, I want all weight components of a bundle to complete before the slot becomes visible, so that asynchronous prefetch cannot expose stale or incomplete data.
8. As a cache manager, I want the current layer’s GPU expert set to be protected during a graph, so that direct-read execution cannot observe an eviction race.
9. As a predictor, I want the current layer’s routing activity to provide candidates for the next layer, so that transfers can overlap with current computation.
10. As a predictor, I want the prediction window to remain one layer by default, so that prefetch traffic does not consume the bandwidth needed by the next immediate layer.
11. As a cache policy, I want actual routing scores to determine long-term residency, so that high-score but not-yet-selected experts are not discarded by pure LRU.
12. As a cache policy, I want only the top `P` current routing scores to update the history, so that low-score noise does not churn the cache score table.
13. As a cache policy, I want demand misses to be able to evict a predicted expert when necessary, so that prediction is a performance hint rather than a correctness barrier.
14. As a cache policy, I want prefetch to avoid evicting current GPU experts, pending slots, and pinned slots, so that prefetch cannot invalidate the current computation.
15. As a model user, I want MRS to be switchable off, so that LRU and MRS can be compared under identical model and memory settings.
16. As a model user, I want static hot experts to be non-pinned by default in MRS mode, so that a small cache can adapt to the current route distribution.
17. As a model user, I want the original top-k expert IDs and weights to remain unchanged, so that enabling the cache cannot change generated text because of approximate routing.
18. As a performance engineer, I want per-layer hit, miss, prediction, prefetch, and CPU fallback counters, so that a speedup can be attributed to a specific mechanism.
19. As a performance engineer, I want route-score transfer time measured separately from expert transfer time, so that score collection overhead is visible.
20. As a performance engineer, I want to compare MRS and LRU at 512 MiB, 2 GiB, and 8 GiB cache budgets, so that the policy can be evaluated where cache pressure is meaningful.
21. As a performance engineer, I want every benchmark to record peak VRAM, so that a throughput result above the memory limit is rejected.
22. As a maintainer, I want the cache to fall back safely when the route-score tensor or prefetch API is unavailable, so that unsupported backends still produce correct results.
23. As a maintainer, I want the predictor implementation to be replaceable, so that a future Fate gate predictor can be evaluated without changing cache storage or eviction semantics.
24. As a maintainer, I want the implementation to avoid changing the model graph’s routing operators, so that the feature remains an inference-system optimization rather than a model architecture change.

### Implementation Decisions

- The cache storage unit is one complete expert bundle within one layer. Each layer owns its own expert-to-slot and slot-to-expert maps.
- Decode uses a small persistent rolling working set per layer. A physical layer-window is optional for one-pass/prefill experiments, but is not the default because autoregressive decode revisits layer 0 on every token.
- A layer slot contains all weight kinds needed by the MoE operator. The GPU path may use an expert only when every required component is resident and not pending.
- The production predictor is now selectable: `Fate` runs the paper’s online cross-layer gate, while `CrossLayer`/`CrossToken` retain the offline transition manifests for comparison. In Fate mode, the named `ffn_moe_gate_input-i` hidden is copied to host when layer `i` routing is available, multiplied by layer `i+1`’s gate weights on the CPU, and reduced to a byte-bounded candidate set. The native router and model graph are unchanged.
- `SMoE` is an additional online predictor for decode: `ffn_smoe_hidden-i = ffn_input-i + cached_gpu_routed-i + shared_expert-i`, followed by the next layer’s native gate matrix. Only its top candidates are written into the next-layer prefetch queue; CPU-only routed misses are intentionally absent from this approximate signal, and the native route remains authoritative. Enable with `LLAMA_MOE_PREDICT_SMOE=1`; it requires split mode and is limited to one-token decode graphs.
- The predictor produces an ordered candidate list. It does not modify router logits, top-k IDs, router weights, CPU/GPU partition correctness, or expert computation.
- The prefetch window is fixed to one subsequent layer. The default candidate count is twice the model’s activated expert count, bounded by the layer slot capacity.
- Prefetch admission is byte-bounded by the explicit cache budget and candidate cap; actual unexpected misses may feed back into the next graph only through a separately bounded fallback budget, disabled by default.
- MRS state is stored independently for every layer and expert. The initial score is zero. On every decode graph, the runtime updates only the top `P` route scores using an averaging coefficient of `alpha = 0.75`; `P` defaults to twice the activated expert count.
- Route scores are read from the most semantically appropriate graph tensor: masked selection probabilities when present, otherwise biased selection probabilities, otherwise the unbiased probability tensor. For this model, the unbiased probability tensor is expected to be sufficient when no expert-group mask is active.
- Telemetry distinguishes route prediction coverage, prediction-ready coverage before the next layer, resident cache hit rate, prefetch bytes, fallback-prefetch admissions, and GPU/CPU wait time. Aggregate cache hit rate is not treated as a proxy for prediction quality.
- In split mode, route IDs, selected weights, and route scores are queued before one backend synchronization. Score collection must not introduce a second synchronization per layer.
- If full route scores are unavailable, the implementation may update a degraded MRS state from selected expert weights, but telemetry must identify that fallback. It must not silently claim full-score MRS behavior.
- Victim selection is per-layer. The order of exclusions is pending slots, pinned slots, experts used by the current GPU path, and then policy candidates. Among eligible experts, MRS chooses the lowest score, with the LRU tick as the tie breaker.
- Prediction protection is soft. Prefetch may skip predicted residents when an alternative victim exists; a demand miss may evict any eligible expert with a lower MRS score.
- Static hot experts are not pinned by default. A separate pin budget remains available for experiments but is excluded from the MRS default configuration.
- Prefetch and cache warming use the existing asynchronous side stream. A bundle becomes resident only after all component transfers finish. The main compute path must never read a pending slot.
- Direct-read is allowed only when the layer’s shared slot map is frozen and all weight kinds use the same layer cache. Otherwise the implementation uses the ordinary device-to-device gathered-copy path.
- Cache allocation is a gross byte budget covering all layer bundles, alignment, and component padding. The runtime clamps the requested cache budget against a 15360 MiB total device target and a workspace/KV safety reserve. If no complete layer slot fits, caching is disabled without changing CPU-MoE behavior.
- The cache policy is opt-in for compatibility. MRS mode is enabled explicitly for A/B tests; the existing LRU path remains available as the control.
- The canonical MRS experiment configuration is: lazy mode enabled, CPU-MoE enabled, split mode enabled, one-layer XT prefetch enabled, MRS enabled, `alpha = 0.75`, `P = 2K`, and static pinning disabled.
- Telemetry must expose cache policy, effective byte budget, effective layer slot count, route-score source, MRS updates, MRS-selected victims, LRU tie breaks, prefetch usefulness, prefetch waste, CPU fallback work, H2D bytes, and peak VRAM.
- The implementation should keep policy logic behind a small cache-policy boundary so that LRU, MRS, and a future predictor can be tested independently from CUDA transfer and tensor-layout code.

### Testing Decisions

Good tests assert externally visible behavior: generated output remains equivalent, cache state never exposes incomplete bundles, memory stays within the limit, and the selected policy changes hit/miss behavior in the expected direction. Tests should not depend on private slot numbers unless the slot mapping is the externally observable safety contract.

The following validation layers are required:

- Build validation: compile the existing release target after each cache-policy change.
- Correctness differential: run the same deterministic prompt with cache disabled, LRU enabled, and MRS enabled. Compare generated token IDs or a stable output digest.
- Cache lifecycle validation: exercise empty slots, demand fills, repeated hits, pending prefetches, bundle completion, layer transitions, and demand eviction while prediction protection is active.
- Policy validation: replay a deterministic route trace through LRU and MRS at multiple capacities. Verify that MRS updates only the configured top `P` scores and that the selected victim is the lowest eligible score.
- Prediction validation: verify that Fate candidates come from the next-layer gate evaluated on the current-layer gate input, SMoE candidates come from the approximate shared-plus-cache-resident hidden, XT candidates target only the next layer, predicted experts are not used as actual route IDs, and a missing manifest disables only manifest prediction rather than the cache or CPU-MoE path. Record gate CPU/GPU time separately from prefetch-ready coverage.
- Direct-read validation: verify that all three weight components resolve the same `(layer, expert)` slot and that a pending or missing expert cannot be read through a direct view.
- Backend fallback validation: run with prefetch unavailable or route-score tensors unavailable and verify correct CPU-MoE execution plus explicit degraded telemetry.
- Memory validation: run the standard model with 512 MiB, 2 GiB, and 8 GiB requested cache budgets while sampling device memory. Every run must use `--lazy-mode on` and `--cpu-moe`; no run is accepted if peak usage exceeds 15360 MiB.
- Performance validation: compare LRU and MRS with identical prompt, context, cache budget, thread count, lazy mode, CPU-MoE mode, and split mode. Record tokens/s, CPU fallback time, H2D bytes, cache hit rate, and peak VRAM.
- Regression validation: run the existing short direct-read and ordinary D2D cache smoke tests after changing the eviction policy.

Prior art for the experiment design is the existing command-line cache smoke-test and trace scripts in the project, together with the current layer-local bundle implementation. No new permanent test fixture under the general test suite is required for this first PRD; deterministic route replay and CLI differential tests are the appropriate seams.

Acceptance criteria:

- MRS and LRU produce equivalent deterministic model output.
- No pending bundle is used by GPU compute or selected as a victim.
- All cache state is layer-local and all required weight kinds share one slot decision.
- The standard benchmark configuration stays at or below 15360 MiB peak device memory.
- MRS telemetry proves that full-score updates, degraded updates, and victim choices are distinguishable.
- Under at least one pressure budget, MRS reduces cache misses or H2D traffic versus LRU without causing a performance regression greater than the measurement noise at the same workload.

### Out of Scope

- HyperMoE’s hypernetwork or HyperExpert model architecture. It changes model computation and is unrelated to this inference cache.
- Training or fine-tuning a hidden-state MLP predictor.
- Replacing the native router, changing top-k, dropping experts, or altering route weights.
- The full HybriMoE dynamic CPU/GPU timeline simulator and its intra-layer scheduling algorithm.
- Default three-layer prefetch. It may be evaluated later only after one-layer traffic and overlap are measured.
- Hard-coded shallow-layer caching. Layer allocation can be revisited after collecting model-specific traces.
- Rewriting the CPU IQ4/IQ3 dequantization kernel. That is a separate performance branch and must not be conflated with cache-policy gains.
- Claims of 20 tokens/s or any fixed speedup before the controlled A/B measurements are complete.
- Configurations that intentionally exceed the 15 GiB peak VRAM target.

### Further Notes

Fate’s key transferable result is that adjacent-layer gate inputs can support low-overhead next-layer expert prefetching, while its shallow-favoring cache behavior is model- and budget-dependent. The implementation now uses the paper’s online CPU gate path when `LLAMA_MOE_PREDICT_FATE=1`; the offline XT manifest remains an explicit A/B control. See [Fate](https://arxiv.org/abs/2502.12224).

HybriMoE defines MRS as an exponentially averaged top-P route-score history and reports the strongest benefit under constrained cache capacity. This PRD ports that policy per layer while retaining the project’s CPU-MoE fallback and layer-local bundle invariant. See [HybriMoE](https://arxiv.org/abs/2504.05897) and the [official implementation](https://github.com/PKU-SEC-Lab/HybriMoE).

The term HyperMoE is intentionally excluded from the implementation decision: the ACL paper describes a hypernetwork-based MoE model that transfers information among experts, not a cache or prefetch kernel. See [HyperMoE](https://aclanthology.org/2024.acl-long.571/).

The current measured layer-local cache baseline remains the reference point for the next A/B run. New MRS results must report both policy delta and memory delta; a higher tokens/s number without the corresponding hit, transfer, and VRAM counters is insufficient evidence.
