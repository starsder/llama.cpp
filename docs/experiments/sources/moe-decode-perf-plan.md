# 历史来源快照：moe-decode-perf-plan.md

> 本文是原始研究笔记的归档，不是当前推荐配置或验收结论。包含后来撤回的数字、未实现的计划、未发布的代码描述及尚未解决的问题。请先读[档案索引](../README.md)及分主题复盘。
> 仅将维护者本机路径替换为 SOURCE_TREE / LOCAL_EVIDENCE / LOCAL_MODELS / OTHER_SOURCE_TREES / USER_HOME；这些是路径标记，不是已上传的资源。原文的错误与前后更正不静默改写。

- 原来源：`docs/moe-decode-perf-plan.md`（本地工作区快照，不能假定与已发布 master 相同）。
- 原文件 SHA-256：`89eda1b24485a05a01b1eda6dc7bd0996b549d4ad18c519d02e2dd59713dda40`。
- 以下分隔线后为历史正文；主题文档引用的 § 编号及原始行号指向未加本页说明的来源。

---

# MoE decode 性能:实测剖面与改动点

日期: 2026-09-11 | 构建: `build-ple-trace-mrs` | 基线 commit: `2f1a363c8`

所有数字都来自**同一个二进制**、**同一份配置**、**31 个 decode 图**的实测。
统计口径是逐图 CSV(`stats-*.csv` 的 `-1`/`-2`/`-3`/`-4` 行),**已剔除 warmup/prefill 图**。

---

## 0. 一句话

decode 当前 **86 ms/token(11.3 t/s,输出正确)**。其中约 **40 ms 是每层都在跟 device 硬会合**,
另有 **~20 ms 是 CPU 半边算本该命中的专家**(命中率只有 26.8%,而预测准确率有 73%)。

目标 25 t/s(40 ms/token)。**只拆会合到不了(21.7 t/s),必须同时把预取交付率提上去。**

---

## 0.1 选定路线:借壳,加一个算子

**不重写引擎,也不继续在 scheduler 里打补丁。** 保留 llama 的壳与算子
(GGUF 解析、IQ3_XXS、flash attention、分词器、采样、除 MoE 外的整张图),
把**这一个模型的 MoE 层**收成**一个自持算子**,由它自己管执行。

### 边界

| 留在 llama | 进算子 |
|---|---|
| GGUF 加载 / mmap / 权重布局 | 专家缓存与驻留策略 |
| IQ3_XXS 反量化与 GEMM kernel | GPU/CPU 分区决策 |
| attention / RoPE / dense / PLE / KV | 预取、插入 worker、淘汰 |
| 分词器、采样、CLI、server | SMoE/Fate 预测的落盘与触发 |
| **图构建**(含 router 的 top-k 节点) | 自持 stream、栅栏、批量回传 |
| 共享专家 FFN(GPU 常驻,独立于 MoE) | 专家拷贝(D2D / H2D) |

算子签名(6-7 个 src,`GGML_MAX_SRC = 10`,放得下):

```
GGML_OP_MOE_QWEN4EXP(cur, selected_experts, weights, gate_exps, up_exps, down_exps[, candidates])
    -> 与 GPU 半边 + CPU 半边之和等价的一个 device 张量
```

共享专家**不进算子** —— 它只依赖 `cur`、GPU 常驻、且是 SMoE 的输入之一,
留在 llama 的图里更自然。SMoE 的 side graph 由 llama 构建,
其候选结果作为下一个 MoE 算子的 `candidates` 输入。

### 为什么这直接实现"防止乱改 llama"

现在 `ggml-backend.cpp` 里那 **3484 行** MoE 缓存 / split / devpart 调度钩子
**全部搬出去**,放进 `ggml/src/ggml-cuda/moe-qwen4exp.*`。
`ggml_backend_sched_compute_splits` 回到接近上游的状态。

顺带消失的东西:
- devpart 的 tensor 生命周期 bug(算子显式持有自己的 buffer,不再和分配器斗)
- 每 split 边界的跨后端拷贝(不再有 split)
- `moe_insert_flush` 每层硬排空(算子自己决定何时同步)
- 那 195 次 × 400µs 的 `generic` 拷贝(批量一次 D2H)

### 分阶段(每步可验证)

| 阶段 | 内容 | 验证 | 估时 |
|---|---|---|---|
| **M0** | 加一个**空算子** `GGML_OP_MOE_QWEN4EXP`,内部直接调现有 MoE 路径,**行为零变化** | 输出 token id 逐位一致;算子在图中可见 | 0.5 天 |
| **M1** | 把缓存 + 分区搬进算子,执行逻辑照旧 | 输出逐位一致;`ggml-backend.cpp` 的 MoE 钩子删除 | 1-2 天 |
| **M2** | 栅栏换型:`synchronize` → `event wait`;自持 stream;批量 D2H | decode 从 86ms 降到 ~46ms | 1-2 天 |
| **M3** | 预取 / 预测搬进来,准入模型重写,窗口调优 | 命中率与 `prefetch_ready` 上升 | 1-2 天 |

**M0 是关键的第一步**:它只证明"缝"存在(算子注册、CUDA 分发、图构建都对),
不改变任何行为,失败成本半天。**先做 M0。**

### 两个真实风险

1. **算子内驱动 CPU 半边。** CPU 半边要用 ggml-cpu 的线程池跑 IQ3_XXS GEMM。
   算子跑在 CUDA 后端,要能拿到并驱动 CPU 后端。若这条路不通,
   退路是**保留 CPU 半边为独立 split**,但让它只吃一个极小的接口
   (批量 id 表),而不是现在的逐层拷贝。
2. **CUDA graph 捕获。** 动态策略 vs 静态图。先走"排除捕获"
   (`ggml_cuda_graph_check_compability` 里对 `MUL_MAT_ID` 已有先例),
   确认功能与性能后,再考虑把内部做成 capture-safe。

### 明确不做的事

- 不写新的 GGUF 解析、分词器、attention kernel
- 不动 IQ3_XXS 的 CPU/CUDA kernel(PRD 第 122 行已划为独立分支)
- 不碰 `--cpu-moe` 对**其它模型**的行为(`LLM_FFN_EXPS_REGEX` 那条路径原样保留)

---

## 1. 实测账本(devpart OFF,输出正确)

```
total  85,990 us
├─ cpu    27,676  (32%)  CPU 半边算未命中的专家(真活)
├─ gpu     7,391  ( 9%)  GPU 入队
└─ pre    49,727  (58%)
    ├─ inputs 29,115
    │    ├─ split_partition     23,233   n=144   ← 里面 host CPU 循环 22,724
    │    ├─ flag_input           4,267   n=108
    │    ├─ generic              1,054   n= 51
    │    └─ (ids_d2h = 0, expert_copy = 0, activation_D2H = 509)
    └─ drain   20,457                    ← SMoE 事件等待 ~17ms + prefetch 提交 ~4ms
```

`split_partition` 有 per-graph 缓存(`part.graph_id == s.graph_id` 直接返回),
**144 次调用里只有 48 次做实事**,其余早退。那 23ms 集中在开头四行:

```cpp
4253: moe_cache_activate_layer(s, layer);
4256: moe_insert_drain(s);
4262: moe_insert_flush(s);              // ← 阻塞条件变量,等 worker 队列清空
4266: moe_cache_slot_events_drain(s);
```

`moe_insert_flush`:

```cpp
s.insert_cv.wait(lock, [&]{ return s.insert_queue.empty() && s.insert_inflight == 0; });
```

注释说明它是**故意**的:避免侧流拷贝越过 CUDA graph 捕获边界。代价是每层一次硬排空。

### devpart ON 的对照(同一二进制)

```
total 122,912
├─ cpu     3,247   ← devpart 省掉 24.5ms  ✓
├─ gpu     7,182
└─ pre   111,370
    ├─ split_partition      0      ← devpart 省掉 23.2ms  ✓
    ├─ generic         78,151  n=195 × ~400us  ← 爆炸  ✗
    └─ drain           32,431
```

**devpart 的账算反了**:分区搬到 device 是对的,但 CPU 半边跑在 host 上,
`ids_cpu/wgt_cpu` 还是要通过调度器的跨后端拷贝送回来 —— 195 次小拷贝 × 400µs,
比原地在 host 算(23ms)贵 3 倍。

---

## 2. 缓存与预取的真实数字(devpart OFF)

```
hits=11982  misses=32688                    → 命中率 26.8%
prefetch_required  = 14570
prefetch_predicted = 10684  (73.3%)         → 预测准确率:好
prefetch_ready     =  3643  (34.1%)         → 交付转化率:差
prefetch_dropped   =  8092                  → 65% 候选被丢
smoe_predictions   =     1                  → 全程只跑 1 次?(需查,疑似计数或真 bug)
```

**关键区分:预测器工作正常(73%),坏的是交付(34%)。**

丢弃发生在准入判据里(3400-3405):

```cpp
const double eta_us = (inflight_copies + n_copies) * gate_copy_us      // gate_copy_us = 70us(硬编码)
                    + (inflight_bytes + bytes) / gate_bw_bps * 1e6;    // gate_bw_bps = 20GB/s(硬编码)
const double deadline_us = layers_until_visit * layer_us_ewma;         // 按"层数"算
return eta_us <= deadline_us;
```

两个模型问题:

1. `gate_copy_us = 70us` 从未标定,而**实测每次小传输是 150-400us** → 低估 2-5 倍。
2. `deadline` 按"层数"算,但真实约束是"剩余时间"。SMoE 在 FFN 末尾才开火,
   **提前量 ≈ 0**,而模型以为有整整一层。

---

## 3. 改动点

### P0 — 纯测量(零风险,先做)

| # | 改动 | 位置 | 目的 |
|---|---|---|---|
| 0.1 | 四个会合点各加计时器 | `ggml-backend.cpp` 4253/4256/4262/4266 | 定案那 23ms 到底是 `insert_flush` 还是别的 |
| 0.2 | `dma_inflight_copies/bytes` 峰值与均值计数 | 每图 CSV 加一行 | 判断丢弃是"积压"还是"提前量不足" |
| 0.3 | 查 `smoe_predictions=1` | `moe_cache_smoe_enqueue` | 付了 17ms 却没产出 |
| 0.4 | 保留整套 decode 级计时器 | `-1/-2/-3/-4` CSV 行 + stderr 分解 | 后续所有改动的基线 |

**0.2 的结果决定 P1 里改哪个**:积压 → 标定/批量化;提前量不足 → 换序前移。

### P1 — 低风险直接修复(语义不变)

| # | 改动 | 预期 | 风险 |
|---|---|---|---|
| 1.1 | `deadline` 从"层数"改成"剩余时间"模型 | 准入判据与现实对齐,丢弃率下降 | 低(只影响预取准入) |
| 1.2 | 标定 `gate_copy_us`(实测 300 左右)或用启动自标定 | 不再过度准入→积压→全丢 | 低 |
| 1.3 | 预取 3 份合成 1 份(gate/up/down 本就是连续 layer bundle) | `eta` 主导项砍到 1/3 | 中(要确认 bundle 连续性) |
| 1.4 | 共享专家 FFN 换序前置 + SMoE 用 `input + shared` 开火 | **提前量从 0 变成整个 routed MoE** | 中(准确率会掉,但转化率应大涨) |
| 1.5 | 窗口 1 层 → 2-3 层 | deadline 翻倍 | 低(1.4 成立后才有意义) |

**1.4 是杠杆最大的一条。** 共享专家 FFN 只依赖 `cur`,和 routed MoE **没有依赖边**,
本来就可以先算。现状是它在 MoE 之后构建,所以 SMoE 只能在 FFN 末尾开火。

取舍算得清:现在 73% 准确 × 34% 交付 = 26.8% 命中;用一个略低的准确率 × ~90% 交付,
净收益为正。

### P2 — 结构性改动(需要设计)

| # | 改动 | 收益 | 说明 |
|---|---|---|---|
| 2.1 | `insert_flush` 每层全排空 → 图级一次 | ~23ms | 只等本层真正需要的 slot 事件,其余留在途 |
| 2.2 | SMoE 回读逐层 → 每 token 一次 | ~17ms | 预测结果服务的是**下一 token**,晚一层不影响正确性 |
| 2.3 | devpart 的 `ids_cpu/wgt_cpu` 批量回传 | 抵消那 78ms | 48 层的 ids 合计只有 ~4KB,图尾一次 D2H |
| 2.4 | 准入模型重写(基于真实带宽与剩余时间) | 命中率 | 见 P1.1/P1.2 的深化 |

**2.1 - 2.3 有一个共同前提**:它们要求的都是"图级(而非层级的)host 会合"和
"跨层批量"。`ggml_backend_sched` 是**逐 split** 的模型,表达不了。这是 B/C 路线之争的实证依据。

### P3 — 正确性/稳定性(与性能无关,单独排队)

| # | 问题 | 证据 |
|---|---|---|
| 3.1 | devpart 输出**不确定** | 同配置同二进制 temp=0:run1 近似正确 / run2 重复循环 |
| 3.2 | devpart **崩溃** | `dsh-G2` exit `0xC0000005` 访问违例 |
| 3.3 | 保活 view **没有**修复乱码 | 当前源码含保活 view(2366-2371),实测仍错 |
| 3.4 | 附带 bug:驻留表初始为 0 | `moe_cache_finalize` 在首图 compute 之后才跑;0 是合法 slot,静默按 slot 0 命中 |
| 3.5 | 共享专家 GPU 驻留是**隐式命名约定** | `--cpu-moe` 正则 `\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps` 不匹配 `ffn_*_shexp`;代码里无任何断言。模型改名即静默落 CPU,且不报错 |

3.5 建议加一句显式检查 —— 因为 SMoE 预测器**依赖**共享专家在 GPU 上
(`smoe_hidden = ffn_input + smoe_gpu_out + ffn_shexp_gated`),它落 CPU 会让预测器多一次逐层 D2H。

### P4 — 预填充(独立立项)

预填充 **不走这条路**:`llama-graph.cpp:2334` 的 `if (moe_split && n_tokens == 1)`。
split / devpart / SMoE 只在单 token decode 上跑。

所以 26 → 60 t/s 是**另一个问题**(CPU 专家 GEMM 吞吐 + 专家 H2D 量)。
PRD 第 122 行已把 CPU IQ3 反量化 kernel 划为独立分支。

**当前没有任何有效的预填充测量** —— `[ Prompt: 26.2 t/s ]` 是 5 token 的 prompt 算出来的。
先补一次真实长 prompt(2k / 8k)的基线,再谈优化。

---

## 4. 目标可达性

```
decode 目标 25 t/s = 40 ms/token,当前 86 ms

P2.1 + P2.2(拆掉 40ms 逐层会合)               → ~46ms = 21.7 t/s   ❌ 差一点
  + P1.1/P1.2/P1.4(把交付率提上去,cpu 降)   → 26-32ms = 31-38 t/s ✅
```

`cpu` 那 27.7ms 与命中率直接挂钩:命中率 26.8% → CPU 算 73% 的专家。
命中率回到 60% 即可把 `cpu` 压到 ~11ms。

**且 P2.1 是 P1 的前提**:逐层排空既直接吃 23ms,又让侧流反复被掐断,
把 `dma_inflight` 行为喂坏,进而喂坏那个准入判据。

---

## 5. 已落地的改动

| 改动 | 位置 | 效果 | 回退 |
|---|---|---|---|
| 单副本 scheduler 也建事件 | `ggml-backend.cpp` 5359 附近 | decode 107.9 → 85.0ms;190 次 `cudaStreamSynchronize` 变 `cudaStreamWaitEvent` | `GGML_SCHED_NO_SINGLE_COPY_EVENTS=1` |
| decode 级计时器 | `ggml-backend.cpp` 多处 + CSV `-2/-3/-4` 行 | 让 23ms 与 17ms 从"藏在 inputs/pre 里"变成可见 | 无(仅 `LLAMA_MOE_CACHE_TIMING=1` 时输出) |

---

## 6. 复现

```powershell
cmd /c build-cli.bat                      # sccache 已配置,-j 16(32 会死机)
powershell -NoProfile -ExecutionPolicy Bypass -File dsh-probe-09.ps1

# decode-only 统计(剔除 gid < 5 的 warmup 图)
#   -1 行: gid,-1,total_us,ids_wait,ids_parse,copy,0,cpu,gpu,pre
#   -2 行: gid,-2,in_wait,in_scan,d2h_enq,d2h_sync,d2h_n,d2h_bytes,in_parse,in_expert,drain,inputs
#   -3 行: gid,-3,flag_us,flag_n,gen_us,gen_n,loop_n,hostw_n
#   -4 行: gid,-4,cpuhalf_us,cpuhalf_evt_us,cpuhalf_n,splitpart_us,splitpart_n,sp_d2h_us,sp_d2h_n
```

模型与运行参数见 `run-2g-devpart.ps1` / `dsh-probe-06.ps1`。

---

## 7. 这份文档否定掉的旧结论

排查过程中被实测推翻的假设,记录在此以免重复:

| 旧结论 | 实际 |
|---|---|
| ids 回读 46ms 是结构瓶颈 | decode 图里回读 **0 次**;旧数字是 warmup 图污染平均值 |
| 专家拷贝 3.5ms/token | decode 期间 **0 次**,该分支根本不进 |
| host_weights 分支每次 16 次 | decode 期间 **0 次** |
| D2H 是瓶颈 | 每次 activation D2H 只有 ~10us,48 次共 509us |
| 保活 view 修复了乱码 | 当前构建含保活 view,实测**仍错且不确定** |
| devpart OFF = 19-21 t/s | 实测 **10-11.6 t/s**,无任何历史输出支持 19-21 |
| 20.7 → 8.4 是性能悬崖 | 20.7 那批是**乱码的旧构建**;同构建下没有这个悬崖 |
| CPU 2.7ms 说明分区失效 | 高命中率下正常;但实际命中率只有 26.8%,该数字另有原因 |
