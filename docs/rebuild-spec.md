# Qwen4Exp 重建规格(rebuild spec)

目标:在**干净的 llama 基点上**,把 Qwen4Exp 的 MoE 执行从 llama 的 split 调度器里
搬出来,做成**一个自持算子**;顺带把 MoE 缓存、预取、预测、TBQ 全部收进一个自洽的模块。

前置阅读:`docs/moe-decode-perf-plan.md`(实测剖面与改动点)、`docs/moe-cache-score-aware-prd.md`(策略 PRD)。

---

## 1. 基点与目标

### 基点

| 候选 | 说明 |
|---|---|
| `a9e9c3c5` | fork 上 `mtp/qwen4exp-nextn` 的 tip。**严格包含 `b76199698`(已验证是祖先)**,多 72 个上游 commit,自带 qwen4exp + MTP。**推荐** |
| `b76199698` | 当前基线,9 天前,也自带 qwen4exp |
| ~~master `e9e0d9923`~~ | **不可用**:没有 qwen4exp,差 230 commit |

```powershell
git worktree add F:/src/llama-qwen4exp-clean a9e9c3c5
```

### 验收标准

| 指标 | 当前 | 目标 |
|---|---|---|
| decode(无 MTP,2G 缓存) | 11.3 t/s | **>= 25 t/s** |
| 输出正确性 | — | 与旧路径 **token id 逐位一致**(temp=0) |
| prefill | 无有效测量 | >= 60 t/s(独立立项) |
| 峰值显存 | — | <= 15360 MiB |

**正确性是硬门槛**:任何"更快但输出变了"的结果一律作废。

---

## 2. 边界:什么留在 llama,什么进算子

| 留在 llama | 进算子 |
|---|---|
| GGUF 解析 / mmap / 权重布局 | 专家缓存与驻留策略 |
| IQ3_XXS 反量化与 GEMM kernel | GPU/CPU 分区决策 |
| attention / RoPE / dense / PLE / KV cache | 预取、插入 worker、淘汰 |
| 分词器 / 采样 / CLI / server | SMoE 预测的触发与消费 |
| **图构建**(含 router top-k 节点) | 自持 stream、栅栏、批量回传 |
| 共享专家 FFN(GPU 常驻,独立于 MoE) | 专家拷贝(D2D / H2D) |
| TBQ 的 KV 侧实现 | —— |

**共享专家不进算子**:它只依赖 `cur`、GPU 常驻、且是 SMoE 的输入之一。
留在 llama 图里更自然,也避免了 SMoE 跨越算子边界。

### 为什么边界划在这里

撞车的**只有 `ggml_backend_sched`**:它同时持有 tensor 生命周期、执行顺序/边界、同步点,
而这三样必须由 MoE 执行自己持有。算子边界就是所有权边界。

---

## 2.2 调度归属:引擎边界(不可协商)

**原则:调度、并行、栅栏全部由我们持有。llama 只负责"调用一次算子"。**

### 结构

```
llama 的图:  ... -> attn(L) -> MOE_OP(L) -> attn(L+1) -> MOE_OP(L+1) -> ...
                                    | 一次分发
                  +----------------------------------------------+
                  |  引擎(持久对象,跨层跨 token 存活)             |
                  |   - 自有 compute stream / prefetch stream     |
                  |   - 专家缓存(分层分页,未来接 SSD)            |
                  |   - 自有 CPU 线程池(GPU/CPU 半边真并行)      |
                  |   - 预取队列 + 事件 + 预测器状态               |
                  |                                              |
                  |   入口: llama stream --event--> 我们的 stream  |
                  |   内部: 自己排程,只在真实依赖处立边界          |
                  |   出口: 我们的 stream --event--> llama stream  |
                  +----------------------------------------------+
```

**引擎是持久对象,不随算子生命周期生灭。**
因此 `MOE_OP(L)` 可为 `L+1` 发起预取,`MOE_OP(L+1)` 直接消费
—— **跨层预取流程在引擎内部闭环,llama 完全不可见。**

### 会被"打断"的五件事,逐一封死

| 打断源 | 封死方式 |
|---|---|
| 调度器在 split 边界插拷贝 | 算子输入(`cur` / `topk`)本就在 GPU 上,不产生跨后端拷贝 |
| **CUDA graph 捕获** | **显式排除本算子**(`ggml_cuda_graph_check_compability` 对 `MUL_MAT_ID` 已有先例);动态排程不可捕获 |
| 图分配器接管 buffer | 缓存 buffer **在图外持久分配**,不交给 `ggml_gallocr` |
| 调度器的事件同步 | 只在出口记一个 event,llama 的 stream 等它 —— **一次**,不是每层 |
| 后端归属不确定 | 输入在 CUDA 上 ⇒ 算子被分到 CUDA;显式断言 |

### 护栏指标(硬验收项)

重建完成后,以下两个 diff 必须接近零:

```
git diff <base> -- ggml/src/ggml-backend.cpp     ->   0 行        (重建前 +3484)
git diff <base> -- src/llama-graph.cpp           ->   仅构建算子的几行 (重建前 +159)
```

**它们一旦重新增长,就说明滑回深度修改。**
这条优先于任何吞吐数字 —— 它决定架构能否撑到第二阶段(SSD tier)。

---

## 2.3 内存层级路线图与模块划分

### 两个阶段

| 阶段 | 存储层级 | 目标 |
|---|---|---|
| **一**(现在) | 128 GiB RAM + 16 GiB VRAM | prefill >= 60 t/s / decode >= 25 t/s |
| **二** | + SSD,专家从 SSD 流式读入 RAM | 同目标下支持更大模型 / 更小 RAM |

### 关键发现:PLE 已经是第二阶段的原型

`src/models/models.h:2293` 的 `ple_row_cache`:

```cpp
void gather(rows, out);
void copy_pages(page_indices, out);   // "Materialize complete raw pages for a lower-level cache.
                                      //  This keeps the CPU L2 in the path;
                                      //  callers never reach into the mmap directly."
void prefetch_async(rows, rows_per_token);
```

**`copy_pages` 就是通往更低一层的桥** —— 它刻意不让调用方直接碰 mmap。
加上 `ple_gpu_row_cache`(GPU L1),PLE 已经是一个两级半结构:

```
SSD/mmap -> CPU L2(LRU 页缓存, 63 KiB/页, 728 行/页) -> GPU L1 -> 算子读
```

**而 MoE 专家缓存是同一件事的另一份独立实现**(单元 =(layer, expert),bundle ~2 MB)。
两份代码在解同一个问题。

### 模块划分(修正后)

不是"一个大 MoE 算子",而是:

```
Qwen4Exp 加速器
├── 分层分页权重存储 + 预取引擎          <- PLE 与 MoE **共用同一套机制**
│   ├── PLE 实例    单元 = token 行,页 63 KiB
│   └── 专家实例    单元 = (layer, expert),bundle ~2 MB
├── 预测器(SMoE / Fate / XT)             <- 共用,输出候选
└── MoE 算子(分区 + GPU/CPU 半边 + merge)
```

差异只有三项:**页大小、访问模式、预测器**。
淘汰策略、tier 管理、异步预取 + 事件栅栏**完全同构**。

llama 侧只看到 MoE 算子(以及 PLE 的 hook),共用引擎在其下。

### 两条硬约束(现在就要满足,否则第二阶段重来)

**1. 预取的 copy 源必须抽象化,不得写死 host 指针。**
现在的 `moe_cache_copy` 直接用 `input->data`(pinned host)。
第二阶段源头是 SSD,**必须是 `copy_pages` 式的"从下一层物化一页"接口**。

**2. 提前量必须参数化,不得写死 1 层。**

```
RAM -> VRAM : ~2 MB @ 10 GB/s  ≈  200 us      1 层提前量够(1.8 ms/层)
SSD -> RAM  : ~2 MB @  3 GB/s  ≈  700 us + 延迟
两跳叠加    : 需要 2-3 层,甚至跨 token 的提前量
```

现有的 `deadline = layers_until_visit * layer_us_ewma` 是**单跳且按层数**的模型。
**重建时就写成多跳、按剩余时间**,否则第二阶段要再改一次数据流。

---

## 2.4 专家聚类存储(第二期,但设计上不许预闭)

### 形态:纯置换,不写自定义 GEMM

专家按**共激活关系聚类**存放(经常一起出现的放一块),顺序变为簇序。
**量化类型不变、张量形状不变,只是 expert 维被置换。**

⇒ **ggml 的 IQ3_XXS kernel 原样可用**,只需多一层 id 映射。

对比:变长簇 / 按簇分精度 才需要自定义布局与 gather kernel —— **本方案不需要。**

### 为什么它正好打当前瓶颈

```cpp
// 现有准入判据的主导项:
eta = (inflight_copies + n_copies) * gate_copy_us    // n_copies = 3 (gate/up/down)
//                                   ^ 每次拷贝 150-400us 的固定开销
```

聚类后一次传输覆盖**一整簇**共激活专家:簇内连续 ⇒ `n_copies` 3 -> 1,
且第二阶段 SSD 从随机读变顺序读。**直接对症 `prefetch_dropped = 8092`(65%)。**

本质上:**聚类布局 = 把离线预测器编码进权重布局**,SMoE 在线预测只是它的精细化。

### 诚实说明:它不构成"单干"的理由

纯置换下,llama 的 GGUF loader **仍然可用** —— 专家张量仍是 `[n_embd, n_ff, n_expert]`,
置换表是独立小张量(48 层 x 512 专家 x 2B ≈ 48 KB)。

**逼我们单干的是调度器**(实测净负 29%、每层会合 40 ms),不是权重格式。

**这带来一个好处:两期解耦,不必一次做完。**
```
第一期:引擎(自有调度 + 缓存 + 预取 + 预测)
第二期:聚类布局 + 自有容器
```

### 三条不许预闭

1. **预取的 copy 源必须可换层**(RAM / SSD),不得写死 host 指针。
2. **提前量必须参数化**(多跳、按剩余时间),不得写死 1 层、不得按层数算。
3. **逻辑 expert id 与存储位置必须分开两层抽象。**
   现在混用(`expert_slot[expert_id]`),一旦混用,第二期就得改缓存核心。
   正确形态:`logical_id --置换表--> storage_slot`,缓存按 storage_slot 键。

### 待实测定夺的取舍:簇大小

```
簇小(2-4 专家,~4-8 MB)    命中精准;传输次数多,per-copy 开销仍在
簇大(8-16 专家,~16-32 MB)  摊销固定开销;带进不需要的专家,占缓存
```

按实测 per-copy 开销(150-400 µs)与 2 GiB 缓存预算,初估 **4-8 专家/簇**为甜点,须实测。

### 第二期需要的额外工作

- 容器:聚类顺序 + 置换表(可扩 GGUF,或自有格式)
- 离线工具:从路由轨迹算共激活矩阵 → 聚类 → 生成置换表
- 加载器:读置换表,建 `logical -> storage` 映射(第一条抽象的下游)
- 缓存:slot 表改按 storage_slot 键

---

## 3. 算子规格

```cpp
// ggml.h
enum ggml_op {
    ...
    GGML_OP_MOE_QWEN4EXP,
};

// src[0..6]:
//   [0] cur                 F32 [n_embd, n_tokens]     层输入激活(已在 device)
//   [1] selected_experts    I32 [k, n_tokens]          router top-k
//   [2] weights             F32 [k, n_tokens]          router 权重
//   [3] gate_exps           host 权重                  路由专家 gate
//   [4] up_exps             host 权重                  路由专家 up
//   [5] down_exps           host 权重                  路由专家 down
//   [6] candidates          I32 [n_cand] (可选)        上一层 SMoE 给的预取候选
//
// 输出: F32 [n_embd, n_tokens]   == GPU 半边 + CPU 半边之和
struct ggml_tensor * ggml_moe_qwen4exp(
    struct ggml_context * ctx,
    struct ggml_tensor  * cur,
    struct ggml_tensor  * selected_experts,
    struct ggml_tensor  * weights,
    struct ggml_tensor  * gate_exps,
    struct ggml_tensor  * up_exps,
    struct ggml_tensor  * down_exps,
    struct ggml_tensor  * candidates);
```

7 个 src,`GGML_MAX_SRC = 10`,放得下。已有的 `GGML_OP_MOE_PARTITION_IDS/WGT`
可以作为**算子内部实现**,不再作为图节点暴露。

### 所有权契约

算子在 `ggml_cuda_compute_forward` 里被分发,拿到 `ctx.stream()` 和 backend 上下文。
**进入算子之后,以下全部由算子自己负责,不再与调度器协商:**

- 专家缓存 buffer 的分配与持有(生命周期不交给 `ggml_gallocr`)
- 侧流(预取/插入)的创建、事件、栅栏
- GPU/CPU 半边的划分与重叠
- device→host 的批量回传

---

## 4. 执行模型

### 原则

> **异步发出,只在真实依赖处立栅栏;栅栏必须是 event wait,不是 stream synchronize。**

已实测的栅栏类型差异(同一二进制、同一调用次数):

```
190 次 cudaStreamSynchronize  ->  59.10 ms
190 次 cudaStreamWaitEvent    ->   0.07 ms
```

### 一个 decode token 的真实依赖集

| 节点 | 依赖 | 需要 host 栅栏? |
|---|---|---|
| partition(L) | router(L) | ❌ device 内部 |
| GPU 半边(L) | slot 索引 + 缓存数据 | ❌ device 内部 |
| **CPU 半边(L)** | **`ids_cpu(L)` 到 host** | ✅ **唯一必须的** |
| merge(L) | GPU 半边 + CPU 半边 | ❌ device 内部 |
| SMoE 预测(L+1) | input + shared | ❌ **不是栅栏** |
| 预取完成 | —— | ❌ **不是栅栏** |

**一个 token 里真正必须的 host 侧栅栏,只有"每层的 `ids_cpu` 回到 host"。**
48 次,每次 80 字节。其余全是实现强加的。

SMoE 与预取**服务的是未来**,与当前 token 的正确性无关,不得进入关键路径。

### 三条硬性要求

1. **栅栏类型**:所有"只为拿数据"的 `ggml_backend_synchronize(split_backend)`
   改成 `cudaEventRecord` + `cudaEventSynchronize`。
2. **前置条件按 slot 粒度**:不得使用 `insert_flush` 式的"等 worker 队列全空"。
   只等**本层 top-k 里 pending 的那些 slot** 的完成事件。
3. **小传输合并**:
   - 48 层的 `ids_cpu/wgt_cpu`(合计 ~4KB)→ 图尾一次 D2H
   - 预取的 gate/up/down 三个 bundle(连续)→ 一次传输而非三次

### 提前量

SMoE 必须在**层内更早**开火,否则预取无提前量。共享专家 FFN 只依赖 `cur`,
与 routed MoE 无依赖边,可以**先算**:

```
attention → [共享专家 FFN 先算] → SMoE 开火(input + shared) → 预取 L+1 → routed MoE
                                                            ↑ 提前量 = 整个 MoE 时长
```

代价是近似里少了 `smoe_gpu_out`,准确率会降;但按实测
(准确率 73% × 交付率 34%),**用准确率换交付率是净收益**。

### 准入模型

现有实现是硬编码模型,必须重写:

```cpp
// 旧(已废)
eta_us      = (inflight_copies + n_copies) * 70us + bytes / 20GB/s
deadline_us = layers_until_visit * layer_us_ewma        // 按"层数"算 —— 错
```

新模型要求:
- `gate_copy_us` **标定**,不用猜(实测每次小传输 150-400µs,旧值 70µs 低估 2-5 倍)
- deadline 用**剩余时间**,不用层数
- `n_copies` 合并后从 3 降到 1

---

## 5. 必须保留的负面结果(不得重踩)

| 旧结论 | 实测 |
|---|---|
| ids 回读 46ms 是瓶颈 | decode 图里回读 **0 次**;旧数字是 warmup 图污染平均值 |
| 专家拷贝 3.5ms/token | decode 期间 **0 次**,该分支不进 |
| D2H 是瓶颈 | 每次 activation D2H ~10µs,48 次共 509µs |
| 保活 view 修复了乱码 | 含保活 view 的构建实测**仍错且不确定** |
| devpart OFF = 19-21 t/s | 实测 **10-11.6 t/s**,无任何历史输出支持 19-21 |
| 20.7 -> 8.4 是性能悬崖 | 20.7 那批是**乱码的旧构建**;同构建下不存在 |
| CPU 2.7ms 说明分区失效 | 高命中率下正常;但实际命中率只有 26.8% |

**方法论教训**:统计必须**按图类型分离**。`tm_graphs` 含 prefill/warmup 会污染 decode 均值
—— 上面两条错结论就是这么来的。

---

## 6. 踩坑清单

1. **共享专家在 GPU 上靠命名约定,不靠声明。**
   `--cpu-moe` 的正则是 `\.ffn_(up|down|gate|gate_up)_(ch|)exps`,
   `ffn_up_shexp` **不匹配** → 留在 GPU。但代码里没有任何断言。
   模型改名即静默落 CPU,且**不报错、不掉正确性、只变慢**,而 SMoE 预测器依赖它在 GPU。
   **必须加显式检查。**

2. **CUDA 后端有两套 D2H,行为不同。**
   ```cpp
   // buffer 级 —— 阻塞
   cudaMemcpyAsync(data, src, n, DeviceToHost, cudaStreamPerThread);
   cudaStreamSynchronize(cudaStreamPerThread);
   // backend 级 get_tensor_async —— 非阻塞,只 enqueue 到 compute stream
   cudaMemcpyAsync(data, src, n, DeviceToHost, cuda_ctx->stream());
   ```
   异步版本本身没问题,**问题在调用方紧接着 `synchronize`**。

3. **CUDA graph 捕获要静态图,策略是动态的。**
   这是 `moe_insert_flush` 每层排空的成因。新算子先走"排除捕获"
   (`ggml_cuda_graph_check_compability` 对 `MUL_MAT_ID` 已有先例),跑通再谈 capture-safe。

4. **`layer_us_ewma` 测的是 host 墙钟,含所有阻塞** —— 用它做 deadline 会把阻塞当成本成本身。

5. **驻留表初始值必须是 `-1`。** 0 是合法 slot,初始为 0 等于"全部命中 slot 0"。
   旧实现 `moe_cache_finalize` 在首图 compute 之后才跑,graph 1 的表是全 0。

6. **`smoe_predictions=1`**:旧实现付了 17ms 的逐层等待,却只产出 1 次预测。需确认是计数 bug 还是真 bug。

---

## 7. 可移植代码清单

### 原样移植(加性、隔离,与新架构零冲突)

```
TBQ      ggml/src/ggml-turboq.c / .h / ggml-turboq-tables.h
         ggml/src/ggml-cuda/turboq-device.cuh
         ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-tbq4_0.cu
         + 类型注册: ggml-common.h / ggml-quants.h / ggml-quants.h /
           dequantize.cuh / convert.cu / cpy-utils.cuh / cpy.cu / getrows.cu /
           set-rows.cu / fattn-common.cuh / fattn-vec.cuh / fattn.cu /
           common.cuh / arch-fallback.h / arch/x86/quants.c / quants.c /
           ggml-cpu/ggml-cpu.c(类型分发)/ repack.cpp
         + 入口: arg.cpp / llama.h(LLAMA_FTYPE_MOSTLY_TBQ*)

devpart  ggml/src/ggml-cuda/moe-partition.cu / .cuh
         → 从"图节点"降级为**新算子的内部内核**

模型     qwen4exp + MTP —— 基点里已有,零移植

harness  run-*.ps1 / dsh-probe-*.ps1 / 计时器 CSV 格式(-1/-2/-3/-4 行)

文档     docs/moe-cache-score-aware-prd.md / handoff.md / moe-decode-perf-plan.md
```

### 要重写

```
ggml-backend.cpp   MoE 缓存 + split + devpart + 调度钩子        3484 行
ggml-cpu.c         MoE CPU 路径(融合算子 GGML_OP_MOE_CPU 可保留) 330 行
llama-graph.cpp    devpart 分支 -> 换成构建 GGML_OP_MOE_QWEN4EXP  159 行
```

### 待定范围

**PLE**(`llama-context.cpp` +117、`llama-memory-hybrid*`)是**另一套缓存**,
与 MoE 缓存逻辑独立。需明确:PLE 也进算子(另一个量级),还是保持现状留在 llama 侧。

**已定:PLE 在范围内**,它是第二阶段(SSD tier)的机制本体,不是可选项。

---

## 7.1 移植机制与 S1 的精确范围

### 机制:基点相同 ⇒ 零合并

新树基点 `b76199698` 与改动集 `2f1a363c8` 的**父 commit 是同一个**,
所以 `git checkout 2f1a363c8 -- <path>` 整文件搬运**零冲突**,不需要打补丁或三方合并。

```powershell
# 在 F:/src/llama-qwen4exp-clean 里
git checkout 2f1a363c8 -- <可移植文件清单>
```

混合文件用 `git diff -U0` 看 hunk 归属,只取可移植的 hunk。

### 实测的三个可分离性结论

1. **`ggml-cpu.c` 的 +329(MOE_CPU 融合算子)完全自包含**
   —— 不引用 `moe_cache` / `sched` / `part_table` / `devpart`。
   主体是 `@@ -1453,0 +1466,285 @@`(285 行实现),其余为零散接入。
   **这是 M0 的 CPU 半边,原样搬运。**

2. **`qwen4exp.cpp` 的 +998 里,PLE 与 MoE 集成干净分离**
   ```
   @@ -7,0 +9,795 @@              PLE 实现主体(795 行)        <- 移植
   @@ -148,0 +945 @@              load_arch_tensors 一行       <- 移植
   @@ -1131..1175 build_layer_ffn MoE/SMoE 集成 ~100 行        <- 跳过(重写)
   @@ -1186..1267 llm_graph_input_ple  PLE ~50 行              <- 移植
   @@ -1324       build_inp_ple       PLE ~50 行               <- 移植
   ```

3. **`ggml-backend.cpp` 的 +3484 无一处可移植** —— 全是 split/devpart/缓存钩子。

### S1 拆成两段

| 段 | 内容 | 目的 |
|---|---|---|
| **S1a** | `GGML_OP_MOE_CPU`(ggml-cpu.c + ggml.h + ggml.c 的 builder)<br>`moe-partition.cu/.cuh`<br>prefetch 后端接口(`ggml-backend-impl.h` / `ggml-backend.h` + CUDA 侧实现) | **解锁 S3(M0 算子)** |
| **S1b** | TBQ(KV 量化类型,~20 文件但机械)<br>PLE(~900 行 + models.h + llama-context.cpp + llama-model.* + llama-memory-hybrid*)<br>harness 脚本 | 产品功能,不在 MoE 关键路径上 |

**S1a 先做** —— 它是 M0 的前置;S1b 可以并行或延后。

### 明确跳过

```
ggml/src/ggml-backend.cpp              +3484   全部重写
src/llama-graph.cpp / .h 的 MoE 部分   +159    全部重写
src/models/qwen4exp.cpp  build_layer_ffn 段    全部重写
```

这三处正是"两个主体争同一份状态"的地方,搬过来只会把老问题带过来。

---

## 8. 分阶段(每阶段必须可验收)

| 阶段 | 内容 | 验收 | 估时 |
|---|---|---|---|
| **S1** | 干净 worktree + 移植可移植清单 | 编译通过;非 qwen4exp 模型行为不变(`test-backend-ops` + 一个稠密模型冒烟) | 0.5 天 |
| **S2** | 差分 oracle:一条命令跑「旧路径 vs 新算子」并 diff token id | 命令可用,当前两条路径输出一致(此时算子还不存在,先跑旧 vs 旧自证) | 0.5 天 |
| **S3** | **M0:朴素正确的 MoE 算子**,无缓存/预取/devpart | `LLAMA_MOE_OP=1` 与 `=0` token id 逐位一致 | 0.5-1 天 |
| **S4** | 缓存 + 分区搬进算子 | 逐位一致;`ggml-backend.cpp` 的 MoE 钩子删除 | 1-2 天 |
| **S5** | 栅栏换型 + 自持 stream + 批量 D2H | **decode <= 40ms**;计时器证明 23ms/17ms 消失 | 1-2 天 |
| **S6** | 预取/预测搬进来,准入模型重写,提前量前移 | 命中率与 `prefetch_ready` 上升 | 1-2 天 |

**S1/S2 必须在 S3 之前。** 理由见下。

### 为什么先建 oracle

上一版的失败模式是:**每一版都写得比验得快**。`devpart` 是三个版本前的东西,
今天仍在产生乱码和崩溃,而没人能证明"哪一版是对的"。

`LLAMA_MOE_OP=0` 的旧路径**不是要删的包袱,是对照组**。任何 S4-S6 的改动,
只要 token id 不一致,一律回退。

---

## 9. 度量口径(沿用,勿改)

`LLAMA_MOE_CACHE_TIMING=1` + `LLAMA_MOE_CACHE_STATS=<csv>`,每图写四行:

```
-1: gid,-1,total_us,ids_wait,ids_parse,copy,0,cpu,gpu,pre
-2: gid,-2,in_wait,in_scan,d2h_enq,d2h_sync,d2h_n,d2h_bytes,in_parse,in_expert,drain,inputs
-3: gid,-3,flag_us,flag_n,gen_us,gen_n,loop_n,hostw_n
-4: gid,-4,cpuhalf_us,cpuhalf_evt_us,cpuhalf_n,splitpart_us,splitpart_n,sp_d2h_us,sp_d2h_n
```

**分析时必须剔除 `gid < 5`(warmup/prefill 图)**,否则 decode 均值会被污染 ——
这是两条错结论的来源。

新增(本重建):
```
-5: gid,-5,inflight_copies_peak,inflight_bytes_peak,prefetch_dropped,prefetch_ready
```

---

## 10. 明确不做的事

- 不写新的 GGUF 解析、分词器、attention kernel
- 不动 IQ3_XXS 的 CPU/CUDA kernel(PRD 第 122 行已划为独立分支)
- 不碰 `--cpu-moe` 对**其它模型**的行为(`LLM_FFN_EXPS_REGEX` 路径原样保留)
- 不改 TBQ 的语义(只搬运)
