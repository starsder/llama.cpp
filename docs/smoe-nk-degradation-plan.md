# SMoE N+k 层预测退化:数据、缺口与做法

日期: 2026-09-11 | 目标:离线模拟 SMoE 预测 N+1 / N+2 / ... / N+K 层专家的准确率退化

---

## 1. 数据在哪

```
F:\models\qwen38\traces\
├── standard-smoe-20-20260910b\r01..r20\      ← 20 个 prompt 的原始轨迹(每请求一目录)
│     tensors.csv      张量元数据(graph_id,sched_id,kind,layer,occurrence,ne*,nb*,type,bytes,file)
│     tokens.csv       每 token 一行
│     *.bin            每个张量的原始数据
│     run-manifest.csv
├── standard-smoe-sim-20-20260910b\r01..r20\  ← 现有模拟结果(overall.csv / per_layer.csv)
├── standard-smoe-20-20260910b-summary\       ← 聚合(report.md / overall.csv / per_layer.csv)
├── simulate_smoe_from_standard_trace.py      ← 模拟器
├── aggregate_standard_smoe_20.py             ← 聚合器
├── offline_smoe_eval.py / direct_smoe_recall.py
└── run_standard_smoe_20.ps1                  ← 采集驱动
```

**采集方式**:专属插桩检出 `F:\src\qwen4exp-smoe-trace`(`LLAMA_TRACE_SMOE=<目录>`),
每请求跑 `-c 4096 -n 16 --temp 0 -st --no-warmup`,20 个 prompt(中英对话/代码/学科)。

**轨迹粒度**:**逐 graph、逐 layer、逐 token**。decode 图每个只含 1 个 token(`ne[2]=1`)。
张量必须按 `(graph_id, layer, kind)` 取 —— 只按 kind+shape 取会跨 graph 张冠李戴。

### 已 dump 的张量白名单

`F:\src\qwen4exp-smoe-trace\ggml\src\ggml-backend.cpp:1607-1610`

```cpp
"ffn_moe_logits-", "ffn_moe_topk-", "ffn_moe_weighted-",
"ffn_shexp_gated-",
"attn_input-", "attn_output-", "hc_after_attn-",
"hc_ffn_inject-", "l_last-",
```

| kind | 含义 | 形状 |
|---|---|---|
| `ffn_moe_logits` | router 原始 logits | [512, nt] |
| `ffn_moe_topk` | 路由出的专家 id | [10, nt] |
| `ffn_moe_weighted` | 每个选中专家的加权贡献 | [2560, 10, nt] |
| `ffn_shexp_gated` | 共享专家输出 | [2560, nt] |
| `hc_after_attn` | attention 后的超残差 | [2560, 4, nt] |
| `hc_ffn_inject` | FFN 超连接的 gate 注入 | [4, nt] |
| `l_last` | 该层输出(FFN combine 后) | [2560, 4, nt] |
| `attn_input` | attention 的输入 | [2560, nt] |

---

## 2. 现有模拟器做了什么、没做什么

`simulate_smoe_from_standard_trace.py`:

- 复现**因果 FIFO 池**(每层独立,slots ∈ {16,22,32,44,64})
- 把未驻留专家的贡献置零,保留共享专家
- 重建该层的 `l_last`,与真实 `l_last` 比 cosine / relative L2
- 统计 `fifo_pool_hit8` / `fifo_ranked_recall8/16`

**它自己在 docstring 和 report.md 里写明了局限:**

> Mode: `teacher_forced_local_smoe`. Each layer uses the standard run's **true pre-FFN residual**;
> **this is not a cumulative counterfactual rollout.**
>
> These numbers do not establish that a cumulative SMoE rollout preserves later-layer routing;
> **that requires rerunning the graph with the approximated residual fed into the next layer.**

---

## 3. 缺口:缺 `ffn_input`

SMoE 在运行时的公式(`src/models/qwen4exp.cpp:1989-1993`):

```cpp
smoe_hidden = ffn_input(L) + smoe_gpu_out(L) + ffn_shexp_gated(L);
smoe_logits = layers[L+1].ffn_gate_inp @ smoe_hidden;      // 用 L+1 的真实 gate
```

- `smoe_gpu_out` → 由 `ffn_moe_weighted` + 驻留掩码得到 ✅ 在白名单里
- `ffn_shexp_gated` → ✅ 在白名单里
- **`ffn_input`(router 的输入,即 `build_hc_mix(hc_after_attn, hc_ffn_*)` 的输出)→ ❌ 不在白名单里**
- `ffn_gate_inp`(gate 矩阵)→ 不在 trace 里,但**可以从 GGUF 抽出** ✅

所以离线做 N+k 模拟,只差 `ffn_input` 这一项。

### 两条补法

**(a) 补 dump(推荐,最省事)**
在 `F:\src\qwen4exp-smoe-trace` 里两处改动:
```cpp
// src/models/qwen4exp.cpp,紧跟 build_hc_mix(..., hc_ffn_*, &inject, il) 之后
cb(cur, "ffn_input", il);
```
```cpp
// ggml/src/ggml-backend.cpp 白名单
"ffn_input-",
```
然后重跑 `run_standard_smoe_20.ps1`(20 请求 × ~1 分钟)。

**(b) 从 `hc_after_attn` 离线重建**
按 `build_hc_mix` 精确公式算。**试过,未验证通过**(量级差 ~20 倍),
原因可能是该检出的 hc 实现与当前工作区有差异。**有了 (a) 就不必走这条路。**

---

## 4. 已准备好的资产

- **48 个 gate 矩阵**:`F:\models\qwen38\traces\gate\blk.N.ffn_gate_inp.weight.npy`
  (从 GGUF 抽出,已归一为 `[n_expert=512, n_embd=2560]`,F32)
  抽取脚本:`F:\models\qwen38\traces\extract_gate.py`
- 模型常量:`n_embd=2560, n_expert=512, hc=4, hc_dim=10240, hc_lr=320, top_k=10`
- 路由概率张量 `ffn_moe_logits` —— 已确认是**原始 logits**(softmax 之前),见 `llama-graph.cpp` 的 `cb(logits,"ffn_moe_logits",il)`

---

## 5. N+k 退化模拟怎么做

**已实测完成。结果见第 7 节。**

```python
for layer L in 0..46:
    h = ffn_input(L) 与 L+1 同图同 token
    block = (ffn_moe_weighted(L) * resident_mask).sum(axis=expert) + ffn_shexp_gated(L)
    smoe_hidden = h + block                              # SMoE 的近似 gate 输入

    for k in 1..K:                                       # N+1 .. N+K
        logits = W_gate[L+k] @ smoe_hidden               # 用 L+k 的真实 gate
        pred   = topk(logits, 10)
        true   = ffn_moe_topk(L+k)
        record recall(pred, true)                        # ← 退化曲线
```

输出:**准确率 vs k 曲线**(以及分层、分 prompt)。这同时给出:

- **SMoE 单独的 N+1 准确率**(不含 32 个静态热专家)← 你说"没测过"的那个数
- **退化速度**:K=2/3 还值不值得预取 ← 决定预取窗口该开几层
- 与 HybriMoE 的"提前 3 层"对比依据

### 两个对照必须一起跑

```
A: smoe_hidden 用 ffn_input + block        ← SMoE 本体
B: smoe_hidden 用 l_last(真实)             ← 上限参考(teacher)
```
A vs B 的差 = SMoE 近似本身的损失;B 的绝对值 = 方法上限。

---

## 6. 一个重要的命名澄清

之前把 `prefetch_predicted / prefetch_required`(= 73%,早期跑到 98-99%)当成了 SMoE 准确率。
**它不是。** 喂给预取的是 `SMoE topk ∪ 静态热集(predict_static=32)` 的并集
(`moe_cache_prefetch_layer` 里合并),所以那个 98% 是**两者之和**。

**要隔离 SMoE,只能靠上面的离线模拟,或者在运行态跑 `PREDICT_STATIC=0`。**

---

## 7. 实测结果(2026-09-11)

数据:`F:\models\qwen38\traces\standard-smoe-8-20260911`(8 个 prompt,新增 `ffn_moe_input`)
采集方式:白名单加一行 `"ffn_moe_input-"`(`ggml-backend.cpp:1607`),
**`qwen4exp.cpp` 完全不用改** —— `build_moe_ffn` 早就把 router 的输入命名为 `ffn_moe_input`
(`llama-graph.cpp:1973`,紧跟 `logits = gate_inp @ cur`)。
模拟器:`F:\models\qwen38\traces\simulate_smoe_nk.py`

### 退化曲线(SMoE 预测 L+k 层,recall@10)

| 变体 | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| **oracle**(用 L+k 的真实输入) | 100.00% | 100.00% | 100.00% | 100.00% |
| **full**(SMoE 现状:h + 全部路由 + 共享) | **68.53%** | 59.94% | 55.70% | 53.77% |
| **fifo**(带 22 slots 缓存掩码) | 68.29% | 59.72% | 55.47% | 53.65% |
| **shared_only**(h + 共享专家) | 68.07% | 59.52% | 55.29% | 53.47% |
| **input_only**(只有 h) | **67.90%** | 59.34% | 55.22% | 53.44% |

样本量:56400 / 55200 / 54000 / 52800(每次 10 个专家的命中数)

### 结论

**1. 管线验证通过**:oracle 在所有 k 上都是 100% —— gate 矩阵、topk、取数链路全对。

**2. SMoE 的 N+1 = 68.5%**,与记忆中的"60% 多"吻合;
   也和运行态实测的 `prefetch_predicted/prefetch_required = 73.3%` 自洽
   (运行态那个 73% 含 32 个静态热专家,离线 68.5% 是 SMoE 单独)。

**3. 退化很平缓**:k=2 仍 60%,k=3 55%,k=4 54%。**2 层窗口完全可用**
   (按 `token = 22ms + 38ms x (1-h)` 的模型,25 t/s 只需要 h ≈ 53%)。

**4. ★ 路由专家和共享专家的贡献几乎为零。**

```
full         68.53%     h + Σ(w_i * e_i) + shared
shared_only  68.07%     h + shared            (-0.46pt)
input_only   67.90%     h                     (-0.63pt)
```

**只用 `ffn_input` 就能拿到 67.90%,和完整 SMoE 的 68.53% 只差 0.63 个百分点。**

### 这意味着什么

预测器**不需要**等 routed MoE 的输出,也**不需要**共享专家的输出:

- `smoe_gpu_out`(`ffn_moe_weighted` 的 GPU 半边之和)→ **可以删**
- `ffn_shexp_gated`(共享专家)→ **可以删**
- 于是**共享专家不必先算**,SMoE 也不再依赖 MoE 的任何输出

`ffn_moe_input` 在 attention 一结束就可用 ⇒ **预测可以在 MoE 开始之前就发出**,
提前量从"约等于 0"变成"整个 routed MoE 的时长 + 下一层的 attention"。

**连带可以删掉的运行态开销**:
- SMoE side graph(每层一次 device 侧的 gate matmul + topk)
- `moe_cache_smoe_enqueue` / `moe_cache_smoe_drain` 的逐层 `event_synchronize`
  —— 实测 **17 ms/token**(而 CPU 处理部分只有 2 ms)

**新形态**:每层只做一次 `W_gate(L+1) @ ffn_moe_input(L)` 的极小 matmul(512x2560),
连回读都不必 —— topk 留 device,结果直接喂预取队列。

### 保留意见

本模拟是**单步 teacher-forced**:每层的 `ffn_input` 取自真实轨迹。
但由于预测**只用于预取提示、不参与计算**,误差不会跨层累积 ——
退化曲线上的 k 就是真实可用的提前量,不需要累积 rollout。
(这一点与 `simulate_smoe_from_standard_trace.py` 里那种"重建 l_last"的情形不同。)
