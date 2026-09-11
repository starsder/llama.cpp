# DevPart 乱码定案 — 交接文档

日期: 2026-09-11 | 仓库: `F:/src/llama.cpp-unsloth-qwen4exp` | 分支: 当前工作区(未提交)

---

## 1. 一句话现状

**乱码根因已定案**：DevPart(设备侧 MoE 分区,kernel 版 `qwen4exp` 路由)之所以乱码,不是 kernel 逻辑错,而是
**`selected_experts`(topk)和 `weights` 这两个 router 输出的显存生命周期被 ggml 分配器提前结束** —— kernel 读到被回收复用的内存。

**修复已验证有效**:在 `src/llama-graph.cpp` 的 devpart 分支为这两个 tensor 追加"保活 view 节点"后,文本完全正确。

**唯一未解决的问题**:保活后吞吐从 **20.7 t/s 掉到 ~8.4 t/s**(2.5 倍悬崖),机制未查明。这是当前唯一 blocker。

---

## 2. 现象与边界(全部有日志支撑)

| 配置 | 文本 | 速度 |
|---|---|---|
| devpart OFF(host 路径,同 build 同缓存) | 完全正常 | ~19-21 t/s |
| devpart ON,2G 缓存 | `The///////////////////////////////` | 20.7 t/s |
| devpart ON,8G 缓存 | 逐字相同的乱码 | ~20.6 t/s |
| devpart ON + 保活修复 | 正常 | **8.4 t/s(问题)** |
| devpart ON + `set_output(topk,weights)` | 正常 | 8.2 t/s(同样慢) |
| devpart ON + `set_output`(全 4 个 pids/pwgt/topk/weights) | 正常 | 8.6 t/s |

- 乱码是**确定性的**(temp=0),与缓存大小无关(2G/8G 逐字相同)。
- 精确边界:token 1(缓存空,全 CPU 路径)永远正确;**token 2 起**一旦有专家命中 GPU 缓存就乱。
- 纯净重编译(去掉全部插桩)复现同样的乱码 → **插桩从未影响 bug 本身**。

---

## 3. 已排除的假设(不要再查)

| 假设 | 否定证据 |
|---|---|
| kernel 没执行 | 哨兵写入(`out[0] += 1000000`)在回读中可见 |
| kernel 逻辑/约定错 | 进程内 selftest 对拍:routing 0 错、weight 0 错;`-1`/dup-slot 约定与 host 路径逐条一致(`ggml-backend.cpp:4361-4387` host 同样写 `-1`) |
| A2 partition kernel 不在 DAG / 依赖边缺失 | probe 显示 `ffn_moe_part_ids-1` 是 split 6(CUDA0)i=218/221 的正式节点;dbg3 里每次 kernel launch 后硬 `cudaStreamSynchronize`(恰在 partition→消费之间)乱码依旧 |
| A1 tensor 元数据(view vs leaf 的 ne/nb)不同 | devpart 的 `reshape_2d(view_1d(pids,k,0),k,1)` 与 host leaf `new_tensor_2d(I32,k,1)` 形状/步长完全一致 |
| 驱逐覆写竞态(evicting insert 与 compute 流竞争) | 实现了"驱逐型 insert 延迟到图结束"(part_deferred),全速验证**乱码依旧**;该代码已删除 |
| 表镜像写不到 device | kernel printf 读到的表值与 flush 写入一致 |
| 插桩污染 | 纯净编译复现乱码,速度也回到 20.7 |

---

## 4. 根因证据链(lifetime)

1. `ggml_set_output(selected_experts); ggml_set_output(weights);`(**只钉输入**,输出不钉)→ 文本干净。
   即"这两个 tensor 必须活到图尾"。
2. 等价的零 flag 写法:在 `src/llama-graph.cpp` devpart 分支追加两个**保活 view 节点**
   (`ggml_view_1d(selected_experts, …)` / `ggml_view_1d(weights, 1, 0)` + `ggml_build_forward_expand`)→ 文本同样干净。
   view 的 `n_views` 计数把 src 的释放点推迟到该节点位置,达到同样效果、不触发 OUTPUT flag 的副作用。
3. 只钉 pids/pwgt 输出的那一轮(round 1')未做——目前证据指向**输入侧**。
4. 机制推断(未验证):partition kernel 在 split 6 读这两个 tensor,而它们的 liveness 在"最后一个声明的消费者"处结束;
   被回收后被后续节点复用,复用发生在并发/异步上下文中(或与 CUDA graph 复放交互),于是 kernel 读到脏数据。

---

## 5. 待解决的性能悬崖(唯一 blocker)

已知:
- 20.7 t/s(乱码/纯净)→ 8.2~8.6 t/s(任何"延长生命周期"的写法都慢,不论 flag 还是 view)。
- 钉住版 + `GGML_CUDA_DISABLE_GRAPHS=1` → 6.9 t/s(比 8.2 更慢)→ **说明钉住后 CUDA graph 捕获仍在工作,悬崖不是"丢失捕获"**。
- 统计 CSV(见下)显示 per-graph `total_us` 119ms(pinned) vs 46ms(pristine),但 gpu/pre 分列只占几 ms → 大部分时间落在未被分列的等待里。
- 权重 pin 行两者一致(`pinned 1 weight buffers (72.6 GiB)`),不是它。
- 待验证的头号假设:**每 token 重新实例化 CUDA graph**。
  `ggml_cuda_graph_update_required()`(`ggml/src/ggml-cuda/ggml-cuda.cu:2790`)逐节点 memcmp 整个 `ggml_tensor` 结构(含 `data` 指针);
  一次 `cudaGraphInstantiate` 约几十 ms,与 70ms/token 的时间差高度吻合。
  验证办法:debug 日志里数 `CUDA Graph id ... reused` 的次数,或对比每次 token 的时间分布(首 token vs 后续)。
- 备选排查:用 `LLAMA_TRACE_EVAL=1` 打出 per-split 时间,对比 pristine 与修复版的 split 分布找差异点。
- 另需澄清:把"20.7 是假速度(乱码导致 GPU 全包/CPU 空转)"排除。反证:host 路径(正确、同样真算 CPU 半边)是 ~19-21 t/s,
  所以修复后掉到 8.4 不能用"现在真的干活了"解释。

---

## 6. 当前代码状态

已落地(工作区已改,未提交):
- `src/llama-graph.cpp` devpart 分支:保活 view 节点(`ffn_moe_part_keep_ids/wgt-%d`)——**修复候选,但带性能悬崖**。
- `ggml/src/ggml-cuda/moe-partition.cu` / `.cuh`(未跟踪新文件):两个单线程 kernel `moe_partition_ids_kernel` /
  `moe_partition_wgt_kernel` + 两个 launcher;**插桩已全部移除**,当前是干净版。
- `ggml/src/ggml-backend.cpp`:本会话加的 probe / selftest / snapshot / 驱逐延迟(part_deferred)**已全部删除**,回到干净状态。

已删除的临时件(需要时可重建,见第 8 节):
- 进程内 selftest(`LLAMA_MOE_DEVPART_SELFTEST`,含 `part_table_snap` + 图尾对拍块)
- probe 块(`[DEVPART-PROBE]`)
- flush 调试打印(`[PARTITION-FLUSH]`)
- kernel 侧 `[KIDS]` printf / `LLAMA_MOE_DEVPART_DEBUG` 开关
- `part_deferred` 驱逐延迟

**注意**:最后一次 `build-cli.bat` 被取消,二进制状态不确定;继续工作前先重新构建。

---

## 7. 复现/验证命令

```powershell
# 构建(sccache 已配置;-j 16,32 会死机)
cmd /c build-cli.bat

# 主要运行脚本(都在仓库根,输出为 UTF-16,读的时候要 iconv)
# run-2g-devpart.ps1  : 2G 缓存 + devpart  (乱码基线)
# run-2g-ref.ps1      : 2G 缓存,devpart OFF (正常对照组)
# run-8g-devpart.ps1  : 8G 缓存 + devpart
# run-2g-nographs.ps1 : devpart + GGML_CUDA_DISABLE_GRAPHS=1
powershell -NoProfile -ExecutionPolicy Bypass -File run-2g-devpart.ps1

# 读输出
iconv -f UTF-16LE -t UTF-8 devpart-out.txt | tr -d '\r' | sed -n '/capital of France/,/Generation/p'
```

模型:`F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf`,
`-ngl 49 --cpu-moe -c 8192 --cache-type-k q8_0 --cache-type-v q8_0 -n 32 --temp 0 --no-mmap`。

关键环境变量:`LLAMA_MOE_DEVPART=1`、`LLAMA_MOE_DIRECT_READ=1`、`LLAMA_MOE_SPLIT=1`、`LLAMA_MOE_CACHE_MIB`、
`LLAMA_MOE_PREDICT_SMOE=1`、`LLAMA_MOE_MRS=1`、`LLAMA_MOE_PREFETCH=1`、`LLAMA_MOE_VRAM_LIMIT_MIB=15360`、
`LLAMA_MOE_CACHE_TIMING=1` + `LLAMA_MOE_CACHE_STATS=<csv>`(时序分列)、`LLAMA_TOKEN_PROF=1`、`LLAMA_TRACE_EVAL=1`。

---

## 8. 下一步建议(按性价比排序)

1. **查明性能悬崖**:先在修复版上跑 `LLAMA_TRACE_EVAL=1`,对比 pristine 的 split 分布;同时确认是否每 token 重新
   `cudaGraphInstantiate`(假设见第 5 节)。若确认,优先找"保活但不改内存布局"的写法。
2. **试更省的保活手段**(逐个实验,每个 ~80s 编译 + 60s 跑):
   - 只保活 `weights` 或只保活 `selected_experts`,看是否单个就够、是否单个就不掉速;
   - 把保活 view 挪到**整个图的最末尾**(而不是每层之后),看是否影响速度;
   - 用持久 scratch buffer 承载(用户提的"方案 1"):在 router 之后用 in-graph `ggml_cpy` 把 topk/weights 拷进
     固定的持久 device buffer,partition kernel 改读 scratch —— 从根上绕开分配器生命周期。
3. 性能问题解决后做验收三件套:无插桩 build、同配置 temp=0 逐位一致(devpart ON vs OFF)、速度回到 ~20 t/s。
4. 顺手修 C(与主 bug 无关但确实是 bug):`moe_cache_finalize` 在首图 compute 之后才跑 → graph 1 的驻留表是全 0
   (0 是合法 slot,会静默按 slot 0 命中);建议表初始化值改 `-1` + finalize 提前到首次 compute 之前。

---

## 9. 工作方式备注

- 用户要求:**每次跑出数据就立刻汇报**(文本 + 速度 + 结论),一起看。
- 用户要求:**不要对上下文做压缩摘要**(压缩会丢关键状态)。
- 机器:5950X / 128GB / RTX A5000 Laptop 16GB(sm_86);`-j 32` 会死机,构建固定 `-j 16`。
- 日志是 UTF-16 + BOM,`grep` 前先 `iconv`。
- `tests/` 未经维护者许可不要加文件(仓库 AGENTS.md 约定)。
