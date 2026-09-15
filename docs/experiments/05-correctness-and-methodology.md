[中文](05-correctness-and-methodology.md) · [English](05-correctness-and-methodology.en.md)

# 05 正确性、测量失效与工程事故

本项目最需要保留的不只是“哪个开关快”，还有**为什么一度相信某个结果，以及后来用什么证据撤回它**。本章与[研发主线](00-research-chronology.md)、[host/devpart](01-host-and-devpart.md)、[预测与缓存](02-prediction-and-cache.md)、[权重内核](03-weight-quantization-and-kernels.md)、[KV 量化](04-kv-tbq-and-nxq.md)配套。

本次工作只归档文档和既有证据，没有重新运行模型、修复以下开放缺陷，或把未验收的实验代码合入 master。

## 1. 证据怎样分级

| 证据 | 能支持什么 | 不能自动支持什么 |
|---|---|---|
| 已保存的原始日志、CSV、JSON，带来源哈希 | 当次配置、结果、退出状态；可复算的计数与差异 | 换模型、换输入、换二进制后仍成立 |
| 历史 handoff、提交说明 | 当时做过什么、当时如何解释；只存报告的实验结果 | 假装仍有完整原始日志；把旧解释当最终结论 |
| 维护者直接补充 | 缺失的研发顺序、动机、报告过的实测结果 | 替缺失协议补造样本数、计数分母、精确容量单位 |
| 源码检查、算术重算 | 生命周期缺陷、类型几何、计数口径、实现覆盖范围 | 没有堆栈就认定一次崩溃根因；没有运行就宣称性能收益 |
| 假设或外推 | 决定下一步值得测什么 | 验收、生产建议、论文方法被证明无效 |

[历史来源](sources/)原样保留后来推翻的说法，并明确标成历史。新复盘发现矛盾时，不静默修改旧笔记；在本文和各专题给出更正。

## 2. C-01：流畅短文本掩盖了少算与错算

**为什么会踩坑。** 很多早期 smoke prompt 只需生成“Paris”一类短答案；一个错误路由、提前 EOS、缺失 CPU 半边的路径，也可能产出看似通顺的短文本。高 token/s 因此不能反证计算正确。

**出现过的反例。** devpart 的若干 19–32 t/s 高值后来发现 CPU 半边无效、权重 staging 未提交或消费了错误数据；host 的 SPLIT=1 也出现独立的旧路由拷贝错误。具体发生顺序与每次修复见[01 章](01-host-and-devpart.md)。这些成绩不能作为优化收益留下。

**另一个掩盖因素。** `--ignore-eos` 适合固定请求长度的吞吐协议，但可能让本应早停的退化继续吐 token。历史 Eiffel Tower 自由生成回归要求不加该选项；固定历史重放又是另一种试验。

**决定。** 分开保留三类问题：自由生成是否退化、同一真实 token 历史下 logits 是否改变、完整 CLI 是否正常退出。不能用其中一个替代另外两个。

**来源。** [handoff](sources/handoff.md) §6.5、§6.17–6.25、§6.30–6.32；[host/devpart 路线表](01-host-and-devpart.md)。

## 3. C-02：host split 的旧路由拷贝——地址合法也能静默算错

**动机。** 把命中专家放到 GPU、漏失专家交给 CPU，保持同一个 MoE 数值计算。

**根因。** `ids_gpu` / `wgt_gpu` 是 GPU 半边输入的 host leaf。调度器可能先排了输入拷贝，分区钩子随后才改写 leaf；设备端于是读到上一轮路由。数据类型、指针和索引都可能合法，结果仍错误。

**修复。** 钩子写入之后，对对应设备输入重新发出 `ggml_backend_tensor_copy`。这是发布代码基线已包含的修复（`0f853bb6b`，位于 `7e01451b2` 的历史内），不是本次文档发布新做的代码改动。

**验证边界。** 历史报告：修前 Eiffel 用例 3/3 退化；修后 4/4 给出正确答案段。但其记录也明确写了答案段长度 443–455，而前一次参考记录是 751 字符。因此只可称当时退化回归通过，不能改写成逐字、逐 token、逐 logits 完全相同。

**更正链。**

1. SPLIT=1 曾被当作 20.3 t/s 的封板路径；发现静默错误后撤回。
2. SPLIT=0 的一次 15.9 t/s 被当作安全退路；后来同配置出现 3.4–6.5，先前的可重复性能判断也撤回。
3. 临时禁止 SPLIT=1；修复输入发布顺序后解除该保护。
4. 修复后的历史 20.8 t/s 可以作为当时记录保留，但不是当前 400 请求 token／6 GiB 档的新验收，也缺少与 9.2 t/s 完全一致的受控 A/B 证明。
5. 错误路径上扫出来的 ahead 曲线，**趋势也可能失真**；不能只撤回绝对速度而保留“趋势可靠”的断言。

**重新开启条件。** 改图切分、leaf 生存期、copy 顺序、graph capture 或 slot binding 时，应再次覆盖此类输入发布时序。不要只检查专家 ID 是否落在合法范围。

**来源。** [handoff](sources/handoff.md) §6.30–6.33；历史原文中“全部已验证”“逐字一致”等较强表述须按上述实际验证范围阅读。

## 4. C-03：UD 与共享池 stride——同一个地址公式有两层前提

**第一次风险：混合类型。** 文件名 `UD-IQ3_XXS` 不是所有张量的真实类型；gate/up/down 各有自己的块大小、行字节和专家字节。发布基线加入了逐张量几何检查和对齐防护，见[03 章](03-weight-quantization-and-kernels.md)。

**第二次风险：跨层共享池。** 后来的本地共享池把多种层布局放进一个物理 pitch。`2,534,400` 字节不被某些 IQ 格式的 `82` 字节块大小整除；先把字节 stride 除成块 stride，会提前截断。旧消费路径还发生过把原专家编号用于紧凑 slot view 的错误，日志中在约第 29 步暴露。

**修复原则。** 通道、样本、专家偏移先按 64 位字节地址计算，再转为对应块指针；不能假设一层成立的块整除条件在共享池里仍成立。direct/gathered 的 `src[0]` 绑定改变后，应让 CUDA graph 缓存看到新的图身份，而不是一概关闭 CUDA graphs。

**实际证明。** 本地 CUDA 探针覆盖 24 个 IQ2_S / IQ3_S / IQ4_NL 的紧凑与 strided 4D、1/4/17 token 及 fused SwiGLU 场景，GPU 对 GPU 的最大差为 0；对应 memcheck 记录为 0 errors。

**不能扩张的结论。** 这不是 CPU 对 GPU 的全模型等价证明，不是完整模型的 memcheck，也不说明最终 CLI 退出路径已修复。这批共享池／stride 后续工作没有随本次文档进入 master。

**来源。** [共享池 PRD 快照](sources/moe-cache-score-aware-prd.md)；[02 章](02-prediction-and-cache.md)；[测量证据](evidence/measurements.json)。

## 5. C-04：预测准确率、准入、驻留、真实使用必须分账

维护者报告的 **SMoE teacher 99%**、某组离线 `recall@k`、预测加静态表的候选覆盖率、在线 cache hit、prefetch ready、实际专家使用，是不同量。

```text
预测中有它
  → 允许准入
  → 传输提交
  → 在消费前完成
  → 未被淘汰且绑定正确
  → 本次计算确实使用
```

任何一箭头都可能失败，也都可能产生额外成本。高 teacher 命中仍然值得实现；它不承诺在线一定提速。早期 PLE 的 SSD→主存命中与后来 GPU PLE 命中也必须按层级分开，见[维护者补充](sources/author-recollection.md)。

生命周期计数进一步揭示：`readmit` 不能与单次 `admit`、驻留去重、真实 GPU 使用混淆；`unused_evict` 的逻辑字节并不等于 PCIe 总线上实测的字节。统计钩子应记录真实消费，而不是 padding、CPU 路由或同图重复调用。

**已保存的内部一致性检查。** 原基线 `10,216 admissions − 7,272 evictions = 2,944 live`；`133,484 GPU uses × 3 = 400,452 component hits`。它们证明这份计数的内部关系，不证明命中本身有净性能收益。

**来源。** [02 章](02-prediction-and-cache.md)；`evidence/measurements.json` 的 `instrumentation-only-baseline` 记录。

## 6. C-05：固定历史对照防住输入漂移，但不等于完整 CLI 验收

### 6.1 两种计时各回答什么

| 协议 | 回答的问题 | 不应声称 |
|---|---|---|
| `llama-cli` 自由生成、完整进程、请求 400 token | 用户实际看到的输出、CLI 打印的生成速度、退出状态 | 输入历史与另一变体完全一致 |
| 同一 prompt IDs＋强制同一后续 IDs，400 decode steps | 同一真实历史下的 latency 与 logits 差异 | 自由生成相同、CLI 销毁路径相同 |
| 128 步、512 MiB 共享池压力 | 淘汰策略确实被触发时的诊断 | 400 步／6 GiB full-RAM 基线验收 |
| 单核算子 microbench | 某 dtype／内核／尺寸下的成本 | 端到端 token/s 提升 |

固定历史的计时器覆盖 `llama_decode + llama_synchronize`；logits 拷贝与落盘在计时外。不能把其 `1000 / mean_ms` 当作完整 CLI 速度。计时可另外报告去掉前 16 步的 steady 值，**正确性始终覆盖所有 decode 步**。

### 6.2 最后几次完整 CLI 记录

共同请求档：400 token、6 GiB 专家缓存、full-RAM、8k、q8_0 KV、AHEAD=2、非阻塞、per-layer MRS；完整环境保存在 JSON 中。**这是本地开发二进制的历史结果，不是重新构建发布 master 的验收。**

| 记录 | CLI Generation | 退出码 | 结论 |
|---|---:|---:|---|
| 原路径＋生命周期计数 | 19.3 t/s | 0 | 有效的当次完整运行 |
| 全来源 frequency gate 候选 | 19.9 t/s | 0 | 自由输出已改变；不能作为孤立提速证据，候选后来撤回 |
| 仅 hot-backfill 候选 r1 | 18.6 t/s | 0 | 未达到当时 19 t/s 门槛 |
| 后置旧基线复跑 | 18.4 t/s | 0 | 基线也有波动，不能选择性只与早前高值比较 |
| 仅 hot-backfill 候选 r2 | 不可用 | `0xC0000005` | 原生崩溃；不得计入成功吞吐 |
| 第一次旧基线配对启动 | 不可用 | `0xC0000135` | 缺 DLL 的启动装置错误，不是模型性能结果 |

JSON 的 `tokens_generated` 未被装置独立解析；`graphs_observed=403` 不应改写成“确认生成 403 个 token”。请求长度、实际 decoder steps、可见字符数、图计数必须分别记录。

### 6.3 两个候选为什么没有通过

- **全来源频率门控**：预取字节大幅下降，但固定历史 3 次对照没有稳定速度收益，在线 hit 与 ready 均下降；相对参考 top-1 只有 398/400 相同。候选自己重复稳定，不等于对原路径正确。
- **仅 hot-backfill 修复**：实际修复了合成 fixture 中的轮转跳层、空槽处理和真实 eligible victim 比较；固定历史命中约 +2.67%，字节约 +0.62%。三次运行均值的中位数约改善 1.05%，落在运行变化范围内；top-1 仍是 398/400，完整 CLI 又有低于门槛与原生崩溃。
- **共享池 LFU_POS 压力测试**：总 hit 约 +20.77%，传输计数仅约 −0.60%，run-median 的中位数改善约 1.92%；128 步存在一次 top-1 翻转。不能把高命中直接包装成稳定吞吐增益。

这些结果保留在[测量 JSON](evidence/measurements.json)，包括基线重复、候选重复、失败退出，不只挑最好的一行。**已撤回以基线误差乘任意倍数（例如 3×）充当正确性阈值的做法。**

## 7. C-06：同步耗时变少，可能只是等待换了位置

pinned 路由读回曾把 enqueue 从约 3785 µs 降到 4 µs，但 synchronize 从约 1 µs 升到 3787 µs；总时间仅约 48.7→48.3 ms。结论是等待位置转移，不是回读凭空快了三个数量级。

同样，异步 CPU、GPU queue、prefetch、ids_wait 是可能重叠的计时区间。相加超过总时间不一定是错误；但某个计数器占比很小，也不能不检查定义就把它当作严格 Amdahl 上限。

**决定。** 记录区间包含关系、线程与 stream、同步点、计时器起止。预测器和自适应控制器的目标应面对端到端代价，而非单个漂亮计数。

**来源。** [handoff](sources/handoff.md) §6.4、§6.11–6.12、§6.34、§6.39；[01](01-host-and-devpart.md)与[03](03-weight-quantization-and-kernels.md)。

## 8. C-07：页锁定、整机内存压力与一次已经修过的退出 UAF

**动机。** 大专家权重留在主存，GPU 预取希望使用页锁定内存；让异步传输真正可用，而不是把同步成本藏在 pageable staging 中。

**事故。** 历史中把 mmap/lazy 与大范围 pin 组合使用，触发了整机内存压力／卡死风险；并发模型实例会进一步放大风险。另有旧退出钩子在 backend 已释放后调用 prefetch wait，形成 UAF；该旧 wait 被移除。

**取舍。** 后来的 full-RAM 受控协议要求：串行持有 `.tools-run.lock`、至少 90,000 MiB 可用内存、`--no-mmap --lazy-mode off`，并记录约 72.6 GiB 的 pinned 权重与 pin failures。此纪律适用于这一大模型运行协议，**不否定早期 PLE 的 SSD/mmap 节省主存方案**。

**边界。** 这次旧 prefetch-wait UAF 与下一节的 profiler 生命周期风险是不同缺陷，不应合并成一个已经全部修好的事故。

**来源。** [handoff](sources/handoff.md) §6.10；[01 章](01-host-and-devpart.md)；历史基线 JSON 的 load flags 与内存记录。

## 9. C-08：最后的原生退出崩溃——已定位风险，尚无崩溃堆栈

**观察。** 候选 r2 返回 `0xC0000005`，CSV 已到 graph 403，没有最终 Generation 行。失败 stderr 停在 DIRECT-VIEW 汇总之后；成功运行接着打印 SERVER-PROF / TOKEN-PROF / MOE-CACHE。这些仅能缩小调查方向。

**源码确认的生命周期风险。** `tools/server/server-context.cpp` 中：

```cpp
static server_context_impl * prof_self;
// constructor, when LLAMA_TOKEN_PROF is enabled:
prof_self = this;
atexit(&server_context_impl::prof_print);
```

`prof_print()` 随后通过这个裸指针读取实例计数。`server_context` 拥有该实现对象，实例销毁后静态指针没有被清空；稍后的 atexit 回调可能访问已释放对象。相关位置：源工作区 `server-context.cpp:868–900,4197–4200`。

**为什么还不能宣布根因已修复。**

- 未捕获这一次崩溃的原生堆栈，也没有运行到 fault handler；源码风险与实际崩溃的因果尚未闭合。
- stdout 有缓冲，退出期失败可能发生在最终计时已经格式化、但尚未 flush 之后。缺少 Generation 行不能反推出模型尚未完成最后 decode。
- 固定历史 helper 每次显式 synchronize，且不走相同 server/profiler 生命周期；helper 退出 0 不能为 CLI 销毁阶段作担保。
- 线程、event 与 backend 所有权还有其他待审计点；不能没有证据就把它们列成已确认根因。

**状态。** 暂停前只准备了 native fault capture/probe 源码，**没有编译、没有运行、没有堆栈、没有提交修复**。重新开启时应先捕获真实 fault，再修对象所有权与回调生存期；关闭统计仅是隔离变量，不是修复证明。

**来源。** `final-r2.json` 及[公开摘要](evidence/measurements.json)；上述源码；[handoff](sources/handoff.md) §6.32 的退出风险记录。原笔记“只影响统计输出”不够严谨：非零退出必须被当作失败。

## 10. C-09：装置错误也会污染结论

| 问题 | 实际处理或现状 | 保留下来的教训 |
|---|---|---|
| 编译目标更新，但 CLI 使用了另一份旧二进制 | 历史发现并区分；日志见 §6.12 | 构建成功不等于运行了新产物 |
| 复制旧 CLI 却缺 DLL 搜索路径，退出 `0xC0000135` | 修正装置 PATH 后才得到旧基线 18.4 t/s | 启动失败不能填成模型 0 t/s，也不能悄悄剔除 |
| wrapper 不传播原生退出码 | 发布面旧 runner 有此风险；后续本地 runner 才记录真实退出状态 | 必须保存原始 child exit code；看到速度行仍不够 |
| 真超时 | 本地 1 秒超时 smoke：124、结果不可用、模型锁释放 | 这是已运行的装置 smoke，不是本次文档新测试 |
| 锁已占用 | 本地 smoke：退出 2，未启动模型 | 防重入应在大模型加载之前生效 |
| raw graph count 与 token count 混用 | 本文不从 403 图补造生成数 | 不同计数器保留原名、来源与可用性 |

**来源。** [handoff](sources/handoff.md) §6.12；[测量 JSON](evidence/measurements.json) 的 loader-failure、timeout、lock 三个记录；[根 README](../../README.md)的运行风险说明。

## 11. C-10：原始证据误删，不能靠重跑伪造恢复

一次清理使用过宽匹配，误删了原先已有的 30 份 `zz1…zz10` stdout、stderr、stats 日志，而不是只清理本次新增临时文件。事故已向维护者披露；维护者明确决定“旧日志无需恢复，继续优化”。

[事故清单](evidence/deleted-logs-incident.json)保留文件名、恢复尝试与未恢复状态。这些日志的缺失不应用后来跑出来的同名文件冒充补回，也不应被其他幸存日志无标注替代。

**本次归档边界。** 在独立发布树中只加入显式列出的文档与证据；保留原工作区的未提交源码和全部现存实验产物，不进行通配符清理。

## 12. C-11：量化质量与容量实验也有独立的有效性门槛

- `-c 262144` 加短生成，只证明某个容量配置曾启动／运行，不等于 256k 长文 PPL/KLD 已验证。
- 1 chunk 与 8 chunk PPL 的 baseline 不同，不能把两张表拼成一个排名。
- TBQ4 的短 Paris 输出不能推翻 PPL≈65、无结果退出等反证；关闭旋转仍坏，也没有把确切 FA 缺陷自动定位到某一行。
- NXQ fake-quant 仍使用本地受限编码器。补 RoPE／换物化表示没有显示明显均值改善，不等于证明本地实现与原参考等价。
- 本模型只有 12 个 full-attention KV 层。把 48 个 transformer blocks 当成 48 个量化 KV 层，进而外推“需保护 80% 层”，不是有效推导。实际 N=0/2/4 的测量仍有价值；撤回的是错误分母和外推，不是删除不方便的数据。

详细数字、格式定义、参考差异与尚未完成的验证见[04 章](04-kv-tbq-and-nxq.md)。

## 13. 以后怎样避免重复踩坑

1. **先定义工作点**：SSD/mmap、full-RAM、GPU PLE、专家缓存、KV 容量分别是哪一层预算。
2. **保存身份**：源码提交＋dirty 状态、运行二进制哈希、参数、清理过的环境、模型元数据与输入 token 历史。
3. **把失败也入账**：崩溃、不可用、超时、装置错误、数值差异都保留，不只摘最高速度。
4. **分别过三关**：固定历史数值／自由生成／完整进程。先保证测量对象算对，再谈性能。
5. **统计真实生命周期**：预测不等于准入，准入不等于 ready，ready 不等于使用，使用不等于赚回传输成本。
6. **公开结论时带协议**：写明 byte、MiB、token、graph、component 的分母；把推算和实测分开。
7. **原始记录不覆写**：新发现写更正和替代结论；用文件哈希和证据索引关联它们。

这些是本项目失败样本导出的操作要求，不是一次源码审查就能宣称满足的形式证明。返回[档案索引](README.md)。
