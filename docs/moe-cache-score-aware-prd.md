# PRD: MoE 层间预测与分层 MRS 缓存

状态：冻结设计 v1  
目标模型：Qwen3.8-Flash MoE 量化模型  
运行约束：`--lazy-mode on`、`--cpu-moe`、峰值显存不超过 15 GiB

## Problem Statement

当前模型拥有大量 MoE 专家，单个 token 只激活少量专家，但专家权重无法全部放入 16 GiB 显存。CPU-MoE 可以保证模型运行，但专家缺页、主机到设备传输和缓存抖动成为主要瓶颈。

现有缓存路径已经具备部分预测、预取和层内 bundle 能力，但仍需要固定以下行为：

- 缓存必须以 `(layer, expert)` 为管理单位。
- 同一层的 gate、up、down 权重必须共享一个 slot 生命周期。
- 预测结果不能改变原始路由和模型输出。
- 预测缓存与实际路由分数缓存不能混为一个策略。
- pending 的异步传输不能被当前计算读取或被淘汰。
- 缓存预算必须服从 15 GiB 总显存上限。

## Solution

实现一个保持原始 MoE 路由不变的层本地专家缓存：

1. 使用 XT 转移表作为低开销的 Fate-style 下一层预测器。
2. 使用单层窗口，当前层预测下一层，不默认进行三层预取。
3. 使用 HybriMoE 的 Minus Recent Score（MRS）作为层内淘汰策略。
4. 以完整路由分数更新 MRS，以预测结果只决定预取候选。
5. 使用每层独立 bundle buffer、slot 表和 pending 状态。
6. cache hit 走设备内复制或 direct-read，cache miss 仍由 CPU-MoE 计算并异步 warm。
7. 动态收紧缓存预算，确保总目标显存不超过 15360 MiB。
8. 增加可选的 SMoE 共享专家引导在线预测：用当前层输入、缓存中 GPU routed 专家贡献和常驻 shared expert 贡献构造近似 hidden，计算下一层 gate，仅把候选用于预取。

## User Stories

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

## Implementation Decisions

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

## Testing Decisions

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

## Out of Scope

- HyperMoE’s hypernetwork or HyperExpert model architecture. It changes model computation and is unrelated to this inference cache.
- Training or fine-tuning a hidden-state MLP predictor.
- Replacing the native router, changing top-k, dropping experts, or altering route weights.
- The full HybriMoE dynamic CPU/GPU timeline simulator and its intra-layer scheduling algorithm.
- Default three-layer prefetch. It may be evaluated later only after one-layer traffic and overlap are measured.
- Hard-coded shallow-layer caching. Layer allocation can be revisited after collecting model-specific traces.
- Rewriting the CPU IQ4/IQ3 dequantization kernel. That is a separate performance branch and must not be conflated with cache-policy gains.
- Claims of 20 tokens/s or any fixed speedup before the controlled A/B measurements are complete.
- Configurations that intentionally exceed the 15 GiB peak VRAM target.

## Further Notes

Fate’s key transferable result is that adjacent-layer gate inputs can support low-overhead next-layer expert prefetching, while its shallow-favoring cache behavior is model- and budget-dependent. The implementation now uses the paper’s online CPU gate path when `LLAMA_MOE_PREDICT_FATE=1`; the offline XT manifest remains an explicit A/B control. See [Fate](https://arxiv.org/abs/2502.12224).

HybriMoE defines MRS as an exponentially averaged top-P route-score history and reports the strongest benefit under constrained cache capacity. This PRD ports that policy per layer while retaining the project’s CPU-MoE fallback and layer-local bundle invariant. See [HybriMoE](https://arxiv.org/abs/2504.05897) and the [official implementation](https://github.com/PKU-SEC-Lab/HybriMoE).

The term HyperMoE is intentionally excluded from the implementation decision: the ACL paper describes a hypernetwork-based MoE model that transfers information among experts, not a cache or prefetch kernel. See [HyperMoE](https://aclanthology.org/2024.acl-long.571/).

The current measured layer-local cache baseline remains the reference point for the next A/B run. New MRS results must report both policy delta and memory delta; a higher tokens/s number without the corresponding hit, transfer, and VRAM counters is insufficient evidence.
