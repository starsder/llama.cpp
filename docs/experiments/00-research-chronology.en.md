[中文](00-research-chronology.md) · [English](00-research-chronology.en.md)

# 00 Research Main Line: From PLE Saving Main Memory, to Prediction, Transfer, and Correctness

**The project's starting point is PLE, not the later host performance ledger.** The original question was: how do you avoid keeping the entire enormous PLE table resident in main memory, while still being able to fetch the rows you need quickly? The later expert cache carried over the same idea of "keep only the small subset of data that is actually needed," but added the hard problems of prediction timing, asynchronous transfer, GPU residency, and CPU/GPU division of labor.

The order of the research and the motivations for each turn are filled in from the [maintainer's recollection](sources/author-recollection.md); the specific mechanisms, numbers, failures, and fixes are supported by the individual topic documents and the [primary evidence index](evidence/README.en.md). Some of the work proceeded in parallel; what follows is the causal main line, not a pretense that every change has a precise, verifiable timestamp.

## I. PLE: solve SSD → main memory first, rather than chasing GPU token/s first

### The original problem

The target GGUF's `per_layer_token_embd.weight` is about 26.8 GiB. If the whole table were left occupying main memory long-term just so that a small number of rows could be used, it would crowd out the expert weights, the system, and other working sets.

The first step was therefore a **host PLE row cache on the SSD mmap loading path**: keep the quantized representation on disk, cache the needed pages/rows and exploit prefetching; this is not a change to the model mathematics, nor a re-quantization of PLE.

### What was obtained

The maintainer reported early measurements of: **about 1G of PLE cache occupancy, reaching a hit rate above 90%**, which can free up a large amount of main memory that would otherwise have to hold PLE resident. This was the initial success criterion for this line, and it cannot be retroactively dismissed using only full-RAM token/s.

The surviving related logs also contain experiments on host PLE, GPU L1, prefetch toggles, and different capacities and lengths. They are not the same set of numbers: for example, the host layer's `ablation-ple-only.csv` shows a cumulative hit rate of about 93.1%, while GPU L1's `ple-gpu-overlap-stats.csv` shows about 93.4%. The latter **cannot** be passed off as evidence of SSD→main-memory hits; for the corresponding protocol, cache level, capacity, and prefetch behavior see the [PLE subsection of chapter 01](01-host-and-devpart.en.md).

### Why it was later not used as the main configuration

Next, a GPU PLE cache was built along similar lines, but the speedup was limited. Later still, the target operating point became full residency of the weights in main memory, so the SSD read problem that had to be mitigated earlier had already changed, and GPU PLE in turn competed with the MoE expert cache, KV, and workspace for VRAM.

As a result, the full-RAM/MoE-priority configuration disables PLE. **This is a change in the benefit premise and in resource allocation, not a failure of the PLE row cache approach.** If SSD/low-main-memory deployment is targeted again in the future, the early PLE results are still worth reusing.

## II. First version of the expert cache: static hot table + XT

PLE proved that "caching the data actually accessed" has value, and the next step turned to MoE expert weights.

The first combination adopted was one that was easy to implement and explain:

- **Static hot table**: count high-frequency experts per layer from traces, providing a residency base.
- **XT**: predict the same-layer experts for the next token from the current token's expert selection, with roughly one full round of layer traversal as lead time.
- **CrossLayer control**: an adjacent-layer transition table, convenient for estimating coverage offline and then comparing against the union of the static tables.

This stage left behind the manifest format, hold-out replay, static/transition table ablations, and lazy-mode capacity sweeps. The gains were not zero-cost: pinning consumes VRAM and slots, and also freezes in a hot set that may not suit a new distribution.

**Why keep looking for a predictor.** The static table is not flexible enough, and XT's coverage is not good enough either; the cache can hit, but it is still some distance from "delivering the right experts to the GPU in time, on demand." They were later kept as seeds or explicit controls, and no longer served as the main predictor. [See PC-01–PC-04 for details](02-prediction-and-cache.en.md).

## III. Fate: predict ahead from the hidden state, the first setback

The next step tried the Fate-style idea: use the current hidden/gate input and the next layer's gate weights ahead of time to predict subsequent experts, avoiding the wait for the real routing to complete before starting the transfer.

The maintainer states explicitly: **the attempt at the time failed to squeeze enough useful information out of the hidden layers, and therefore stalled.** The archive preserves this reason for the turn, rather than writing the history as a smooth succession of swapping a few predictors.

The conclusion is bounded at the same time: this was this project's local approach at the time, it is not a proof that the Fate paper is invalid, and it does not mean that all hidden-state prediction is impossible. No fully corresponding enablement record was found in the surviving run logs, so the precise hit rate at the time cannot be fabricated in hindsight. [See PC-05 for details](02-prediction-and-cache.en.md).

## IV. Shared-expert SMoE: a 99% teacher rate made the online implementation worth trying

After the setback, the work turned to **using shared experts in the SMoE manner**: shared experts are always computed; then, combining the current input with the part of the routed contribution that is already available, a surrogate input for the subsequent gate is constructed, without having to wait for all the cold experts to finish.

**The key turning point was a 99% hit rate in the teacher test.** This is a measurement the maintainer reported directly, and it is also the motivation for starting the online implementation. The archive does not use a later, different set of `recall@10` numbers to replace or refute it; the exactly corresponding candidate set, denominator, and test protocol still await matching against the primary records.

Later, a traceable N+k teacher-forced control was also run, studying the degradation as the lead time increases, and it found that on a certain dataset the gap between input-only and the full surrogate is very small. This raised the possibility of removing a dependency altogether, but **a close offline approximation is not the same as having changed the online path to input-only**, let alone having verified cumulative generation quality.

**This step proved "worth implementing"; it did not prove "the online path is necessarily fast."** [See PC-06–PC-11 for details](02-prediction-and-cache.en.md).

## V. The real engineering difficulty: accurate prediction can still deliver late and transfer too much

The online pipeline imposes an entire extra set of conditions beyond the teacher test:

```text
candidate expert prediction
    ↓
admission and deduplication
    ↓
weight copy submission/queueing
    ↓
complete and stay resident before consumption
    ↓
correct routing and slot binding
    ↓
full computation on both CPU/GPU
```

Along the way, one after another: per-layer read-back and event waits, asynchronous workers, pinned staging, non-blocking results arriving one beat late, rank cutoffs, in-flight deduplication, the static hot table contaminating the prediction coverage accounting, a wrong gate-softmax eviction score, and hot-region backfill and cold start.

The most counterintuitive historical record is: prefetch of about **161→464 MB/token**, hit rate **40.2%→58.2%**, yet speed **14.3→10.6 t/s**. This is not a controlled benchmark of the current release build, but it changed the objective from "the higher the hit rate the better" to "whether the benefit that hits bring can cover the transfer and the waiting." The reciprocal of the timing slope is about 13.8 GB/s, which is not a measured PCIe peak.

Another example: synchronous `AHEAD=1` can significantly improve prediction rank hits, but the added synchronization cost in turn offsets the gain. **Accuracy and usable lead time must be considered together.** [For mechanisms and corrections see chapters 01, 02, 05](02-prediction-and-cache.en.md).

## VI. The two gates: from "hit as much as possible" to "net benefit and can it make it in time"

This algorithmic line is the point where the above contradiction converges, and it should not be drowned in a list of environment variables.

### Gate one: is it worth transferring

The source code `moe_cache_effective_rank_cut()` estimates the hit benefit per byte by predicted rank, compares it against a threshold, and decides how far down the ranking to take.

- A fixed cutoff can be retained, avoiding an implicit dependency on `n_used`.
- YIELD_AUTO can be used to probe token time slightly, tuning `yield_min`.
- TREND_AUTO can be used to fit `time ≈ a − V × hits + P × MB`, taking `P/V` as the break-even threshold, together with hysteresis, constraints, and fit rejection.
- The byte budget limits the transfer rate and is not the same question as "is a single candidate worth it."

### Gate two: is there enough time

`moe_prefetch_feasible()` estimates the arrival time of the in-flight work plus this batch's copies, and compares it against the deadline given by how many layers remain until consumption and the per-layer EWMA time. When the predicted transfer cannot make it, there is no need to keep piling onto the queue, and the experts still have a CPU compute path.

Here `70 µs/copy` and `20 GB/s` are model constants, not bandwidth learned from measurements on this machine; the adaptive part covers layer time and in-flight volume. The call scope and bypass conditions must also be read from the source code; one cannot claim that all cache writes are governed by these two gates.

### Why not all adaptive features are on by default

Extremum search has a probing cost, and regression is affected by warm-up, collinearity, and state changes; several early threshold/budget schemes also lost to a fixed cut. The algorithm itself preserves the correct decomposition of the problem, which does not mean every controller is already better than a fixed configuration.

**Keep this separate from the later frequency gate.** The all-sources frequency gate is a different admission experiment, of "is the candidate hotter than the victim," and is not the two-gate scheme itself; it later reduced bytes without a stable speedup, and has been rejected. [For criteria, formulas, defaults, and the failure table see PC-12–PC-24](02-prediction-and-cache.en.md).

## VII. The dev path: meant to reduce host involvement, but ran into timing correctness first

The host path has to read back routing, partitioning, and publishing leaves before starting computation on both sides. devpart tries to put more partitioning/residency decisions on the GPU, reducing host participation.

What followed was not one bug but a string of interface-contract problems: whether the host leaf or the GPU routing writes first and reads first, whether the WGT staging is really committed, whether the device residency table is published, whether looking up activations by name finds the wrong one, whether the slot view matches the original expert ID, and whether the CPU half really computes.

There is therefore a correction chain that must be preserved in full: **slow first version → seemingly blazing fast but under-computing/miscomputing → item-by-item fixes → the correct path is no longer fast → not made the default.** The highest 28–32 t/s cannot be kept on its own.

At the same time, the host split uncovered a separate, pre-existing routing copy problem, which forced the previously "frozen" 20.3 t/s to be retracted as well. The final master chose the conservative host baseline that includes that routing copy fix and the UD geometry guard, not the commit that first looked fast. [For the full iteration see chapter 01; for the correctness criteria see chapter 05](01-host-and-devpart.en.md).

## VIII. Weight quantization and CPU kernels: not ignored, but SIMD cannot be discussed apart from the bottleneck

The work then went back to examine quantization and kernels: given that a CPU half exists, could IQ decoding and dot products be the real slow point?

First, one premise needs correcting: `UD-IQ3_XXS` is a mixed-recipe label; the largest share of the actual bytes comes from IQ4_NL, the expert gate/up are mostly IQ2_S, with some IQ3_S as well. Inferring kernels from the file name leads in the wrong direction from the very start.

Single-core data does show that some grid-codebook IQ kernels on Zen 3 are far slower than K-quants of similar bit width: the codebook reads, index assembly, and unpacking methods differ. Gather and de-insertion were tried; one was slower, the other essentially unchanged. The IQ4_XS 8×8 repack later showed a microbench gain, but the established model/buffer/operator path did not automatically become faster because of it.

**The conclusion is not "all IQ is slow" or "AVX2 is hopeless."** It depends on the specific dtype, backend, operator, and buffer choice, and on whether this cost really lies on the critical path. [For quantization choices and all counterexamples see chapter 03](03-weight-quantization-and-kernels.en.md).

## IX. TQ4/TBQ4: saved KV, but failed on quality and the execution path

After longer contexts and vision squeezed the VRAM budget, KV compression could free up slots for the MoE cache. So f16, q8_0, q4_0, TBQ3, and TBQ4 were compared.

- What users call TQ3/TQ4 are `tbq3_0` / `tbq4_0` in this set of code, not the upstream ternary weight quantization TQ1/TQ2.
- TBQ4 looks normal on short outputs, yet counterevidence appeared: PPL around 65, and evaluations of somewhat greater length exiting with no results.
- TBQ3 has gaps in the CUDA FlashAttention whitelist/instance coverage; TBQ4 has instead already entered a different FA path. The two cannot be assumed to run the same implementation just because their bit widths are close.
- Disabling the rotation still broke; disabling FA directly was in turn rejected by the quantized V cache requirement. Ruling out some hypotheses does not mean the exact root cause has been fixed.

**256k here is mainly a capacity configuration; the quality numbers come from a short-corpus protocol.** TBQ3 was then the capacity-first candidate, q4/q8 are different fidelity trade-offs; TBQ4 cannot be treated as a usable configuration just because it saves slots. [For the full matrix and failure probes see chapter 04](04-kv-tbq-and-nxq.en.md).

## X. NXQ: get the endpoint right first, then discover that the reference protocol and the accounting are not equivalent

The NXQ/E8 exploration continued to pursue low-bit KV, but once again exposed two layers of problems.

**The engineering endpoint.** Coset information, confusion between 256-element blocks and 32-element indices, scale underflow, CPY/SET_ROWS, selection after coordinate projection, whole-head rotation — each step has its own separate fix and evidence. Storing packed KV on the GPU, materializing it as f16 on read, and then calling the existing FA does not amount to having implemented a native packed FlashAttention either.

**Experimental interpretation.** The NexusQuant 0.6.3 reference runtime stores fp16 fake-quant results; int8 + temporal differencing + zstd is an offline compression accounting. The real footprint of the local fixed-length NXQ cannot be directly interchanged with the reference's nominal bit count. The later fake-quant control experiment on the local graph still uses the locally constrained encoder, and cannot claim to fully reproduce the original.

Adding K without RoPE did not show a lower mean KLD; boundary protection showed a mean improvement at the measured points, but the layer count had been computed wrong: **of the 48 blocks, only 12 are full-attention KV layers**. The 48/44/40 in the original table should be interpreted as 12/11/10 according to the actual quantization target; the claim of "protecting about 80% of layers" derived from the wrong denominator has no basis. The data is retained, the extrapolation is retracted, and it also cannot be presumed in the opposite direction that further protection is necessarily ineffective.

**This line still belongs to uncommitted research; it is not a feature delivered in master.** [See chapter 04](04-kv-tbq-and-nxq.en.md).

## XI. Later shared pool and residency period: the problem moved further down to "who was moved, and was it really used"

The shared pool and LFU_POS tried to move slots from cold layers to hot layers that will be accessed later. The stress experiment raised the hit rate, but the DMA count barely dropped, and the speed improvement did not stably exceed run-to-run variation.

So a residency-period ledger was added, distinguishing predicted/non-predicted admission, deduplication, unused eviction, used once/used multiple times, and short-interval reloads. On that basis, an all-sources frequency gate and a hot-only backfill fix were then attempted: the former was rejected, the latter resolved some logic fixture problems, but fixed histories still show logits differences, and the full CLI in turn showed below-threshold and exit crashes.

These attempts that did not pass acceptance were not packaged as achievements, nor did they disappear from the archive because they were paused. [See PC-28–PC-36](02-prediction-and-cache.en.md) and [methodology and incidents](05-correctness-and-methodology.en.md).

## What is finally retained is not a single line saying "how much faster it got"

- The value of PLE is first of all avoiding massive main-memory residency in SSD scenarios; the trade-offs of full-RAM cannot erase this starting point.
- A high teacher hit rate is a strong motivation for moving toward an engineering implementation, but online gains must pass through admission, transfer, deadline, residency, and correct computation.
- The two gates separate "worth transferring" from "can it make it in time"; the lifecycle ledger in turn separates "was once moved in" from "was really used."
- Timing errors, incomplete quantization paths, and wrong statistical denominators all manufacture pretty numbers. Retractions and corrections are themselves research results.
- Only with a concrete protocol, failure records, and version boundaries can the next maintainer decide whether a given line should be retried, or need not be stepped on again.

Return to the [full archive index](README.en.md).
