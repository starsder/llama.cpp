[中文](README.md) · [English](README.en.md)

# Qwen MoE Hybrid Inference: The Complete Experiment Archive

This is not a scorecard of "how much was improved in the end"; it is a research record that preserves the R&D order, the failed paths, the engineering incidents and the retracted conclusions. Each topic explains: **why it was attempted → how it was implemented/measured → what actually happened → why it was kept or abandoned → what remains unresolved**.

**Read the [chronology](00-research-chronology.en.md) first.** The ordering follows the maintainer's supplements: SSD → main-memory PLE → static hot table + XT → Fate → SMoE teacher → online cache and dual gating → devpart timing/correctness → weight kernels and TBQ → NXQ. The 45 subsections of `handoff.md` are only a part of it, not the starting point of the whole R&D process.

## 1. Reading map

| Document | Question it answers |
|---|---|
| [00: R&D chronology](00-research-chronology.en.md) | Why the shift from PLE to expert prediction, and why KV was studied; how the maintainer-added 1G/90%+ and the teacher 99% are booked |
| [01: PLE, host and devpart](01-host-and-devpart.en.md) | SSD row cache, GPU PLE, per-layer rendezvous, async, page locking, split merge, device partitioning, prefill, 256k, MTP, and the unimplemented operators/tiered-storage design |
| [02: Prediction and cache](02-prediction-and-cache.en.md) | Static/XT/CrossLayer/Fate/SMoE, offline and online definitions, real usage frequency, eviction, admission, dual gating, global pool, LFU_POS, backfill, fixed history and the last-400-token candidate |
| [03: Weight quantization and kernels](03-weight-quantization-and-kernels.en.md) | Why UD-IQ3_XXS is used; the name does not equal the actual dtype; the CPU cost of IQ2_S/IQ4_NL and the Q series; addressing, compression, repack and the failed AVX2 rewrite |
| [04: TBQ and NXQ](04-kv-tbq-and-nxq.en.md) | The capacity, speed, PPL and KLD of TBQ3/TBQ4; the FlashAttention path; NXQ endpoints, materialization, rotation, reference-chain gaps, fake-quant and boundary protection |
| [05: Correctness and methodology](05-correctness-and-methodology.en.md) | Why fluent output is not enough; silent miscalculation, shared-pool stride, invalid controls, measurement-rig errors, memory incidents, accidental log deletion, exit crashes not localized to a stack |
| [Evidence index](evidence/README.en.md) | 125 small logs, aggregated JSON, historical snapshots, read-only parse output, hashes, and materials not found or not uploaded |

When investigating a specific question, you can search the topics for route numbers: `H01–H17`, `D01–D05`, `PC-01–PC-36`, `WQ-01–WQ-11`, `R-KV-01–R-KV-04`, `R-NXQ-01–R-NXQ-07`, `C-01–C-11`. These numbers cover both research routes and diagnostic events; they do not represent 91 independent and valid optimizations.

## 2. Version boundaries: recorded does not mean released

| Object | Identity |
|---|---|
| [`7e01451b2`](https://github.com/starsder/llama.cpp/commit/7e01451b2d7aab6a4ff58ecfdd2f3b13fcaeb0bf) | The native code baseline preserved by this archive: host routing fix and mixed-quantization addressing protection; not a comprehensive acceptance proof |
| [`0862af564`](https://github.com/starsder/llama.cpp/commit/0862af564b48b898f7d84a57a895420b86c23824) | The parent revision of this documentation commit, which updated the README on top of the above code baseline |
| Source workspace `17ca0de85` + uncommitted changes | The origin of the later repack/TBQ/NXQ and cache-policy research; being written into the documentation does not put it into master |
| This archive | Adds only documentation and selected evidence; does not upload experiment source code, models, binaries or large logits arrays |

"Released" only means the relevant code is within the above release baseline; "off by default", "already reverted", "design only", "local candidate" and "not accepted" are marked separately. Original wording in historical documents such as "completed", "sealed", "correct" and "ceiling" must be read together with the later corrections.

This archive **did not re-run inference, performance benchmarks or model tests, and did not compile the model program**. It only performed read-only parsing of existing GGUF headers and existing scheduling logs, and checked the consistency of the documentation and evidence files. The release baseline does not gain any new speed or correctness endorsement from this cleanup.

## 3. The conclusions most easily misread

1. **PLE has two distinct levels.** The earliest was the SSD → main-memory row cache, which the maintainer reports as about 1G with a 90%+ hit rate, and which did save PLE resident memory; the later GPU PLE is another level. The GPU-level 93.1% log cannot be used to prove a denominator for the earlier hit rate, nor can a comparison of 4× capacity plus a different gather implementation be attributed to prefetching itself. An all-RAM working point makes the original SSD benefit no longer prominent; that does not mean the original approach was invalid.
2. **The teacher 99% is the motivation for going online, not an online cache hit rate.** The figure is kept as recalled by the maintainer. The surviving N+k offline reports have other results such as 68.53%; the full protocol of the 99% round is missing, so the two can neither be merged nor asserted to necessarily use different denominators. Only after Fate failed to obtain a useful signal did SMoE become the main line.
3. **Dual gating is a value gate plus a deadline gate.** One decides whether each byte of prefetch is worth it, the other decides whether it can catch up with consumption; the byte-rate budget is a separate control quantity. The default switch, cold start and worker bypass must be read from the source conditions. "All-source frequency admission" is a separate later failed candidate; do not conflate the two.
4. **Hit rate, bytes, waiting and throughput are not the same objective.** A higher hit rate can come with more transfer and lower speed; cutting 53% of the logical prefetch bytes also did not establish a stable speedup. Submitted bytes are not a PCIe bus count, and the reciprocal of the timing slope is not a bandwidth measurement.
5. **Good-looking host/devpart results were once overturned by correctness problems.** Zero data on the CPU half and routing-copy timing errors both once produced "fluent and fast". The old host control for devpart is also invalid; that set of numbers cannot be used to judge the net merit of the two paths after the fix.
6. **The UD label does not mean all weights share the same 3-bit format.** The actual bulk is IQ2_S and IQ4_NL, with IQ3_S as a further exception. A low bpw does not guarantee a fast CPU; the index/table-lookup cost of IQ2_S matters greatly, but two failed AVX2 attempts cannot prove that all implementations are "maxed out". The kernel tool prints GiB/s as GB/s; the topic keeps the original numbers and corrects the conversion.
7. **TQ3/TQ4 here means TBQ3/TBQ4 (not the regular weight Q3/Q4).** TBQ4 is unusable under the tested 512-context 1/8-chunk quality protocol; the same-batch single-chunk value is 63.7301 while 65.0026 comes from another batch probe. TBQ3 lacks the CUDA FA whitelist, and the released source routes to CPU `FLASH_ATTN_EXT`; the historical binary's actual placement lacks logs, so it cannot be made up as "GPU dequantizes first, then non-FA".
8. **NXQ is not a faithful reproduction of the original project, nor is it a native packed FA.** Locally the GPU stores packed KV first, then materializes a full F16 and hands it to the existing FA. The reference project's runtime fake-quant and offline compression accounting differ from the local fixed-length format; the encoding/rotation alignment and local controls did not close all the gaps, and the protection boundary still has a quality tail problem.
9. **The latest candidate did not obtain full acceptance.** The 18.6 t/s CLI is below the 19 screening line; the fixed-history 400-step run had 398/400 top-1 agreement, and another CLI exited with `0xC0000005`. Neither the passing logic fixtures nor the 24 stride GPU probes without error can be extrapolated into numerical model equivalence or a stable speedup.

The corrections did not quietly overwrite the original ledger: the topics state the conclusions that can currently be supported, while [sources](sources/handoff.md) keeps the historical statements and status flags, so that it is easy to see how conclusions were retracted.

## 4. Item-by-item destination of all 45 handoff sections

The table below keeps the judgmental wording of the historical titles; **the titles are not conclusions endorsed by this archive**; the "observations and boundaries / kept or abandoned" sections of the target topics are authoritative. The full original text is in the [handoff historical snapshot](sources/handoff.md).

| Original section | Historical title | Topic destination |
|---|---|---|
| §6.1 | Completed measurements (2026-09-12; same binary, `run-cur-ref.ps1` at the same configuration level) | [01/H01–H05](01-host-and-devpart.en.md) |
| §6.2 | Implemented but neutral changes | [01/H06](01-host-and-devpart.en.md) |
| §6.3 | Next steps (direction already confirmed with the user) | [01/H09](01-host-and-devpart.en.md) |
| §6.4 | 2026-09-12: final conclusion on the latency structure (**independent of the copy mechanism**) | [01/H02/H03/H07/H09](01-host-and-devpart.en.md) |
| §6.5 | 2026-09-12: the devpart host-leaf experiment (speed already met the target, correctness unresolved) | [01/H09: early inflated speed](01-host-and-devpart.en.md) |
| §6.6 | 2026-09-12: prefetch volume and the admission deadline (conclusion of this session, **volume is a cost, not a benefit**) | [02/PC-12–PC-16/PC-24](02-prediction-and-cache.en.md) |
| §6.7 | 2026-09-12 evening: hot-region root-cause fix — **the eviction score was wrong** (the biggest gain of this session) | [02/PC-17–PC-21](02-prediction-and-cache.en.md) |
| §6.8 | 2026-09-12 late night: adaptive threshold (**extremum-seeking controller**, converged) | [02/PC-22](02-prediction-and-cache.en.md) |
| §6.9 | 2026-09-12 late night 2: model-driven self-tuning (B) — threshold and budget both computed by itself | [02/PC-23/PC-24](02-prediction-and-cache.en.md) |
| §6.10 | 2026-09-12 late night 3: localizing the exit-phase crash + whole-machine incident and anti-recurrence gate | [05/C-07](05-correctness-and-methodology.en.md) |
| §6.11 | 2026-09-12 late night 4: CPU_ASYNC promoted to production + itemized overhead inventory (currently 48.7ms/token) | [01/H06/H08](01-host-and-devpart.en.md) |
| §6.12 | 2026-09-12 late night 5: pinned read-back (ineffective) + split structure + compatibility of devpart with the adaptive system | [01/H02/H08/H10](01-host-and-devpart.en.md) |
| §6.13 | 2026-09-12 late night 6: split merge investigation — **those 13.1ms were bought with VRAM; do not do it** | [01/H08](01-host-and-devpart.en.md) |
| §6.14 | 2026-09-13: separation of prefill/decode preference (measured) | [01/H13](01-host-and-devpart.en.md) |
| §6.15 | 2026-09-13: impact of SMoE/cache on prefill (measured: **neutral**) | [01/H13](01-host-and-devpart.en.md) |
| §6.16 | 2026-09-13: prefill reading the cache (D2D staging) — **measured zero benefit, reverted** | [01/H12](01-host-and-devpart.en.md) |
| §6.17 | 2026-09-13: devpart re-check — **gain confirmed +54%, the correctness defect localized to the visibility of device-partition output** | [01/H09: old gain invalidated](01-host-and-devpart.en.md) |
| §6.18 | 2026-09-13: devpart correctness fix — root cause localized, half fixed (graph build crashed) | [01/H09: graph build anomaly](01-host-and-devpart.en.md) |
| §6.19 | 2026-09-13: devpart correctness — **fixed (output coherent)**, what remains is floating-point ordering differences + synchronization overhead | [05/C-01: coherent does not equal correct](05-correctness-and-methodology.en.md) |
| §6.20 | 2026-09-13: devpart — speed path opened up (+70%), CPU-half data still to be diagnosed | [01/H09: CPU zero-data inflated speed](01-host-and-devpart.en.md) |
| §6.21 | 2026-09-13: remaining devpart defects converge to a **single point** — premature read-back was not committed | [01/H09: premature read-back not committed](01-host-and-devpart.en.md) |
| §6.22 | 2026-09-13: devpart wrap-up status — the pipeline works, defects converged to "the content of the residency table on the device side" | [01/H09: device residency table](01-host-and-devpart.en.md) |
| §6.23 | 2026-09-13: devpart fixed (correctness aligned with the host, verbatim identical) | [01/H09: corrected by the probe at the time](01-host-and-devpart.en.md) |
| §6.24 | 2026-09-13: devpart 400-token steady state and the root cause of "cache not being filled" | [01/H09/H10: residency decision still on the host](01-host-and-devpart.en.md) |
| §6.25 | 2026-09-13: sealing record (conclusions and status of this round) | [01/H09/H11: retracted after sealing](01-host-and-devpart.en.md) |
| §6.26 | 2026-09-13: MTP compatibility investigation (MTP runs; cache × speculative front end still crashes, bisected to the point) | [01/H15: MTP investigation](01-host-and-devpart.en.md) |
| §6.27 | 2026-09-13: MTP×cache — decision and bisection conclusion (current status) | [01/H15: bisection](01-host-and-devpart.en.md) |
| §6.28 | 2026-09-13: **decision: abandon the MTP×cache approach** (all WIP reverted) | [01/H15: revert](01-host-and-devpart.en.md) |
| §6.29 | 2026-09-13: cache parameters under the vision encoder + 256k + TBQ4 KV (conclusion: use auto, do not hand-tune) | [04/R-KV-04: auto definition corrected](04-kv-tbq-and-nxq.en.md) |
| §6.30 | 2026-09-13: major finding — `LLAMA_MOE_SPLIT=1` **silently computes wrong results** (consistency/race bug) | [05/C-02: routing silent error](05-correctness-and-methodology.en.md) |
| §6.31 | 2026-09-13 night: deployability matrix + safety lock (user already asleep; this round wrapped up autonomously) | [01/H11: deployment conclusion retracted](01-host-and-devpart.en.md) |
| §6.32 | 2026-09-13 late night (autonomous): **fix the split silent miscalculation + restore speed + look-ahead adaptation** (all verified) | [02/PC-25/PC-26: fix and ahead](02-prediction-and-cache.en.md) |
| §6.33 | Check item 1 closed: the r1 anomaly of `SMOE_AHEAD=1` = the "one-beat late" of non-blocking read-back | [02/PC-27: read-back one beat late](02-prediction-and-cache.en.md) |
| §6.34 | Assessment: the AVX2 dot/repack kernels for IQ4_NL **do not help** us (measured) | [03/WQ-05: initial judgment](03-weight-quantization-and-kernels.en.md) |
| §6.35 | Weight-type measurement (unsloth UD = dynamic quantization, mixed multi-format) + correction of 6.34 | [03/WQ-01/WQ-02/WQ-06: dtype correction](03-weight-quantization-and-kernels.en.md) |
| §6.36 | Impact of mixed types (UD) on the read strategy: audit + runtime safeguard | [03/WQ-03: addressing](03-weight-quantization-and-kernels.en.md) |
| §6.37 | Cache compression headroom assessment: **essentially none** (measured) | [03/WQ-04: compression and padding correction](03-weight-quantization-and-kernels.en.md) |
| §6.38 | Kernel coverage matrix: whether every combination of our formats × compute paths × CPU/GPU has an accelerated kernel | [03/WQ-07: coverage matrix](03-weight-quantization-and-kernels.en.md) |
| §6.39 | CPU kernel efficiency measurement + feasibility assessment of "maxing out 4n/repass" (conclusion: not the current bottleneck) | [03/WQ-08: CPU efficiency and units](03-weight-quantization-and-kernels.en.md) |
| §6.40 | AVX2 kernel feasibility boundary + "what it is worth if done" (deciding whether to invest effort) | [03/WQ-09: feasibility boundary](03-weight-quantization-and-kernels.en.md) |
| §6.41 | AVX2 i-quant kernel optimization: both experiments were done, and the conclusion is that **this path is maxed out on AVX2** | [03/WQ-10/WQ-11: two failures, not a general upper bound](03-weight-quantization-and-kernels.en.md) |
| §6.42 | KV quantization type evaluation: speed / MoE cache slots / PPL / KL divergence (including one bad point for TBQ4) | [04/R-KV-01–R-KV-03](04-kv-tbq-and-nxq.en.md) |
| §6.43 | NXQ GPU KV compression integration | [04/R-NXQ-01/R-NXQ-02/R-NXQ-07](04-kv-tbq-and-nxq.en.md) |
| §6.44 | NexusQuant alignment: the reference's real path, this round's two modifications, two structural gaps | [04/R-NXQ-02/R-NXQ-03/R-NXQ-06](04-kv-tbq-and-nxq.en.md) |
| §6.45 | NexusQuant faithful reproduction (fp16 fake-quant path) and conclusions | [04/R-NXQ-04–R-NXQ-06](04-kv-tbq-and-nxq.en.md) |

## 5. Routes not among these 45 sections but also included

| Supplementary source/route | Destination |
|---|---|
| The maintainer's supplementary account of the earliest PLE row cache, about 1G/90%+, the later small GPU PLE gain, and the shift to all-RAM | [00](00-research-chronology.en.md), [01 H14](01-host-and-devpart.en.md), [verbatim record](sources/author-recollection.md) |
| Static hot table, XT, CrossLayer, Fate; the motivation for the SMoE teacher 99% | [02 PC-01–PC-06](02-prediction-and-cache.en.md) |
| SMoE N+k, FIFO simulation, trace gaps, the ffn_input single item and the full three items | [02 PC-07–PC-11](02-prediction-and-cache.en.md), [degradation experiment plan snapshot](sources/smoe-nk-degradation-plan.md) |
| Shared global pool, LFU_POS, cross-layer victims, position value, byte-stride and UID risk | [02 PC-28–PC-31](02-prediction-and-cache.en.md), [05 C-03](05-correctness-and-methodology.en.md) |
| Lifecycle diagnostics, the failed all-source frequency gate, the empty-cache/rotation backfill fix, 400-token fixed history and the exit failure | [02 PC-32–PC-36](02-prediction-and-cache.en.md), [05 C-04/C-05/C-08/C-09](05-correctness-and-methodology.en.md) |
| IQ4_XS AArch64 repack fix and its boundary against the local AVX2 gain | [03 WQ-11](03-weight-quantization-and-kernels.en.md), [original commit notes](evidence/commit-notes.txt) |
| Standalone MoE operators, RAM/VRAM/SSD tiering, expert clustering, shared-expert front-loading, deadline rewrite | [01 D01–D05](01-host-and-devpart.en.md), [performance plan](sources/moe-decode-perf-plan.md), [rebuild spec](sources/rebuild-spec.md) |
| The accidental log-deletion incident and the facts not recovered, invalid noise tolerance, correctness before speed | [05 C-09–C-11](05-correctness-and-methodology.en.md), [incident record](evidence/deleted-logs-incident.json) |

## 6. How to use this archive

- To reuse code: first check the symbols and default values in master; do not infer from a WIP description that a feature has been released.
- To compare speed: first check the binary hash, prompt, context, KV, cache budget, step count and run phase. `auto`, 6144 MiB, 2048 MiB and different batches cannot be stitched into a single A/B pair.
- To cite quality: distinguish roundtrip, teacher-forced, free generation, PPL, KLD, top-1 and logit difference; the `±` after KLD is, in the corresponding tool, the standard error of the mean, not the standard deviation of 8 chunks, and not an equivalence test.
- To reopen a failed route: reopen it under the preconditions stated in the topic; do not carry over retracted speeds, wrong units, or high hit-rate figures that lack a protocol.
- To audit the evidence: start from the [evidence index](evidence/README.en.md). Original hashes and public-copy hashes are listed separately; missing material is explicitly written as missing, and reruns are not disguised as recovery.

## 7. Sources and acknowledgements

This fork derives directly from [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp); the base engine comes from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). Unsloth UD in the model file name refers to the mixed-quantization release definition, not a single format invented by this archive.

The NXQ reference project is [NexusQuant](https://github.com/jagmarques/nexusquant), corresponding to [nexusquant-kv 0.6.3](https://pypi.org/project/nexusquant-kv/0.6.3/); see chapter 04 for the sources, license and gaps against the actual implementation. Only public material and local experiments are analyzed here; the original project's compression-rate/quality commitments are not counted as this fork's results, nor was the original project run to fake a reproduction.

Back to the [repository README](../../README.en.md).
