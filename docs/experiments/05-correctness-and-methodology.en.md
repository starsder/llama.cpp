[中文](05-correctness-and-methodology.md) · [English](05-correctness-and-methodology.en.md)

# 05 Correctness, Measurement Failures, and Engineering Incidents

What this project most needs to preserve is not only "which switch is fast", but also **why a result was once believed, and what evidence was later used to retract it**. This chapter accompanies the [R&D main line](00-research-chronology.en.md), [host/devpart](01-host-and-devpart.en.md), [prediction and cache](02-prediction-and-cache.en.md), [weight kernels](03-weight-quantization-and-kernels.en.md), and [KV quantization](04-kv-tbq-and-nxq.en.md).

This round of work only archived documents and existing evidence; it did not re-run the model, fix the open defects below, or merge unaccepted experiment code into master.

## 1. How evidence is graded

| Evidence | What it can support | What it does not automatically support |
|---|---|---|
| Saved raw logs, CSV, JSON with provenance hashes | The configuration, result, and exit status of that run; counts and differences that can be recomputed | Still holding after changing the model, the input, or the binary |
| Historical handoffs, commit messages | What was done at the time and how it was interpreted then; only reported experiment results are preserved | Pretending complete raw logs still exist; treating an old explanation as a final conclusion |
| Direct supplements from the maintainer | Missing R&D ordering, motivation, reported measured results | Fabricating sample counts, count denominators, or exact capacity units for a missing protocol |
| Source inspection, arithmetic recomputation | Lifecycle defects, type geometry, counting conventions, implementation coverage | Declaring a crash root cause without a stack; claiming a performance gain without running |
| Hypotheses or extrapolation | Deciding what is worth testing next | Acceptance, production recommendations, a paper's method being proven invalid |

The [historical sources](sources/) keep claims that were later overturned exactly as written, and mark them explicitly as historical. When a new retrospective finds a contradiction, the old notes are not silently edited; corrections are given in this document and in the topic-specific chapters.

## 2. C-01: Fluent short text masked under-computation and mis-computation

**Why this trap gets stepped into.** Many early smoke prompts only needed to generate a short answer such as "Paris"; a path with wrong routing, premature EOS, or a missing CPU half could also produce short text that looks fluent. A high token/s therefore cannot disprove that the computation is correct.

**Counterexamples that occurred.** Several 19–32 t/s highs on devpart were later found to have an inert CPU half, weight staging that was never committed, or consumption of wrong data; host's SPLIT=1 also had a separate stale-routing copy error. See [chapter 01](01-host-and-devpart.en.md) for the exact order in which these occurred and each fix. These results cannot remain as optimization gains.

**Another masking factor.** `--ignore-eos` suits a throughput protocol with a fixed request length, but it may let a degradation that should have stopped early keep emitting tokens. The historical Eiffel Tower free-generation regression required not adding that option; fixed-history replay is yet another kind of trial.

**Decision.** Keep three classes of questions separate: whether free generation degrades, whether logits change under the same real token history, and whether the full CLI exits normally. One of them cannot be substituted for the other two.

**Sources.** [handoff](sources/handoff.md) §6.5, §6.17–6.25, §6.30–6.32; the [host/devpart route table](01-host-and-devpart.en.md).

## 3. C-02: The stale routing copy on host split — valid addresses can still compute silently wrong

**Motivation.** Put hit experts on the GPU and miss experts on the CPU, while keeping the same MoE numerical computation.

**Root cause.** `ids_gpu` / `wgt_gpu` are the host leaves for the GPU half's inputs. The scheduler may have queued the input copy first, and the partition hook only rewrote the leaf afterwards; the device side therefore read the previous round's routing. Data types, pointers, and indices can all be valid and the result is still wrong.

**Fix.** After the hook writes, re-issue `ggml_backend_tensor_copy` for the corresponding device input. This is a fix already contained in the released code baseline (`0f853bb6b`, inside the history of `7e01451b2`), not a code change newly made for this documentation release.

**Verification boundary.** The historical report: before the fix the Eiffel case degraded 3/3; after the fix 4/4 gave the correct answer passage. But its record also explicitly states the answer passage length as 443–455, whereas the previous reference record was 751 characters. So it can only be called a pass of the degradation regression at that time; it must not be rewritten as being identical character-for-character, token-for-token, or logits-for-logits.

**Chain of corrections.**

1. SPLIT=1 was once treated as a sealed 20.3 t/s path; withdrawn after the silent error was found.
2. A single 15.9 t/s on SPLIT=0 was treated as a safe fallback; later the same configuration produced 3.4–6.5, and the earlier repeatable-performance judgment was withdrawn as well.
3. SPLIT=1 was temporarily banned; that protection was lifted after the input publish order was fixed.
4. The post-fix historical 20.8 t/s can be kept as a record of that time, but it is not a new acceptance for the current 400 request token / 6 GiB tier, and it also lacks a controlled A/B proof exactly matching the 9.2 t/s one.
5. For ahead curves swept on a wrong path, **the trend can be distorted too**; one cannot withdraw only the absolute speed while keeping the assertion that "the trend is reliable".

**Reopening conditions.** When changing graph partitioning, leaf lifetime, copy order, graph capture, or slot binding, this class of input publish ordering should be covered again. Do not only check whether expert IDs fall within a valid range.

**Sources.** [handoff](sources/handoff.md) §6.30–6.33; the stronger statements in the historical text, such as "all verified" and "identical character-for-character", must be read against the actual verification scope described above.

## 4. C-03: UD and shared-pool stride — the same address formula has two layers of preconditions

**First risk: mixed types.** The file name `UD-IQ3_XXS` is not the true type of every tensor; gate/up/down each have their own block size, row bytes, and expert bytes. The released baseline added per-tensor geometry checks and alignment guards, see [chapter 03](03-weight-quantization-and-kernels.en.md).

**Second risk: the cross-layer shared pool.** The later local shared pool put several layer layouts into one physical pitch. `2,534,400` bytes is not divisible by the `82`-byte block size of certain IQ formats; dividing the byte stride into a block stride first truncates early. The old consumption path also had an error where the original expert number was used for a compact slot view, exposed in the logs at about step 29.

**Fix principle.** Compute channel, sample, and expert offsets first as 64-bit byte addresses, then convert to the corresponding block pointers; one cannot assume that a block-divisibility condition holding for one layer still holds in the shared pool. After the direct/gathered `src[0]` binding changes, the CUDA graph cache should be made to see a new graph identity, rather than disabling CUDA graphs wholesale.

**What was actually proven.** The local CUDA probe covered 24 IQ2_S / IQ3_S / IQ4_NL compact and strided 4D, 1/4/17 token, and fused SwiGLU cases, with a maximum GPU-to-GPU difference of 0; the corresponding memcheck record is 0 errors.

**Conclusions that must not be stretched.** This is not a whole-model CPU-to-GPU equivalence proof, not a memcheck of the complete model, and it does not mean the final CLI exit path has been fixed. This shared-pool/stride follow-up work did not enter master with this documentation.

**Sources.** [Shared-pool PRD snapshot](sources/moe-cache-score-aware-prd.md); [chapter 02](02-prediction-and-cache.en.md); [measurement evidence](evidence/measurements.json).

## 5. C-04: Prediction accuracy, admission, residency, and actual use must be kept on separate accounts

The **SMoE teacher 99%** reported by the maintainer, some set of offline `recall@k`, the candidate coverage of prediction plus a static table, online cache hit, prefetch ready, and actual expert use are different quantities.

```text
It is in the prediction
  → allowed admission
  → transfer committed
  → completed before consumption
  → not evicted and bound correctly
  → actually used by this computation
```

Any one arrow can fail, and any one arrow can also incur extra cost. A high teacher hit rate is still worth implementing; it does not promise an online speedup. The early PLE SSD→main-memory hits and the later GPU PLE hits must also be kept separate by tier, see the [maintainer's supplement](sources/author-recollection.md).

Lifecycle counting further reveals: `readmit` must not be conflated with a single `admit`, residency deduplication, or real GPU use; the logical bytes of `unused_evict` do not equal the bytes actually measured on the PCIe bus. Instrumentation hooks should record real consumption, not padding, CPU routing, or repeated calls within the same graph.

**Preserved internal consistency checks.** The original baseline `10,216 admissions − 7,272 evictions = 2,944 live`; `133,484 GPU uses × 3 = 400,452 component hits`. They prove the internal relations of this counting, not that the hits themselves yield a net performance gain.

**Sources.** [chapter 02](02-prediction-and-cache.en.md); the `instrumentation-only-baseline` record in `evidence/measurements.json`.

## 6. C-05: The fixed-history comparison guards against input drift, but is not equivalent to full CLI acceptance

### 6.1 What the two kinds of timing each answer

| Protocol | Question it answers | Must not claim |
|---|---|---|
| `llama-cli` free generation, full process, 400 token request | The output the user actually sees, the generation speed printed by the CLI, the exit status | That the input history is exactly the same as in another variant |
| Same prompt IDs plus forced same subsequent IDs, 400 decode steps | Latency and logits differences under the same real history | That free generation is the same, that the CLI teardown path is the same |
| 128 steps, 512 MiB shared-pool stress | Diagnostics for when the eviction policy is genuinely triggered | 400 step / 6 GiB full-RAM baseline acceptance |
| Single-core operator microbench | The cost for a given dtype / kernel / size | An end-to-end token/s improvement |

The fixed-history timer covers `llama_decode + llama_synchronize`; logits copying and writing to disk are outside the timing. Its `1000 / mean_ms` must not be taken as full CLI speed. The timing may additionally report a steady value with the first 16 steps removed, while **correctness always covers all decode steps**.

### 6.2 The last few full CLI records

Common request tier: 400 token, 6 GiB expert cache, full-RAM, 8k, q8_0 KV, AHEAD=2, non-blocking, per-layer MRS; the complete environment is saved in JSON. **These are historical results from a local development binary, not acceptance for a rebuild of the release master.**

| Record | CLI Generation | Exit code | Conclusion |
|---|---:|---:|---|
| Original path plus lifecycle counting | 19.3 t/s | 0 | A valid full run of that time |
| All-source frequency gate candidate | 19.9 t/s | 0 | Free output has already changed; cannot serve as isolated speedup evidence, candidate later withdrawn |
| Hot-backfill-only candidate r1 | 18.6 t/s | 0 | Did not reach the then 19 t/s threshold |
| Later re-run of the old baseline | 18.4 t/s | 0 | The baseline fluctuates too; one cannot selectively compare only against an earlier high value |
| Hot-backfill-only candidate r2 | Unavailable | `0xC0000005` | Native crash; must not be counted as successful throughput |
| First paired launch of the old baseline | Unavailable | `0xC0000135` | A launcher error from a missing DLL, not a model performance result |

The JSON's `tokens_generated` was not independently parsed by the harness; `graphs_observed=403` must not be rewritten as "confirmed 403 tokens generated". Request length, actual decoder steps, number of visible characters, and graph count must be recorded separately.

### 6.3 Why the two candidates did not pass

- **All-source frequency gating**: prefetch bytes dropped substantially, but a 3-run fixed-history comparison showed no stable speed gain, and both online hit and ready declined; against the reference top-1 only 398/400 were identical. A candidate being stable across its own repeats does not mean it is correct relative to the original path.
- **Hot-backfill-only fix**: it genuinely fixed rotation layer skipping in the synthetic fixture, empty-slot handling, and comparison against real eligible victims; fixed-history hits about +2.67%, bytes about +0.62%. The median of the three-run means improved by about 1.05%, which falls within run variation; top-1 is still 398/400, and the full CLI again had a below-threshold run and a native crash.
- **Shared-pool LFU_POS stress test**: total hit about +20.77%, transfer count only about −0.60%, the median of the run-medians improved by about 1.92%; at 128 steps there was one top-1 flip. A high hit rate must not be packaged directly as a stable throughput gain.

These results are preserved in the [measurement JSON](evidence/measurements.json), including baseline repeats, candidate repeats, and failed exits, not just the best row. **The practice of using the baseline error multiplied by an arbitrary factor (for example 3×) as a correctness threshold has been withdrawn.**

## 7. C-06: Less synchronization time may just mean the wait moved elsewhere

A pinned routing read-back once cut enqueue from about 3785 µs to 4 µs, but raised synchronize from about 1 µs to 3787 µs; total time only went from about 48.7 to 48.3 ms. The conclusion is that the wait position moved, not that the read-back became three orders of magnitude faster out of thin air.

Likewise, async CPU, GPU queue, prefetch, and ids_wait are timing intervals that can overlap. Their sum exceeding the total time is not necessarily an error; but a counter with a tiny share must also not be treated as a strict Amdahl cap without checking its definition.

**Decision.** Record interval containment relations, threads and streams, synchronization points, and timer start/end. The targets of the predictor and the adaptive controller should face the end-to-end cost, not a single pretty counter.

**Sources.** [handoff](sources/handoff.md) §6.4, §6.11–6.12, §6.34, §6.39; [01](01-host-and-devpart.en.md) and [03](03-weight-quantization-and-kernels.en.md).

## 8. C-07: Page locking, whole-machine memory pressure, and an exit UAF that had already been fixed

**Motivation.** Keep large expert weights in main memory, with GPU prefetch wanting to use page-locked memory; make asynchronous transfer genuinely usable, rather than hiding synchronization cost in pageable staging.

**Incident.** Historically, combining mmap/lazy with large-range pin triggered whole-machine memory pressure / hang risk; concurrent model instances further amplify the risk. Separately, an old exit hook called prefetch wait after the backend had already been released, forming a UAF; that old wait was removed.

**Trade-off.** The later full-RAM controlled protocol requires: holding `.tools-run.lock` serially, at least 90,000 MiB of available memory, `--no-mmap --lazy-mode off`, and recording about 72.6 GiB of pinned weights and pin failures. This discipline applies to this large-model run protocol, and **does not negate the early PLE SSD/mmap scheme for saving main memory**.

**Boundary.** This old prefetch-wait UAF and the profiler lifecycle risk in the next section are different defects and should not be merged into a single incident that has all been fixed.

**Sources.** [handoff](sources/handoff.md) §6.10; [chapter 01](01-host-and-devpart.en.md); the load flags and memory records of the historical baseline JSON.

## 9. C-08: The final native exit crash — risk located, no crash stack yet

**Observation.** Candidate r2 returned `0xC0000005`, the CSV had already reached graph 403, and there was no final Generation line. The failure stderr stops after the DIRECT-VIEW summary; a successful run goes on to print SERVER-PROF / TOKEN-PROF / MOE-CACHE. These can only narrow the direction of the investigation.

**Lifecycle risk confirmed in source.** In `tools/server/server-context.cpp`:

```cpp
static server_context_impl * prof_self;
// constructor, when LLAMA_TOKEN_PROF is enabled:
prof_self = this;
atexit(&server_context_impl::prof_print);
```

`prof_print()` then reads instance counters through this raw pointer. `server_context` owns the implementation object, and the static pointer is not cleared after the instance is destroyed; a later atexit callback may access a freed object. Relevant locations: source workspace `server-context.cpp:868–900,4197–4200`.

**Why the root cause cannot yet be declared fixed.**

- The native stack of this particular crash was not captured, and execution never reached a fault handler; the causal link between the source risk and the actual crash is not yet closed.
- stdout is buffered, so an exit-time failure may occur after the final timing has already been formatted but before it is flushed. A missing Generation line does not allow inferring that the model has not yet completed its final decode.
- The fixed-history helper explicitly synchronizes every time and does not go through the same server/profiler lifecycle; a helper exit of 0 cannot vouch for the CLI teardown phase.
- Threads, events, and backend ownership have other points still to be audited; they must not be listed as confirmed root causes without evidence.

**Status.** Before the pause, only the native fault capture/probe source was prepared, **with no compilation, no run, no stack, and no committed fix**. On reopening, the real fault should be captured first, and then object ownership and callback lifetime fixed; turning statistics off is merely isolating a variable, not proof of a fix.

**Sources.** `final-r2.json` and the [public summary](evidence/measurements.json); the source code above; the exit risk record in [handoff](sources/handoff.md) §6.32. The original note "only affects the statistics output" is not rigorous enough: a non-zero exit must be treated as a failure.

## 10. C-09: Harness errors also contaminate conclusions

| Problem | Actual handling or current state | Lesson kept |
|---|---|---|
| The build target was updated, but the CLI used a different, old binary | Discovered and distinguished historically; see §6.12 for the logs | A successful build does not mean the new artifact was run |
| An old CLI was copied but the DLL search path was missing, exiting `0xC0000135` | Only after fixing the harness PATH was the old baseline 18.4 t/s obtained | A launch failure must not be filled in as a model 0 t/s, nor silently dropped |
| The wrapper does not propagate native exit codes | The old runner on the release side has this risk; only the later local runner records the real exit status | The raw child exit code must be saved; seeing a speed line is still not enough |
| A genuine timeout | Local 1-second timeout smoke: 124, result unavailable, model lock released | This is a harness smoke that has already been run, not a new test for this documentation |
| The lock is already held | Local smoke: exit 2, model not launched | Re-entrancy protection should take effect before the large model is loaded |
| Raw graph count and token count conflated | This document does not fabricate a generation count from the 403 graphs | Different counters keep their original names, sources, and availability |

**Sources.** [handoff](sources/handoff.md) §6.12; the loader-failure, timeout, and lock records in the [measurement JSON](evidence/measurements.json); the run risk notes in the [root README](../../README.en.md).

## 11. C-10: Raw evidence deleted by mistake; recovery cannot be faked by re-running

One cleanup used an over-broad match, mistakenly deleting 30 pre-existing `zz1…zz10` stdout, stderr, and stats logs, instead of cleaning only the temporary files newly added by that session. The incident was disclosed to the maintainer; the maintainer explicitly decided that "the old logs need not be recovered; continue optimizing".

The [incident list](evidence/deleted-logs-incident.json) preserves file names, recovery attempts, and unrecovered status. The absence of these logs must not be passed off as filled in by files of the same name produced by later runs, nor silently substituted with other surviving logs.

**Archiving boundary for this work.** Only the explicitly listed documents and evidence are added to the separate release tree; the uncommitted source in the original workspace and all existing experiment artifacts are kept, with no wildcard cleanup.

## 12. C-11: Quantization quality and capacity experiments also have their own validity thresholds

- `-c 262144` with a short generation only proves that some capacity configuration once launched / ran; it does not equal 256k long-context PPL/KLD having been verified.
- The 1 chunk and 8 chunk PPL baselines differ, so the two tables must not be spliced into one ranking.
- TBQ4's short Paris output cannot overturn counter-evidence such as PPL≈65 and exit with no result; it is still broken with rotation disabled, and it also did not automatically localize the exact FA defect to a particular line.
- NXQ fake-quant still uses the local restricted encoder. Adding RoPE / changing the materialized representation showed no clear mean improvement, which does not amount to proving the local implementation is equivalent to the original reference.
- This model has only 12 full-attention KV layers. Treating the 48 transformer blocks as 48 quantized KV layers and then extrapolating "80% of layers need protection" is not a valid derivation. The actual N=0/2/4 measurements are still valuable; what was withdrawn is the wrong denominator and the extrapolation, not the deletion of inconvenient data.

See [chapter 04](04-kv-tbq-and-nxq.en.md) for the detailed numbers, format definitions, reference differences, and verification not yet completed.

## 13. How to avoid repeating these traps from now on

1. **Define the working point first**: which budget tier SSD/mmap, full-RAM, GPU PLE, expert cache, and KV capacity each belong to.
2. **Save identity**: source commit plus dirty state, run binary hash, parameters, cleaned environment, model metadata, and input token history.
3. **Book failures as well**: crashes, unavailability, timeouts, harness errors, and numerical differences are all kept; do not just excerpt the highest speed.
4. **Pass the three gates separately**: fixed-history numerics / free generation / full process. First make sure the measured object computes correctly, then talk about performance.
5. **Count the real lifecycle**: prediction is not admission, admission is not ready, ready is not use, and use is not earning back the transfer cost.
6. **Carry the protocol when publishing conclusions**: state the denominators in byte, MiB, token, graph, component; keep estimates and measurements separate.
7. **Do not overwrite raw records**: write new findings as corrections and superseding conclusions; link them via file hashes and the evidence index.

These are operational requirements derived from this project's failure samples, not a formal proof that can be claimed satisfied by a single source review. Return to the [archive index](README.en.md).
