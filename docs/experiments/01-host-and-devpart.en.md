[中文](01-host-and-devpart.md) · [English](01-host-and-devpart.en.md)

# 01 — host path, split, devpart, CPU async, per-layer readback, pinned, worker, prefill and speculation

This file is an experiment log, not an implementation description. Each route gives the same set of
fields: route ID, status, why it was attempted, technical mechanism, experiment conditions and
evidence, observation and boundaries of the conclusion, reason for keeping or dropping it, open
questions and conditions for reopening it.

This file contains no new experiments, no new optimizations and no source changes. All numbers come
from existing logs or existing reports in the source workspace, and must be cited together with their
run context; retracted numbers are collected in §4.

Navigation: [research main line](00-research-chronology.en.md) · [prediction and dual gating](02-prediction-and-cache.en.md) ·
[weight kernels](03-weight-quantization-and-kernels.en.md) · [correctness and corrections](05-correctness-and-methodology.en.md) ·
[preserved evidence](evidence/README.en.md) · [historical handoff](sources/handoff.md).

---

## 0. Reading conventions

### 0.1 Status vocabulary (strictly distinguished; "the source exists" is not "released")

| Status | Basis for the determination |
|---|---|
| **Released (master baseline)** | The identifier/behavior is inside the native code baseline of the document parent version `0862af564`; source line numbers in this chapter follow that version |
| **WIP only** | Exists only in later experiment branches or in the uncommitted workspace; unreachable from the release baseline |
| **Design only** | Appears only in design documents (`moe-decode-perf-plan.md`, `rebuild-spec.md`, `moe-cache-score-aware-prd.md`); implemented on no tree |
| **Retracted** | Was once implemented and measured, then reverted or dropped (including "reverted after the experiment" and "decided not to do it") |
| **Still open** | Known defect or unreviewed item, with no conclusion |

The source workspace HEAD is `17ca0de85`, plus uncommitted NXQ/TBQ, cache and toolchain research
changes. The release baseline `0862af564` = native code baseline `7e01451b2` + one documentation
commit; `7e01451b2` is an ancestor of the source workspace HEAD, so "released" in §0.1 includes the
entire committed history of qwen4exp (`c08171aa8` … `7e01451b2`), but does not include the uncommitted
changes on top of the source workspace. Where host/split/devpart behavior is concerned, this chapter
always describes the master-reachable version and explicitly marks WIP.

### 0.2 Hardware and threshold

- Hardware (user-provided): AMD Ryzen 9 5950X, DDR4-2666 128 GB, RTX A5000 Laptop 16 GB, PCIe 4.0 x8.
- Current threshold: **19 t/s under the 400 token / 6 GiB cache / full RAM (`--no-mmap`) regime**.
  Later local builds already have records under that regime (see chapters 02/05), but that is not the
  same as rebuilding and accepting the release baseline; this chapter gives no "meets threshold"
  conclusion for master.
- Numbers that must not be cited as results: `20.3 t/s` (sealed value, see the retraction in §2.11),
  the `20.1–22.0 t/s` series (same reason), devpart's `19–23 / 28.2 / 29.7 / 32.0 t/s` (inflated under
  a wrong implementation, see §2.9), `20.7 t/s` (garbled old build).

### 0.3 Measurement-regime discipline (lessons from existing reports)

1. Performance is reported separately for warmup, prefill and decode; a steady subset may be reported
   additionally, but correctness must retain all decode steps. Early ledgers such as the 46 ms
   readback once mixed in non-steady graphs.
2. Detecting degradation-type defects **must not** use `--ignore-eos`, otherwise the degraded tail is
   masked.
3. At 256k context the run-to-run variance of t/s is about 35% (suspected Laptop GPU downclocking); a
   single measurement is not enough to fix an "optimal point".
4. Comparing speeds requires the same configuration: `run-cur-ref.ps1` defaults to `CACHE_MIB=2048`
   (hit rate ~25%, 13.7 t/s), which is not the same operating point as the `6144 + HOT_BACKFILL=8` tier.

### 0.4 Citation conventions

This chapter cites sources as "document name §number" and writes no cross-directory links. These
document names correspond to the following in the release tree: `sources/handoff.md`,
`sources/moe-decode-perf-plan.md`, `sources/rebuild-spec.md`, `sources/moe-cache-score-aware-prd.md`,
`sources/smoe-nk-degradation-plan.md`. Code citations are written as `path:line` (relative to the
repository root).

### 0.5 The research order (user-supplied authority) and its relation to this file

The route IDs in this chapter are organized by **mechanism**, not by research time; the main-line
narrative is carried by `00-research-chronology.md`. To avoid mistaking the late chapters of the
handoff for the starting point, the authoritative order supplied by the user is recorded here:

1. **PLE cache** (first stage): done for SSD→memory reads; the PLE weight table is 26.8 GiB, and the
   "rows that are used" need to be kept in memory/VRAM without changing the quantization.
2. **MoE/SMoE cache**: originally a **static table + XT transition table** (offline manifest/transition
   probabilities).
3. **Fate**: switched to trying a hidden-layer predictor; the conclusion was that **no useful
   predictive information can be squeezed out of the hidden layer** (a setback).
4. **SMoE (exploiting the shared experts)**: teacher test (teacher-forced) hit rate **99%**.
5. Only after that came the online implementation: engineering pitfalls, the hit rate rising but
   getting slower, and finally the dual-gated adaptive transfer threshold.
6. After that: the timing error on the devpart path, the TQ4 crash, NXQ and various fixes.

**Regime discipline concerning "99%" (must be observed)**:

- 99% is a **user-reported fact**; it is kept, and it must not be contradicted with numbers from other
  experiments.
- `full = 68.53%` in `smoe-nk-degradation-plan.md` §7 is the **offline single-step teacher-forced
  recall@10 (per-layer top-10 overlap rate, 8 prompts, 56400 samples)**; **whether it uses the same
  protocol and the same denominator** as the "teacher test hit rate 99%" **has not been confirmed**,
  and the two cannot substitute for or contradict each other. For 99% the note
  "**user recollection, protocol/denominator pending correspondence with the original record**" is
  retained, rather than defining the metric on its behalf.
- The online hit rates (27.5% / 70.4% / 80.8% / 90.9% etc.) are a third metric: they are jointly
  determined by cache state, admission and delivery rate, and must not be conflated with the teacher
  test hit rate.
- When §2.3 of this chapter cites 68.53%, it uses it for exactly one purpose: "the sensitivity of
  predictor accuracy to the input terms".

Likewise, **PLE must not be written up as an overall failure**: see §1A.

---

## 1. Route index

| ID | Route | Status | Main source sections | One-sentence conclusion |
|---|---|---|---|---|
| H01 | host path baseline ledger (86 ms/token breakdown) | Released | handoff §3/§6.1; perf-plan §1 | True baseline 11.3 t/s; the 40 ms is the per-layer host↔device rendezvous |
| H02 | per-layer router readback rendezvous (`ids_wait`/MRS) | Released (problem remains) | handoff §6.1/§6.4/§6.11/§6.12 | 17–19 ms at the time; the tested pinned change established no total-duration gain |
| H03 | SMoE-side graph readback event wait + lookahead distance | Released | handoff §6.1/§6.4/§6.6/§6.30 addendum/§6.32/§6.33 | 15.3 ms is waiting for the GPU; the fix is more lead time (ahead defaults to 3) |
| H04 | `insert_flush` hard drain per layer | Released (hypothesis void) | handoff §6.1; perf-plan §1/§P2.1 | Measured 0.07 ms/graph; "saves 20 ms" is void |
| H05 | `split_partition`'s "host CPU loop 22.7 ms" | Released (hypothesis void) | perf-plan §1; handoff §6.5 table | Actually waiting for the GPU to produce the router, not host CPU time |
| H06 | Asynchronizing the CPU half (worker) | Released (on by default) | handoff §6.2/§6.6/§6.7/§6.10/§6.11/§6.24/§6.31 | Neutral early on, later one round shortened it by about 1.4 ms; depends on the operating point |
| H07 | Weight pinning (`pin_weights`) and the memory gate | Released | handoff §5/§6.6/§6.10 | Pins 72.6 GiB, mutually exclusive with mmap; a single-instance lock and a memory precheck have been added |
| H08 | split segmented accounting and merging | Reverted after the experiment | handoff §6.11/§6.12/§6.13 | 13.1 ms cannot all be treated as removable launch tax; this merge OOMed |
| H09 | devpart + host leaf | Released (off by default) | handoff §6.3–§6.5/§6.17–§6.25 | The fake speedup is retracted; no same-regime net gain versus the fixed host has been established |
| H10 | Compatibility of devpart with hot-region signals/adaptive systems | Design only (not implemented) | handoff §6.12 ③/§6.24 | Device-side histogram + one readback per token, otherwise you trade hit rate for 17 ms |
| H11 | host leaf old-routing silent error (`SPLIT=1`) | Released (fixed) | handoff §6.30–§6.32; see also chapter 05 | All speed evidence of 20.1–22.0 is retracted; the fix = resend the copy immediately after writing the leaf |
| H12 | prefill read cache (D2D staging) | Retracted | handoff §6.16 | Zero gain: prefill's H2D is already asynchronously overlapped |
| H13 | prefill/decode phase separation (hot set) | Released | handoff §6.14/§6.15 | The concentration of the two phases is fundamentally different; what is separated is the statistics and the use, not the admission |
| **H14** | **PLE layered cache (CPU L2 / GPU L1) — the first stage of the research** | **The body is released; operator entry is design only** | handoff §6.29/§6.34/§6.35; rebuild-spec §2.3/§7; upstream `4e1865e34`, fork `2f1a363c8` | Effective at the SSD/mmap operating point (user reports 1G → 90%+ hits, at the **host row-cache layer**); small gain at the full RAM operating point (turning GPU L1 off actually gives 17.0 → 17.9 t/s) — **the log is filed separately in §1A** |
| H15 | MTP/speculation × expert cache | Retracted (all WIP reverted) | handoff §6.26–§6.28 | The only trigger condition is `SPLIT=1`; the speculative front end and the cache never coexisted |
| H16 | Adaptive admission (yield threshold/extremum search/trend regression/automatic ahead) | Released (off by default) | handoff §6.6/§6.7/§6.8/§6.9/§6.32 ④ | The threshold and the budget can be computed by the model itself; a fixed ahead=3 is still slightly better than online search |
| H17 | Higher hits yet slower, and dual gating | A released experiment interface | handoff §6.6–§6.9 | Value gate + time-limit gate; the rate budget is a third independent control quantity |
| D01 | Standalone MoE operator (getting an operator out of the shell) | Design only | perf-plan §0.1/§0; rebuild-spec §3/§4/§7.1/§8 | Only the device-side partition kernel landed; the operator itself is unimplemented |
| D02 | Memory hierarchy roadmap (RAM+VRAM → +SSD) | Design only | rebuild-spec §2.3 | PLE is already a prototype; neither of the two hard constraints on the MoE cache side is satisfied |
| D03 | Expert clustering storage | Design only | rebuild-spec §2.4 | Pure permutation, no kernel change; no tool, no permutation table, no implementation |
| D04 | Shared expert first + SMoE firing early | Design only (not implemented) | perf-plan §P1.4; nk-plan §7 | Offline measurement shows using only `ffn_input` is off by just 0.63 pt, but the code still adds the three terms |
| D05 | Admission model rewrite (`gate_copy_us` calibration / remaining-time deadline) | Partially designed, not adopted | perf-plan §2/§P1.1/§P1.2; rebuild-spec §4/§6 | The code is still 70 µs + by layer count; the measurement switched to a rank cutoff line and lookahead distance |

---

## 1A. The first stage of the research: PLE layered cache (H14, log filed separately)

> This section is placed first according to the authoritative research order supplied by the user: the
> PLE cache was the earliest stage, before the MoE/SMoE cache (see §0.5). It should not be written up
> as a "late prefill accessory", and it is not an overall failure — it is **effective at the
> SSD/mmap (lazy) operating point** and gives **a small gain at the full RAM operating point**.

### 1A.1 Why it was attempted (motivation)

- The target model has a **26.8 GiB** per-layer token embedding table `per_layer_token_embd.weight`
  (type `iq4_nl`; there is also the type ledger in section 6.35: the whole model is `iq4_nl`
  47.9 GiB / 63%, of which this table is 26.8 GiB). It cannot fit entirely into 16 GB of VRAM; main
  memory can hold it, but keeping it fully resident would crowd out other working sets, so the
  SSD/mmap scenario uses an on-demand cache.
- **The upstream motivation is SSD→memory reads** (`4e1865e34`, 2026-08-28, unsloth/Daniel Han,
  title "llama: batched readahead for lazily read gather tables"): `TENSOR_READ_LAZY` skips the eager
  pull-in of large tables and marks their range `MADV_RANDOM`; that also turns off kernel readahead,
  hence "a sparse gather faults synchronously once per row". The approach is to **first merge the rows
  that a batch is about to gather into whole pages**, then from `set_input` issue `MADV_WILLNEED`
  (Windows `PrefetchVirtualMemory`) hints in one go ("16 gathers become a couple of hints"). The two
  consumers are qwen4exp and gemma4; **only hints are issued, the result is unchanged**.
- On the fork side it was turned into a bounded cache (`2f1a363c8`): `ple_row_cache` at
  `src/models/models.h:2285-2332`, whose class comment reads in essence —
  *"A bounded host-RAM cache for the lazily mmap'ed PLE hash-embedding table. Entries are
  64 KiB-aligned groups of complete table rows. **The cache retains the original on-disk
  representation, so quantization is never changed.**"*
  The comment of `copy_pages` (`models.h:2299`) there is the key to the next layer down:
  *"Materialize complete raw pages for a lower-level cache. This keeps the CPU L2 in the path;
  callers never reach into the mmap directly."*
- Page geometry: `target_page_bytes = 64 KiB`, `rows_per_page = 64 KiB / row_bytes`
  (`src/models/qwen4exp.cpp:430-450`); under `iq4_nl` it is 728 rows/page, and the log prints
  `[PLE-LRU] enabled 2048 MiB: 32776 pages x 63 KiB, 728 rows/page` (63 KiB is a rounded display).
  The capacity is controlled by `LLAMA_PLE_CACHE_MIB`, and **unset means disabled** (`configure`
  returns directly when the env is empty).

### 1A.2 Measurements and evidence

- Statistics mechanism: `LLAMA_PLE_CACHE_STATS_FILE` appends one line per gather,
  `rows,resident_pages,capacity_pages,hits,misses,hit_rate,total_hits,total_misses,total_prefetch_pages`
  (`src/models/qwen4exp.cpp:184-197`; the code comment states outright *"A page is the LRU unit, so
  these are page (not individual-row) hits"*); GPU L1 additionally has
  `LLAMA_PLE_GPU_CACHE_STATS_FILE`, `LLAMA_PLE_GPU_CACHE_TIMING_FILE`, `LLAMA_PLE_GPU_PREFETCH_FILE`,
  `LLAMA_PLE_PREFETCH_DEBUG_FILE`.
- **User report (authoritative measurement; original wording kept)**: with a **1G** PLE cache, the
  **SSD→main-memory (mmap path)** hit rate reaches **90%+**, so a large amount of PLE-resident memory
  can be released. → marked: "**user recollection, protocol/denominator/sample count pending
  correspondence with the original record**". **The "1G" is not automatically refined into "1 GiB"**
  (the user's original unit/wording is preserved).
- **The layer and the denominator must be kept apart** (the two statistics outlets have different
  schemas; the layer can be determined from the file's column count):

  | Layer | Statistics outlet (source) | schema | Denominator of the hit counts |
  |---|---|---|---|
  | **Host row cache (CPU L2, the SSD/mmap→main-memory level)** | `LLAMA_PLE_CACHE_STATS_FILE` (`src/models/qwen4exp.cpp:184-197`) | 9 columns, including `total_prefetch_pages` | Hits/faults of pages (LRU units) in this layer |
  | **GPU L1 (VRAM local cache, another level)** | `LLAMA_PLE_GPU_CACHE_STATS_FILE` (`src/models/qwen4exp.cpp:604-617`) | 8 columns, **without** `total_prefetch_pages` | The fraction of this batch's rows already in GPU slots |

  **Host row-cache layer (the same level as "SSD→main memory 90%+")**:

  | File | Cache tier | `total_hits / total_misses` | Page hit rate | Note |
  |---|---|---|---|---|
  | `ablation-ple-only.csv` (09-03 12:02) | capacity 65552 pages | 14899 / 1101 | **93.1%** | `total_prefetch_pages=12774`, prefetch is running; **same level and same direction** as that report |
  | `ple-lru-1g-no-prefetch.csv` | capacity 16388 pages, **prefetch off** | 1505 / 13327 | **10.1%** | The capacity and the gather protocol differ too, so the causal contribution of the prefetch switch cannot be isolated |
  | `ple-cpu-l2-with-gpu-l1.csv` | early combination | 48 / 6384 | 0.75% | An early version that never took shape |

  **GPU L1 layer (different protocol; must not be used to prove the SSD→main-memory 90%+)**:
  `ple-gpu-overlap-stats.csv` (09-03 15:06, capacity 16388 pages, 14791/1049 = **93.4%**, 13832 pages
  resident), `ple-gpu-l1-rawpages-1g.csv` (3153/18912 = 16.7%),
  `ple-gpu-l1-rawpages-1g-async.csv` (2076/15840 = 13.1%), `ple-gpu-lookahead-1g.csv`.
  **GPU-local 90%+ and SSD/main-memory 90%+ are different layers with different denominators and
  different protocols; neither corroborates the other.**

  `ple-lru-256m-o1.csv` belongs, by schema (8 columns), to the GPU writer, but the file name suggests
  a CPU LRU ⇒ it is classified as "schema ownership clear, cache layer pending confirmation from the
  original run record" and **is not used as a conclusion for either layer**.

  ⇒ **The 1G (user's original wording) tier's SSD→main-memory 90%+ is retained as measured by the
  maintainer.** The 93.1% and 10.1% in the surviving host logs come from different capacities (65552
  versus 16388 pages) and different batches, and are not a single-variable prefetch A/B; they cannot
  be used to infer back the dependency conditions of that 1G experiment. The GPU L1's 93.4% only shows
  an observation of a different cache layer and cannot substitute for the host-level result.
- The repository root also has four early ablation CSVs (`abl-2g-ple2g.csv`, `abl-2g-ple512.csv`,
  `abl-512-ple2g.csv`, `abl-512-ple512.csv`, 2026-09-11) whose **contents are completely identical**:
  35 gathers, totaling 147/1293 (10.2%), `total_prefetch_pages=0`, and a last-line residency of 1293
  pages **far below** the smallest capacity of 8194 ⇒ short generation, the cache never filled up,
  **the capacity tiers cannot be distinguished**; they only prove "the statistics apparatus works and
  page-granularity counting is running".
- There are lazy-mode ablation records from the period when the MoE cache and PLE coexisted:
  `ablation-lazy-moe8-gpuple.log` (2026-09-10) used PLE-LRU 4096 MiB + PLE-GPU-L1 512 MiB + MoE cache
  8 GiB, with `Generation: 17.0 t/s` (`Prompt: 11.4 t/s`) — this is direct evidence that "PLE and the
  MoE cache coexist".
- **GPU L1 (the second layer of the same idea, `ple_gpu_row_cache`, `models.h:2334-2394`)**: the
  comment says it is an "Optional GPU L1 over raw quantized PLE pages"; rows are mapped through a
  `page_id → GPU-slot` table and the raw format is gathered/dequantized in place by CUDA; prefetch
  uses an **independent CUDA copy stream** to fill during the current batch's computation, and
  `gather()` fences before consuming (and there is `active_pages` protection against in-flight pages
  being overwritten by lookahead). Measured (handoff §6.29, 256k + vision, 256 token, full RAM):
  **the speedup is not large** — GPU L1 "is worth only +0.3 t/s (user's conclusion)" and fights for
  bandwidth; at the same cache size, **turning it off actually gives 17.0 → 17.9 t/s**.
  ⇒ It helps little for full-RAM residency; PLE's real use is trading 1–4 GB of memory for ~90% hits
  in lazy mode.

### 1A.3 When it was squeezed out of the budget by the MoE cache (disabled)

- The **time and scope** of the budget yielding (verified item by item; do not over-generalize):
  - During the MoE cache experiment period from 2026-09-03 to 09-10, PLE **was on**: most scripts set
    `LLAMA_PLE_CACHE_MIB=2048` + `LLAMA_PLE_GPU_CACHE_MIB=1024` (40 scripts in the repository still
    keep that setting), and the lazy-mode ablation even used 4096 MiB + 512 MiB (§1A.2's
    `ablation-lazy-moe8-gpuple.log`).
  - **The first commit in which `LLAMA_PLE_CACHE_MIB='0'` appears in a script = `d78c8bd42`**
    ("qwen4exp: hot-region expert cache — true-usage eviction, backfill, adaptive admission",
    2026-09-13 00:26 +0800), in **bisect/ablation scripts** (`bisect-garbage.ps1`, `bisect2..4.ps1`,
    `dsh-probe-bz8.ps1`) — that is, when the MoE hot-region/backfill/adaptive admission landed, PLE
    was first zeroed out in these control scripts.
  - Making PLE=0 a **recommended configuration** happened in the **256k + vision + TBQ4 KV** round
    (handoff §6.29), on the grounds that the GPU L1 gain is small and it fights for bandwidth, PLE is
    useless under full RAM, and 16 GB of VRAM must be shared with the MoE cache.
  - The accurate statement is therefore: **PLE and the MoE cache coexisted for a long time and PLE was
    later squeezed out by the MoE cache in configurations short of VRAM/budget**, not "PLE was
    replaced as soon as it shipped".
- The scale of the budget competition: on the MoE side, logs from the same period show
  `[MOE-CACHE] 144 weight tensors grouped into 48 layer bundles, 95123456 bytes per layer-slot
  round, 22 slots/layer, 1995.8 MiB physical cache (requested=2048 effective=2048)`
  (`abl-2g-ple.log`, the early 2 GiB tier) → later 6144 MiB → at 256k, `auto` gives only 37–40 slots;
  16 GB of VRAM is the common ceiling for PLE-GPU-L1 and the MoE cache.
- Configuration conclusion (handoff §6.29): both PLE settings are recommended to be 0; the auto branch
  of `moe_cache_apply_vram_limit` only adapts upward when **a positive number is not passed manually**
  (`requested < 0 → budget = limit − used − guard`), while a manual cap can only be lowered —
  **this is exactly why "PLE frees 1 GB but it does not go into the cache"**.
- **Environment variable name correction**: handoff §6.29 writes `LLAMA_MOE_PLE_CACHE_MIB` /
  `LLAMA_MOE_PLE_GPU_CACHE_MIB`; the code actually reads `LLAMA_PLE_CACHE_MIB`
  (`src/models/qwen4exp.cpp:43`) and `LLAMA_PLE_GPU_CACHE_MIB` (`:432`), and the repository scripts
  uniformly use the latter. This log follows the code.

### 1A.4 Its relation to the MoE cache, and the "PLE into the operator" design

- rebuild-spec §2.3 recognizes PLE as the **mechanism itself of the second stage (SSD tier)**:
  `copy_pages` is the "bridge to a lower level"; the module split is not "one big MoE operator" but
  "layered paged weight storage + prefetch engine (**shared** by PLE and the experts) + predictor +
  MoE operator", differing only in page size (63 KiB vs ~2 MB bundle), access pattern and predictor.
  That conclusion has already been fixed in rebuild-spec as "PLE is in scope, not optional", but
  **the implementation is still two separate codebases**.
- The relation between IQ4_NL repack and PLE needs a historical correction: that load's log has no
  `CPU_REPACK` allocation, but registration and buffer selection are done through the generic
  extra-buffer interface, so "no direct function call" cannot be asserted as "not registered".
  The existing repack only hooks `MUL_MAT` / `MUL_MAT_ID`, whereas PLE is a row gather / `GET_ROWS`;
  **re-enabling PLE does not automatically use repack**. Under CPU_ASYNC, `cpu=` also includes the
  join-wait regime, so 0.7/55.4 cannot be taken as all CPU computation or as a strict upper bound on
  the gain. See [chapter 03](03-weight-quantization-and-kernels.en.md).
- Remaining design items: the two hard constraints `logical_id → storage_slot` and "the prefetch copy
  source can change layers" are **already satisfied on the PLE side** (`copy_pages` + the page
  abstraction) and **not satisfied on the MoE expert cache side** (see D02); when the two are merged
  into a single engine, the PLE side is the template rather than a burden.

| Field | Content |
|---|---|
| Route ID | H14 (the first stage of the research) |
| Status/version | The cache body is **released** (upstream `4e1865e34`'s lazy readahead + fork `2f1a363c8`'s `ple_row_cache`/env/GPU L1; master-reachable); "PLE into the operator/merged with the expert cache" is **design only**; off by default (env unset = off) |
| Why it was attempted | The 26.8 GiB PLE table exceeds 16 GB of VRAM but fits in 128 GB of main memory; the SSD/mmap scenario uses a bounded cache to save main memory, rather than it being absolutely impossible to fit by capacity |
| Technical mechanism | Upstream: `TENSOR_READ_LAZY` + `MADV_RANDOM` + batched `MADV_WILLNEED`/`PrefetchVirtualMemory` hints; fork: a host-RAM LRU page cache of 64 KiB-aligned whole rows (preserving the original on-disk quantization), plus an optional raw-page GPU L1 (independent copy stream + fence) |
| Experiment conditions and evidence | User reports **1G (original wording)** → SSD→main memory 90%+; the surviving host logs additionally have 93.1% and 10.1%, but the capacities and protocols differ, so it is not a prefetch-switch A/B. The GPU L1's 93.4% does not corroborate it; under full RAM the GPU PLE gain is small — for the trade-off see §1A.3 |
| Observation and boundaries of the conclusion | **Effective at the SSD/mmap (lazy) operating point; small gain at the full RAM (`--no-mmap`) operating point.** It must not be written as "PLE failed"; nor may 1G/90% (the host row-cache layer) be mixed with the GPU L1 hit rate (another protocol/denominator), the MoE online hit rate, or the offline recall@10 (§0.5, §1A.2) |
| Reason kept/dropped | The implementation is kept (off by default); in configuration, the budget is yielded to the MoE cache in full RAM scenarios (scripts default to 0 from `d78c8bd42` onward) |
| Open questions/reopening conditions | Evaluate for SSD/lazy deployments, or when the D01 paging-engine design is restarted; first match the host PLE's capacity, page hits and prefetch protocol, and do not substitute GPU L1 data. Repack must separately satisfy the type, buffer and operator conditions, and cannot be inferred from "PLE is enabled" |

---

## 2. Per-route log

### 2.1 H01 host path baseline ledger

| Field | Content |
|---|---|
| Route ID | H01 |
| Status/version | Released; baseline build `build-ple-trace-mrs` (corresponding to `2f1a363c8` / its later instrumented version in the source workspace) |
| Why it was attempted | Any structural rework needs a credible per-segment ledger first, otherwise the warmup-graph pollution of the early reports gets mistaken for the bottleneck |
| Technical mechanism | Timers added at the four rendezvous points and the input loop of `ggml/src/ggml-backend.cpp`, writing a CSV per graph (rows `-1/-2/-3/-4`) |

**Experiment conditions and evidence** (same binary, same configuration, 31 decode graphs; `dsh-r8.txt` / `dsh-r9.txt`;
same-tier configuration as `run-cur-ref.ps1`, `CACHE_MIB=2048`, `SPLIT=1`, `MRS=1`, `devpart=0`):

```
host path (devpart OFF)  total 86.0 ms/token ≈ 11.3 t/s    ← true baseline
  cpu   27.7   CPU half computes the missed experts (real work)
  gpu    7.4   GPU enqueue
  pre   49.7
    ├ inputs 29.1
    │   ├ split_partition 23.2 (n=144, of which host-side loop 22.7)
    │   ├ flag_input       4.3 (n=108)
    │   └ generic          1.1 (n=51)
    └ drain  20.5            SMoE event wait ~17 + prefetch submit ~4

devpart ON               total 122.9 ms/token ≈ 8.1 t/s
  cpu 3.2 | gpu 7.2 | pre 111.4 → inputs 78.8, of which generic 78.2 (n=195, ≈400 µs each)
```

Same-period addendum (handoff §6.1, 2026-09-12): `split_partition` 21.3 ms/graph (of which only
3.3 ms was timed), `ids_wait ≈19 ms` (18.57 ms measured with `MRS=0`), `drain+flush 0.07 ms`,
`activate/on_ids/body = 0.00/0.00/1.1 ms`, `act_d2h 0.43 ms`,
SMoE event wait 15.3 ms/graph (533/35), hit rate 27.5% (12306/32364), prediction accuracy 73% but a
low delivery rate, `MRS=0` (LRU) 10.7 t/s (MRS net gain +0.4 t/s), `smoe-off` crash `0xC0000005`.

Regime improvement already landed (perf-plan §5): single-copy scheduling also creates events →
`decode 107.9 → 85.0 ms`, 190 `cudaStreamSynchronize` become `cudaStreamWaitEvent` (revert switch
`GGML_SCHED_NO_SINGLE_COPY_EVENTS=1`); decode-level timers (CSV `-2/-3/-4`).

Old conclusions overturned by perf-plan §7 (must not be re-trodden): `ids readback 46 ms` (0 times in
decode graphs, warmup pollution), `expert copy 3.5 ms/token` (0 times), `host_weights` branch 16 times
per graph (0 times), `D2H is the bottleneck` (each activation D2H ~10 µs, 48 times for a total of
509 µs), `keepalive views fixed the garbling` (builds with keepalive views are still wrong and
non-deterministic), `devpart OFF = 19–21 t/s` (measured 10–11.6 t/s, unsupported by any historical
output), `20.7 → 8.4 is a performance cliff` (the 20.7 batch was a garbled old build),
`CPU 2.7 ms means partitioning is broken` (normal at a high hit rate).

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The ledger holds only for the `CACHE_MIB=2048`, 27.5% hit-rate operating point; after that the hit rate rose to 70–90% and all the absolute values of every segment changed (see §2.6) |
| Reason kept/dropped | Kept: all subsequent routes use it as the reference; it also yields the structural conclusion of a "40 ms per-layer rendezvous" |
| Open questions | The timing regime is easily polluted by warmup (already excluded with `gid < 5`, but that relies on manual work); outside `LLAMA_MOE_CACHE_TIMING=1` there are no steady-state statistics |

### 2.2 H02 Per-layer router readback rendezvous

| Field | Content |
|---|---|
| Route ID | H02 |
| Status/version | Released; the mechanism is still there (`moe_prefetch_feasible`, `moe_cache_d2h_begin` etc., `ggml/src/ggml-backend.cpp:4015`, `:4760`) |
| Why it was attempted | The ledger showed `split_partition ≈21 ms`, early attributed to `insert_flush`/the host loop; a controlled experiment is needed to separate "waiting for the GPU" from "waiting for the copy" |
| Technical mechanism | The host must obtain this layer's `ffn_moe_topk`/`weighted` to do the GPU/CPU partitioning, hence `get_async` + `ggml_backend_synchronize` per layer; the MRS path must also read 256 gate scores back to the host, and the wait positions of the two readbacks can be moved onto each other |

**Experiment conditions and evidence** (handoff §6.4, 30 decode graphs, 82.5 ms/token):

```
MRS full-score readback ON :  prologue=18.04 ms  ids_wait= 1.31 ms   → split_partition ≈ 21 ms
MRS full-score readback OFF:  prologue= 0.83 ms  ids_wait=17.01 ms   → split_partition ≈ 21 ms
```

The wait merely moves from "the hidden blocking of pageable D2H" to an explicit
`ggml_backend_synchronize`, and the total is unchanged (SMoE's staging is pinned to begin with; its
15 ms wait is waiting for the GPU, not for the copy).

Retest with a pinned destination (handoff §6.12 ①, changing the destinations of the scheduler ids
readback and the MRS score readback to pinned, three places in `moe_cache_d2h_begin/end`):

```
before: d2h_enq=3785 µs  d2h_sync=1 µs     ids_wait=1.2 ms   total=48.7 ms
after:  d2h_enq=4 µs     d2h_sync=3787 µs  ids_wait=17.3 ms  total=48.3 ms
```

Reconfirmation at the new operating point (handoff §6.10 ④, 6 GB + backfill 8): `MRS_FULL=0` → 19.9 t/s,
no difference from the 19.9/20.0 baseline. The whole MRS set is itself about 0.4 ms and has been
confirmed to be dead weight (coverage 0.3%, see §2.3/H16), but removing it alone gives a net gain of ≈0.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The conclusion concerns only the pipeline structure "the host must obtain the router result layer by layer"; whether the *content* of MRS is useful is a separate question (§2.3, H16). Pinning does not change the total, it only makes the wait accounting honest |
| Reason kept/dropped | The pinned destination is kept; device-side partitioning is a structural option that has been tried, not a proven unique option, and no net gain has been established against the fixed host |
| Open questions | Under `SPLIT=1` this 17–19 ms rendezvous is still there (it persists after the silent error is fixed); `SPLIT=0` is correct but gives 3.4–6.5 t/s (§2.11) |

### 2.3 H03 SMoE-side graph readback event wait and lookahead distance

| Field | Content |
|---|---|
| Route ID | H03 |
| Status/version | Released (`moe_cache_smoe_process` `ggml/src/ggml-backend.cpp:5217`; lookahead defaults to 3) |
| Why it was attempted | In the ledger `drain ≈20.5 ms` (SMoE event wait ~17 ms) is on the critical path; prediction serves "the future" and in principle should not enter the critical path |
| Technical mechanism | The SMoE-side graph computes candidates for the next layer; the readback copy is enqueued after compute, so the event necessarily waits for that layer's GPU to finish. The tunable quantities are "how many layers of lead" (`LLAMA_MOE_SMOE_AHEAD`) and whether the readback blocks (`LLAMA_MOE_SMOE_NONBLOCK`) |

**Experiment conditions and evidence**:

- Historical joint variation of prefetch volume and elapsed time (handoff §6.6①, `NONBLOCK=1 AHEAD=2`):
  161 MB/tok → 40.2% hits, 70.0 ms, 14.3 t/s; 254 MB → 46.9%, 77.7 ms, 12.9; 464 MB → 58.2%, 94.1 ms, 10.6.
  The margin is about 0.073 ms/MB, whose reciprocal is about 13.8 GB/s, **not a measured PCIe bandwidth**.
  The longer `event_synchronize` wait is recorded; the specific PCIe/VRAM contention path was the
  explanation at the time and was not isolated by hardware profiling.
- Rank decay and active truncation (handoff §6.6 ②③): `cutoff 1 → 4.69 hits/MB`, `2 → 5.25`, `3 → 2.21`,
  `4 → 1.78`; `take = min(n_slots, n_topk, take_max)`, `take_max = n_used+2`; `n_used` is only assigned
  in the partition hook, so **on the first token / on layers that have not passed the hook `n_used==0`
  → the cutoff line silently disappears (bug, fixed)**; now `rank_cut = LLAMA_MOE_TAKE_MAX` (default 2).
  Truncating first and then skipping the already-resident is better than skipping first and then
  filling the budget.
- Deduplication hard numbers (handoff §6.6 ④): candidates 2852 = `dup_resident 1496` (52.5% costing no
  bytes) + transfers 1356; `dup_list=dup_admit=dup_pending=0`; `readmit=86` (6.3%).
- Capacity interaction (§6.6 ⑤): at 30 token, 2048 and 6144 MiB show no difference (6 GB ≈67 slots/layer,
  only ~2 admitted per layer per token, `readmit=0`); at 150 token, 2048 is optimal at cutoff 1 (14.6),
  and 6144 plateaus at 2–4 (14.8–14.9). VRAM peak 2048 → 9132 MiB, 6144 → 13278 MiB (limit 15360,
  guard 1024).
- Lookahead curve (handoff §6.30 addendum, 8k/q8_0/64 slots, `SPLIT=1` and carrying the silent error
  at the time):

  | ahead | r1 | r2 | r3 | hit rate | gen t/s |
  |---|---|---|---|---|---|
  | 1 | 58.5% | 49.7 | 43.6 | 68.2% | 17.2 |
  | 2 (old default) | 84.7% | 76.5 | 70.2 | 77.4% | 18.1 |
  | 3 | 80.9% | 67.9 | 60.7 | 85.4% | 19.2 |
  | 4 | 78.6% | 68.7 | 59.5 | 85.7% | 19.9 |
  | 6 | 56.1% | 46.2 | 39.7 | 80.6% | 18.5 |

  The optimum is 3–4 layers: the hit rate is not determined by the r1 term alone — ahead=2 has the
  highest r1 yet the lowest hit rate, while at 3–4 layers r1 drops slightly but the lead is enough for
  the prefetch to actually complete.
- Retest after the fix (handoff §6.32 ②, 8k/q8_0, `auto`=97 slots): `SPLIT=1+AHEAD=3` → 90.9% hits,
  **20.8 t/s** (verified correct); `AHEAD=2` → 83.5%, 18.4; `AHEAD=4` → 87.8%, 20.2. Relative to
  9.2 t/s with the cache off = +126%.
- The `ahead=1` r1 anomaly closed out (handoff §6.33): after making the readback synchronous, ahead=1's
  r1 goes from 61.2% → **93.1%**, while ahead=2 is essentially unchanged (86.0 → 84.7) ⇒ the
  non-blocking readback lets the prediction be consumed one beat late; the ahead=1 window expires by
  exactly one beat. Trade-off (2 runs back to back each): the default `NONBLOCK=1 + AHEAD=3` still has
  the best mean at 20.1 t/s; `NONBLOCK=0 + AHEAD=1` is comparable in speed (19.1) but has r1 93.1%,
  left for scenarios that value long-term placement quality more.
- Offline control (`smoe-nk-degradation-plan.md` §7, 8 prompts, the new `ffn_moe_input` whitelist):
  oracle k=1..4 all 100%; full 68.53/59.94/55.70/53.77%; `input_only` 67.90/59.34/55.22/53.44%
  ⇒ the predictor barely depends on the routing/shared-expert outputs (0.63 pt apart), so the firing
  point could have been moved earlier to the end of attention (see D04).
  **Regime reminder**: the 68.53% here is the "offline single-step teacher-forced recall@10", and
  whether it uses the same protocol as the user-reported "teacher test hit rate 99%" has not been
  confirmed, so the two cannot substitute for each other; the online hit is in turn affected by
  residency and delivery (§0.5).

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The §6.30 addendum was measured under the wrong SPLIT=1, so both the trend and the absolute values may be distorted; the post-fix §6.32/§6.33 are independent evidence. The "non-blocking one beat late" explanation for ahead=1 rests on the synchronous control |
| Reason kept/dropped | Kept: increasing the lead is the realistic way to eliminate the 15 ms wait; D04 (firing earlier) is the extreme form of it that was never taken to the end |
| Open questions | The `SMOE_AUTO`-type online search is off by default; the side graph still synchronizes per layer; the `AHEAD_AUTO` objective must use the window's incremental hit rate |

### 2.4 H04 `insert_flush` per-layer hard drain (hypothesis void)

| Field | Content |
|---|---|
| Route ID | H04 |
| Status/version | Released; hypothesis void (the "~23 ms" of perf-plan §P2.1 did not hold) |
| Why it was attempted | The code comment says `moe_insert_flush` blocks deliberately (to keep side-stream copies from crossing the CUDA graph capture boundary), one hard drain per layer, which looks very suspicious on paper |
| Technical mechanism | `s.insert_cv.wait(...)` waits for the insert worker queue to empty + in-flight to hit zero; changing it to "once per graph" would preserve correctness |

**Evidence**: `drain+flush` measures **0.07 ms/graph (0.3%)** (handoff §6.1). perf-plan §1 also notes
that of the 144 calls to `split_partition` only 48 do real work and the rest return early, and that
the 23 ms is concentrated in the first four lines, but after the §P0.1 timers landed it was attributed
to the router rendezvous (§2.2), not to the flush.

| Field | Content |
|---|---|
| Reason kept/dropped | Dropped: measured non-bottleneck. The to-do table in handoff §6.5 marks this item as "measured 0.07 ms, void" |
| Open questions | None (kept as an excluded item, to avoid suspecting it again) |

### 2.5 H05 `split_partition`'s "host CPU loop 22.7 ms" (hypothesis void)

| Field | Content |
|---|---|
| Route ID | H05 |
| Status/version | Released; hypothesis void |
| Why it was attempted | perf-plan §1 recorded the 23 ms of `split_partition` as "inside it a host CPU loop of 22,724 µs", suggesting the host CPU is the bottleneck and could be solved with a faster loop/parallelism |
| Technical mechanism | The partition hook executes on the host, but the loop itself is extremely cheap; the real cost is the wait in front of it |

**Evidence**: row 4 of the to-do table in handoff §6.5 states outright "`split_partition`'s host CPU loop
→ measured as non-host-CPU time, void"; §6.10 ⑤ gives the breakdown:
`prologue 17.9 ms (of which mrs_queue 17.0 = waiting for the GPU to produce this layer's router)`
+ `ids_wait 1.4 + partition 1.2 + act_d2h 0.5`. Both independent experiments, MRS and pinned, only move
the wait (§2.2).

| Field | Content |
|---|---|
| Reason kept/dropped | Drop the attribution; keep the fact that "partitioning itself is cheap", to reject changes of the "optimize the partition loop" kind |
| Open questions | None |

### 2.6 H06 Asynchronizing the CPU half (worker)

| Field | Content |
|---|---|
| Route ID | H06 |
| Status/version | Released, `LLAMA_MOE_CPU_ASYNC` defaults to 1 |
| Why it was attempted | The CPU half (27.7 ms) is the largest single item in the ledger; moving it off the scheduler thread and overlapping it with the GPU is the obvious direction |
| Technical mechanism | During partitioning the MoE CPU half is dispatched to its own worker (`moe_cpu_half_submit`), and the CPU split joins; the scheduler thread only joins |

**Experiment conditions and evidence (in chronological order; note the differing operating points)**:

| Time/section | Operating point | Result | Explanation |
|---|---|---|---|
| handoff §6.2 (2026-09-12) | `CACHE_MIB=2048`, 27.5% hits | `CPU_ASYNC=1` → text identical word for word, **11.0 vs 11.1 t/s (neutral)** | The CPU half already overlaps GPU execution, it is not on the critical path |
| handoff §6.6 ① | prefetch-volume sweep | workers 1→3: low volume 13.3→13.6; high volume 9.9→10.5 (+6%) | Adding workers does not rescue PCIe contention |
| handoff §6.7 unresolved | prefetch on | `LLAMA_MOE_INSERT_WORKERS` **spins idly** under `PREFETCH=1` (every worker spawn point requires `!prefetch`) | The earlier conclusion "workers are not the bottleneck" is void; prefetch copies are submitted inline within the host thread |
| handoff §6.10 ④ | 6 GB + backfill 8 | `CPU_ASYNC=1` → **20.5** (baseline 19.9/20.0); host thread cpu 12.4→5.7 ms, join costs an extra 2.7 ms, net saving 1.4 ms | The operating point changed: the hit rate and admission made the scheduler thread the bottleneck |
| handoff §6.11 ① | same, switching it to default-on | **21.5 / 21.4** (previously 19.9–20.0); total 51.0→48.7 ms; cpu 12.4→6.2; compute 17.2→13.1; gpu_queue 5.3→6.9 | User approved making it permanent |
| handoff §6.24 | devpart 400 token | worker dispatched at the staging point (`moe_cpu_half_submit_layer_staged`), `dispatched=19153` took effect | No speedup observed, because the resident set was empty and the CPU work itself was 3× larger |
| handoff §6.31 | during the split silent-error period | `CPU_ASYNC=0` **is not a usable fallback**: it produces no output and the cache spins idly (`hits=0`) | Can only be kept as an excluded item |

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | **Do not write this as a general law**: the same switch is neutral at the 27.5% hit-rate operating point and saves a net 1.4 ms/graph at the 6 GB + backfill 8 operating point. The criterion is "whether the CPU half still overlaps the GPU sufficiently and whether the scheduler thread has become the bottleneck", which has nothing to do with the switch itself |
| Reason kept/dropped | Kept and on by default; it is also the prerequisite infrastructure for devpart's asynchronous CPU half (§2.9) |
| Open questions | The dead branch of the insert worker spinning idly under `PREFETCH=1` still exists (neither cleaned up nor enabled); the combination `CPU_ASYNC=0` with `SPLIT=1` is unusable (hits=0) |

### 2.7 H07 Weight pinning (`pin_weights`) and the memory gate

| Field | Content |
|---|---|
| Route ID | H07 |
| Status/version | Released (`moe_cache_pin_weights` `ggml/src/ggml-backend.cpp:3979`, `LLAMA_MOE_PIN_WEIGHTS` on by default) |
| Why it was attempted | The expert weights must be resident in host memory and usable for DMA; the enqueue blocking of pageable memory is a measurable cost (§2.2's `d2h_enq 3.7 ms`) |
| Technical mechanism | `cudaHostRegister` locks the entire set of expert weights (log: `pinned 1 weight buffers (72.6 GiB)`), making the copies truly asynchronous |

**Experiment conditions and evidence**:

- Memory topology and risk (handoff §5): pinning and mmap are **mutually exclusive**; mixing the two
  holds both 76 GB of file pages and 72.6 GB of locked memory → blowing past 128 GB. All current
  scripts use `--no-mmap` + default pinning.
- Environment constraints: never run two inferences at the same time; do not use `-ngl 0` without
  `--cpu-moe`.
- Whole-machine incident (handoff §6.10 ②): a background test loop overlapped with another batch of
  instances → two `llama-cli` processes each pinning 72.6 GiB → 2×72.6 > 131 GiB of physical memory →
  commit charge exhausted, machine frozen. **Unrelated to the model/VRAM/this optimization**, it was an
  operational mistake.
- Recurrence prevention (same section ③, tested): the `tools-run.py` single-instance lock
  (`.tools-run.lock` + PID liveness check, stale locks reclaimed automatically) + an idle-memory
  precheck (`--min-free-mib` defaults to 90000, `--wait-mem` defaults to 120 s; if insufficient it
  waits and then errors out).
- The upside of the pinned ring is only a few ms (handoff §6.6 ①); SMoE staging is pinned to begin with
  (§6.4).
- Changing the scheduler ids / MRS score readback destinations to pinned: enqueue 3785 µs → 4 µs, but
  the wait moves to synchronize (§2.2), kept as neutral.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | "Pinning can rescue the rendezvous" is explicitly rejected (§6.4/§6.12); pinning solves whether DMA can be truly asynchronous, not the pipeline structure |
| Reason kept/dropped | Kept (on by default); the accompanying memory gate and serialization discipline are among this project's most important protections |
| Open questions | The single-instance lock exists only inside `tools-run.py`; paths that run `llama-cli` directly have no protection (it relies on manual discipline) |

### 2.8 H08 split segment cost and merging

| Field | Content |
|---|---|
| Route ID | H08 |
| Status/version | Reverted after the experiment (not adopted); the status quo is `SPLIT=1` with CUDA graph effective |
| Why it was attempted | The `compute` segment is 13.1 ms/graph; if the 3 weight kinds per layer could be merged into 1 split, about 2/3 could be saved |
| Technical mechanism | With `--cpu-moe` the expert weights are resident on the CPU (WEIGHTS + backend incompatible) → scheduler pass 5's rule cuts once per `MUL_MAT_ID` (`ggml_backend_sched_split_graph`); merging means removing that rule or rewriting the input sources |

**Experiment conditions and evidence**:

- Cost structure (handoff §6.11 ②/§6.12 ②): the hook runs **142.6 times per graph ≈ 48 layers × 3
  weight kinds** (real=47.5 truly partitioning, early=95.1 returning early at 0.00 ms);
  `ggml_backend_graph_compute_async` 13.1 ms ≈ 92 µs each. The CUDA graph is already effective: with
  `GGML_CUDA_DISABLE_GRAPHS=1`, total 81.8 ms and compute 62.7 ms.
- Structure dump (handoff §6.13, using `LLAMA_MOE_DUMP_SPLITS=1`, previously "written but never run"):
  per decode graph **242 splits / 8978 nodes / 1601 leafs** (prefill 145; the two decode graphs are
  completely identical → the graph does not grow); 5 splits per layer:
  `attn block (119 nodes) + gate(1) + up(2) + down(21+VIEWs) + CPU MOE_CPU(1) + router block (136~218)`.
- Both attempts failed:
  1. Patching `src[0]`→cache view before graph construction: `n_splits` is still 242 (the hook hits the
     wrong tensor) and it introduces a fail-fast crash; to succeed it must be done during llama graph
     construction and take over the fallback copies for non-direct layers → risk no lower than devpart.
  2. Turning off that scheduler rule (formally low-risk): **VRAM peak 13200 → 15868 MiB (> 15360
     limit) → OOM crash**; the reason: after merging, the three weight inputs must be resident
     simultaneously, adding ~2× expert bundles per layer.
     **This is exactly why the rule exists: trade splitting for weight-staging memory.**
- State after the revert: `21.6 t/s, VRAM 13198 MiB, correct text`.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | "The 13.1 ms is free money" is rejected; the conclusion holds only under the VRAM budget of the "cache capacity = 6 GB tier" — if the cache were deliberately shrunk, merging might become feasible again |
| Reason kept/dropped | Decided not to do it in this round. Merging after shrinking the cache is a candidate trade-off that was not measured; the 13.1 ms cannot be listed in full as recoverable launch tax |
| Open questions | That trade was never measured; the `LLAMA_MOE_DUMP_SPLITS` diagnostic is kept |

### 2.9 H09 devpart (device-side partitioning) + host leaf — the complete correction chain

| Field | Content |
|---|---|
| Route ID | H09 |
| Status/version | Released, `LLAMA_MOE_DEVPART` off by default. The data defect located at the time has been fixed; the user stopping work on it does not mean a comprehensive numerical regression was completed |
| Why it was attempted | The host waits layer by layer at the router rendezvous, and device-side partitioning was the structural change intended at the time; the "only solution" framing is an old design judgment, not a proof arrived at after excluding other implementations |
| Technical mechanism | A new device-side partition kernel (`ggml/src/ggml-cuda/moe-partition.cu`, 81 lines, the kernel at :8/:41, the residency-table read at :19), with the GPU directly producing `part_ids/part_wgt/table`; the CPU half instead consumes a host leaf, asynchronously backfilled by the MoE side using pinned staging + per-layer events |

**Correction chain (chronological, with every failure and fix)**:

| # | Section/date | Status and numbers | Failure/fix content |
|---|---|---|---|
| 1 | handoff §3 (devpart ON, device view) | 8.1 t/s (122.9 ms/token) | Correct text but slow: `generic` 78.2 ms = 195 cross-backend copies × ~400 µs; perf-plan §1 judged that "the books were done backwards" |
| 2 | §6.3/§6.4 plan | — | Three steps: device partitioning (saves 19 ms) → change the CPU half's input to its own asynchronous D2H + worker + boundary join → then fold the SMoE wait in |
| 3 | §6.5 (host-leaf experiment) | Reported **19–23 t/s** (`CPU_ASYNC=2` filling 17–19; mode 1 worker 14.9) → **retracted** | Proved the 78 ms did come from those 195 copies; but the output broke from the 2nd token (`The////`). Ruled out invalid ids filling, the CPU half not computing, the paltry keepalive view; did not get to the bottom of the semantics of `cur` being taken from `src[1]`, the `wgt_cpu` offset, the CUDA graph capture/replay interaction, or whether `moe_cpu_half` hits layer by layer. The repository was restored to the two usable states host 11.2 / devpart 8.1 |
| 4 | §6.17 (re-inspection) | Reported 29.7 t/s, steady state **32.0 t/s** (vs host 20.8, +54%) → **retracted** | The breakage was unrelated to the hot-region mechanism (still broke with `HOT_BACKFILL=0 PREFETCH=0`, `The划分为其职 Dess Dess…`); `LLAMA_MOE_DUMP_CPUHALF` showed the CPU half's ids all 0 (the host reference values were normal) → localized to "the device-partition output has not yet been written/visible when the CPU half consumes it" |
| 5 | §6.18 (root cause + fix step 1) | Symptom moved forward | Root cause: the devpart branch's `ids_cpu/wgt_cpu` are **a device view and have no `ggml_set_input`** → they can only rely on a scheduler cross-backend copy, which does not wait for the device to write → the CPU half always reads 0 (the device kernel itself is correct: first k = GPU slots / last k = CPU expert ids, `-1` marks). Fix 1: switch to a host leaf + the second half of pinned asynchronous D2H (≈160 B per layer) → the breakage form became `viewer viewer…` (ids/weights in place, the remaining inputs wrong). Fix 2 (adding a `cur_cpu` leaf) **crashed during load** (VRAM only 6.9 GB, not OOM → assertion/shape assumption). Performance concern: `ggml_backend_synchronize` on every tensor (3 times per layer, waiting on this layer's GPU half), must be changed to once per graph |
| 6 | §6.19 (output coherence fixed) | 26.0 t/s (host 19.6) | Graph construction uses host leaves (`ids_cpu/wgt_cpu/cur_cpu`), `ids_gpu/wgt_gpu` remain device views; filling reads the activation from "this layer's `MUL_MAT_ID` `src[1]` (a device tensor)". Pitfall: mistakenly reading the CPU op's own `src[3]` (already a host leaf) → `get_async` hits the CUDA assertion `ggml-cuda.cu:2652 unsupported buffer type` (already written into a code comment). The first token matches the host, divergence at about the 3rd token = difference in floating-point summation order (inevitable under greedy), not a data error |
| 7 | §6.20 (speed path opened up) | Reported **28.2 t/s** (host 16.8, +70%) → **retracted** | The trio: the partition op gets its own split (pass 5 closing it, `LLAMA_MOE_PART_SPLIT=1` on by default); `moe_cache_devpart_readback` immediately after the split containing the partition op issues 3 groups of D2H into per-layer pinned staging and records a per-layer event; the CPU half waits only on that layer's event (not the whole stream). Under devpart, `cpu_half_async=0` is forced (the worker would be dispatched before the leaf is published). The text is still `The/////` |
| 8 | §6.21 (narrowed to a single point) | Diagnosis | Under `LLAMA_MOE_DUMP_CH=1` the host row prints but **the devpart row never prints** ⇒ `rb.ready` is permanently false ⇒ `moe_cache_devpart_readback` exits early at some `continue` ⇒ the leaf keeps its zero values. Four bail points are listed |
| 9 | §6.22 (pipeline fixed) | 28.2 t/s (still wrong) | Fallback path added: the scheduler may schedule the CPU half before the partition split (the leaf is a host tensor → no dependency edge) → when staged is not ready, call `tensor_get` directly (preserving correctness). **The real cause of "the readback is never submitted"**: the layer number was looked up via `manifest.n_layers` (0 when no manifest is loaded) → changed to look up `s.layers`. The diagnostic chain proved the pipeline works and the defect is "what the kernel reads from the residency table ≠ the host mirror" (the fallback reads `ids: -1 -1 … wgt: 0.000 cur: 0.0000`, whereas the host reference is the real ids/weights) |
| 10 | §6.23 (three real defects fixed) | After the corrections: 60 token host 14.1 / devpart **15.0**, text **completely identical**; 240 token + SMoE trio: host 15.6 / devpart **13.6**, text completely identical | **Defect 1, the residency table is uninitialized**: `ffn_moe_part_table` in the graph memory is not -1, so the kernel reads it as "every expert is in slot 0" ⇒ everything judged resident ⇒ the CPU half only gets -1/0. Fix: at the start of the graph publish all `ffn_moe_part_table*` as all -1 (once), after which they are still overwritten by the per-layer flush. **Defect 2, the activation is taken from the wrong source**: under devpart the CPU half's own `cur_cpu` is a host leaf, and the real activation is in the GPU half's `MUL_MAT_ID`; looking it up by name is wrong (the `ffn_moe_topk-<real layer>` and the partition-family tensor numbering conventions differ). Fix: use **pointer identity** (the GPU half's `src[2] == moe_graph_find("ffn_moe_ids_gpu", layer)`'s `MUL_MAT_ID`, take its `src[1]`, i.e. `moe_cpu_activation()`). **Defect 3, staging hooks the wrong operator**: `pids` and `pwgt` get split by the scheduler into different splits, and reading `pwgt` in the split containing `MOE_PARTITION_IDS` reads garbage from before the kernel produced it (prefill went through the fallback so it was not exposed; decode was entirely wrong). Fix: move the readback onto `MOE_PARTITION_WGT`, take ids from its `src[2]` (pointer); staging slots are invalidated per graph (`rb.graph`) + a blocking fallback when not ready |
| 11 | §6.23 performance conclusion | **Once correct it is no faster than the host** (14–15 vs 14–16, level within noise); the earlier 28.2 **cannot be trusted** | `LLAMA_MOE_DEVPART` stays off by default; this round's temporary probes (`CH-PUB`/`CH-W`/`CH-ACT`/`CH-TBL`/`CH-RB`/`CH-CPU`) were deleted, keeping the `LLAMA_MOE_DUMP_CH` dual-path comparison dump |
| 12 | §6.24 (400 token steady state) | devpart **16.8 t/s** (total 62.3 ms; cpu 19.0 vs host 6.7); in the same section host 20.3 t/s | The gain side is real: `ids_wait=0.0`, `partition=0`, `split_partition n=0` (the 17.9 ms rendezvous is indeed eliminated), but it is eaten back by `cpu 19.0 ms`. **Root cause**: the devpart resident set is permanently empty (`hits=87 / misses=300219`, versus 404757/169833 = 70.4% on host) — the device partition kernel only **reads** the residency table, while the whole insertion/eviction/prefetch-insert sequence lives in the host partition hook, and devpart skips that hook entirely to save 17.9 ms ⇒ the table is always all -1 ⇒ splitting freezes ⇒ GPU utilization becomes a flat line |
| 13 | §6.24 attempts (unresolved, code kept, devpart-only) | — | ① dispatch the CPU half to a worker from the staging point (`dispatched=19153` took effect, but the CPU work itself is 3× larger); ② replay the host-side hit/miss counts and `moe_cache_warm_miss` (`moe_cache_devpart_account`); ③ additionally call `moe_insert_drain/flush` at the end of the graph; ④ stage the real routing back and feed it to `moe_cache_on_ids` |
| 14 | §6.25 ceiling measurement | `CEILING_DIV=10` saves only 6.0 ms/graph (+14%), while `ids_wait 17.9 ms` does not budge at all | Conclusion: the per-layer router rendezvous is a **GPU-side pipeline latency**, unrelated to the CPU half's workload; devpart merely re-records it under a different segment ⇒ the upside of "predicting one beat ahead" is part of that ≈ 6 ms (+8~12%), not worth touching the kernel for |
| 15 | §6.31/§6.32 | During the safety-lock period devpart was still allowed ("another path already verified correct"); the state is unchanged after the split fix | devpart stays off by default and is no longer pursued (user's decision) |

**Status note**: the device-side partition kernel and the whole devpart readback/fallback logic are
master-reachable (`moe-partition.cu`, `moe_cache_devpart_readback` `:4644`,
`moe_cpu_half_submit_layer_staged` `:5552`, the call site `:7051`, the only occurrence of
`LLAMA_MOE_PART_SPLIT`); `LLAMA_MOE_DEVPART` defaults to 0.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | 19–23, 29.7, 32.0 and 28.2 come from a wrong implementation; 16.8 is the historical record after the corrections, but its host 20.3 control was later retracted as well. So the direction of the net gain against the fixed host still lacks same-regime evidence |
| Reason kept/dropped | The code is kept, off by default; passing the text probe at the time is not a guarantee of comprehensive correctness. Residency and prefetch decisions still depend on the host side, which is the structural obstacle to pushing further |
| Open questions/reopening conditions | Reopening conditions: move the admission/prefetch decisions onto a device-drivable path as well (see H10), or prove under the same regime that devpart still does not fall behind after the host fix |

### 2.10 H10 devpart's compatibility with hot-region signals / the adaptive system

| Field | Content |
|---|---|
| Route ID | H10 |
| Status/version | **Design only** (not implemented); proposed in §6.12 ③, with empirical counter-evidence given in §6.24 |
| Why it was attempted | devpart moves partitioning to the device side → the host no longer reads back `part.ids` → the input signal for the entire hot-region mechanism disappears |
| Technical mechanism | The dependents are listed one by one: the true-frequency eviction score (§6.7), the backfill ordering, and the per-rank accuracy/yield statistics (§6.8/§6.9) all depend on `part.ids`. The compatible approach: compute the usage histogram on the **device** (scatter-add the router ids into a per-layer count buffer, or reuse the top-k kernel), then **read it back once per token in a batch** (48 layers × 256 × 4 B ≈ 49 KB ≈ negligible) |

**Evidence**: the judgment of §6.12 ③ — "otherwise you are trading +20pp of hit rate for 17 ms, which
does not pay off"; the empirical counter-evidence of §6.24 — after devpart skips the host hook the
residency table is permanently empty, with a hit rate of 87/300219, which directly shows that this
coupling is real rather than a theoretical worry.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | This is the precondition for devpart to "actually use the cache"; the current implementation only achieves half of it, "read back the real routing and feed it to `moe_cache_on_ids`" (§6.24 attempt ④), and the admission decision is still not on the device side |
| Reason kept/dropped | Not implemented: it is independent structural work and the cost/benefit ratio is not good enough to bet on |
| Reopening conditions | Do it merged with D01 (a self-contained operator), or first do the minimal closed loop "device-side histogram + one readback per token" and prove the hit rate does not drop |

### 2.11 H11 host leaf old-routing silent error (`SPLIT=1`) — retraction of the speed evidence

| Field | Content |
|---|---|
| Route ID | H11 |
| Status/version | Released (the defect is fixed, the safety lock is lifted); the correctness-methodology details are written up in chapter 05, and this section records only the **retraction of the speed evidence** and the fix highlights |
| Why it was attempted | While accepting the "cache gain", the output length for the same prompt was found to differ by an order of magnitude across the three configuration tiers |
| Technical mechanism | In the host `SPLIT` path, the GPU half's `ids_gpu/wgt_gpu` are **inputs of this split** (a `ggml_set_input` host leaf), and their values are written by the partition hook **in the middle of the input loop**; the scheduler may already have scheduled the copies of these two leaves first ⇒ the MoE GEMM gets **the previous layer's routing** |

**Experiment conditions and evidence**:

- Reproduction (handoff §6.30, single prompt, greedy, `-n 160`, 8k/q8_0, host mode):
  cache OFF → 751 characters, completely correct, 9.2 t/s; cache ON + `SPLIT=0` → 751 characters
  correct, 15.9 t/s; cache ON + `SPLIT=1`(direct) → **27-character degeneration, 22.0 t/s**;
  `SPLIT=1`(gather) → 27 characters.
  Reproduction essentials: `-p 'Write a short factual paragraph about the Eiffel Tower.' -n 160 --temp 0`,
  and **`--ignore-eos` must not be used**, otherwise the degenerate tail is masked — which invalidates
  all previous "identical text" verifications.
- Bisection (same section): `DIRECT_READ=0` still wrong (unrelated to the direct slot-view);
  `PREDICT_SMOE=0` still wrong; `HOT_BACKFILL=0` still wrong; `SPLIT=0` correct; `CPU_ASYNC=0` produces
  nothing and `hits=0`.
- Nature of the defect: a classic race (the publish/consume timing between the host leaf the CPU half
  consumes and "this layer's/this token's partition result"), or, in direct mode when the slot-view
  patch is not in effect, indexing the expert dimension of the raw weight tensor ⇒ reading the wrong
  expert at a legal address ⇒ no crash, silently wrong.
- Deployability matrix (handoff §6.31): cache off 9.2 (correct); `SPLIT=1` 22.0 (wrong) silently
  wrong; `SPLIT=0` **3.4–6.5 t/s** (correct) but slower than not using the cache at all (the experts
  are still computed on the CPU and the weights were moved to VRAM ⇒ every expert read becomes a PCIe
  copy); `SPLIT=1` + 0 slots (64 MiB) no output (wrong). ⇒ **the entire gain of the cache depends on
  split**; 22 t/s is a fake speed, and 15.9 is one fast sample in the long tail (other runs of the same
  configuration gave 6.5/4.0/3.7/3.4, not reproducible).
- Safety lock (delivered that round, later lifted): `src/llama-graph.cpp` makes `LLAMA_MOE_SPLIT=1`
  refuse to take effect and prints the reason (debugging requires an explicit `LLAMA_MOE_SPLIT=2`);
  `tools-run.py`/`run-cur-ref.ps1` default to `SPLIT=0`; the regression case = the prompt above
  (without `--ignore-eos`), 751 characters when correct and 27 when wrong, three lengths 27/652/751 ⇒
  a partial race.
- Fix (handoff §6.32 ①): after writing the two leaves the hook **immediately resends the copy**
  (`tensor_copy(...)` + `ggml_backend_tensor_copy(ids_gpu_t, dst_ids)`, likewise for wgt, a few hundred
  bytes). Verification: 3/3 reproduced 27 characters before the fix → **4/4 passed** after the fix
  (giving the same correct answer as the `SPLIT=0` reference); the safety lock was lifted and the
  scripts went back to `SPLIT=1`.
- Historical speed after the fix (handoff §6.32②, 8k/q8_0, `auto`=97 slots): cache off 9.2;
  `AHEAD=3` gives 20.8 t/s with 90.9% hits; `AHEAD=2` gives 18.4 and `AHEAD=4` gives 20.2.
  `20.8/9.2−1≈126%` is the arithmetic of the numbers on that page, lacking a fully consistent
  controlled A/B, and is not a new acceptance for the current 6 GiB / 400 requested token tier.

**Retraction list (compare chapter 05)**: the pre-fix SPLIT=1 high values and the sealed conclusion
are retracted; under the same routing error the ahead curve's trend and absolute values may both be
distorted. Numbers with the same name after the fix must be judged from their own logs, and "20.3"
alone cannot tell you which round it belongs to. For the 400 requested token results of the latest
local development build see chapters 02 and 05; this round did not rebuild the release master, nor
does it pass these later results off as release-build acceptance. SPLIT=0 once gave 15.9, and
afterwards only 3.4–6.5, so it cannot be treated as a stable performance fallback either.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | This section deals only with "whether the speed evidence holds"; the full account of the defect's reproduction, bisection, race mechanism and regression case is in chapter 05 |
| Reason kept/dropped | Keep the fix (`SPLIT=1` is currently the only path with a gain); keep the lesson of the safety lock (an environment variable can refuse to take effect, which is better than silently going wrong) |
| Open questions | Similar risks remain: variable-length batches/multiple graphs (speculative verification) with the persisted slot-view patch (§2.15); with 0 slots (e.g. `CACHE_MIB=64`) split produces nothing (it should degrade to all-CPU) |

### 2.12 H12 prefill read cache (D2D staging) — retracted

| Field | Content |
|---|---|
| Route ID | H12 |
| Status/version | **Retracted** (on master both `LLAMA_MOE_PREFILL_CACHED` and `moe_cache_prefill_d2d` occur 0 times, verified) |
| Why it was attempted | prefill copies the "union of experts used" from the host H2D into device staging; the part already resident in the cache could instead go D2D (the cache is read-only, never written, satisfying the "no transfer during prefill computation" constraint), and at a 25% residency rate that was estimated to save ~4.8 GB ≈ 370 ms per prompt |
| Technical mechanism | In `moe_copy_experts_grouped`, segment by residency: resident → D2D, non-resident → the original H2D, keeping "contiguous experts copied in one go" |

**Experiment conditions and evidence** (handoff §6.16, two interactive rounds as self-control, same
prompt, round 1 prefill cold and round 2 warm):

```
turn 1: Prompt: 37.3 t/s | Generation: ...
turn 2: Prompt: 37.3 t/s | Generation: ...     ← completely identical, no measurable gain
```

Reason: prefill's staging copy uses `ggml_backend_tensor_set_async` (asynchronous) → the H2D **already
overlaps with GPU computation**, so switching to D2D saves no time. This also explains the "prefill is
neutral" of §2.13: prefill is GPU-compute-bound and the weight transfers are hidden.
→ Reverted (no pointless complexity left behind). **Two tool capabilities kept along the way**:
interactive multi-turn testing (`hub start` launching `llama-cli` without `-p`, reading stdin under a
PTY, printing `Prompt: X t/s | Generation: Y t/s` each round; note that this build has no `-i`/`-cnv`);
and the `prefill/decode tendency` diagnostic line, now permanent.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The conclusion is "the prefill side needs no cache awareness", not "the cache is harmful to prefill" |
| Reason kept/dropped | Drop the implementation; keep the conclusion (to avoid trying it again) |
| Reopening conditions | Re-evaluate if prefill becomes bandwidth-bound (for example if weights no longer go through asynchronous H2D, or a change on the Q side breaks the overlap) |

### 2.13 H13 prefill/decode phase separation (hot set)

| Field | Content |
|---|---|
| Route ID | H13 |
| Status/version | Released (`LLAMA_MOE_PREFILL_WEIGHT`, prefill routing enters the statistics) |
| Why it was attempted | The prompt length determines the cold-start cost; the access concentration of the two phases may be fundamentally different, and treating both with the same policy may be wrong |
| Technical mechanism | New diagnostics: the prefill-side counts are taken where `used_ids` is parsed (**the partition hook only runs in decode**, so prefill routing was previously entirely invisible); the two sides are counted separately with Q8 fixed-point counting (decode row = 256, prefill row = 256×`PREFILL_WEIGHT`), and the decay/eviction/backfill ordering scale is unchanged |

**Experiment conditions and evidence** (handoff §6.14, same configuration, only the prompt differing;
the 70.4% hit rate after the change matches before the change ⇒ no regression):

| Workload | decode topC | prompt hot set coverage | prefill touches/layer | decode touches/layer |
|---|---|---|---|---|
| 6-token short prompt | 85.8% | **17.9%** | 150.4 | 211.6 |
| ~150-token long prompt | 72.2% | **52.0%** | 209.8 | 247.8 |

Conclusion: prefill is parallel → each layer touches ~82% (209.8/256) and the distribution is nearly
flat; decode is serial → it concentrates on the top 64 (covering 72.2% of the routing); the value of
the prompt as a cold-start seed grows sharply with length (17.9% → 52.0%, = 72% of the reachable
ceiling 72.2%); prefill routing was landed into the statistics + `LLAMA_MOE_PREFILL_WEIGHT` (defaults
to 1.0; 0.1 is recommended for long-prompt scenarios; under this workload 0.1 and 1.0 give the same
60.3% hit rate, because prefill quality itself only accounts for ~13%).

Effect of SMoE/the cache on prefill (handoff §6.15, long prompt, `-n 8`):

| Configuration | Prefill (Prompt t/s) | Decode (Gen t/s) | VRAM |
|---|---|---|---|
| All on (SMoE+cache+backfill) | 44.5 | 11.8 | 9257 |
| SMoE off (`PREDICT_SMOE=0`) | 44.4 | 12.1 | 9255 |
| SMoE off + cache off (`CACHE_MIB=0`) | 44.7 | 8.9 | 7223 |
| All off (stock `--cpu-moe`) | 42.7 | 8.7 | 7223 |

⇒ prefill is not held back by this mechanism (44.5 vs stock 42.7, the difference is noise); the decode
gain comes from **the cache**; prefill **does not enter the cache** (it goes through
`moe_copy_experts_grouped` into the scheduler inputs and does not pollute the cache) — this is correct
behavior, and what the two phases should separate is **statistics and use** (prior/seed), not admission.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The decode numbers in the table (11.8/8.9/8.7) come from a `-n 8` short generation, dominated by cold-start filling, and must not be cited as steady-state speeds |
| Reason kept/dropped | Kept (low risk, no regression); "using this prefill's top-C as a seed" is left as an undone item |
| Open questions/reopening conditions | Undone: replacing/superimposing `LLAMA_MOE_PIN_STATIC`'s static manifest with the prefill hot set (the ceiling was measured at 52%). Reopening conditions: for "long prompt + short generation" scenarios |

### 2.14 H14 PLE layered cache — the complete log has been moved forward to §1A

This section keeps only the supplementary data related to the **256k + vision encoder + TBQ4 KV**
configuration; the mechanism, motivation, early measurements, the point at which it was squeezed out of
the MoE budget, and the status notes are in **§1A** (the first stage, filed separately).

- 256k VRAM ledger (handoff §6.29, measured 14442/16384 MiB): MoE cache 3.9 GB + PLE-GPU 1 GB
  (recommended off) + vision encoder 0.85 GB + non-MoE weights/KV (256k, TBQ4)/compute ≈ 9 GB ⇒ base
  ≈11 GB. **TQ4 256k itself is not beyond expectations** (the bulk of the ≈9 GB is weights and the
  compute buffer).
- The budget log names differ from the actual parameters: `auto-256k-vis` is an explicit request for
  8192 MiB followed by a clamp, yielding 40 slots / 3881 MiB / 64.6% / 13.2 t/s; the two cap2700 runs
  give 28 slots / 56.5–58.4% / 14.7–15.3 t/s. It has **12** more slots, yet that high-budget sample is
  slower; on that basis one cannot claim that auto is faster or that there is a stable paired gain.
- At the same element count, `iq4_nl` at 4.5 bpw gives about a **75.6%** byte increase over `iq2_s` at
  2.5625 bpw; the old 47% came from a different denominator and cannot be reused. Lowering bpw is a
  candidate direction, not a case where other layout options have been exhausted.
  The raw operator's original log shows `12.62 / 11.70 / 32.00 GB/s`, while the tool actually computes
  in GiB/s; it is not directly comparable to decimal PCIe bandwidth or to token throughput, see
  [chapter 03](03-weight-quantization-and-kernels.en.md).
- `LLAMA_PLE_GPU_CACHE_TIMING_FILE` / `LLAMA_PLE_GPU_PREFETCH_FILE` /
  `LLAMA_PLE_PREFETCH_DEBUG_FILE` are the PLE-side diagnostic outlets
  (`src/models/qwen4exp.cpp:589/604/719/2125`), matching the statistics regime of §1A.2.

### 2.15 H15 MTP / speculation × expert cache — retracted

| Field | Content |
|---|---|
| Route ID | H15 |
| Status/version | **Retracted**: `ggml/src/ggml-backend.cpp` was reverted to the sealed commit `c08171aa8`; the later-layer registration, `mtp_mode`, the slot out-of-range diagnostic and the temporary scripts were all removed. Verification: on master `LLAMA_MOE_LATE_LAYERS`, `moe_cache_finalize_new_layers` and `mtp_mode` occur 0 times |
| Why it was attempted | This MoE-cache design had reached a plateau on the host path (the ceiling measurement of §2.9 #14), so going up another step required changing axes: the fork already had a NextN/MTP draft head (`model : add the qwen4exp NextN/MTP draft head`, `qwen4exp: allow loading a draft-only MTP export`) |
| Technical mechanism | The MTP layer = `blk.48` (`n_layer_all=49`, `n_layer_nextn=1`), with its own 512-expert MoE, weights ≈1750 MiB (≈3.4 MiB per expert; 64 slots ≈220 MiB); its tensor names are isomorphic to the trunk (`ffn_moe_*-48`), and matching the cache by name is itself compatible. The cache layout is frozen at "the first single-token graph" (after `moe_cache_ensure` finalizes it is always null) ⇒ `blk.48` can never enter the cache, so a "later layer" path was implemented: `moe_cache_finalize_new_layers` (right after `moe_cache_init`, avoiding the CUDA graph capture window) builds the bundle once the set of weight kinds is stable (→ `alloc_persistent_layer`, slot ceiling 32), and until it is built `ensure` always returns nullptr |

**Experiment conditions and evidence**:

- MTP itself is usable (handoff §6.26): `llama-cli` **does not support speculative decoding** (`tools/cli`
  has no `common_speculative` call), and when `-md` fails to load the draft it **silently degrades** to
  ordinary generation (measured 13.3 t/s) — this is the real cause of "loading the draft errors out".
  The correct way to run it = `examples/speculative-simple`:
  `llama-speculative-simple.exe -m <IQ3_XXS 00001> -md <mtp-...-Q4_K_M.gguf> --spec-type draft-mtp
   -ngl 49 --cpu-moe --no-mmap -c 8192 -n 64 --temp 0 --spec-draft-n-max 8`,
  measured `n_drafted=27 n_accept=27 accept=100.000%`, correct text, **8.55 t/s without the cache**
  (the draft head is a MoE layer and also runs on the CPU ⇒ without speculation it is even slower).
  Parameters: `--spec-draft-n-max` (`--draft-max` is deprecated); the draft borrows the target's
  `token_embd/output` (`nextn_shared_target_tensors` + `borrow_shared_tensor`, requiring
  `ml.model_shared`). Note: a known bug in the fork: `common/speculative.cpp:2547` takes the draft path
  from `model_path` and then actually loads using `params.model.path` (the target).
- Blocking point (§6.26/§6.27): `llama-speculative-simple` + cache inevitably crashes with
  `CUDA error: an illegal memory access` (`ggml_cuda_kernel_launch`) in **the target's own graph**
  (about 17–19 s, before the draft has run). Bisection:

  | Bisection item | Result |
  |---|---|
  | `CACHE_MIB` only (**without SPLIT**) | Runs through (`decoded 19 tokens in 2.154 s`) |
  | SPLIT + direct | (wrong) CRASH |
  | SPLIT + no direct (gather) | (wrong) CRASH |
  | SPLIT + `CPU_ASYNC=0` | (wrong) CRASH |
  | SPLIT + MRS/prefetch/SMOE off | (wrong) CRASH |
  | `LLAMA_MOE_LATE_LAYERS=0/1` | (wrong) both CRASH (unrelated to the later layer) |

  ⇒ The breaking point converges on **the GPU/CPU split path itself** (`moe_split_partition` + the CPU
  half's leaf/activation copy/ids readback), unrelated to the direct patch, the asynchronous CPU half,
  the optional features or the MTP layer.
- Diagnosis (§6.27 addendum): `compute-sanitizer --tool memcheck` runs to the end **with no memory
  report**, giving only `CUDA error: unknown error` (at 42 s, `cudaStreamSynchronize`); combined with
  `CUDA_LAUNCH_BLOCKING=1` the error stops at `ggml_cuda_kernel_launch` (`common.cuh:1700`) ⇒ it looks
  more like **an illegal kernel argument/pointer**. `LLAMA_MOE_DUMP_CH=1` checked the GPU half's slot
  index for out-of-range ⇒ measured **0 out-of-range** (excluded). `SPLIT=0` + cache + MTP **runs but
  only at 8.8 t/s** (8.55 without the cache) ⇒ a cache without splitting has almost no gain for MTP.
- Decision and reason for dropping (§6.28): the only trigger condition is `LLAMA_MOE_SPLIT=1`; and
  without splitting the cache has almost no gain for MTP; fixing it is independent structural work
  whose cost/benefit ratio is not good enough to bet on. Moreover, **that cache never coexisted with the
  speculative front end** (it was only verified on `llama-cli`'s fixed single-token decode graph) —
  this is a pre-existing limitation, not one introduced in this round.
- Usable knowledge left behind: the only usable MTP front end = `llama-speculative-simple`; the draft is
  a serial bottleneck (8.55 vs 13.3); the draft layer's size and naming; the recovery path (first fix
  the crash of "split path × speculative front end", suspecting the `cur_cpu` activation D2H byte count,
  the `ids_cpu/wgt_cpu` host leaf sizes, and the validity of the `node->src[0]=input_cpy` patch under
  variable-length batches). This round's MTP measurements have **no independent raw log file name**,
  only the excerpts inside the reports mentioned above (the commands and outputs of handoff
  §6.26–§6.28).

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | "MTP does not work" is wrong: MTP itself accepts 100% with correct text; what does not work is the "cache × speculative front end" combination. Nor should 8.55 t/s be taken as the speculation gain — the draft runs entirely on the CPU and is slower than the target alone at 13.3 t/s |
| Reason kept/dropped | Retracted (all WIP reverted); the investigation conclusions are kept for reference |
| Open questions/reopening conditions | Reopening conditions: first solve the failure of the slot-view patch under variable-length batches/multiple graphs (the same class of risk as §2.11), or first do the D01 operator (explicitly holding buffers, no longer fighting the allocator) |

### 2.16 H16 Adaptive admission (yield threshold / extremum search / trend regression / automatic ahead)

> The motivation and measurements of the dual gating (value threshold + rate budget) are in §2.17; this
> section records the controller's own implementation and convergence evidence.

| Field | Content |
|---|---|
| Route ID | H16 |
| Status/version | Released, all off by default (`LLAMA_MOE_YIELD_AUTO`, `LLAMA_MOE_TREND_AUTO`, `LLAMA_MOE_AHEAD_AUTO`, `LLAMA_MOE_ADMIT_BUDGET_MIB` etc.); the default behavior matches that before their introduction (regression-verified) |
| Why it was attempted | The marginal value of a hit is state-dependent (≈0.074 ms per hit when the CPU half has work; ≈0 marginal value after a ~80% hit rate), while the marginal cost of a transfer always exists (≈0.08 ms/MB) ⇒ the optimal depth is where "marginal value crosses marginal cost", and it moves with the distribution/prompt/cache state/phase |
| Technical mechanism | Three parallel controllers: ① extremum search (decision variable = per-byte efficiency threshold in hits/MiB, objective = median per-token time, comparing the previous and next window every 16 graphs, multiplicative step ±10%, clamped to [0.5,32]); ② model-driven (sample `(ms, hits, MB)` per graph, do a centered least squares on `ms ≈ a − V·hits + P·MB` over a 32-graph sliding window, Cramer solve, EWMA 0.75/0.25 → threshold = P/V, budget = frac×ms_hat/P); ③ ahead extremum search (objective = **the window's incremental hit rate**) |

**Experiment conditions and evidence**:

- Measurements and lessons on the threshold/budget (handoff §6.7 ③/§6.8/§6.9):
  an accuracy-based threshold (`RANK_ADAPTIVE`) has a chicken-and-egg problem (a narrow cutoff offers
  only a few ranks → the statistics starve; fixed: accuracy is measured over the **complete candidate
  table**, decoupled from admission);
  with a measured yield threshold (`YIELD_MIN=4`), r1/r2 measure 17.9/16.7 ≫ 4, but deeper ranks cost no
  bytes → yield=0 → judged 0 → cut sticks at 1 (an **unexpected bonus**: cut=1 plus the saved transfers
  = 20.6 t/s > cut=2's 19.3);
  estimating yield as "accuracy × measured hits per admission" with a byte budget: a 64 MiB budget →
  12 t/s (early layers eat it all, later layers starve), 128 MiB → 19.0 t/s, **worse than 1/2**.
  A shard smaller than 1 bundle (1.9 MB) rejects all admissions (64 MiB/48 = 1.33 MB → 12 t/s) ⇒ the
  shard floor must be ≥1 bundle.
- Occupancy observation (user-measured): bf8/bf16 raises decode occupancy from 40–45% to 60%, so the
  earlier "100%" was actually prefill-full load ⇒ **the old estimate of a "GPU 40 ms hard floor / 25 t/s
  ceiling" is void**; `ADMIT_BUDGET_MIB` pins occupancy at the middle plateau (it rate-limits prefetch
  admission → working-set growth is rate-limited). Conclusion: **the value threshold (which ranks are
  worth it) and the rate control (how much per token) must be kept separate** — the rate goes to
  backfill, the threshold only handles value.
- Extremum-search convergence (§6.8, 400 token, 6 GB+backfill 8): prose `67.8 → 38.7 ms` (threshold
  4.4 → 10.9), converging to cut=1, yield_min=11.82, 18.8 t/s; code's ~4.3–4.8 is the yield threshold,
  not milliseconds ⇒ the two distributions yield different thresholds (the manual constant 4 is 3× too
  loose on prose).
- trend model (§6.9, 400 token, 6 GB+backfill 8, frac=0.25): `trendV=0.0158 ms/hit`,
  `trendP=0.0936 ms/MB` → threshold = P/V = 5.92 hits/MB (turning cut=4 on), budget = 116.5 MB;
  `fits=191 rej=212`; 19.6 t/s (same configuration historically 18.8–19.9, no regression).
  Arithmetic correction: 0.0936 is about **28.2%** higher than 0.073; it is only 17% relative to 0.08.
  Across operating points it is approximately the same order of magnitude, not independent confirmation
  of hardware bandwidth. Guards: a window of <16 points, insufficient spread (hits spread <15% or MB
  spread <8%), too small a determinant, or an out-of-range solution → that fit is rejected.
- automatic ahead (§6.32 ④): the objective must be the window's incremental hit rate (cumulative hit
  count/cumulative hit rate would be biased by cache warmup — the measured controller climbed all the
  way to the clamp value 6 and then walked back down to 1); after the fix it **probes back and forth
  between 3↔4** (4×19, 3×18, 5×6, 6×4, 2×1) = it correctly finds the optimal region; but **a fixed
  default of 3 is still slightly better** (20.8 vs 19.1 t/s) ⇒ the controller is off by default, left as
  an optional extra.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | All signals come from the host partition hook ⇒ it inherently conflicts with devpart (H10); the ahead curve and the yield-related experiments were mostly completed during the `SPLIT=1` silent-error period or at an old operating point ⇒ the controller itself is usable, but the absolute values of the "optimal point" must be cited under the §6.32 retest regime |
| Reason kept/dropped | Keep the implementation (off by default); the default configuration = fixed `SMOE_AHEAD=3` + `HOT_BACKFILL=8` + default `rank_cut=2` |
| Open questions/reopening conditions | The `PREFETCH_JOIN=1` path crashes (defaults to 0, not used for now); the `HOT_IDLE` idle thread has had no end-to-end verification (it needs an interactive session of "generate → idle → generate again") |

### 2.17 H17 "Higher hits yet slower" and the dual-gated adaptive transfer threshold

| Field | Content |
|---|---|
| Route ID | H17 |
| Status/version | A released experiment control interface: the time-limit gate is on by default; value adaptation such as YIELD_AUTO / TREND_AUTO is off by default. For the full criteria see [chapter 02 PC-24](02-prediction-and-cache.en.md) |
| Why it was attempted | One must not chase the highest hit rate alone; the value of prefetching per byte and whether it can catch up with consumption must be distinguished |
| Technical mechanism | The **value gate** decides the cutoff from the per-rank yield and a threshold; the **time-limit gate** compares the estimated in-flight plus current-batch transfer time against the layer distance × the per-layer time EWMA. `P/V` can produce a value threshold, and `frac×ms_hat/P` is an independent byte budget, not a second gate |

**Historical observations** (handoff §6.6–6.9, not a new controlled acceptance):

| Prefetch volume | Hit rate | total | t/s | flag_input |
|---|---|---|---|---|
| 161 MB/tok | 40.2% | 70.0 ms | 14.3 | 15.8 ms |
| 254 MB/tok | 46.9% | 77.7 ms | 12.9 | 23.3 ms |
| 464 MB/tok | 58.2% | 94.1 ms | 10.6 | 37.5 ms |

- The observation is "volume, hits and waiting rise together while speed drops"; the specific contention
  mechanism is still an explanation, not a hardware-profiling conclusion.
- The 0.074 ms per hit and the diminishing marginal value after a higher hit rate are estimates at the
  operating point of the time, not a fixed gain or a universal 80% threshold.
- Backfill can produce thrash at small capacities; the old 6 GB/backfill 8 values of 80.8%/23.3 t/s come
  from a stage later found to have a routing error, so **neither the trend nor the absolute values can
  serve as a performance basis for the correct path**.
- One TREND fit gives `V=0.0158`, `P=0.0936`, `P/V=5.92`, budget 116.5 MB, `fits=191/rej=212`, 19.6 t/s
  that time. It shows the controller produced these estimates; it does not prove it is better than all
  fixed parameters.
- The time-limit gate check has a scope of applicability: `prediction && prefetch_gate && !queue_on_worker`;
  it lets things through on cold start when there is no valid per-layer time history. hot-backfill / warm
  / seed cannot be generalized as being entirely under dual gating.
- `70 µs/copy` and `20 GB/s` are model constants; the layer time and in-flight volume vary with the run,
  and 20 GB/s was not measured online.

| Field | Content |
|---|---|
| Reason kept | Keep the decomposition "worth transferring / can catch up / how much can be transferred per round"; the defaults and individual performances of the different controllers are recorded separately and cannot be lumped together as "dual gating off by default" |
| Open questions | The value estimate is affected by warmup, collinearity and the input distribution; the byte budget may favor early layers; a budget smaller than one bundle cannot form a single valid admission |
| Reopening conditions | First confirm at a correct operating point with the same input that prefetch is the main cost, then calibrate the model constants, check the bypass, and compare against fixed thresholds |

## 3. Designed but unimplemented routes (`moe-decode-perf-plan.md` and `rebuild-spec.md`)

### 3.1 D01 Standalone MoE operator (getting an operator out of the shell)

| Field | Content |
|---|---|
| Route ID | D01 |
| Status/version | **Design only**. Verification: on master and on the source workspace HEAD, `GGML_OP_MOE_QWEN4EXP` and `moe_qwen4exp` both occur **0 times**; what landed is only the device-side partition kernel `ggml/src/ggml-cuda/moe-partition.cu` (81 lines) and the primitives `GGML_OP_MOE_PARTITION_IDS/WGT` (`GGML_OP_MOE_PARTITION` occurs 11 times on master) |
| Why it was attempted | The only thing that collides is `ggml_backend_sched`: it simultaneously holds tensor lifetimes, execution order/boundaries and synchronization points, and those three must be held by the MoE execution itself (perf-plan §0.1; rebuild-spec §2.2). The goal is to move the entire MoE cache/split/devpart scheduling hook block of `ggml-backend.cpp` on master out |
| Technical mechanism | Operator signature (7 srcs, fits within `GGML_MAX_SRC=10`): `GGML_OP_MOE_QWEN4EXP(cur, selected_experts, weights, gate_exps, up_exps, down_exps[, candidates])` → a device tensor equivalent to the sum of the GPU half + the CPU half. The shared experts **do not enter the operator** (they depend only on `cur`, are GPU-resident, and are one of the SMoE inputs). The engine is a **persistent object** (surviving across layers and tokens): its own compute/prefetch stream, expert cache, its own CPU thread pool, prefetch queue and events; entry `llama stream --event--> our stream`, and on the exit side the reverse **records only one event**. The five classes of "interruption sources" are sealed off one by one (split boundary copies / explicit exclusion from CUDA graph capture / gallocr does not take over the cache buffers / the scheduler event only once at the exit / explicit backend-ownership assertions). `ggml_backend_sched_compute_splits` returns to a state close to upstream |

**Stage breakdown (two numbering schemes, neither executed)**:
perf-plan §0.1: M0 empty operator (zero behavioral change, only proving the "seam" exists) → M1 move
cache + partitioning → M2 change the fence type (`synchronize` → `event wait`, self-owned stream,
batched D2H, 86 → ~46 ms) → M3 move prefetch/prediction, rewrite admission.
rebuild-spec §8: S1 clean worktree + port list → **S2 differential oracle** (one command runs old vs new
and diffs the token ids) → S3 the `M0` naive correct operator (`LLAMA_MOE_OP=1/0` bit-identical) →
S4 move cache+partitioning in (delete the hooks) → S5 change the fence type + self-owned stream +
batched D2H (decode ≤ 40 ms) → S6 move prefetch/prediction in.
S1a (`GGML_OP_MOE_CPU` + `moe-partition.cu` + the prefetch backend interface) is the prerequisite for M0.

**Why it was not continued**:

1. **The clean line was abandoned**: the `clear/` migration snapshot inside
   `OTHER_SOURCE_TREES/llama-qwen4exp-clean` (worktree @ `b76199698`) (TQ4/MoE/PLE sources + docs +
   tools + patches, 35 files) is **not self-consistent** (crashes on load at 0.7 s), and the HEAD itself
   cannot run this model (handoff §2). The snapshot serves only as a reference for "our code list".
2. **Its priority was eaten by the host path's gains**: without touching the operator, the host path went
   from 11.3 t/s to ~20 t/s (the post-fix regime of §2.11), while what the operator could win back is
   exactly the remaining two structural costs (the 17–19 ms rendezvous + the 13.1 ms split startup);
   among those, devpart's old host control has been invalidated and the split merge ran into a VRAM
   obstacle; neither can be counted directly as a realizable gain ⇒ the operator's marginal benefit
   becomes unclear.
3. The design's own bar (rebuild-spec §2.2 guardrail metrics: `git diff -- ggml-backend.cpp` going to
   zero, `llama-graph.cpp` left with only the few lines that construct the operator) is an architectural
   goal, not a measurement; without a "measure first" benefit proof it is not worth betting on.
4. **Correctness first**: the reason the S2 differential oracle exists is precisely that "every previous
   version was written faster than it was verified" (devpart is three versions old and still produces
   garbling and crashes today) — advancing to M0 before the oracle is built would repeat the same failure
   mode.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | Among the things "borrowing the shell" could eliminate (the devpart lifetime bug, per-split cross-backend copies, `insert_flush`, the 195 generic copies), the first two have already been covered by the host fix or by the devpart experiment; the `insert_flush` 0.07 ms measurement does not hold |
| Reopening conditions | Evidence appears that "the host path can no longer improve and the bottleneck clearly lies at the operator boundary" (for example needing cross-layer batched D2H or an SSD tier) |

### 3.2 D02 Memory hierarchy roadmap (RAM+VRAM → +SSD) and its two hard constraints

| Field | Content |
|---|---|
| Route ID | D02 |
| Status/version | **Design only** (rebuild-spec §2.3). The cross-layer interface on the PLE side is released (`copy_pages`/`prefetch_async`), the MoE cache side is unimplemented |
| Why it was attempted | The second stage needs to add an SSD tier (larger model / smaller RAM); if the first stage does not reserve the abstraction, the data flow must be changed a second time |
| Technical mechanism | Module split: not "one big MoE operator" but "layered paged weight storage + prefetch engine (**shared** by PLE and the experts) + predictor + MoE operator". Only three things differ: page size (63 KiB vs ~2 MB bundle), access pattern, predictor; the eviction policy, tier management and asynchronous prefetch + event fence are **completely isomorphic** |

**Two hard constraints (must be satisfied now, otherwise the second stage starts over)**:

1. **The prefetch copy source must be abstracted and must not hard-code a host pointer** — currently
   `moe_cache_copy` uses `input->data` (pinned host) directly, whereas in the second stage the source is
   an SSD, which must be a `copy_pages`-style "materialize one page from the lower tier" interface.
2. **The lead distance must be parameterized and must not be hard-coded to 1 layer** —
   `RAM→VRAM ~2 MB @10 GB/s ≈ 200 µs` (1 layer is enough);
   `SSD→RAM ~2 MB @3 GB/s ≈ 700 µs + latency`; stacking the two hops requires a lead of 2–3 layers or
   even across tokens. The existing `deadline = layers_until_visit * layer_us_ewma` is a **single-hop,
   layer-count-based** model, and a rebuild must write it as multi-hop and remaining-time based.

**Implementation status check (master)**: constraint 1 is **not satisfied** —
`moe_copy_experts_grouped` still issues `ggml_backend_tensor_set_async` from `input->data` (a host
pointer); constraint 2 is **not satisfied** — `moe_prefetch_feasible` at `ggml/src/ggml-backend.cpp:4015`
is still `eta_us = (dma_inflight_copies + n_copies) * gate_copy_us + bytes/gate_bw_bps` and
`deadline_us = max(1, layers_until_visit) * layer_us_ewma` (defaults `gate_copy_us = 70 µs`,
`gate_bw_bps = 20 GB/s`). There is no SSD-related code; PLE's `copy_pages` is the only
"layer-swappable" interface.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The judgment that PLE and this cache are "another independent implementation of the same thing" rests only on the interface shape and the unit size; no merging experiment was done |
| Reopening conditions | A deployment target with insufficient RAM appears (e.g. moving to a larger model), or work starts on D01 |

### 3.3 D03 Expert clustering storage

| Field | Content |
|---|---|
| Route ID | D03 |
| Status/version | **Design only** (rebuild-spec §2.4). No clustering tool, no permutation table, no loader mapping, no cache keyed by storage_slot |
| Why it was attempted | The dominant term in the current admission criterion is `n_copies = 3` (gate/up/down) at a fixed cost of 150–400 µs each; after clustering, one transfer covers a whole cluster of co-activated experts ⇒ `n_copies` 3→1, and in the SSD stage random reads become sequential reads, directly targeting `prefetch_dropped=8092` (65%) |
| Technical mechanism | Pure permutation: experts are stored clustered by co-activation, **the quantization type is unchanged, the tensor shape is unchanged, only the expert dimension is permuted** ⇒ the ggml kernels work as-is, and only one more id mapping layer is needed (`logical_id --permutation table--> storage_slot`, the permutation table being 48 layers × 512 experts × 2 B ≈ 48 KB). **No custom GEMM is written**; only variable-length clusters / per-cluster precision require a custom layout, which this scheme does not need. The cluster size is to be decided by measurement (2–4 experts ~4–8 MB hits precisely but the per-copy overhead remains; 8–16 experts ~16–32 MB amortizes the overhead but drags in unneeded experts), with an initial estimate of 4–8 experts/cluster |

**Second constraint (related to D02, not satisfied)**: the logical expert id and the storage location
must be separated into two layers of abstraction; currently they are mixed (`expert_slot[expert_id]`,
on master the cache is keyed by logical expert number) ⇒ once mixed, the second phase has to change the
cache core. To be honest: under pure permutation llama's GGUF loader **is still usable**, and "what forces
us to go it alone is the scheduler, not the weight format", so the two phases can be decoupled (phase one
engine, phase two clustered layout + our own container).

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | "Clustering hits the current bottleneck exactly" is an inference based on the per-copy fixed cost (150–400 µs); the marginal cost later measured in this project is 0.073 ms/MB ≈ 13.8 GB/s (leaning toward bandwidth rather than pure fixed cost), so the upper bound of the amortization gain needs to be reassessed |
| Reopening conditions | When prefetch is still the bottleneck and an SSD tier is needed; or first verify the gain of `n_copies` 3→1 using "one transfer for the three weight tensors of the same layer" |

### 3.4 D04 Shared experts first + SMoE firing early

| Field | Content |
|---|---|
| Route ID | D04 |
| Status/version | **Design only/not implemented**. Verification: on master `src/models/qwen4exp.cpp:1993-2000` is still `smoe_hidden = ffn_input + smoe_gpu_out + ffn_shexp_gated` |
| Why it was attempted | perf-plan §P1.4 judged it "the highest-leverage one": at present the shared-expert FFN is built after the MoE, so SMoE can only fire at the end of the FFN, with a lead of ≈ 0; the shared experts depend only on `cur` and have no dependency edge with the routed MoE, so they could be computed first. Trade-off: trade a slightly lower accuracy for a higher delivery rate (prediction coverage and on-time delivery still need a combined assessment; the old "73%×34%=26.8%" arithmetic does not hold, and it is not proven that the two denominators can be multiplied directly) |
| Technical mechanism | Reordering: `attention → [shared-expert FFN computed first] → SMoE fires (input + shared) → prefetch L+1 → routed MoE`, with the lead going from ~0 to the duration of the entire routed MoE + the next layer's attention; consequently the SMoE side graph and the per-layer `event_synchronize` of `moe_cache_smoe_enqueue/drain` can be deleted (measured 17 ms/token, while the CPU processing part is only 2 ms) |

**Offline support (`smoe-nk-degradation-plan.md` §7, 8 prompts, `ffn_moe_input` whitelist, 56400 samples)**:
oracle all 100%; **full 68.53%**, `shared_only 68.07%`, **`input_only 67.90%`** (k=1)
⇒ on that data `input_only` differs from full by **0.63 pt**, which supports continuing to test the
possibility of deleting dependencies.
k=2/3/4 are 59.94/55.70/53.77%, which supports continuing to verify the lead, but does not prove that a
two-layer window is "fully usable" online.
These are single-step teacher-forced results; only if premises such as the actual computation path
remaining unchanged hold does a prediction hint avoid feeding the approximate activations directly back
into the model. The cumulative online impact has not been verified.

**Why it was not continued**: this route would change three high-risk things at once (simplifying the
SMoE input, reordering the shared experts, deleting the side graph), while the 17 ms wait has already
been solved by "adding lead" at a far lower risk (§2.3: ahead defaults to 3 → 90.9% hits, 20.8 t/s).
Under this project's acceptance bar there is no evidence that another tier can be won; moreover it is
tangled up with the boundary D01 draws (shared experts stay in llama, SMoE does not enter the operator),
so it is a change that should be made together with the operator.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | 68.53% vs 67.90% are **offline single-step** results; before enabling it online, the "change in the actual delivery rate and hit rate once the lead grows" must be verified |
| Reopening conditions | When prefetch/wait overhead needs to be pushed lower, or when work starts on D01 |

### 3.5 D05 Admission model rewrite (`gate_copy_us` calibration / remaining-time deadline)

| Field | Content |
|---|---|
| Route ID | D05 |
| Status/version | **Partially designed, not adopted**. Verification: master's `ggml/src/ggml-backend.cpp:4015-4024` still holds the old model; the environment overrides `LLAMA_MOE_GATE_COPY_US`/`LLAMA_MOE_GATE_BW_GBPS` exist (`:2725`/`:2728`) but **no script calibrates them** |
| Why it was attempted | Two problems with the old model: `gate_copy_us = 70 µs` was never calibrated, while each small transfer actually measures 150–400 µs (underestimated by 2–5×); the `deadline` is computed in "layers", whereas the real constraint is "remaining time", and SMoE only fires at the end of the FFN so the lead is ≈ 0 |
| Technical mechanism | perf-plan §P1.1/P1.2: change the deadline to a remaining-time model and calibrate `gate_copy_us`; rebuild-spec §4 adds "preconditions at slot granularity (do not use an `insert_flush`-style wait-until-empty)" and "merge small transfers" |

**The path actually taken**: the project did not rewrite the deadline model, but instead (a) made the
rank cutoff line explicit as `rank_cut` (§6.6 ③, and fixed the bug where `n_used==0` made the cutoff
line silently disappear); (b) deduplicated (52.5% of candidates cost no bytes); (c) hot-region backfill
(no time limit, not subject to the budget); (d) increased the lead with `SMOE_NONBLOCK=1 + AHEAD=3`;
(e) turned "threshold/budget" into an observed quantity with the two online controllers of §2.16.
The two operating points give estimates of about 0.073 and 0.0936 ms/MB; the latter is about 28.2%
higher than the former. This is a cost-magnitude reference, not an independent measurement of PCIe
bandwidth or of the causal mechanism.

| Field | Content |
|---|---|
| Observation and boundaries of the conclusion | The old constants and the deadline model remain to be calibrated; the joint variation of prefetch, waiting and speed suggests that further investigation is needed, but their individual causal costs have not been isolated |
| Reopening conditions | When prefetch becomes the main bottleneck again (for example once the hit rate has reached ~80% and admission is no longer the limit) |

---

## 4. Retracted / unreleased summary (must be checked before citing)

| Object | Status | Basis |
|---|---|---|
| The numbers `20.3 t/s` (sealed), `20.2 t/s` (seal review), `20.1–22.0 t/s` | **Void** | §2.11 (fake speed during the `SPLIT=1` silent-error period) |
| The numbers devpart `19–23 / 26.0 / 28.2 / 29.7 / 32.0 t/s` | **Void** | §2.9 (intermediate implementations with zero CPU-half data) |
| The number `20.7 t/s` | Void | perf-plan §7 (garbled old build) |
| The hypothesis "`insert_flush` draining each layer = ~20 ms" | Void | §2.4 (measured 0.07 ms) |
| The hypothesis "`split_partition`'s host CPU loop = 23 ms" | Void | §2.5 (actually waiting for the GPU to produce the router) |
| The hypothesis "pinning can eliminate the rendezvous" | Void | §2.2 (the wait merely moves) |
| "The worker count is not the bottleneck" | Void (scope of the conclusion) | §2.6 (`INSERT_WORKERS` spins idly under `PREFETCH=1`) |
| prefill D2D staging (`LLAMA_MOE_PREFILL_CACHED`, `moe_cache_prefill_d2d`) | **Retracted** (0 occurrences on master) | §2.12 |
| MTP×cache later-layer addition (`LLAMA_MOE_LATE_LAYERS`, `moe_cache_finalize_new_layers`, `moe_cache_build_late_layer`, `mtp_mode`) | **Retracted** (0 occurrences on master, code reverted to `c08171aa8`) | §2.15 |
| split merging (removing the scheduler pass 5 rule) | Reverted after the experiment (VRAM OOM) | §2.8 |
| The `SPLIT=1` safety lock (refusing to take effect) | Lifted (after the §6.32 fix) | §2.11 |
| `GGML_OP_MOE_QWEN4EXP` / SSD tier / expert clustering / PLE into the operator | **Design only** (0 implementations) | §3.1–§3.3 |
| The PLE layered cache body | **Released** (upstream `4e1865e34` + fork `2f1a363c8`); off by default (env unset = off); **small gain at the full RAM operating point, effective at the lazy/mmap operating point** | §1A |
| "PLE failed overall / PLE was replaced by MoE as soon as it shipped" | **Wrong statement, prohibited**: the two coexisted for a long time (09-03…09-10), and PLE was only squeezed out in configurations short of VRAM/budget | §1A.3 |
| "The teacher test hit rate 99% is refuted by the offline recall@10" | **Wrong inference, prohibited**: the two have different metrics/denominators (see §0.5 for details) | §0.5 |
| D04 shared experts first + SMoE firing early; D05 deadline model rewrite | Design only/not implemented | §3.4, §3.5 |
| The CSV `-5` row (inflight peak) of `rebuild-spec.md` §9 | Not implemented (no `inflight_copies_peak`; only the `prefetch_dropped` count) | Verified in this chapter |
| The uncommitted NXQ/TBQ, cache and tool changes in the source workspace | WIP only, not in the release baseline | §0.1 |
| Kept but unverified: the `LLAMA_MOE_HOT_IDLE` idle thread, `PREFETCH_JOIN=1` (crashes), 0-slot split (no output), `AHEAD=1` side-graph off-by-one (closed as one beat late) | Still open | §2.3, §2.16 |

---

## 5. Relation to chapter 05 (correctness and methodology)

This chapter records only "the impact of the `SPLIT=1` silent error on **the speed evidence**":
retracting the 20.1–22.0 t/s series and the sealed values 20.3/20.2, marking the 400 token / 6 GiB
regime as unverified, and emphasizing the origin of the measurement discipline "must not use
`--ignore-eos`".
The defect's reproduction steps, bisection table, race mechanism, one-line-level fix diff and regression
case belong to chapter 05; section 2.11 of this chapter keeps only a summary and a pointer, so that the
two accounts do not overwrite each other.

Similar risks (for cross-reference by chapter 05): the failure of the persisted slot-view patch under
variable-length batches/multiple graphs (the MTP crash of §2.15), split producing nothing with 0 slots
(§2.11 leftovers), and `0xC0000005` at exit. A non-zero exit does not mean "only the statistics are
lost"; the root cause and the scope of the impact are not closed out, see chapter 05.

---

## 6. Open questions and reopening conditions

| # | Question | Current state | Reopening conditions |
|---|---|---|---|
| 1 | Acceptance of the release baseline at 400-token/6GiB/full-RAM | A local candidate record is not the same as acceptance after rebuilding master | Set up a same-condition control separately later; this round only archives |
| 2 | devpart's residency decision depends on the host path | Off by default; the historical 16.8 cannot be judged a net loss against a retracted host result | If reopened, first close the residency/prefetch decision chain |
| 3 | The same-regime comparison of devpart against the fixed host | Missing (the 20.3 control is void) | Retest under the same configuration |
| 4 | The per-layer router rendezvous | The local fixes tried established no net gain, which does not amount to excluding all structural fixes | Needs a new structural scheme and a controlled comparison |
| 5 | split segmentation and merging | The 13.1 ms is not a fully eliminable startup tax; this merge OOMed | The trade of shrinking the cache and then merging has not been measured |
| 6 | The speculative front end × cache crash (`SPLIT=1` is the only trigger) | WIP has been reverted; the cache has never coexisted with speculation | First make the slot-view patch invalidate per (graph, shape), or do the D01 operator |
| 7 | With 0 slots (`CACHE_MIB=64`) split produces nothing | Not fixed (it should degrade to all-CPU) | Handle together when fixing the split path |
| 8 | The `LLAMA_MOE_HOT_IDLE` idle thread | Released but not end-to-end verified | An interactive "generate → idle → generate again" session |
| 9 | `PREFETCH_JOIN=1` crashes | Defaults to 0 | Investigate that path separately |
| 10 | Occasional `0xC0000005` at exit (losing statistics rows) | Not fixed at the root; the `LLAMA_MOE_CRASH_TRACE=1` tracer is kept | Use semi-automatic retries plus the tracer to capture a stack |
| 11 | The `AHEAD=1` side-graph off-by-one | Closed as "the non-blocking readback is one beat late", not a defect | Already closed in §6.33 (only quality-preference scenarios switch to synchronous + ahead=1) |
| 12 | All the D01–D05 design items | The list of unimplemented items and the reasons are in §3 | Already written in their respective entries |
| 13 | The correspondence of the PLE 1G (maintainer's original wording) host row-cache 90%+ to its original run | The recollection is kept; same-layer logs agree in direction, but the protocol has not been matched item by item. GPU L1 hits are not used as evidence for this conclusion | If necessary, complete the protocol from the raw runs under `LOCAL_MODELS/qwen38/traces/` |
| 14 | Merging PLE and the expert cache into a single paging engine (rebuild-spec §2.3) | Design only; the PLE-side interface (`copy_pages`/page abstraction) is ready, while the expert side still hard-codes host pointers | Lazy/SSD deployment, or when work starts on D01 |

---

## 7. Source section coverage list

| Source | Covered sections | Location in this document |
|---|---|---|
| `handoff.md` | §6.1 completed measurements | §2.1, §2.2, §2.4, part of H03 |
| | §6.2 CPU_ASYNC neutral | §2.6 |
| | §6.3 devpart next steps | §2.9 #2 |
| | §6.4 final conclusion on the latency structure (independent of the copy mechanism) | §2.2, §2.7, §2.9 |
| | §6.5 devpart host-leaf experiment (8.1 → 19–23) | §2.9 #3 (retracted) |
| | §6.10 exit-time crash localization + whole-machine incident and gate | §2.7, §2.6 table |
| | §6.11 CPU_ASYNC made permanent + cost inventory | §2.6, §2.8, §2.2 |
| | §6.12 pinned readback (ineffective) + split structure + devpart compatibility | §2.2, §2.8, §2.10 |
| | §6.13 split merge investigation (bought with VRAM, not done) | §2.8 |
| | §6.14 prefill/decode tendency separation | §2.13 |
| | §6.15 SMoE/cache neutral for prefill | §2.13 |
| | §6.16 prefill read cache (D2D) zero gain, reverted | §2.12 |
| | §6.17 devpart re-inspection (29.7/32.0, retracted) + localization | §2.9 #4 (retracted) |
| | §6.18 devpart correctness fix (root cause + graph-construction crash) | §2.9 #5 |
| | §6.19 devpart fixed (26.0 t/s, floating-point order difference) | §2.9 #6 |
| | §6.20 speed path opened (28.2, retracted) + CPU-half data pending diagnosis | §2.9 #7 (retracted) |
| | §6.21 converged to a single point (readback never submitted) | §2.9 #8 |
| | §6.22 wrap-up (pipeline works, the defect is in the residency-table content) | §2.9 #9 |
| | §6.23 devpart fixed (three defects + identical text + not faster) | §2.9 #10/#11 |
| | §6.24 devpart 400 token steady state and the "cache does not fill" root cause | §2.9 #12/#13, §2.10 |
| | §6.25 seal record (including the ceiling measurement, devpart off by default) | §2.9 #14/#15, §2.11 |
| | §6.26 MTP compatibility investigation (runs; cache×speculation crashes) | §2.15 |
| | §6.27 MTP×cache decision and bisection | §2.15 |
| | §6.28 abandoning MTP×cache (WIP reverted) | §2.15 |
| Adjacent sections (this chapter needs them to complete the routes) | §6.6 prefetch volume and cutoff line; §6.7 hot-region root fix; §6.8/§6.9 adaptation; §6.29 PLE/256k configuration; §6.30–§6.33 silent error and fix; §6.34/§6.35 IQ4_NL repack and PLE | §2.3, §2.16, §2.14, §2.11, §2.14 |
| `moe-decode-perf-plan.md` | §0.1 chosen route (getting an operator out of the shell), §1 measurement ledger, §2 real numbers for the cache and prefetch, §3 P0–P4 change points, §4 target reachability, §5 changes already landed, §6 reproduction, §7 old conclusions rejected | §2.1, §3.1, §4 (including the items not adopted) |
| `smoe-nk-degradation-plan.md` | §7 N+k degradation measurement | §2.3, §3.4 |
| `rebuild-spec.md` | §1 baseline and acceptance, §2 boundaries, §2.2 scheduling ownership, §2.3 memory hierarchy, §2.4 expert clustering, §3 operator specification, §4 execution model and admission, §5 negative results, §6 pitfalls, §7/§7.1 port list and S1, §8 staging, §9 measurement regime | §3.1–§3.5, §2.15, §2.11, §2.7 |
| `moe-cache-score-aware-prd.md` | Strategy background (routing unchanged, hits/delivery, admission semantics; the implementation decisions and the reference experiment configuration of §60/§82/§86; the relation of §128 Fate/XT) | Cited only where the mechanism needs explaining (§2.3/§2.16, the predictor lineage of §0.5) |
| The user-supplied authoritative research order (PLE → static table+XT → Fate → SMoE/shared experts → online implementation → dual gating → devpart/TQ4/NXQ) | §0.5 (regime and boundaries) and §1A (PLE's first stage, filed separately) of this document; the main-line narrative is in `00-research-chronology.md` | §0.5, §1A |

**Division of labor with the other chapters**: the predictor lineage (static table/XT → Fate → SMoE, including the correspondence to the original record of the teacher test's 99%)
goes to `02-prediction-and-cache.md`; weights/quantization and the operator go to
`03-weight-quantization-and-kernels.md`; KV/TBQ/NXQ go to `04-kv-tbq-and-nxq.md`; the reproduction,
bisection, mechanism and regression case of the `SPLIT=1` silent error go to
`05-correctness-and-methodology.md`. This chapter keeps only the parts directly related to the
host/split/devpart performance evidence.

---

## 8. Raw log candidates and preservation boundaries

The following paths are relative to `SOURCE_TREE`. `verified` means the content/encoding was read;
unverified entries are matched by name and date, and the specific values still follow the corresponding
section's record.

| Log | Content | Corresponding route |
|---|---|---|
| `dsh-r8.txt`, `dsh-r9.txt` `verified` (UTF-16LE; containing `[1-]/[2-]/[3-]/[4-]` rows) | Per-graph ledger of 31 decode graphs (the 86.0 ms/token breakdown) | H01 |
| `dsh-r3-timing.txt`, `dsh-r4-timing.txt`, `dsh-r5-timing.txt`, `dsh-r6.txt`, `dsh-r6-csv.txt`, `dsh-r6-decode.txt`, `dsh-r7.txt`, `dsh-timing.txt`, `dsh-hitrate.txt` | Timing/hit-rate iterations | H01, H02, H03, H16 |
| `dsh-r3-text.txt`, `dsh-r5-text.txt`, `dsh-r7-text.txt`, `dsh-r5-bd.txt`, `dsh-r5-e3bd.txt` | Text-correctness controls | H11, H13 |
| `dsh-A-*`, `dsh-A2-run1-*`, `dsh-A2-run2-*`, `dsh-A3-*`, `dsh-A4-*`, `dsh-B-*`, `dsh-C-*`, `dsh-C2-*`, `dsh-C4-*`, `dsh-E1-*`, `dsh-E2-*`, `dsh-E3-*`, `dsh-F1-*`, `dsh-F2-*`, `dsh-G1-*`, `dsh-G2-*`, `dsh-H1-*`, `dsh-H2-*`, `dsh-I1-*`, `dsh-I2-*` (each with `-out.txt` / `-err.txt`) | P0-stage rendezvous/prefetch/hit experiments (`dsh-G2` is the log name of the devpart access violation) | H02, H03, H04, early H09 |
| `L2048-err.txt`, `L6144-err.txt`, `L2048-out.txt`, `L6144-out.txt` | Cache capacity tier controls | H03, H16 |
| `T2048_1..4`, `T6144_1..4/6/8` (each with `-out`/`-err`) | Capacity × rank cutoff line sweep | H03, H16 |
| `a400a-err.txt`, `a400a-out.txt`, `a400b-err.txt`, `a400b-out.txt` `verified` (UTF-16) | 400 token steady state, containing the `hot-set oracle` and per-layer slot/hit rows | H16, the control regime of §2.9 #12 |
| `ab-a-*`, `ab-b-*`, `ab-c-*` `verified` (1-line summaries) | SPLIT/load-failure A/B records | H11 |
| `cur-ref-out.txt`, `cur-ref-err.txt` `verified` | devpart OFF baseline (containing the `[PLE-LRU]`/`[PLE-GPU-L1]` enable rows, `[MOE-CACHE] enabled: budget=2048 … devpart=0`, and the `ids readback` row) | H01, H14 |
| `run-cur-ref.ps1`, `run-cur-devpart.ps1`, `tools-run.py` | Reproduction harness (the full set of configurations and environment variables) | all |
| `sweep-ahead.csv`, `sweep-ahead2.csv`, `sweep-c1.csv`, `sweep-c1nb.csv`, `sweep-corrupt.csv`, `sweep-fix.csv`, `sweep-hd.csv`, `sweep-t2.csv`, `sweep-v.csv`, `sweep-x.csv`, `sweep-neutral.csv`, `sweep-reg.csv`, `sweep-smoke.csv`, `sweep-u.csv`, `sweep-press.csv`, `sweep-budget.csv`, `sweep-auto.csv`, `sweep-auto8k.csv`, `sweep-limit.csv`, `sweep-tune.csv`, `sweep-ab.csv`, `sweep-2x2.log` | Sweeps of lookahead, cutoff line, non-blocking, silent error, fix, adaptation/budget, VRAM limit, etc. | H03, H11, H16 |
| `sweep-vision-cache.py`, `sweep-vision.log`, `sweep-vision256k.csv`, `cases-*.txt` | Cache/PLE controls under 256k + vision (a comment in the script states outright that "both PLE caches are dead weight under full RAM") | H14 |
| `sweep-kv.csv`, `sweep-k8q8.csv`, `sweep-k8tbq.csv`, `sweep-k256q8.csv` | KV type × cache parameters (crossing with the KV/TBQ of chapter 04; this chapter cites them only at PLE/256k) | H14 (crossing) |
| `dsh-probe-01..09.ps1`, `dsh-probe-bz8.ps1`, `bisect-garbage.ps1`, `bisect2..4.ps1` | P0-stage probes and bisection scripts (including PLE switches and `PREFETCH_JOIN` combinations; `dsh-probe-bz8.ps1`/`bisect*.ps1` were the first scripts to set PLE to 0) | H01, H11, H14 |
| `build-cli.bat` | Build (sccache, `-j 16`; `-j 32` freezes the machine) | all |

**PLE (first stage) dedicated log candidates** — note that **the PLE statistics files are not in the
repository; the main directory is `LOCAL_MODELS/qwen38/traces/`** (the `LLAMA_PLE_CACHE_STATS_FILE` in
`run-moe-ple.bat`/`run-nomoe-ple-cold.bat` points to that directory):

| Log | Content | Note |
|---|---|---|
| `ple-gpu-overlap-stats.csv` (09-03 15:06) | **GPU L1 layer** (8-column schema), overlap prefetch: 14791/1049 = **93.4%**; **not used to prove the SSD→main-memory 90%+** | §1A.2 |
| `ablation-ple-only.csv` (09-03 12:02) | **Host row-cache layer** (9-column schema): 14899/1101 = **93.1%**, `prefetch_pages=12774` | §1A.2 |
| `ple-lru-256m-o1.csv` (09-03 09:10) | 8-column schema (layer pending confirmation from the original record): 27.6% (capacity-limited control) | §1A.2 |
| `ple-lru-1g-no-prefetch.csv` | **Host row-cache layer** (9 columns), 10.1% with prefetch off; the capacity and the batch differ too, so it cannot be attributed to the prefetch switch alone | §1A.2 |
| `ple-gpu-l1-rawpages-1g.csv`, `-1g-async.csv`, `ple-gpu-lookahead-1g.csv`, `ple-cpu-l2-with-gpu-l1.csv` | Early variants before prefetch/alignment were in place (13–17% / 0.75%) | §1A.2 |
| `ple-cpu-stats.log`, `ple-cpu-stats-long.log`, `ple-cpu-stats-nopf.log` (09-04) | PLE page statistics of long/control runs | §1A.2 |
| `ple-gpu-l1-timing.csv`, `ple-gpu-l1-pinned-timing.csv`, `ple-gpu-overlap-prefetch.csv`, `ple-gpu-overlap-consumer.csv` | GPU L1 timing/overlap specifics | §1A.2 |
| `ablation-lazy-moe8-gpuple.log`, `ablation-lazy-moe8-ple.csv`, `ablation-lazy-moe8-ple-2.csv`, `ablation-lazy-moe7-cpuple{,-rerun}.log` (09-10) | Lazy-mode ablation with **PLE + MoE cache coexisting** (8 GiB / 7 GiB tiers) | §1A.2, §1A.3 |
| `run-moe-ple.bat`, `run-moe-ple-nopf.bat`, `run-moe-ple-long{,-ctl}.bat`, `run-nomoe-ple-cold.bat` (in the repository) | Reproduction harness for the PLE ablations (including `LLAMA_PLE_CACHE_MIB=4096`, `LLAMA_PLE_PREFETCH`, and the `LLAMA_PLE_CACHE_STATS_FILE` path) | §1A.2 |
| In the repository: `abl-2g-ple2g.csv`, `abl-2g-ple512.csv`, `abl-512-ple2g.csv`, `abl-512-ple512.csv`, `abl-2g-{ple,nople,nojoin,ple2g-gpu1g}.log`, `abl-512-{ple,ple2g,stats}.log`, `layer-bundle-ab-lazy.log`, `vram-samples.txt`, `ple0-guard512-{err,out}.txt`, `ple0-guard1024-{err,out}.txt`, `ple256-guard512-{err,out}.txt` | MoE×PLE ablations and VRAM sampling (note: the four CSVs are short generations whose cache never filled up, so they cannot support a capacity conclusion) | §1A.2, H14 |

**Parts with no raw logs** (explicitly existing only as historical reports): the MTP/speculation round
(handoff §6.26–§6.28 leave only command and output excerpts); devpart's
`LLAMA_MOE_DUMP_CH`/`DUMP_SPLITS`/`CRASH_TRACE` dumps (the output is quoted inside the report, with no
separate file kept); the `prefill/decode tendency` rows (scattered across `sweep-vision*` and the 256k
run logs, not archived separately per route).
