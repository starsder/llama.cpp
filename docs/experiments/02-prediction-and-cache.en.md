[中文](02-prediction-and-cache.md) · [English](02-prediction-and-cache.en.md)

# 02 Prediction and Cache: Static Table/XT → Fate → SMoE → Online Cache and Dual Gating

This chapter covers every attempted route for MoE/SMoE expert prediction and device-side expert caching: from the static hot table and the XT transition manifest, to the Fate online cross-layer gate, shared-expert SMoE prediction, and then the engineering of the online cache (prefetch volume, rank cutoff, deduplication, eviction score, hot-region backfill, dual-gate adaptive transfer threshold), and finally the shared pool / position weights plus the lifetime statistics of `LOCAL_EVIDENCE/moe-cache/optimization400` and two uncommitted candidates.

The chronology and causality of the main line are narrated uniformly by `00-research-chronology.md`; this chapter only provides the traceable experiment archive for the parts of that main line that **belong to prediction and caching**. The PLE cache was the first item in the development order and belongs to `01-host-and-devpart.md`; this chapter only cites it at the stage-one summary and does not repeat its evidence.

## 0. Reading conventions

### 0.1 Evidence path markers

For the path markers see [Evidence Index](evidence/README.en.md): `SOURCE_TREE` is the experiment source working tree, `LOCAL_EVIDENCE` is the temporary evidence root (the relevant reports live under its `moe-cache/`), `LOCAL_MODELS` is the model root, and `OTHER_SOURCE_TREES` is other checkouts. The source documents of this chapter use relative links.

- `sources/handoff.md` (full snapshot of the source working tree handoff memo): cited by §number + snapshot line range.
- `sources/smoe-nk-degradation-plan.md` (snapshot of the SMoE N+k degradation plan).
- `sources/moe-cache-score-aware-prd.md` (snapshot of the current PRD: true frequencies, shared pool and position weights; its final §History v1 preserves the frozen design text).
- `sources/moe-decode-perf-plan.md`, `sources/rebuild-spec.md` (corroborating evidence: predictor placement and the list of retracted old conclusions).
- The in-repo `../moe-cache-score-aware-prd.md` is an **earlier frozen v1 copy** (only sections such as Problem/Solution/Decisions); when citing the current policy and shared-pool content, always use the `sources/` snapshot.

### 0.2 Status vocabulary (this chapter uses only these five classes, and keeps them separate from "the code exists in the release tree")

| Label | Meaning |
|---|---|
| Published | The code is in a commit of this release tree (latest `7e01451b2`) and the switch has an explicit default value; **this does not mean its performance conclusion has been accepted** |
| WIP only | Exists only as **uncommitted** modifications in `SOURCE_TREE`, or only in a side-branch checkout; the release tree has no such code |
| Design only | Written in documents only, never landed as runnable code |
| Retracted | Was once used as a conclusion and later explicitly retracted (e.g. the host 20.3 / 32 tps speeds from before the routing-misplacement fix, and the old capacity criterion) |
| Still open | The phenomenon is recorded but the cause is not closed, or the verification conditions are not in place |

**Discipline (two rules, observed throughout this document)**:
1. **Source code existing ≠ implementation released, and a document being archived ≠ implementation released**. "Published" in the table above only means that the switch/code appears in a commit of this release tree; whether it has been adopted as a recommendation and whether there is usable evidence must be looked up separately in the "keep/abandon/pause reason" field. A side-branch checkout on the collection side (e.g. the SMoE trace instrumentation) remains a side branch even if its document made it into the snapshot, and does not count as released.
2. No optimization candidate was adopted into the release tree in this round; the DLL under `SOURCE_TREE/build-ple-trace-mrs/bin/` may already have been replaced by a WIP candidate at the time of the run, so citing a run result must give the binary hash **at the time of that run**, and must not claim that the directory "currently" still holds the same file (see §11 item 9).

### 0.3 Uniform fields for each route

Route ID, status/version, why it was tried, technical mechanism, experimental conditions and evidence, observations and conclusion boundaries, keep/abandon/pause reason, remaining issues and conditions for reopening.

### 0.4 Hardware and acceptance stance (the working point provided by the user)

This machine: AMD Ryzen 9 5950X (16 cores), DDR4-2666 128GB, RTX A5000 Laptop 16GB, PCIe 4.0 x8. Current 400-token working point: `--no-mmap --lazy-mode off`, weights fully resident in main memory, `CACHE_MIB=6144` (the 400-token acceptance baseline is fixed at 6144; auto appears only in the 128-token series), 16 threads, single-instance lock + a free-memory gate of at least 90000 MiB; the current internal screening floor is **19 token/s** (lowered from 20, see `sources/moe-cache-score-aware-prd.md` §Current performance acceptance baseline). **The old 20.3 / 32 tps and similar results have been retracted and must not be cited as valid performance evidence**; historical numbers must carry their context (context length, KV, budget, number of steps, whether the build carried the bug).

## 1. Route overview

| Route ID | Topic | Status | Source (snapshot §) | Key raw logs/evidence |
|---|---|---|---|---|
| PC-01 | Static hot table (manifest.hot, pinned) | Published (default `PREDICT_STATIC=32`; pin defaults to 0) | `sources/moe-cache-score-aware-prd.md` §History v1 §Solution | `LOCAL_MODELS/qwen38/traces/moe-predict-v1.bin`, `-static96.bin`; `SOURCE_TREE/cpunode-run.log`, `fused-final.log` |
| PC-02 | XT cross-token transition table (MOEPRED2) | Published (`PREDICT_XT`) | same as above §Solution 1, §Implementation Decisions | `moe-predict-xt.bin`; `ablation-lazy-moe2-xt.log`; `moe-xt-12288.csv` |
| PC-03 | CrossLayer transition table (MOEPRED1) + offline control | Published (`LLAMA_MOE_PREDICT`) | same as above §Implementation Decisions | `build_expert_manifest.py`, `analyze_cross_token.py`, `moe-predict-v1.bin` |
| PC-04 | Static/XT stage performance and hits (including the intersection with the PLE main line) | Retracted (as criterion) / still open (domain of applicability of the conclusion) | `sources/handoff.md` near §6.1, `sources/moe-decode-perf-plan.md` §1–2 | `ablation-lazy-moe{4,6,8}.log`, `moe-static90-*.csv` |
| PC-05 | Fate online cross-layer gate | Published but off by default; the negative conclusion is a user recollection | `sources/moe-cache-score-aware-prd.md` §History v1 §Further Notes | code `moe_cache_predict_fate`; **no `fate=1` run log found** |
| PC-06 | SMoE teacher test (user-recollected 99%) | User recollection, no matching protocol found | `00-research-chronology.md` main line; this chapter §4.1 | No protocol/raw log matching that 99% found |
| PC-07 | SMoE N+k offline degradation (recall@10) | Plan document archived (domain of applicability of the conclusion is limited) | `sources/smoe-nk-degradation-plan.md` §7 (snapshot lines 173–238) | `LOCAL_MODELS/qwen38/traces/simulate_smoe_nk.py`, `standard-smoe-8-20260911` |
| PC-08 | Early FIFO pool simulator and its self-stated limitations | Plan document archived (superseded by PC-07) | same as above §2 (lines 65–83), §5 (lines 130–162) | `simulate_smoe_from_standard_trace.py`, `standard-smoe-sim-20-20260910b` |
| PC-09 | The `ffn_input` gap and the two ways to fill it | WIP only (the collection-side change is in a side-branch checkout, not in the release) | same as above §3 (lines 84–99) | `SOURCE_TREE/qwen4exp-smoe-trace` (dedicated-branch checkout) |
| PC-10 | SMoE online implementation and its fixed overhead | Published (`PREDICT_SMOE=1`, requires split) | `sources/moe-cache-score-aware-prd.md` §History v1 §Implementation Decisions | `L2048-err.txt`, `L6144-err.txt`, `a400a-err.txt` |
| PC-11 | Conflation of the static hot set with the SMoE accuracy definition (98–99%/73%) | Clarified (the definition), no matching numbers found | same as above §6 (lines 163–172) | code `prefetch_predicted/required`; `global-pressure128-v2.summary.json` |
| PC-12 | Marginal cost of prefetch volume = PCIe contention | Published (switch) / conclusion valid | `sources/handoff.md` §6.6 (lines 209–258) | Historical reports only (no raw log names) |
| PC-13 | Rank cutoff `TAKE_MAX` (implicit → explicit) | Published (default 2) | same as above §6.6③ | same as above |
| PC-14 | Deduplication (resident/pending/list/admit) | Published | same as above §6.6④ | same as above + 400-token lifetime (PC-30) |
| PC-15 | Capacity × cutoff interaction | Published (parameter) / key point not re-confirmed | same as above §6.6⑤ | same as above |
| PC-16 | No benefit from prefetch worker count + correction that `INSERT_WORKERS` spins idle | Published (correction) | same as above §6.6①, §6.7 open items | same as above |
| PC-17 | Eviction score: gate-softmax MRS → ground-truth routing frequency | Published (`EVICT_SCORE=0` means frequency) | same as above §6.7 (lines 259–343) | Historical reports only + the oracle/per-rank lines of `a400a-err.txt` |
| PC-18 | Hot-region backfill + cold-start seed + idle fill | Published (off by default) | same as above §6.7 | Historical reports only |
| PC-19 | Per-rank accuracy / per-byte efficiency | Published (diagnostic) | same as above §6.7 | `a400a-err.txt`, `a400b-err.txt` (per-rank lines) |
| PC-20 | Three attempts at adaptive truncation (accuracy / measured yield / byte budget) | Published (off by default); all three lost to a fixed cut | same as above §6.7 | Historical reports only |
| PC-21 | Occupancy observation: value threshold separated from rate control | Published (conclusion) / old ceiling estimate void | same as above §6.7 | User measurement (historical report) |
| PC-22 | YIELD_AUTO extremum-searching threshold | Published (off by default) | same as above §6.8 (lines 344–372) | Historical reports only |
| PC-23 | TREND_AUTO model-driven threshold and budget | Published (off by default) | same as above §6.9 (lines 373–404) | Historical reports only |
| PC-24 | Dual-gate adaptive transfer threshold (source-code criterion) | Published (`PREFETCH_GATE` defaults to 1) | code + §6.8/§6.9 | `SOURCE_TREE/ggml/src/ggml-backend.cpp` |
| PC-25 | `SPLIT=1` silent miscalculation fix + `AHEAD` default 3 | Published (fix ships with 7e01451b2) | `sources/handoff.md` §6.32 (lines 1124–1173) | Historical reports only (Eiffel case) |
| PC-26 | `AHEAD_AUTO` extremum search | Published (off by default; measured slightly worse than a fixed 3) | same as above §6.32④ | same as above |
| PC-27 | Non-blocking read-back "one beat late" | Published (not changed by default) | same as above §6.33 (lines 1174–1200) | same as above |
| PC-28 | Shared pool `GLOBAL_POOL` | Published (default 0); **the 128 results ran on a later WIP-fixed pool and do not vouch for the old pool in the release tree** | `sources/moe-cache-score-aware-prd.md` §Current policy | 128-series summary (§9.4) |
| PC-29 | Position weights `LAYER_AWARE`/`LFU_POS` | WIP only (uncommitted) | same as above §Current policy | `global-pressure128-v2.summary.json` |
| PC-30 | Shared-pool memory-safety boundaries (zero slot/reading/int64 stride/graph UID) | WIP only (uncommitted); the incident and the fix are evidenced | same as above §Memory-safety boundaries of the shared pool | `stride-memcheck.log`, `final-verification.json` |
| PC-31 | 128 pressure: hits +20.77% but DMA almost unchanged | Still open | same as above §Earlier short-replay diagnostics, §Open issues to investigate | `global-pressure128-v2.summary.json` |
| PC-32 | Residency-time statistics `CACHE_LIFETIME` + 400-token baseline | WIP only (uncommitted) / the measurement definition has been written into the current PRD | same as above §400-token residency observation | `optimization400/lifetime-before.json` |
| PC-33 | All-source frequency gate candidate (rejected) | WIP only / rejected (uncommitted, paused) | same as above §Admission and attribution (does not change the default) | `optimization400/lifetime-after.json`, `fixed400/experiment.json` |
| PC-34 | Hot-backfill-only fix candidate (rotation/empty slot/eligible victim) | WIP only / paused (uncommitted) | same as above §Admission and attribution | `optimization400/logic-final.log`, `fixed400-final/experiment.json`, `final-r1.json` |
| PC-35 | Fixed-history 400-step control method and the baseline's own fluctuation | Kept (methodology) | — | `fixed400*/experiment.json` |
| PC-36 | Run gates and record integrity (lock/timeout/accidentally deleted logs) | Kept (methodology) | — | `lock-smoke.json`, `timeout-smoke.json`, `deleted-logs-incident.json` |

## 2. Stage one: static hot table + XT transition manifest (earliest form)

### PC-01 Static hot table (top-N experts by per-layer frequency)

- **Status/version**: Published. `LLAMA_MOE_PREDICT_STATIC` (default 32, see `SOURCE_TREE/ggml/src/ggml-backend.cpp:1708`) folds the static hot experts into every prediction; `LLAMA_MOE_PIN_STATIC` defaults to 0 (the current PRD §Current policy states explicitly that "static hot experts are not pinned by default"; the early historical implementation did pin by default).
- **Why it was tried**: A single token activates only 10/512 experts per layer, but there are stably high-frequency experts across layers; pinning the handful of "always hot" experts in VRAM first yields a hit floor without depending on any predictor. Note that pinning is **not free**: it occupies limited slots (pushing up VRAM usage and load/residency time) and makes that share of capacity unavailable for adaptation.
- **Technical mechanism**: The `static` section of the MANIFEST file (`MOEPRED1`/`MOEPRED2`) is "the N expert ids per layer in descending activation frequency" (`build_expert_manifest.py` comment: `static: n_layers * n_static * u16 expert ids (frequency desc)`). The early implementation pinned straight from that table into slots (log `pinned 3840 / 9216 / 12960 static hot expert copies`).
- **Experimental conditions and evidence**: The `experts-20260903-082346.csv` collected on 2026-09-03/04 is the training set (hold-out by graph_id, train ≤ 720); manifests `moe-predict-v1.bin` (layers=48 experts=512 trans=32 static=64), `moe-predict-static96.bin` (static=96). The original logs outside `LOCAL_EVIDENCE` are in `SOURCE_TREE`: `cpunode-run.log`, `long-run.log`, `fused-final.log`, `memprobe-run*.log`, all of which print `loaded prediction manifest ... static=64/96/128`. Statistical evidence: `LOCAL_MODELS/qwen38/traces/moe-static90-8192.csv` (1682 (graph,layer) records) sums to hits=31890, misses=18540 → cache hit rate 63.2%. **`pred_hits=0` only means that the file contains no record of "a routed id hitting the prediction bitmap"; it is not grounds for attributing all hits to pinning**: the hit count is a read hit over the resident set (the set formed jointly by pinning and demand admission), and this CSV alone cannot separate the two parts.
- **Observations and conclusion boundaries**: The static table provides a hit floor that **does not depend on a predictor**; but it is insensitive to distribution drift, and pinned slots cannot adapt. The hit rate above holds only for "the build of that time, the per-layer/per-tensor cache capacity of that time, and the weight residency scale of that time", and cannot be extrapolated.
- **Keep/abandon reason**: Kept as **part of the cold-start seed and the prediction union** (`PREDICT_STATIC` defaults to 32; `PIN_STATIC` as an optional experiment), abandoned "pin the big table by default" (consumes capacity, freezes the hot set).
- **Remaining/reopen conditions**: The end-to-end benefit of `PIN_STATIC` has not been accepted at the working point; reopening requires a paired experiment combined with `PREDICT_STATIC` at 400-token / 6144 MiB.

### PC-02 XT cross-token transition table (MOEPRED2, `PREDICT_XT`)

- **Status/version**: Published. `LLAMA_MOE_PREDICT_XT=1`; the code comment (prediction section of `ggml-backend.cpp`) states explicitly: "cross-token manifest (predict_xt): predict THIS layer of the next token".
- **Why it was tried**: A cross-layer (L→L+1) prediction gives only one layer of lead time; but the same layer has a "same-layer cross-token" reuse structure between token t→t+1, which can provide a full round (~48 layers) of lead time.
- **Technical mechanism**: `build_xt_manifest.py` emits `MOEPRED2` (number of trans rows = n_layers, intra-layer self-transitions): `trans[layer][src_expert] = top-N dst experts by count`, with sample pairs decode→decode (token t layer L → token t+1 layer L), `N_TRANS=32, N_STATIC=128`.
- **Experimental conditions and evidence**: Manifest `moe-predict-xt.bin` (layers=48 experts=512 trans=32 static=128). Run evidence: `ablation-lazy-moe2-xt.log` (2026-09-10 12:27, budget=2048 MiB, `predictor=1 prefetch=0 topk=26 static=0`, `pinned 3243 static hot expert copies`), `ablation-lazy-moe4/6/8.log` (same manifest, budget 4096/6144/8192). Prediction-coverage statistics: `moe-xt-12288.csv` sums to `pred_hits=6031 / pred_total=16340 = 36.9%`, and the same file's cache hits=32988 / misses=17442 = 65.4%. The column semantics come from the emission code: `pred_hits` = the number of routed expert ids that land on that layer's prediction bitmap, `pred_total` = the number of routing slots taking part in scoring (`k = ids_tensor->ne[0] = 10`); **this is a coverage definition, not an admission count and not a cache hit**.
- **Observations and conclusion boundaries**: XT gives 36.9% same-layer cross-token coverage; the row count of `moe-xt-8192.csv` (3026) differs from the other files (1682) (it appears to include extra graphs/stages), so its 10.0% coverage **cannot be compared directly with 12288**.
- **Keep/abandon reason**: The manifest path is kept as an explicit A/B control (PRD §Implementation Decisions states explicitly "offline XT manifest remains an explicit A/B control"); it is not the default (insufficient coverage, and the predictor was replaced by the online SMoE).
- **Remaining/reopen conditions**: No independent reopening condition; if this is redone in the future, the CSV row count and stage marker need to be written into the file name/header, to avoid another incomparable file.

### PC-03 CrossLayer transition table (MOEPRED1) and the offline control method

- **Status/version**: Published. `LLAMA_MOE_PREDICT=<path>` is a `MOEPRED1` manifest; a code comment points out that MOEPRED1 has no row for the last layer in XT mode (`trans_rows = n_layers - (MOEPRED1 ? 1 : 0)`).
- **Why it was tried**: Moving "adjacent-layer gates are predictable" from the paper to this model first requires quantifying offline "how much of the next layer's routing a transition table of the previous layer's experts can cover".
- **Technical mechanism**: `build_expert_manifest.py` uses a `(n_layers-1) * n_experts * n_trans` table of `(cand_id, count)` plus a per-layer static list, and **reports the expected hit rate inside the script via a hold-out replay**; `analyze_cross_token.py` puts three controls side by side: the persistence baseline (the token t's own set, i.e. the part LRU can already get), the cross-token transition table, and transition table ∪ static hot set (all held out by graph_id, train ≤ 720).
- **Experimental conditions and evidence**: `SOURCE_TREE` (data root `LOCAL_MODELS/qwen38/traces/experts-20260903-082346.csv`); `moe-predict-v1.bin` (trans=32 static=64). For runtime use see `long-run.log`/`phase-run.log` (`predictor=1`, `pinned 9216 static hot expert copies`).
- **Observations and conclusion boundaries**: **The script's built-in hold-out replay** is the most valuable methodological asset of this stage (judge the predictor's ceiling offline first, then go to the machine); but the replay rests on the assumption of "adjacent layers/adjacent tokens from the same collection run" and does not incorporate the numerical divergence after the CPU/GPU division of labour changed (see PC-31).
- **Keep/abandon reason**: Kept as a control; the default predictor was changed to the online SMoE.
- **Remaining/reopen conditions**: Reopening requires paired data for the four-link chain "predictor ceiling → runtime coverage → hits → throughput" at the same working point.

### PC-04 Static/XT stage performance and hits (including the PLE main-line intersection)

- **Status/version**: **Retracted (as a criterion)**; the numbers themselves are kept as history.
- **Why it was tried**: Contemporaneously with the PLE cache (the first item of the development main line, see `01-host-and-devpart.md`), to verify whether "static table + transition table + per-layer bundle cache" is worth continuing.
- **Technical mechanism**: Under `lazy-mode`, a per-layer expert bundle cache + static pin + XT manifest load (`prefetch=0`, i.e. at the time only residency was done, no prefetching).
- **Experimental conditions and evidence**: The 2026-09-10 ablation series (`LOCAL_MODELS/qwen38/traces/ablation-lazy-moe4.log|moe6|moe8`), the same manifest `moe-predict-xt.bin`, budget 4096/6144/8192 MiB, Generation **15.2 / 16.2 / 16.7 t/s**; in `ablation-lazy-moe4.log` the per-tensor hit rate is 21.9%–56.8% (rising with layer). The control CSVs for PLE-only / MoE-only / combined are `ablation-ple-only.csv`, `ablation-moe-only.csv`, `ablation-lazy-ple.csv`.
- **Observations and conclusion boundaries**: These t/s come from lazy mode, prefetch off, and an old build, and **cannot** serve as a control for the current host path (19 t/s order of magnitude, SPLIT=1, prefetch+SMoE); `sources/moe-decode-perf-plan.md` §7 specifically registers the old conclusions that this document system has retracted.
- **Keep/abandon reason**: Kept as the historical anchor for "static table + XT can only reach this level"; the criterion has been retracted (later changed to SPLIT=1 + direct-read + SMoE prefetch + frequency eviction + backfill).
- **Remaining/reopen conditions**: To reuse that stage's numbers, they must first be re-measured at the same 400-token / 6144 MiB / all-in-memory working point.

## 3. Stage two: the Fate online cross-layer gate, and its frustration

### PC-05 Fate online cross-layer gate (`PREDICT_FATE`)

- **Status/version**: Published (default **off**). Enabled by `LLAMA_MOE_PREDICT_FATE=1`; the controlled baseline of `tools-run.py` hard-codes `"LLAMA_MOE_PREDICT_FATE": "0"` explicitly.
- **Why it was tried**: XT/the static table has insufficient coverage (PC-02), and the transition table is an offline artifact. The key transferable conclusion of the Fate paper is that "adjacent-layer gate inputs suffice to support low-overhead prefetching of the next layer's experts", so it was changed to **online**: multiply the current layer's gate input by the next layer's gate weights to obtain next-layer candidates.
- **Technical mechanism** (PRD §History v1 §Implementation Decisions, snapshot lines 223–248): In Fate mode, the named hidden tensor `ffn_moe_gate_input-i` is copied to the host when layer `i`'s routing becomes available, multiplied on the **CPU** by layer `i+1`'s gate weights, and then compressed into a byte-limited candidate set; the native router and the model graph are unchanged. Counters: `fate_predictions`, `fate_gate_inputs`, `fate_gate_ms` (gate computation time is listed separately and not mixed into prefetch coverage).
- **Experimental conditions and evidence**: Code path `moe_cache_predict_fate` (introduced in `2f1a363c8`, 2026-09-11; refined in `d78c8bd42`, 2026-09-13). **Evidence gap**: searching the whole `SOURCE_TREE` for `fate=1` gives **zero hits**, and searching `LOCAL_MODELS/qwen38/traces/*.log|*.txt` (i.e. all log text of the XT/static stage and the SMoE stage) for `fate=1`/`fate_predictions=[1-9]`/`fate_gate_inputs=[1-9]` gives **likewise zero hits**: every retrievable `policy=` line reads `fate=0 fate_predictions=0 fate_gate_inputs=0 fate_gate_ms=0.00`. Therefore "Fate cannot squeeze useful prediction information out of the hidden layers on this model" currently **exists only as a user recollection, with no matching protocol/raw run log found** ([user recollection, awaiting a matching original record]). This report does not attribute a cause to "why there is no record".
- **Observations and conclusion boundaries**: What can be established is: (a) the Fate online path **exists in code** and compiles in this release tree's commit; (b) it is **explicitly switched off at the current working point** (`fate=0`, and §History v1 §Further Notes keeps only the transferable conclusion "adjacent-layer gate inputs can support low-overhead prefetching", marking the shallow-favoring behaviour as "model- and budget-dependent"); (c) the existing documents do not state Fate's hit rate as a conclusion. What cannot be established is its measured hit rate and the quantitative basis for the point at which it was abandoned.
- **Keep/abandon reason**: Keep the code and the switch (the design goal of a replaceable predictor); off by default and not a recommended path.
- **Remaining/reopen conditions**: Before reopening, a raw log with `fate=1` must first be supplied (same 400-token working point, containing `fate_gate_inputs/fate_predictions/fate_gate_ms`), otherwise any statement about Fate's accuracy can only be a recollection.

## 4. Stage three: SMoE — constructing an approximate hidden state from the shared expert

### PC-06 SMoE teacher test (user-recollected 99%)

- **Status/version**: **A measured fact recollected by the user**; the method is unknown and no matching protocol/raw log has been found.
- **Why it was tried**: After the Fate setback, the approximate input to the next layer's gate was instead constructed from "this layer's input + the shared expert's output" (without waiting for the routed MoE output), to first look at the hit-rate ceiling under teacher-forced conditions.
- **Technical mechanism**: `ffn_smoe_hidden-i = ffn_input-i + cached_gpu_routed-i + shared_expert-i`, then multiplied by the next layer's native gate matrix to take candidates (PRD §History v1 §Implementation Decisions).
- **Experimental conditions and evidence**: **User recollection: teacher test hit rate 99%**. In the three snapshots `sources/handoff.md`, `sources/smoe-nk-degradation-plan.md`, `sources/moe-cache-score-aware-prd.md` and in the `SOURCE_TREE` logs, **no protocol/raw log matching that 99% was found** (**the number of sampled layers, the number of candidates, whether the denominator is the top-10 slots or the expert set, and whether the static hot set is included are all unknown). This section keeps it as a "user-reported fact" and marks it [protocol unconfirmed].
- **Observations and conclusion boundaries**: Whether the 99% and the offline `recall@10 = 68.53%` of `sources/smoe-nk-degradation-plan.md` §7 (k=1, teacher-forced single step) **are under the same protocol is unconfirmed** (the denominator, the candidate set, and whether the static hot set is included are all unknown), so the latter **must not** be used to refute the former, and the 99% **must not** be conflated with an online hit rate such as the runtime `hits/misses` (see PC-11).
- **Keep/abandon reason**: Kept as a main-line fact (the SMoE route started from it).
- **Remaining/reopen conditions**: Finding the raw protocol matching that 99% would let it be promoted to a citable measurement; otherwise it can only be cited as "motivation" and cannot serve as an accuracy baseline.

### PC-07 SMoE N+k offline degradation (recall@10)

- **Status/version**: Plan document archived; the domain of applicability of the conclusion is limited by that document's §reservations (a document being archived does not mean the implementation was released).
- **Why it was tried**: Before the online implementation, two questions needed answers — what the N+1 accuracy of SMoE alone is; and how many layers the prefetch window may be opened for, given how fast the degradation proceeds.
- **Technical mechanism**: `simulate_smoe_nk.py`: for layer L take `h = ffn_input(L)` (same graph, same token as L+1), `block = (ffn_moe_weighted(L) * resident_mask).sum(expert) + ffn_shexp_gated(L)`, `smoe_hidden = h + block`, then use the **true gate of L+k** to compute logits, take top-10, and compare with `ffn_moe_topk(L+k)` for recall@10.
- **Experimental conditions and evidence**: Data `LOCAL_MODELS/qwen38/traces/standard-smoe-8-20260911` (8 prompts, one `"ffn_moe_input-"` line added to the whitelist, `qwen4exp.cpp` needs no change). Table (`sources/smoe-nk-degradation-plan.md` §7, snapshot lines 181–192):

| Variant | k=1 | k=2 | k=3 | k=4 |
|---|---:|---:|---:|---:|
| oracle (use the true input of L+k) | 100.00% | 100.00% | 100.00% | 100.00% |
| full (h + all routed + shared) | **68.53%** | 59.94% | 55.70% | 53.77% |
| fifo (with a 22 slots mask) | 68.29% | 59.72% | 55.47% | 53.65% |
| shared_only (h + shared) | 68.07% | 59.52% | 55.29% | 53.47% |
| input_only (only h) | 67.90% | 59.34% | 55.22% | 53.44% |

  Sample sizes 56400/55200/54000/52800. oracle is 100% for all of k=1..4, showing that on this data and at this step the matching of "true input + true gate + topk/data fetch" is self-consistent; **this only covers that matching link, and neither proves the whole online pipeline correct nor proves conclusions under a cumulative rollout**.
- **Observations and conclusion boundaries**: N+1 = 68.53%, and degradation is gentle (k=2 is still 59.94%); on this data **using only `ffn_input` gives 67.90%, 0.63pt away from full**. The boundary (consistent with PC-08, not opposed to it): **this is a teacher-forced single-step measurement** (each layer's `ffn_input` is taken from the true trajectory), so it describes "the single-step quality of a per-layer predictor under true input" and **is not a cumulative-rollout conclusion**; "the prediction only serves as a hint, and error does not accumulate across layers" is the argument given by the plan document ([that argument is a conditional derivation, not verified by a cumulative rollout]), and this chapter does not treat it as an established fact. The 0.63pt gap holds only on this dataset and **is not enough to assert that the routed/shared terms can be deleted straight from the implementation**.
- **Keep/abandon reason**: The offline result supports continuing to study looking two layers ahead, rather than proving online usability. The online side did not delete the routed/shared dependency on this basis; it is still the sum of three terms (PC-10).
- **Remaining/reopen conditions**: To switch to an online definition, a runtime coverage run with `PREDICT_STATIC=0` must be added on the same data and its denominator aligned with the offline recall.

### PC-08 Early FIFO pool simulator and its self-stated limitations

- **Status/version**: Plan document archived (a document being archived does not mean the implementation was released) / superseded by PC-07.
- **Why it was tried**: First answer "given an intra-layer FIFO pool capacity, what are the hits and the reconstruction error".
- **Technical mechanism**: `simulate_smoe_from_standard_trace.py` reproduces an independent causal FIFO pool per layer (slots ∈ {16,22,32,44,64}), zeroes the contribution of non-resident experts, reconstructs that layer's `l_last` and compares cosine / relative L2 against the true value, and collects `fifo_pool_hit8`, `fifo_ranked_recall8/16`.
- **Experimental conditions and evidence**: `sources/smoe-nk-degradation-plan.md` §2 (snapshot lines 65–83) describes its own mode as `teacher_forced_local_smoe`, "Each layer uses the standard run's true pre-FFN residual; this is not a cumulative counterfactual rollout", and states explicitly that "causal SMoE rollout needs to feed the approximate residual into the next layer and re-run the graph". Data directory `standard-smoe-sim-20-20260910b`, aggregate `standard-smoe-20-20260910b-summary/report.md`.
- **Observations and conclusion boundaries**: This simulator can measure **intra-layer** placement quality, and **cannot** prove that the later layers' routing is preserved under a cumulative rollout; its conclusions hold only in the sense of "intra-layer loss".
- **Keep/abandon reason**: Superseded by the N+k curve of PC-07 (which answers the lead-time question head-on). Its limitation statement is kept as a methodological citation.
- **Remaining/reopen conditions**: A cumulative definition would require actually feeding the approximate residual back into the graph (i.e. implementing and running a cumulative rollout), which has not been done.

### PC-09 The `ffn_input` gap and the two ways to fill it

- **Status/version**: **WIP only**: the collection-side change exists only in the side-branch checkout `OTHER_SOURCE_TREES/qwen4exp-smoe-trace` and is not part of the release tree; its plan document making it into the snapshot does not mean the implementation was released.
- **Why it was tried**: The approximate gate input for SMoE needs the router's input (the output of `build_hc_mix(hc_after_attn, hc_ffn_*)`), while the original whitelist only dumped logits/topk/weighted/shexp and the like.
- **Technical mechanism**: Two paths. (a) Add a dump: adding `"ffn_moe_input-"` to the whitelist (`ggml/src/ggml-backend.cpp:1607` snapshot line number) suffices, and `qwen4exp.cpp` needs no change at all, because `build_moe_ffn` has long named the router input `ffn_moe_input` (`llama-graph.cpp:1973`); (b) reconstruct offline from `hc_after_attn` using the `build_hc_mix` formula.
- **Experimental conditions and evidence**: `sources/smoe-nk-degradation-plan.md` §3 (snapshot lines 84–99) records the result of (b): **tried, not verified as passing, off by roughly 20× in magnitude** (that checkout's hc implementation may differ from the current working tree). In the end (a) was adopted.
- **Observations and conclusion boundaries**: The cost of the added dump is re-running 20 requests × ~1 minute; the cause of the reconstruction method's failure is not settled ([inference, unconfirmed]: it may be related to differences in the checkout's hc implementation).
- **Keep/abandon reason**: Keep (a); abandon (b).
- **Remaining/reopen conditions**: None.

### PC-10 SMoE online implementation and its fixed overhead

- **Status/version**: Published. `LLAMA_MOE_PREDICT_SMOE=1`, requiring split mode and restricted to the single-token decode graph.
- **Why it was tried**: Move PC-07's offline conclusion online: do just one tiny `W_gate(L+1) @ smoe_hidden(L)` matmul and feed the candidates straight into the prefetch queue.
- **Technical mechanism**: The approximate hidden **is still the sum of three terms** `ffn_smoe_hidden-i = ffn_input-i + cached_gpu_routed-i + shared_expert-i` (PRD §History v1 §Implementation Decisions). **The PC-07 conclusion that "using only `ffn_input` is enough" was not implemented**: the code was not changed to an input-only path. The side graph computes the gate matmul + topk on the device side; on the runtime side `moe_cache_smoe_enqueue/drain` does a per-layer `event_synchronize` read-back. Counters `smoe_predictions`, `smoe_logits`.
- **Experimental conditions and evidence**: `SOURCE_TREE/L2048-err.txt`, `L6144-err.txt` (21/64 slots per tensor) and the `T2048_*`, `T6144_*` groups, log line `[MOE-CACHE] smoe per DECODE graph: total=24.64/26.09/27.67/26.80 ms (event_wait=0.85/0.94/1.16/0.89 ms process=1.25/1.44/1.42/1.44 ms) nonblock=1 deferred=1603–1608 ahead=2`. 400-token working point: `smoe_predictions=18354`, `smoe_logits=9397248` (`lifetime-before.json`/`final-r1.json`). A historically worse configuration (`a400a/b-err.txt`) reads `total=284.98/306.46 ms (event_wait=8.70/11.44 ms process=20.64/21.62 ms) deferred≈18343`.
- **Observations and conclusion boundaries**: The **bulk of SMoE's fixed overhead is the read-back/synchronization, not the matmul**: `sources/smoe-nk-degradation-plan.md` §214–231 (snapshot) records "per-layer `event_synchronize` measured at 17 ms/token, while the CPU processing part is only 2 ms". This is exactly why PC-27 (non-blocking read-back one beat late) and PC-25 (AHEAD default 3) exist.
- **Keep/abandon reason**: Kept (`PREDICT_SMOE=1` is part of the current recommended configuration).
- **Remaining/reopen conditions**: `event_wait` varies with ahead/distribution, and no end-to-end bucketing optimization has been done; for reopening conditions see PC-27.

### PC-11 Conflation of the static hot set with the SMoE accuracy definition (98–99% / 73%)

- **Status/version**: The definition has been clarified; the specific numbers await matching original records.
- **Why it was tried**: It was necessary to judge "whether SMoE prediction is actually accurate", while the runtime side offers only a single merged counter.
- **Technical mechanism/definition**: In `prefetch_predicted/prefetch_required`, `prefetch_required` is the number of routing slots taking part in scoring and `prefetch_predicted` is the number of routed ids landing on the prediction bitmap; and **the bitmap itself is the union of the SMoE candidates ∪ the static hot set (`PREDICT_STATIC`, default 32)**. So this ratio is neither "the SMoE accuracy" nor an admission count (`prefetch_experts` is the total number admitted through the common write path, and it includes non-predicted filling).
- **Experimental conditions and evidence**: `sources/smoe-nk-degradation-plan.md` §6 (snapshot lines 163–172) states in the original: early on this ratio (73%, and 98–99% even earlier) was taken to be the SMoE accuracy, and "**it is not**", and it gives the isolation method (offline simulation, or runtime `PREDICT_STATIC=0`). Measured at the 400-token working point: `prefetch_predicted=24971 = prefetch_ready=24971`, `prefetch_required=183540` → **13.6% coverage** (`optimization400/lifetime-before.json`); for the list of counter semantics in the 128 pressure/auto series see the `counter_semantics` field of `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json`.
- **Observations and conclusion boundaries**: 13.6% and 73% are **two readings of the same definition under different configurations** (the candidate set, `TAKE_MAX`, predictor combination and working point all differ) and **cannot be converted into one another**; still less can either be used to refute the 99% of PC-06 (that is a teacher-forced definition).
- **Keep/abandon reason**: Kept as a definition specification; every conclusion about predictor quality must state clearly "what the denominator is".
- **Remaining/reopen conditions**: Reopening requires printing two counters under a fixed configuration — "SMoE-only bitmap coverage" and "coverage after ∪ static hot" (currently only the merged value is printed).

## 5. Stage four: engineering the online cache — volume and cutoff

### PC-12 Rising prefetch volume, rising waits, and the contention hypothesis

- **Status/version**: Published (switch); the conclusion is "volume is a cost, not a benefit".
- **Why it was tried**: The hit rate rises monotonically with prefetch volume, so intuitively "transfer more, hit more, should be faster".
- **Technical mechanism**: The same binary, the `tools-run.py` configuration + `NONBLOCK=1 AHEAD=2`, varying the prefetch volume (MB/token).
- **Experimental conditions and evidence** (`sources/handoff.md` §6.6①, snapshot lines 209–232; **historical report only, no raw log names recorded**):

| Prefetch volume | Hit rate | total | t/s | Host-side `flag_input` |
|---|---|---|---|---|
| 161 MB/tok | 40.2% | 70.0 ms | **14.3** | 15.8 ms |
| 254 MB/tok | 46.9% | 77.7 ms | 12.9 | 23.3 ms |
| 464 MB/tok | 58.2% | 94.1 ms | 10.6 | 37.5 ms |

  Marginal ≈ **0.073 ms/MB**; taking the reciprocal of that marginal time gives **≈ 13.8 GB/s** — **this is the conversion of "each extra 1 MB costs 0.073 ms", not a measured hardware PCIe bandwidth** (the snapshot contains no bandwidth profiling/counter evidence). The **explanation** the snapshot gives is: the growing term is not the copy call itself (80 bytes) but the `ggml_backend_event_synchronize` upstream of `moe_cache_prefetch_layer`, i.e. the prefetch DMA and the GPU's own memory traffic contend for PCIe → the GPU slows down → the host `event wait` grows longer ([inference: the contention mechanism was not directly confirmed by profiling]).
- **Observations and conclusion boundaries**: In the excess region the hit rate and throughput move in **opposite** directions; "the hit rate must come from being more accurate, not from more bytes". The numbers come from an early configuration (non-SPLIT, non-auto budget) and their absolute values cannot be carried directly to the current working point; the snapshot gives neither bucketing of `event wait` nor bandwidth counters, so "PCIe saturation/contention" remains an explanatory conclusion.
- **Keep/abandon reason**: Kept as mechanism evidence (and as the motivation for dual gating).
- **Remaining/reopen conditions**: The prefetch-volume sweep has not been redone at the current working point.

### PC-13 Rank cutoff `TAKE_MAX` (implicit → explicit)

- **Status/version**: Published, default 2.
- **Why it was tried**: The deeper a prediction rank is, the less accurate it is, yet every rank costs the same bytes, so "how many ranks to take" has to be set explicitly.
- **Technical mechanism**: In `moe_cache_smoe_process`, `take = min(n_slots, n_topk, take_max)`; the old implementation had `take_max = n_used + 2` (=12), which implicitly acted as "only take the top N ranks", while `n_used` is assigned only inside the partition hook, so **for the first token / for layers that never passed the hook, `n_used==0` → the cutoff silently disappears (bug, fixed)**; now `rank_cut = LLAMA_MOE_TAKE_MAX` (default 2), independent of `n_used`.
- **Experimental conditions and evidence**: §6.6③ (snapshot lines 233–245) adds another ordering conclusion: "truncate first, then skip the already-resident" is **measured better than** "skip the already-resident and keep filling the budget further down" (the latter moves the admitted set from ranks 1–4 to ranks 5–10, yielding fewer hits for the same bytes). Decay by rank: cutoff 1 → 4.69 hits/MB, cutoff 2 → 5.25 (separately measured 3.05), cutoff 3 → 2.21, cutoff 4 → 1.78 (§6.6②).
- **Observations and conclusion boundaries**: `cutoff 2 → 5.25/3.05` differ by nearly 1.7× between the two measurements, showing that this metric is noisy and that only the monotone relation between 1/2 and between 3/4 is stable.
- **Keep/abandon reason**: Kept; the default of 2 continues to be used at the later 400-token working point (`policy_env` of `fixed400*/experiment.json`).
- **Remaining/reopen conditions**: None.

### PC-14 Deduplication (resident / pending / list / admit / readmit)

- **Status/version**: Published.
- **Why it was tried**: The hot-spot hit count is extremely high, so it had to be shown that "hitting repeatedly" does not turn into "shuttling repeatedly".
- **Technical mechanism**: `expert_slot[e] >= 0` is **the authority for deduplication**: the slot is assigned at admission, **before** the copy is issued, so a duplicate request within the same token/same window is always rejected; counters `dup_resident` (the candidate is already in the resident set), `dup_list` (duplicate within the candidate list), `dup_admit` (self-copy into the same slot), `dup_pending` (duplicate request in flight), `readmit` (reinstall after eviction).
- **Experimental conditions and evidence**: §6.6④ (snapshot lines 246–258) hard numbers: candidates 2852 = `dup_resident` 1496 (52.5% of candidates cost no bytes) + transfers 1356; `dup_list=0`, `dup_admit=0`, `dup_pending=0`; `readmit=86` (eviction churn 6.3%). 400-token working point: `dup_resident=29682/28941/30221`, `dup_pending=0`, `dup_admit=0`, `readmit=4734/4475/164` (see `lifetime-before.json`, `final-r1.json`, `lifetime-after.json` respectively).
- **Observations and conclusion boundaries**: Deduplication is a **request-level** guarantee and does not mean "the transferred volume is already optimal" — `readmit` shows that a considerable share of bytes still comes from reinstallation after eviction (see §10.1).
- **Keep/abandon reason**: Kept (directly related to PC-31's "hits up but DMA not down", and one of the grounds for ruling that phenomenon out).
- **Remaining/reopen conditions**: For the items still open see PC-31 (the correspondence between hits/deduplication/reinstallation/actual transfer is not yet closed).

### PC-15 Capacity × cutoff interaction

- **Status/version**: Published (parameter conclusion); the key point has not been re-confirmed.
- **Why it was tried**: Whether the optimal cutoff moves with the cache capacity.
- **Technical mechanism**: The same configuration, sweeping 2048 and 6144 MiB and cutoff 1/2/4/6/8.
- **Experimental conditions and evidence**: §6.6⑤ (snapshot lines 246–258): at 30 tokens there is no difference between 2048 and 6144 (**the big cache was not filled up**: 6 GB ≈ 67 slots/layer, only ~2 are admitted per token per layer, so 30 tokens cannot fill it; `readmit=0` proves there was never any eviction). At 150 tokens (which does fill up): 2048 is optimal at cutoff 1 (14.6 t/s); 6144 plateaus at 2–4 (14.8–14.9), 6 → 14.5, 8 → 14.1 → **the optimal cutoff moves upward with capacity** (within-group noise ±0.6 t/s, the key point not yet re-confirmed). VRAM peak: 2048 → **9132 MiB**; 6144 → **13278 MiB** (limit 15360, guard 1024).
- **Observations and conclusion boundaries**: This is an early working point (the 14–15 t/s era); the capacity sweep has not been redone at the current 400-token / 6144 MiB / 19 t/s order of magnitude.
- **Keep/abandon reason**: Kept as evidence that "capacity determines the acceptable depth".
- **Remaining/reopen conditions**: Reopening requires a capacity×cutoff grid with ≥3 repetitions at the same working point.

### PC-16 No benefit from prefetch worker count + correction that `INSERT_WORKERS` spins idle

- **Status/version**: Published (correction).
- **Why it was tried**: Whether prefetching is slow because of insufficient parallel submission.
- **Technical mechanism**: Changing `LLAMA_MOE_INSERT_WORKERS` (1/3/9).
- **Experimental conditions and evidence**: §6.6①: at low volume 1→3 workers = 13.3→13.6 t/s; at high volume = 9.9→10.5 (+6%); the pinned ring ceiling is only a few ms. §6.7 open items corrects this further: **`INSERT_WORKERS` is a no-op under `PREFETCH=1`** (all the worker's spawn points require `!prefetch`) — the prefetch copy is **submitted inline** by the host thread, so the earlier conclusion that "workers are not the bottleneck" is **void**.
- **Observations and conclusion boundaries**: Submission is currently inline on the host thread, so worker count is not an effective variable under this configuration. Sampling shows waiting; the specific PCIe/GPU contention mechanism has not been isolated (PC-12).
- **Keep/abandon reason**: Kept as a correction record, so that later readers do not sweep the worker count again.
- **Remaining/reopen conditions**: None (unless the worker submission model is restored).

## 6. Stage five: eviction score, hot regions and adaptation

### PC-17 Eviction score: gate-softmax MRS → ground-truth routing frequency

- **Status/version**: Published. `LLAMA_MOE_EVICT_SCORE` (0 = true frequency, 1 = old mrs), `LLAMA_MOE_HOT_HALFLIFE` (default 512 token base-period halving).
- **Why it was tried**: The ceiling on hits is decided by "whether what is in the cache is really the hot experts", and the eviction score is the only input that decides who gets cached.
- **Technical mechanism**: The old score `mrs_score` = EMA of the top 20 of the gate's full softmax; the new score = `use_count` accumulated at zero cost from `part.ids` inside the partition hook, with sliding-window base-period halving to prevent "hot-set freeze".
- **Experimental conditions and evidence**: §6.7 (snapshot lines 259–343). Measuring the old score against true routing: `mrs_topC = 0.3%` (at the same capacity, taking the top C ranks ordered by mrs covers only 0.3% of the true routing). Example of the three exit diagnostic lines (same section):
  `[MOE-CACHE] hot-set oracle: routed=… layers=48 slots/layer=C | oracle_topC=66.9% resident_set=53.0% mrs_topC=0.3% actual_hit=48.9%`.
  400-token (`--ignore-eos`, statistics taken only over the last 150 graphs = steady state):

| Configuration | Steady-state hit rate | Steady-state ms/token | Steady-state t/s | Reported t/s |
|---|---|---|---|---|
| 2GB + mrs score (old) | 30.5% | ~119 | 8.4 | 13.5 |
| 2GB + frequency score | 53.1% | 51.5 | 19.4 | 17.7 |
| 6GB + frequency score | — | — | — | 17.8 |
| **6GB + frequency score + backfill 8** | **80.8%** | **43.0** | **23.3** | 19.9 |

- **Observations and conclusion boundaries**: `oracle_topC` is **content-dependent** (a policy change → a change in the CPU/GPU division of labour → tiny numerical differences → greedy long runs diverge → **the oracles of different runs cannot be compared directly**, each can only be compared against its own oracle). The raw log is not named in the snapshot ([historical report only]); runtime per-rank/oracle lines of the same mechanism can be seen in `SOURCE_TREE/a400a-err.txt`, `a400b-err.txt` (`oracle_topC=83.4%/77.9%`, `mrs_topC=0.3%/0.2%`, `actual_hit=67.5%/61.8%`), but that is **another run**, and the numbers differ from the table above.
- **Keep/abandon reason**: Kept (the largest gain on this route).
- **Remaining/reopen conditions**: The time-order sensitivity of the `EVICT_SCORE` A/B; reopening requires comparison inside the same oracle.

### PC-18 Hot-region backfill + cold-start seed + idle fill

- **Status/version**: Published, all off by default (`HOT_BACKFILL=0`, `HOT_FILL_BOOT=0`, `HOT_IDLE=0`, `PIN_STATIC=0`).
- **Why it was tried**: Prefetching is limited by the cutoff deadline and the budget and cannot fill a large cache; whereas experts that are "hotter than the currently coldest resident" could have been added at zero risk.
- **Technical mechanism**: `moe_cache_hot_backfill` at the **end of the graph** (after all splits, outside capture) adds hotter experts from the candidate table, **self-terminating** (zero overhead once converged); copies go on a side stream and their completion is reaped by **polling** `slot_events`; a two-stage budget `HOT_FILL_BOOT` (startup period) → `HOT_BACKFILL` (steady state); with `HOT_IDLE=1` a background thread keeps filling while the model is idle (judged as no compute activity for 150 ms), all paths check `graph_active`, the mutex covers only the bookkeeping section, and transfers are never waited on. Seeding: `manifest.hot` → `PIN_STATIC=N` (previously dead code), at finalize N are seeded into each layer and **given no eviction protection**.
- **Experimental conditions and evidence**: §6.7. Positive: the table above, 6GB+backfill 8 → steady-state hit 80.8%, 23.3 t/s. Negative: **under 2GB backfill is harmful** (thrash, `readmit` 3383→10743, steady-state hit 56% and slower) → backfill must be paired with enough capacity. 400-token runtime: `hot_fill=3190` (baseline), `hot_idle=4`; `admit_reject_freq` appears in later candidates (PC-33).
- **Observations and conclusion boundaries**: Backfill is **not** subject to the cutoff gate or the admission budget (the numeric budget only constrains the `prediction` path); this is a design choice, not an oversight.
- **Keep/abandon reason**: Kept (the recommended configuration includes `HOT_BACKFILL=8`).
- **Remaining/reopen conditions**: `HOT_IDLE` has not been verified end to end (it would need an interactive "generate→idle→generate again" session); the reopening condition is that such a session can be run.

### PC-19 Per-rank accuracy / per-byte efficiency

- **Status/version**: Published (diagnostic output).
- **Why it was tried**: To supply data for the cutoff (which rank is still worth taking).
- **Technical mechanism**: Count, by rank in the full candidate table, "the fraction of routing that hits that rank", and at the same time record that rank's measured hits/MiB (`y`) and its estimate (`ye`, estimated as "accuracy × hits per admission" when there are no bytes), through a monotone envelope.
- **Experimental conditions and evidence**: §6.7 gives (400 token, fixed cut=2, samples = 48 layers×400):
  `r1 77.9% y17.93  r2 66.5% y16.65  r3 58.5%  r4 52.1%  r5 46.7%  r6 42.4%  r7 38.0%  r8 35.9%  r9 31.5% r10 28.2% r11 26.4% r12 23.3% r13 21.5% r14 19.7% r15 18.2% r16 16.6%` (y = measured hits/MiB; ranks that were not admitted have no bytes → not measurable).
  The same-format raw runtime text can be seen in `SOURCE_TREE/a400a-err.txt` (`r1=83.5%/y27.55 …`), `a400b-err.txt` (`r1=79.4%/y20.72 …`), containing `ye` and a sample count of 18354.
- **Observations and conclusion boundaries**: **Per-rank yield is not monotone** (a code comment records r3 measured at 7.9 while r4 measured 17.0; in `a400b` r3=10.41 and r4=11.76, the same direction), so any implementation that "lets ranks through one by one according to yield" must use a monotone envelope, otherwise noise will be taken for signal.
- **Keep/abandon reason**: Kept as a diagnostic baseline.
- **Remaining/reopen conditions**: None.

### PC-20 Three attempts at adaptive truncation (all lost to a fixed cut)

- **Status/version**: Published, all off by default (`RANK_ADAPTIVE`, `YIELD_MIN`, `ADMIT_BUDGET_MIB`).
- **Why it was tried**: To turn PC-19's table into an automatic cutoff.
- **Technical mechanism and failure modes** (§6.7):
  1. Accuracy threshold (`RANK_ADAPTIVE`): chicken-and-egg — a narrow cutoff only offers a few ranks, so the statistics starve → the fix is to measure accuracy over the **full candidate table**, decoupled from admission;
  2. Measured yield threshold (`YIELD_MIN=4`, the user's proposal "admit when per-byte efficiency > 4"): r1/r2 measure 17.9/16.7 ≫ 4, but deeper ranks have no bytes → yield=0 is scored 0 → the cut sticks at 1. **An unexpected gain**: `cut=1` + the bytes saved = **20.6 t/s** > the 19.3 of `cut=2`;
  3. "accuracy × measured hits per admission" to estimate yield + a byte budget (`ADMIT_BUDGET_MIB`): 64 MiB → **12 t/s** (early layers eat the whole allowance and later layers starve; 64/48 = 1.33 MB < 1 bundle of 1.9 MB), 128 MiB → 19.0 t/s; **both are worse than 1/2**.
- **Observations and conclusion boundaries**: The three paths fail at different points (statistical starvation / no bytes / fragments too small), so "adaptive truncation" cannot rely on swapping the objective function alone; it must solve both "the availability of the estimate" and "the lower bound of the budget fragment".
- **Keep/abandon reason**: All three are paused; the default returns to a fixed `TAKE_MAX`.
- **Remaining/reopen conditions**: Reopening requires first guaranteeing a per-layer budget lower bound of ≥1 bundle, and decoupling the accuracy statistics from admission.

### PC-21 Occupancy observation: value threshold separated from rate control

- **Status/version**: Published (conclusion).
- **Why it was tried**: To explain "why the budget was not converted directly into speedup".
- **Technical mechanism/evidence**: The user's measurement in §6.7 (2026-09-12): bf8/bf16 raised the decode occupancy from 40–45% to 60%; the "100%" mentioned earlier was actually prefill saturation — **decode was never saturated**, so the old estimate of a "GPU 40 ms hard floor / 25 t/s ceiling" is **void**, and the true ceiling is higher. Moreover `ADMIT_BUDGET_MIB` pins occupancy **on a middle plateau** (what it throttles is prefetch admission → the growth of the GPU-side working set is throttled → it neither dips into the cold valley nor rises to the hot peak).
- **Observations and conclusion boundaries**: The conclusion is a division of design labour: "the value threshold (which ranks are worth it) and rate control (how many per token) must be separated — rate is handed to backfill, the threshold only governs value".
- **Keep/abandon reason**: Kept; the old ceiling estimate has been retracted.
- **Remaining/reopen conditions**: The occupancy curve has not been re-measured at the 400-token working point.

### PC-22 YIELD_AUTO extremum-searching threshold

- **Status/version**: Published, off by default.
- **Why it was tried**: A constant threshold cannot be right — the marginal value of a hit is **state-dependent**: when the CPU half has work, one hit ≈ 0.074 ms, and once the hit rate reaches ~80% the CPU half goes empty → the marginal value is ≈ 0; whereas the marginal cost of transfer always exists (≈ 0.08 ms/MB). The optimal depth lies where "marginal value crosses marginal cost", and it moves with the distribution/prompt/cache state/stage. (The three numbers — 0.074 ms/hit, 0.08 ms/MB, and "zero after 80%" — are the **local fits/observations** given by §6.8 at the then-current 400-token working point, not model constants; TREND_AUTO's V/P are their online estimates.)
- **Technical mechanism**: The decision variable = the per-byte efficiency threshold (hits/MiB); the objective = **median per-token latency** (robust against single-point noise); every `YIELD_AUTO_PERIOD` (default 16) graphs it compares the **medians** of the two windows before and after: if faster, continue in the same direction; if slower, reverse; the step is multiplicative ±10%, clamped to [0.5,32], and each probe logs (`[MOE-CACHE] yield-auto: probe …`). The hard constraint is still the deadline gate (PC-24), and it acts only on prefetching.
- **Experimental conditions and evidence**: §6.8 (snapshot lines 344–372): `prose: 67.8 → 38.7 ms (threshold 4.4 → 10.9), converges to cut=1, yield_min=11.82, 18.8 t/s`; `code: converges at a threshold of ~4.3–4.8 ms`. The same algorithm computes different thresholds on the two distributions (11.8 vs 4.5) — **the manual constant 4 is 3× too loose on prose**.
- **Observations and conclusion boundaries**: All of the benefit comes from "saving the bytes that should not have been transferred"; the controller itself has a back-and-forth probing cost (see the analogous cost in PC-26).
- **Keep/abandon reason**: Kept as an optional item; choose one of this and PC-23, both off by default.
- **Remaining/reopen conditions**: YIELD_AUTO has not been turned on for acceptance at the current working point.

### PC-23 TREND_AUTO model-driven threshold and budget

- **Status/version**: Published, off by default; `BUDGET_FRAC` defaults to 0 (does not set a budget automatically).
- **Why it was tried**: Extremum search is black-box hill climbing, slow to converge and carrying a probing cost; if the "hit value V" and the "transfer cost P" can be estimated directly, both the threshold and the budget can be computed in one step.
- **Technical mechanism**: Every graph samples `(ms, hit count, admitted MB)`, and over a `TREND_WINDOW` (default 32) sliding window performs a **centred least-squares** fit of `ms ≈ a − V·hits + P·MB` (3×3, Cramer's rule, ~50 flops/graph → no measurable effect on speed), smoothed with EWMA(0.75/0.25) (code `moe_cache_trend_fit` + `trend_V/trend_P` update). Guards (keep the old model when the data is untrustworthy, and never slow down because of them): window <16 points, hits spread <15% or MB spread <8%, determinant too small, solution out of range (V ∉ [0.005,2] ms or P ∉ [0.001,0.5] ms/MB) → reject that fit (counter `rej=`).
- **Experimental conditions and evidence**: §6.9 (snapshot lines 373–393): `trendV=0.0158 ms/hit  trendP=0.0936 ms/MB` → threshold `P/V = 5.92 hits/MB` (cut=4 on) → budget `frac × ms_hat / P = 116.5 MB` (frac=0.25; almost identical to the hand-derived 128 MiB fuse); `fits=191 rej=212`; speed 19.6 t/s (historical 18.8–19.9 for the same configuration, no regression). **Independent cross-validation**: P=0.0936 differs by 17% from the 0.08 ms/MB derived by hand from the capacity sweep in §6.6 — two independent paths give the same cost.
- **Observations and conclusion boundaries**: `fits/rej` being nearly 1:1 shows that the guards reject frequently (when the data spread is insufficient), so in short-window/low-fluctuation periods this model will stay at its old values for a long time; that is by design (better not to update than to update badly).
- **Keep/abandon reason**: Kept as the "adaptive version" recommended combination (`TREND_AUTO=1 + BUDGET_FRAC=0.25`), but **it did not enter the release defaults**.
- **Remaining/reopen conditions**: It shares `ms_hat` with PC-24, and the behaviour of the budget drifting with the working point has not been verified over a long run.

## 7. The dual gate's adaptive transfer threshold (source-code criterion)

This section answers: **which two criteria and which two adaptive quantities the "dual gate's adaptive transfer threshold" actually consists of in the source code**. Note that **the dual gate and the "all-source frequency gate" rejected in this round are not the same thing**: the frequency gate is an **extra admission condition added at admission time** (it compares the candidate's retention value with the **actually evictable victim**, see PC-33); it is neither a source of values for the value gate, nor does it change the criteria of the two gates themselves.

### PC-24 Gate 1 (value gate) and gate 2 (deadline gate)

- **Status/version**: Published. `LLAMA_MOE_PREFETCH_GATE` defaults to **1** (`prefetch_gate = true`, the init log prints `gate=1`); the model constants `LLAMA_MOE_GATE_COPY_US` default to 70 µs and `LLAMA_MOE_GATE_BW_GBPS` to 20 GB/s.
- **Why it was tried**: The higher the hit rate the slower (PC-12/PC-21) shows that "transferring more" is itself a cost; two mutually independent criteria are needed: **whether it is worth transferring** (value) and **whether there is time** (deadline). Looking at only one of them degenerates into a constant policy.
- **Technical mechanism (criterion 1: the value gate, deciding admission depth)**: `moe_cache_effective_rank_cut()` checks rank by rank
  `yield_est(r) ≥ yield_min`, where `yield_est` comes from `moe_cache_rank_yield_est()`: when measured bytes exist it uses the measured value (rank hits / rank bytes, in hits/MiB); when there are no bytes it estimates as `accuracy` × `hits per admission` ÷ `MiB per admission`; a **monotone envelope** is taken rank by rank (`env = min(env, yield_est(r))`), and as soon as the envelope falls below the threshold it stops and returns `cut ≥ 1`. **The adaptive quantity = `yield_min` (the hits/MiB threshold)**, whose value follows this order of precedence:
  1. `TREND_AUTO` with valid V and P → `trend_P / trend_V` (the break-even point), with 10% hysteresis (only updated when the change is >10%, to prevent jitter from regression noise), clamped to [0.5, 64];
  2. `YIELD_AUTO` → the extremum-search probe `yield_auto_cur` (multiplicative ±10%, clamped to [0.5, 32]);
  3. constants: `RANK_YIELD_MIN` / the accuracy threshold of `RANK_ADAPTIVE` / the fixed `TAKE_MAX` (default 2).
  **This gate answers**: "is the per-byte hit efficiency of one rank deeper still above the break-even point that makes the token faster".
- **Technical mechanism (criterion 2: the deadline gate, deciding whether this batch of copies is issued)**: `moe_prefetch_feasible(bytes, n_copies, layers_until_visit)`:
  ```
  eta_us    = (dma_inflight_copies + n_copies) * gate_copy_us
            + (dma_inflight_bytes  + bytes) / gate_bw_bps * 1e6
  deadline  = max(1, layers_until_visit) * layer_us_ewma
  可行  ⟺  eta_us ≤ deadline_us
  ```
  If not feasible → **drop that predicted transfer** and let the expert take the CPU path, avoiding blowing up the side stream queue. **The adaptive quantities = `layer_us_ewma` (per-layer scheduler-thread wall clock, 0.75/0.25 EWMA, updated online) and the in-flight amounts `dma_inflight_{copies,bytes}` (added on submit, subtracted on completion)**; `gate_copy_us`/`gate_bw_bps` are model constants (not adapted at runtime, only overridden by env). **This gate answers**: "can it arrive before it is needed". A source comment (above the function) explicitly rejects the scheme of "estimating capacity from the measured completion rate": that would **death-spiral** (it measures demand rather than bandwidth: fewer transfers → lower estimate → more dropping).
- **The two gates have different scopes of application (a key design decision)**: The deadline gate and the byte budget **act only on the prefetch (`prediction`) path**; hot backfill (`hot_backfill`), explicit warm, and static seeding are **not** constrained by the deadline gate (backfill has no deadline and can be done in bulk at the end of a graph). The third adaptive quantity is a **rate budget** (not a gate): `admit_budget_bytes = frac × ms_hat / P` (`TREND_AUTO`, clamped to [16,512] MB) or the manual `ADMIT_BUDGET_MIB`.
- **Deadline-gate bypass**: The call site also requires `!queue_on_worker`; a cold start with no valid layer history is let through directly (release baseline `ggml-backend.cpp:4015–4024,4105–4110`). The fact that a log says `gate=1` must not be taken to mean that every write passed the deadline check.
- **Experimental conditions and evidence**: Source `SOURCE_TREE/ggml/src/ggml-backend.cpp`: `moe_state` fields `prefetch_gate/layer_us_ewma/dma_inflight_bytes/dma_inflight_copies/gate_copy_us/gate_bw_bps` (around lines 1910–1925); `moe_prefetch_feasible` (around lines 4414–4425); the admission condition (around line 4512 `if (prediction && s.prefetch_gate && !queue_on_worker)`); the in-flight add/subtract (around lines 4646/4658, 6595); the `layer_us_ewma` update (around line 6149); `moe_cache_rank_yield_est` / `moe_cache_effective_rank_cut` (around lines 5432–5480); `moe_cache_yield_auto_step` (around lines 5537–5565); the TREND fit and budget (around lines 7700–7719). Behavioural evidence (historical reports): the `yield-auto: probe …` convergence of §6.8, the `trendV/trendP/budget` and `fits/rej` counters of §6.9, and the "budget rate limiting pins occupancy on the middle plateau" of §6.7.
- **Observations and conclusion boundaries**: The value threshold follows the source-code precedence of TREND's **P/V**, the YIELD probe, or a fixed threshold; the deadline gate uses the per-layer EWMA latency and the in-flight amounts, and the copy constants are not learned online. The two criteria act differently, but jointly they affect the subsequent cache state and the measurement samples, so they cannot be said to be statistically independent.
- **Keep/abandon reason**: Kept, with the deadline gate on by default (`gate=1`); the value gate degenerates by default to the fixed `TAKE_MAX=2`.
- **Remaining/reopen conditions**: `gate_copy_us=70 µs` / `gate_bw_bps=20 GB/s` are hand-set constants and have not been calibrated on this machine; to reproduce them outside PCIe 4.0 x8, they should be recalibrated and cross-checked against the 0.073 ms/MB of §6.6.

### Why "a higher hit rate" can be "slower" (observations, explanation and counterexample)

1. **Transfer volume and wait rise together** (PC-12): 161→464 MB/token, hit 40.2→58.2%, speed 14.3→10.6; upstream wait 15.8→37.5 ms. PCIe/VRAM contention was the explanation at the time and was not isolated by hardware profiling; the reciprocal of 0.073 ms/MB is not a measured bandwidth.
2. **Marginal value can change with state** (PC-22): about 0.074 ms/hit, with the gain shrinking after a higher hit rate, and about 0.08 ms/MB, are all local estimates, not universal thresholds or constant costs. The value gate therefore needs to be sensitive to the running state, not merely chase the hit rate.
3. **Hit count ≠ transfer count** (PC-14, §10.1): a repeated hit on a resident expert **produces no** new DMA (the deduplication authority is `expert_slot[e] ≥ 0`, `dup_pending=0`); DMA happens only at admission or when reinstalling after eviction.
4. **Counterexample**: cutting the bytes by 53% is still not enough to speed things up (PC-33: 26.9→12.6 GB, no improvement in latency), showing that the cost structure also contains waiting and critical-path components, and that one must not stare only at DMA bytes.

## 8. Stage six: split timing and "one beat late"

### PC-25 `SPLIT=1` silent miscalculation fix + `AHEAD` default 3

- **Status/version**: Published (the fix ships with this release tree's commit; before that `SPLIT=1` had been temporarily disabled as a safety lock).
- **Why it was tried**: `SPLIT=1` can overlap the MoE CPU half with the GPU half, but it first had to be shown to compute correctly.
- **Technical mechanism (line-level root cause)**: In the host path the GPU half's `ids_gpu`/`wgt_gpu` are **the inputs of this split** (host leaves of `ggml_set_input`), and their values are written by the partition hook **in the middle of the input loop** — the scheduler may **already have scheduled the copies of these two leaves first**, so the MoE GEMM receives **the previous layer's routing** and silently miscalculates (partial, and dependent on graph order). Fix: after the hook writes these two leaves it **immediately re-issues the copy once** (`tensor_copy` + `ggml_backend_tensor_copy`, a few hundred bytes).
- **Experimental conditions and evidence**: §6.32① (snapshot lines 1124–1147): for the Eiffel case, before the fix it **reproduced 3/3** (only 27 characters emitted), after the fix it **passed 4/4** (the answer lands in the 443–455 character range, its beginning matches the `SPLIT=0` reference, and it gives the same passage of the correct answer — **though not word-for-word identical to the reference**). Speed (8k/q8_0, auto=97 slots): cache off 9.2 t/s; `SPLIT=1 + auto + AHEAD=3` hits **90.9%**, **20.8 t/s**; `AHEAD=2` 83.5% / 18.4; `AHEAD=4` 87.8% / 20.2. The "**+126%** relative to cache off" given by the snapshot is **arithmetic inside that page's historical table** (a single run, not a strict A/B, and not the current 6144 MiB / 400-token working point), and **must not be taken as an acceptance conclusion for this working point**. The `SMOE_AHEAD` default was changed from 1 to 3.
- **Observations and conclusion boundaries**: **The earlier 20.3 t/s was a false speed with the bug and has been retracted**; §6.32 also lists the items still not done: the r1 anomaly at `AHEAD=1` (see PC-27), split producing nothing at 0 slots (e.g. `CACHE_MIB=64`) (it should degrade to all-CPU), auto giving only 37–40 slots at 256k+vision (a VRAM constraint rather than a bug), and an occasional `0xC0000005` during exit.
- **Keep/abandon reason**: Kept (`SPLIT=1` restored as the default path).
- **Remaining/reopen conditions**: The 0-slot degradation and the exit-time crash remain open items.

### PC-26 `AHEAD_AUTO` extremum search

- **Status/version**: Published, off by default (a fixed 3 is slightly better).
- **Why it was tried**: To apply the same family of controller as PC-22 to the look-ahead distance.
- **Technical mechanism**: `AHEAD_AUTO=1`, with `AHEAD_AUTO_PERIOD` defaulting to 32 graphs/window; **the objective function must be the window's incremental hit rate** — using the cumulative hit count/cumulative hit rate would be biased by cache warm-up.
- **Experimental conditions and evidence**: §6.32④: with an uncorrected objective the controller climbs all the way to the clamp of 6 and then walks back down to 1; once corrected it is measured probing back and forth between **3↔4** (visit distribution 4×19, 3×18, 5×6, 6×4, 2×1) = it has found the optimal region, but **the fixed default of 3 is still slightly better (20.8 vs 19.1 t/s)**, and probing back and forth has a cost.
- **Observations and conclusion boundaries**: This is the classic case of "the controller is correct but not worthwhile": finding the optimal region ≠ being faster than the fixed optimum.
- **Keep/abandon reason**: Kept as an optional switch; not a default.
- **Remaining/reopen conditions**: Re-evaluate if the distribution changes sharply (long runs, mixed workloads).

### PC-27 Non-blocking read-back "one beat late"

- **Status/version**: Published, not changed by default (`SMOE_NONBLOCK=1 + AHEAD=3`).
- **Why it was tried**: At `AHEAD=1` the r1 (the hit rate of the 1st predicted rank) is anomalously low, and it had to be determined whether this is an implementation bug or a consequence of the mechanism.
- **Technical mechanism/localization**: §6.33 (snapshot lines 1174–1189). The phenomenon (still present after the SPLIT fix): r1 at ahead=1/2/3/4 = **61.2 / 86.0 / 79.5 / 72.3%** — 2→3→4 decay smoothly, and only 1 derails. By elimination: the attribution offset is correct (`target_layer = pending.layer + s.smoe_ahead`); the delivery statistics are almost identical across the four ahead values (`deferred=11`, `predict≈11–12k`) ⇒ predictions are not being lost. **The decisive experiment**: after changing the SMoE read-back to synchronous (`SMOE_NONBLOCK=0`), **r1 at ahead=1 goes from 61.2% → 93.1%** (the highest of all), while ahead=2 is essentially unchanged (86.0 → 84.7) ⇒ conclusion: **the non-blocking read-back causes predictions to be consumed one beat late**; the window at `ahead=1` is only ~1 layer, so exactly one whole beat expires (what is actually used is the prediction from L−1), whereas at `ahead≥2` the one-beat-late consumption is still a prediction "targeting L", hence normal.
- **Trade-off (2 back-to-back runs each, 8k/q8_0/auto=97 slots)**:

| Configuration | gen (t/s) | Hit | r1 |
|---|---|---|---|
| **`SMOE_NONBLOCK=1` + `AHEAD=3` (current default)** | **19.9 / 20.3** | 88.5 / 89.7% | 79.5% |
| `SMOE_NONBLOCK=0` + `AHEAD=1` | 20.0 / 18.1 | 89.5 / 83.7% | **93.1%** |
| `SMOE_NONBLOCK=0` + `AHEAD=2` | 18.1 / 18.5 | 83.5 / 83.8% | 84.7% |

- **Observations and conclusion boundaries**: On speed the current default is best (mean 20.1 vs 19.1 / 18.3, and the synchronous path has larger variance) ⇒ **do not change the default**; if a scenario cares more about **prediction accuracy/long-term placement quality** (placement quality compounds in very long generations), `SMOE_NONBLOCK=0 + AHEAD=1` can be used. The 20.x in this section are **post-fix** host-path measurements and must not be mixed with the retracted old 20.3/32 tps.
- **Keep/abandon reason**: Kept; the default is not changed.
- **Remaining/reopen conditions**: Whether "a high r1 compounds into a higher hit rate" under long context has not been verified.

## 9. Stage seven: shared pool, position weights and memory-safety boundaries

### PC-28 Shared pool `GLOBAL_POOL`

- **Status/version**: Published (default 0 = independent cache per layer).
- **Why it was tried**: An independent cache per layer makes cold layers and hot layers each hold a fixed number of slots; sharing physical slots lets hot layers take more and cold layers take fewer.
- **Technical mechanism**: With `LLAMA_MOE_GLOBAL_POOL=1` all `(layer, expert)` pairs share one pool of physical slots; `moe_cache_state` takes the `global_pool` branch (with specializations in slot allocation/eviction/direct-read views/graph binding). devpart and the shared pool are mutually exclusive (`s.devpart = s.devpart && ... && !s.global_pool`).
- **Experimental conditions and evidence**: `sources/moe-cache-score-aware-prd.md` §Current policy and §Earlier short-replay diagnostics. 128-step auto: per-layer MRS = 9270.1 MiB / 97 slots/layer / median 61.6195 ms / hits 76245 / evictions 9; shared pool MRS = 9281.2 MiB / 3839 slots total / 62.9095 ms / 76245 / evictions **0**. The two hit counts are exactly the same. **Important qualification**: although the release-tree commit does contain `global_pool` code, this set of 128 results was run on a **later WIP-fixed pool** (including the zero-slot/stride/graph-binding fixes), so they **cannot vouch for the old pool path in the release tree**, nor serve as performance evidence for a "released shared pool".
- **Observations and conclusion boundaries**: Under the auto capacity the shared pool **never evicts**, so it **cannot** be shown to have an effective eviction policy; the candidate (LFU_POS) is +0.03% in median latency relative to the shared pool MRS (no difference).
- **Keep/abandon reason**: Kept as an explicit experiment switch; **per-layer cache by default** (the last item of the current PRD §Earlier short-replay diagnostics states explicitly "do not enable the shared pool or position weights automatically").
- **Remaining/reopen conditions**: A pressure working point that actually produces evictions is needed (512 MiB), and PC-31 must be explained.

### PC-29 Position weights `LAYER_AWARE` / `LFU_POS`

- **Status/version**: **WIP only** (`SOURCE_TREE` uncommitted; searching the release-tree commit `7e01451b2` for `LAYER_AWARE`/`lfu_pos` gives 0 hits).
- **Why it was tried**: In a shared pool, "the layer about to be visited in the next round" deserves retention more than "the layer just passed", whereas a pure frequency treats all layers as equivalent.
- **Technical mechanism** (snapshot §Current policy): `future = l > c`; `d = future ? l - c : l - c + N`; `weight = (future ? 1.0 : 0.5) * N / (N + d)`; `retention_score = actual_use_frequency * weight`. The index comes from the **ordered layer table** (no assumption that model layer numbers are contiguous); the current layer takes `N` (coldest); within the same class, the closer the distance the higher the priority; **the position factor does not modify the raw frequency**, so a sufficiently hot expert in an old layer can still outweigh a cold future-layer expert; the weight is **recomputed only when the execution cursor changes**. It requires `GLOBAL_POOL=1`, `MRS=1`, `FIFO=0`, `EVICT_SCORE=0`, and warns explicitly and ignores the setting when incompatibilities arise; the log marker is `policy=LFU_POS`.
- **Experimental conditions and evidence**: 512 MiB pressure (fixed replay, 128 steps, 3 rounds per side, first 16 steps dropped): median hits 31527 → 38076 (**+20.77%**), evictions 8610 → 8557, median decode 77.6790 → 76.1885 ms (**−1.92%**, smaller than the baseline's round-to-round fluctuation). Source `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json`.
- **Observations and conclusion boundaries**: Hits rose by 20.77% but throughput did not speed up stably (see PC-31); also, in all three rounds of the pressure candidate 1/128 steps had a different top-1 from the reference (step 79), so it **does not satisfy numerical equivalence either**.
- **Keep/abandon reason**: Paused (uncommitted, default not switched). The current PRD's conclusion is "this round confirmed the function and the change in hits under pressure, but did not obtain the performance and numerical-equivalence evidence needed to switch the default".
- **Remaining/reopen conditions**: See the closing conditions of PC-31.

### PC-30 Shared-pool memory-safety boundaries (zero slot / reading protection / stride / graph UID)

- **Status/version**: WIP only (uncommitted), but the evidence chain for the incident and the fix is complete.
- **Why it was tried**: The shared pool changes addressing by row/expert offset from a fixed per-layer layout to a global pitch, so anywhere that "treats the row size as a constant" or "divides by the block size before multiplying by the channel" will **silently mis-address**.
- **Technical mechanism** (snapshot §Memory-safety boundaries of the shared pool):
  - **Zero slot**: the direct-read shared pool keeps one all-zero padding slot; its capacity **counts toward the physical pool but not toward the number of resident-able experts** (512 MiB measured: 211 physical / 210 usable; on this machine auto 3840/3839).
  - **reading protection**: a physical slot the GPU is currently reading, a slot whose write has not completed, and fixed slots cannot be evicted; **read protection is released at the existing backend synchronization point**, and the historical prediction list is not treated as a permanent protection set.
  - **stride fix**: CUDA MMVQ/MMQ weight channel/sample strides use **int64 byte offsets** throughout, locating the byte base address first and then interpreting the quantization block; the in-row stride still follows the quantization block; **one must not divide by the block size first and then multiply by the channel**, nor hide the problem by inflating slots through the whole-model LCM; the shared-pool pitch of this model is **2,534,400 bytes**, which is not divisible by the 82-byte quantization block.
  - **graph UID**: direct-view and gathered-copy `src[0]` bindings update the graph UID when switching, preserving CUDA Graph functionality and preventing an old graph from treating an expert ID as a shared-pool slot ID.
- **Experimental conditions and evidence**: The old shared pool produced a CUDA illegal read around decode step **29**; the raw memcheck captured **addressing of expert ID 450** when the slot capacity was 211. After the fix **seven 128-step pressure runs all completed**; an independent CUDA smoke covering IQ2_S/IQ3_S/IQ4_NL MUL_MAT_ID, 4D broadcast, 1/4/17 token, and single-token fused SwiGLU had **all 24 cases item-by-item consistent with the compact layout, memcheck 0 errors** (`LOCAL_EVIDENCE/moe-cache/stride-memcheck.log`; summary `final-verification.json`, containing `cuda_stride_smoke: {cases:24, failures:0, memcheck_errors:0}`).
- **Observations and conclusion boundaries**: This check is **not** a full-model memcheck, and it does not cover multi-GPU or real-hardware NVFP4 execution; "wrong expert data at a legal address" can still go unreported (see the correctness risk in PC-31).
- **Keep/abandon reason**: Kept (the fix itself is effective); the shared pool is still off by default.
- **Remaining/reopen conditions**: To enable the shared pool by default, the smoke above would have to be extended to the full model (including timing/graph binding).

### PC-31 128 pressure: hits +20.77% but DMA almost unchanged

- **Status/version**: **Still open**.
- **Why it was tried**: To verify whether "position weights raise the hit rate" can be converted into a speedup.
- **Technical mechanism/definition**: Fixed external replay, 128 decode steps, first 16 steps dropped, timing including only `llama_decode + llama_synchronize` (excluding the logits copy and file writing), taking the median of 3 rounds per side and then the median of those medians.
- **Experimental conditions and evidence** (snapshot §Earlier short-replay diagnostics + §Open issues to investigate; RTX A5000 Laptop 16GB / 5950X / 16 threads / UD-IQ3_XXS / an 84-token code prompt):

| Configuration | Physical cache MiB | Resident-able slots | Median decode ms | Median hits | Median evictions | VRAM peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Per-layer MRS, auto | 9270.1 | 97 per layer | 61.6195 | 76245 | 9 | 15378 |
| Shared pool MRS, auto | 9281.2 | 3839 total | 62.9095 | 76245 | 0 | 15306 |
| Shared pool LFU_POS, auto | 9281.2 | 3839 total | 62.9310 | 76245 | 0 | 15306 |
| Shared pool MRS, 512 MiB | 510.0 | 210 total | 77.6790 | 31527 | 8610 | 6534 |
| Shared pool LFU_POS, 512 MiB | 510.0 | 210 total | 76.1885 | 38076 | 8557 | 6534 |

  Core divergence: at 512 MiB the median hits rise by **+20.77%**, but the **median prefetch bytes fall only from 17399519232 to 17294964224 (about −0.60%)**, and evictions only from 8610 → 8557; the median latency is −1.92%, **smaller than the baseline's repeat fluctuation** (MRS three rounds 77.6790/80.7960/76.3055; candidate 76.8140/75.6135/76.1885) ⇒ no evidence of a stable speedup.
- **Correctness risk (a timing mismatch cannot be excluded first)**: In **all three rounds** of the pressure candidate a top-1 flip appears at decode step **79** counted from zero (baseline token 271, candidate token 248046); the candidate against the pressure reference has mean KL 0.002296–0.004157 and a maximum absolute logit difference of 3.193697; prefill agrees. **MRS repeats themselves also carry numerical differences** (`baseline_repeat_max_maxdiff = 2.887` in `global-pressure128-v2.summary.json`, candidate vs reference worst 3.194) — so one can neither declare the candidate "lossless" on this basis, nor attribute the difference to the candidate.
- **Definition warning**: The `noise_tolerance_used = 8.66` and `candidate_within_baseline_repeat_noise = true` in that summary belong to **old, inflated noise-tolerance fields and are not used as acceptance grounds** (the current script has removed that judgement).
- **Observations and conclusion boundaries**: "Repeatedly hitting a resident expert does not itself produce a repeated transfer" (the deduplication semantics, PC-14); the earlier explanation of "hitting repeatedly yet still shuttling continuously" **conflated compute hits with prefetch deduplication** and cannot serve as the root cause. At present **neither deduplication failure nor the transfer bandwidth having reached its limit has been proven**.
- **Keep/abandon reason**: Kept as an open problem (the default is not switched); the closing conditions are already written in the snapshot §Follow-up investigation and closing conditions: associate, by (layer, expert) and slot-occupancy generation, "prefetch request → resident/pending deduplication → actual copy submission and completion → eviction → reinstallation → actual hit", verify which of request/submission/completion the counter fields represent, and split the true transfer into "first installation" and "reinstallation after eviction".
- **Remaining/reopen conditions**: First localize the first logit divergence (check the expert identity, the slot owner, and the write-completion state), use forced synchronization/disabling CUDA Graph as a diagnostic control, and only then decide whether an operator-level CPU/GPU comparison is needed.

## 10. Stage eight: `optimization400` lifetime statistics and two uncommitted candidates

### 10.1 Fixing the definition first: why "hot-spot hits" cannot be equated with "repeated transfers"

This section gives directly citable quantitative relations (all from `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json`, build = `ggml-base.lifetime-baseline.dll`, sha256 prefix `ee22deb6`, 400 token / 6144 MiB / all-in-memory):

- **The hit count is a count of "reads of resident slots", and one expert is read three times**: `gpu_uses = 133484`, `hits = 400452 = 3 × 133484` (the gate/up/down weight blocks are each counted once). So hits are **weight-tensor reads**, not transfer counts.
- **Transfers happen only at admission**: `admissions = 10216`, `prefetch_bytes = 20,224,040,448` (≈ 1.98 MB per admission, consistent with the order of magnitude of a 1.9 MB bundle). **hits/admissions ≈ 39.2**.
- **Repeated hits produce no new bytes**: `dup_resident = 29682` (the candidate is already in the resident set → dropped outright, costing no bytes), `dup_pending = 0` (no duplicate requests in flight), `dup_admit = 0` (no self-copy into the same slot). The deduplication authority is `expert_slot[e] ≥ 0`: the slot is assigned at admission, **before** the copy is issued.
- **The real "repeated transfer" is reinstallation after eviction**: `readmits = 4734` (46.3% of admissions), `readmit_bytes = 9,372,176,612` (46.3% of admitted bytes); of these `readmit_within_1_graph = 696`, `readmit_within_4_graphs = 1638`.
- **There is also a large amount of "pointless transfer"**: `evicted_unused = 4530` (62.3% of evictions, 44.3% of admissions), `evicted_unused_bytes = 8,972,577,948` (62.3% of evicted bytes); `evicted_used_once = 1527`, `evicted_used_many = 1215`.
- Consistency: `admissions − evictions = 10216 − 7272 = 2944 = live` (holds); `duplicate_admissions = 0`, `evict_untracked = 0`, `use_not_resident = 0`, `use_complete = 1` (holds).
- Byte-definition warning (current PRD §400-token residency observation): the totals are accumulated by weight-component stride and are **neither the shared-pool physical pitch nor bytes measured on the PCIe link**; an admission is a logical slot occupancy, and pending data still cannot be read by computations.

Conclusion: **a rise in the hit rate can come entirely from "the same batch of resident experts being read repeatedly", at zero cost; only admissions (including reinstallation after eviction) move bytes.** Therefore "hits up" entails neither "transfers up" nor "faster" — to judge benefit/cost one must look at the admission count, the admitted bytes, the reinstallation fraction and the unused-eviction fraction together.

### PC-32 Residency-time statistics `CACHE_LIFETIME` + 400-token baseline

- **Status/version**: WIP only (`LLAMA_MOE_CACHE_LIFETIME` has 0 hits in the release tree `7e01451b2`; the measurement definition has been written into the current PRD §400-token residency observation).
- **Why it was tried**: A per-(layer, expert) residency ledger that "does not print per event but emits a single line" was needed, to separate "unused evictions/reinstallations/actual GPU use" instead of looking only at hits/misses.
- **Technical mechanism**: When enabled, it tracks occupancy, eviction, reinstallation and real GPU use by `(layer, expert)`, and emits one `[MOE-LIFETIME]` summary line at exit; when disabled it does **not allocate** the per-expert tracking arrays. Discriminants: the actual CPU routing frequency ≠ the GPU residency-time use count; zero-weight padding does not count as actual use; devpaths that cannot be fully observed are explicitly degraded (the `use_observations_devpart/gathered` counters).
- **Experimental conditions and evidence**: `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json` (400 token, `--ignore-eos`, 6144 MiB, `SPLIT=1 DIRECT_READ=1 MRS=1 PREFETCH=1 SMOE_NONBLOCK=1 AHEAD=2 HOT_BACKFILL=8`, `GLOBAL_POOL=0`): Generation **19.3 t/s** (`acceptance.status=pass`, threshold 19.0), VRAM peak 12458 MiB, 403 graphs. Counts: admission/eviction/final occupancy = 10216/7272/2944; predicted/non-predicted admissions = 7026/3190; unused evictions 4530 (predicted 3559 / non-predicted 971); reinstallations 4734 (≤1 graph 696, ≤4 graphs 1638); actual GPU uses 133484; `dup_resident=29682`. A repeat at the same working point without the observation instrumentation gave 19.2 t/s; **two single results do not constitute evidence of a speedup**.
- **Observations and conclusion boundaries**: `available / use_complete / integrity error counts` must be checked together; in this file `use_complete=1`, `use_observations_partition=19153`, `use_observations_devpart=0`. The raw output/error/statistics CSVs point to `SOURCE_TREE/opt400-lifetime-before-out.txt|err.txt`, `stats-opt400-lifetime-before.csv`.
- **Keep/abandon reason**: Kept (it is the instrument needed to answer PC-31); the code itself is uncommitted.
- **Remaining/reopen conditions**: An association from `(layer, expert) → slot generation → actual copy submission/completion` is needed (currently CACHE_LIFETIME stops at admission/eviction/use, with no distinction between "submitted" and "completed").

### PC-33 All-source frequency gate candidate (**rejected**)

- **Status/version**: WIP only / **rejected** (the release tree has no such code; the binary file name `ggml-base.rejected-all-frequency.dll` is the maintainer's own label).
- **Why it was tried**: To make admissions from **all sources** (including hot/idle/seed/phase filling) pass a "frequency gate", only letting in candidates hotter than the currently evictable victim, thereby cutting the pointless bytes.
- **Technical mechanism**: The source gains a `prefetch_reject_frequency` counter and an admission criterion of "compare retention value against the chosen victim" (effective only when `mrs && !fifo`); the hit log gains `admit_reject_freq=`.
- **Experimental conditions and evidence**:
  - 400-token CLI (`lifetime-after.json`, same controlled environment): Generation **19.9 t/s** (pass), admissions **4788** (predicted 2336 / non-predicted 2452), evictions 1845, reinstalls **164**, hits 402648, prefetch bytes **9,482,722,816 (−53%)**, `admit_reject_freq=4151`, `hot_fill=2452`, `dup_resident=30221`.
  - Fixed-history 400 steps (`optimization400/fixed400/experiment.json`, `real-qsa-check.exe`, candidate hash `ac46f065`, the three candidate runs' counts exactly identical): hits **322554** (baseline 340299, **−5.2%**), misses 254886 (baseline 237141), evictions 3425 (baseline 10585), prefetch bytes **12,623,473,152** (baseline 26,904,137,728, **−53.1%**), `prefetch_ready` 24155 (baseline 29862); steady-state medians 53.850/55.884/54.123 ms vs baseline 54.053/53.853/54.637 ms.
  - Correctness (with `decode_all` as the primary statement and `decode_steady` only as a supplement): candidate vs reference `decode_all` top1 identical **398/400**, max logit difference 4.8496 (worst row i=369, input_token 328); `decode_steady` top1 382/384, bitwise identical 34/384 — **not used as an acceptance definition**. The baseline itself also varies: repeated baseline runs had `decode_steady` bitwise identical 341/384 (see PC-35).
- **Observations and conclusion boundaries**: **Saving bytes works** (−53%), but **the conclusions on hits and latency depend on the working point**: 19.9 t/s on the CLI self-generated workload, whereas on the fixed 400-step workload hits are −5.2% with no improvement in latency. **Do not treat 19.9 and 19.3 / 18.4 as one controlled A/B group**: `lifetime-before.json` (19.3) is a baseline with observation instrumentation, and `baseline-paired-r2.json` (18.4) is a later single run of the old baseline; the three are not alternating paired runs of the same round. Passing the 19 threshold also does not mean the candidate was adopted.
- **Keep/abandon reason**: **Abandoned** (uncommitted, ultimately paused). Reason: it works as a "save bytes" measure, but as a "speed up" measure there is no stable evidence at any working point; and the drop in hits shows it also rejected some valid admissions.
- **Remaining/reopen conditions**: To reopen it, one must first show "how many of the 4151 rejected admissions were valid (were actually used afterwards)", i.e. associate `admit_reject_freq` with subsequent actual use.

### PC-34 Hot-backfill-only fix candidate (rotation / empty slot / eligible victim, **uncommitted, paused**)

- **Status/version**: WIP only / paused. Binary `ggml-base.hot-backfill-candidate.dll` (sha256 prefix `e0d7a83f`); **note**: **at the time of the run** `SOURCE_TREE/build-ple-trace-mrs/bin/ggml-base.dll` was this candidate (the hash I read during review was md5 `5c21782f…` / sha256 `e0d7a83f…`); that directory is written by a local build script and **may be replaced at any moment**, so the correspondence above holds only "at the time of the run" and it must not be claimed to still be that file "currently". The release-tree source does not contain this candidate.
- **Why it was tried**: `moe_cache_hot_backfill` has three logic defects that make backfill "starve" on some layers or behave incorrectly with an empty cache or an equal-value replacement.
- **Technical mechanism**: The WIP adds a `hot_cursor` **rotating cursor** (rotating by layer, to avoid filling only the first few layers), and adds the criterion that "frequency-driven backfill must beat the **actually evictable victim**" (`eligible victim`, rather than comparing against a protected cold resident). A separate `position_cursor` (position weights recomputed only when the execution cursor changes) belongs to **independent LFU_POS/shared-pool work** (PC-29) and is not newly added by this hotfix.
- **Experimental conditions and evidence**:
  - **Logic smoke** (`optimization400/cache-logic-smoke.cpp` + `build-logic-smoke.cmd`, a pure-CPU unit test that does not run the model):
    - Before the fix (`logic-empty-before.log`, **3 FAIL**): `FAIL: actual-use backfill populates an entirely empty cache`, `FAIL: backfill uses free capacity without requiring a hotter candidate`, `FAIL: hot backfill stops when eligible replacement would only tie`; also `hot backfill covered 48/48 layers` (in that version rotation was already fine).
    - Even earlier (`logic-round-robin-before.log`, **2 FAIL**): the output lines `hot backfill covered 28/48 layers in six eight-expert budgets`, `FAIL: round-robin backfill does not starve eligible layers`, `FAIL: backfill visits sparse actual layer ids`. **28/48 is the behaviour of that unit test's synthetic fixture (six eight-expert budgets, 48 synthetic layer keys), not an observation of "28 layers starved" in the real model**.
    - After the fix (`logic-final.log`, **failures=0**, **15 PASS items** + 1 coverage line + 1 `failures=0` line, 17 lines in total): newly added `actual-use backfill populates an entirely empty cache`, `backfill uses free capacity without requiring a hotter candidate`, `hot backfill stops when eligible replacement would only tie`, `round-robin backfill does not starve eligible layers`, `backfill visits sparse actual layer ids`; and retained `compare against eligible victim, not protected cold resident`, `zero-frequency cold start uses empty slot`, `FIFO is not changed into frequency admission`, `imminent prediction is not blocked by historical frequency`, `global pool compares actual victim owner`, `position-weighted admission favors next layer and preserves zero slot`.
  - **400-token CLI** (`optimization400/final-r1.json`): exit 0, Generation **18.6 t/s** (`acceptance.status=fail`, threshold 19.0), peak 12458 MiB; hits 371631, misses 202959, admissions 10963, evictions 7891, reinstalls 4475, `admit_reject_freq=405`; oracle line `oracle_topC=72.9% resident_set=72.9% mrs_topC=0.1% actual_hit=64.7%`. **The baseline from the same batch, `baseline-paired-r2.json` (18.4 t/s), is a later single run of the old baseline and does not form a controlled A/B with 18.6**. The second run, `final-r2.json`, exited **3221225477 (0xC0000005)** with no throughput line → `acceptance.status=unavailable` (failed runs do not take part in acceptance).
  - **Fixed-history 400 steps** (`optimization400/fixed400-final/experiment.json`, candidate hash `e0d7a83f`): the three runs' hits are **349374/349377/349377** — not every counter is identical item by item; the baseline is 340299, about +2.7%. Prefetch bytes 27,071,999,488 (baseline 26,904,137,728, about +0.6%), evictions 10541 (baseline 10585); steady medians 53.383/54.297/53.365 ms against 53.946/54.058/55.587. Correctness is stated primarily by the **398/400 identical top-1** of `decode_all`, with a max logit difference of 5.5479; the 382/384 of `decode_steady` is only a supplement.
- **Observations and conclusion boundaries**: **The fix of the logic defects has independent evidence** (the unit test goes from 3 FAIL → 0 FAIL, and it does not depend on the GPU); **but this CPU fixture passing only shows that the cache bookkeeping/admission logic is self-consistent, and does not mean the native numerics are wholly correct** (it does not run the model and does not cover CUDA/MoE operator numerics). **Neither "stable speedup" nor "numerical equivalence" materialized end to end**: the CLI's 18.6 t/s missed the 19 threshold, and the fixed 400-step run was only +2.7% hits / −1.25% median latency, falling inside the baseline's own fluctuation band (PC-35).
- **Keep/abandon reason**: **Paused** (uncommitted). Reason to keep: the fix itself is correct and the unit test is repeatable; reason not to take it into the default: there is no acceptable end-to-end benefit.
- **Remaining/reopen conditions**: Reopening requires (a) eliminating the 0xC0000005 of `final-r2` (an exit-time/asynchronous-finalization path, **root cause undetermined**; this belongs to `05-correctness-and-methodology.md`), and (b) separating the latency difference from baseline fluctuation with ≥3 repetitions.

### PC-35 The fixed-history 400-step control method, and "the baseline itself fluctuates too"

- **Status/version**: Kept (methodology).
- **Why it was tried**: To compare the two WIP candidates against the baseline using the same input sequence, the same 400 steps and an alternating order (baseline → candidate → baseline → candidate → baseline → candidate), avoiding "a single run = a conclusion".
- **Technical mechanism**: `optimization400/verify-fixed400.py` reuses `tools/tuning/moe-cache-compare.py`: `real-qsa-check.exe` (different exe directories distinguish the backends), ctx 8192, `--decode 400`, 16 threads, `--n-gpu-layers 49 --cpu-moe`, `CACHE_MIB=6144 GLOBAL_POOL=0 LAYER_AWARE=0 HOT_BACKFILL=8`, 64 slots per layer / 6116.3 MiB physical cache / peak 12196 MiB; each round it enforces `exit_code==0`, complete artifacts, `decode_steps==400`, and no NaN/Inf in the logits, and uses the presence of `admit_reject_freq` as a **revision marker** (present for the candidate, absent for the baseline); before launch it checks for ≥90000 MiB of free memory.
- **Experimental conditions and evidence**: `fixed400/experiment.json` records `backend_sha256`: baseline = `d2a4af54…` (identical in two experiments), candidate₁ = `ac46f065…` (the PC-33 frequency gate), candidate₂ = `e0d7a83f…` (the PC-34 backfill fix).
  **Key methodological observation**: **Repeats of the same baseline also diverge** — for `fixed400`, baseline2 vs ref had a max logit difference of 0.29 and 357/400 bitwise identical (top1 still all identical); for `fixed400-final`, baseline3 vs ref had a max difference of 0.533 and 66/400 bitwise identical; while some other repeats are bitwise fully identical (400/400). Therefore **a single 400-step logit comparison is not enough to judge "equivalence"**, and the candidate's 2/400 top1 flips (definition = `decode_all`, which this chapter uniformly uses as the primary statement of correctness; the 382/384 of `decode_steady` is only a supplement and not used for acceptance) must be evaluated in a control that can explain the baseline's own fluctuation.
- **Observations and conclusion boundaries**: This experiment is a "fixed-input numerical/latency check", and **not** the CLI's 19 t/s acceptance (`experiment.json.scope` keeps them explicitly separate); not running a validation does not mean the candidate has been accepted.
- **Keep/abandon reason**: Kept as methodology (and as the source of the boundaries of the PC-31/PC-34 conclusions).
- **Remaining/reopen conditions**: To call a candidate "equivalent", one would first have to quantify the baseline repeat distribution (more repetitions) and give an acceptable upper bound on the difference — the current script deliberately does **not** set such a tolerance.

### PC-36 Run gates and record integrity

- **Status/version**: Kept (methodology).
- **Why it was tried**: Performance comparisons presuppose that "failed runs do not masquerade as data", "concurrency does not contaminate itself", and "historical logs are not lost".
- **Technical mechanism and evidence**:
  - **Single-instance lock**: `optimization400/lock-smoke.json` shows that when an existing run holds the lock, a new process gets `exit_code=2`, `exit_code_source=lock-busy`, `run_started=false` (it does not start and produces no partial data).
  - **Timeout**: `optimization400/timeout-smoke.json` (`--timeout 1s`) child process `exit_code=124`, `tools-run` likewise 124, the summary is saved, `acceptance.status=unavailable`; "on a timeout/child failure, subsequent runs stop immediately, and missing data is recorded as unavailable rather than filled in as zero".
  - **Accidental-deletion incident**: `optimization400/deleted-logs-incident.json` records that a subagent, while cleaning up, **accidentally deleted 30 existing user logs** (`zz*-out.txt` and the like); same-name school copies, FileHistory and Win32 shadow copies were all unavailable (`vssadmin` needs administrator rights), the user decided not to restore them, and **no substitute runs were fabricated** (`restored=false`, `replacement_runs_fabricated=false`). Wherever historical numbers touch those logs, this chapter cites them only as "historical report".
- **Observations and conclusion boundaries**: The lock and timeout behaviours both have independent smoke evidence; historical conclusions within the range of the deleted logs **cannot be re-checked**.
- **Keep/abandon reason**: Kept.
- **Remaining/reopen conditions**: None.

## 11. Summary of definition traps (cross-route)

1. **Hot-spot hits ≠ repeated transfers** (§10.1): `hits = 3 × gpu_uses` (three weight blocks), and DMA happens only at admission/reinstallation; the criteria are `admissions/readmit_bytes/evicted_unused`, not hits.
2. **`hits/misses` cannot be attributed to predictor quality** (current PRD §Admission and attribution): hits come jointly from predicted prefetch, fallback, hot backfill, phase fill, seeding and post-miss admission; `prefetch_experts` is the **total admissions** of the common write path and includes non-predicted filling.
3. **`prefetch_predicted/prefetch_required` is a coverage definition that includes the static hot set** (PC-11), and it, the teacher-forced recall@10, and the online hit rate, the three of them, cannot be substituted for one another.
4. **`oracle_topC` is content-dependent** (PC-17): change the policy and the division of labour and the numerics change, so oracles from different runs cannot be compared across runs.
5. **The old inflated noise-tolerance fields are invalid** (PC-31): `noise_tolerance_used` / `candidate_within_baseline_repeat_noise` are not used as acceptance grounds.
6. **Retracted values**: the old host 20.3 / 32 tps (before the routing-misplacement fix), and the old "GPU 40 ms hard floor / 25 t/s ceiling". 19 t/s is this machine's internal screening floor, not "passing means correct".
7. **Historical working points that must not be mixed**: different rounds differ in context, KV type, cache budget, step count, and lazy/all-in-memory settings (the `sweep-*.csv` problem named in the current PRD §Current performance acceptance baseline); the `AHEAD=3` candidate must be compared at the same 400-token / 6144 MiB / all-in-memory working point.
8. **teacher-forced single step ≠ cumulative rollout** (PC-07/PC-08): these trials take the true input layer by layer. Only when the prediction serves merely as a prefetch hint and does not change the true computation is the approximate activation not fed directly back into the model; the actual effect of the cache changing the CPU/GPU division of labour and the numerical path still needs online verification, and one must not claim there is no cumulative effect on the basis of an offline curve.
9. **Binary and source decoupled**: `SOURCE_TREE/build-ple-trace-mrs/bin/ggml-base.dll` is written by a local build and **is replaced as candidates change**; one may only say "**at the time of the run** the file in that directory was <hash>" (what I read during review was the backfill candidate with md5 `5c21782f…` / sha256 `e0d7a83f…`), and must not claim it "currently" still is. Citing a run result must carry the hash, and the contents of that directory are **not** the release tree.
10. **Source code existing ≠ implementation released; a document being archived ≠ implementation released**: whether something is published is judged only by release-tree commits; side-branch checkouts (trace instrumentation etc.) and plan documents do not constitute a release.

## 12. Coverage list (route ID ↔ source sections)

| Source | Section/lines (snapshot) | Corresponding routes |
|---|---|---|
| `sources/handoff.md` | §6.6 (209–258) | PC-12, PC-13, PC-14, PC-15, PC-16 |
| | §6.7 (259–343) | PC-17, PC-18, PC-19, PC-20, PC-21 |
| | §6.8 (344–372) | PC-22, PC-24 |
| | §6.9 (373–404) | PC-23, PC-24 |
| | §6.32 (1124–1173) | PC-25, PC-26 |
| | §6.33 (1174–1200) | PC-27 |
| `sources/smoe-nk-degradation-plan.md` | §2 (65–83), §5 (130–162) | PC-08 |
| | §3 (84–99) | PC-09 |
| | §6 (163–172) | PC-11 |
| | §7 (173–238, including the 181–192 table and the 214–231 consequences) | PC-07, PC-10 |
| `sources/moe-cache-score-aware-prd.md` | §Current policy (16–34) | PC-01, PC-28, PC-29 |
| | §Memory-safety boundaries of the shared pool (35–44) | PC-30 |
| | §Admission and attribution (45–52) | PC-11, PC-33, PC-34 |
| | §Current performance acceptance baseline (53–101) | §0.4, PC-32 |
| | §400-token residency observation (74–101) | PC-32, §10.1 |
| | §Earlier short-replay diagnostics (102–133) | PC-28, PC-29, PC-31 |
| | §Open issues to investigate (134–163, including 138–144, 145–151, 152–163) | PC-31 |
| | §History v1 (164–297: §Solution 183, §Implementation Decisions 223, §Testing Decisions 249, §Out of Scope 277, §Further Notes 289) | PC-01, PC-02, PC-03, PC-05, PC-06, PC-10 |
| `sources/moe-decode-perf-plan.md` | §1–2, §7 | PC-04, §0.4 |
| `sources/rebuild-spec.md` | §Predictor (SMoE / Fate / XT) | PC-02, PC-05, PC-06 |
| Source code | `SOURCE_TREE/ggml/src/ggml-backend.cpp` (1700–1725 switch list, 1910–1925 gate fields, 4414–4425 deadline gate, 4512/4646/6595 in-flight amounts, 5432–5480 value gate, 5537–5565 extremum search, 6149 layer_us_ewma, 7700–7719 fit and budget) | PC-24 and every "Published" determination |
| | `SOURCE_TREE/common/arg.cpp`, `tools-run.py` (`PREDICT_FATE=0` default baseline) | PC-05, PC-27 |

## 13. Small logs recommended for preservation (exact paths)

Rule: list only **small files** (≤ ~80 KB preferred, ≤ 2 MB where necessary). **Do not** archive `*.logits.bin` (129–399 MB each), `*stats*.csv` (400+ KB each), or the 0.5–1 MB raw 128-series summaries, unless the fields are trimmed first (the `localization/trimming suggestion` column).

### 13.1 optimization400 (the core evidence for the prediction/cache chapter, all ≤ 1.7 MB)

- `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-before.json` (8 285 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/lifetime-after.json` (8 284 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/final-r1.json` (8 300 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/final-r2.json` (3 595 B, the 0xC0000005 failure sample)
- `LOCAL_EVIDENCE/moe-cache/optimization400/baseline-paired.json` (3 541 B), `baseline-paired-r2.json` (5 623 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/lock-smoke.json` (788 B), `timeout-smoke.json` (3 512 B), `deleted-logs-incident.json` (2 286 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/fixed400/experiment.json` (1 607 168 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/fixed400-final/experiment.json` (1 638 363 B)
  (trimming suggestion: keep only `runs[].{name,revision,exit_code,wall_s,vram,latency.decode_ms_steady,cache.counters}` and `comparisons[].{prefill,decode_all,decode_steady}`, which compresses to ~50 KB per file)
- `LOCAL_EVIDENCE/moe-cache/optimization400/cache-logic-smoke.cpp` (10 403 B), `build-logic-smoke.cmd` (732 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/logic-empty-before.log` (933 B), `logic-round-robin-before.log` (671 B), `logic-final.log` (967 B)
- `LOCAL_EVIDENCE/moe-cache/optimization400/verify-fixed400.py` (4 720 B), `run-baseline.py` (646 B)
- `LOCAL_EVIDENCE/moe-cache/final-verification.json` (1 708 B, containing the 24-case CUDA smoke and the LFU_POS CLI wrap-up)
- `LOCAL_EVIDENCE/moe-cache/baseline400-6g.summary.json` (4 503 B, the baseline summary from the era of the historical 20 t/s criterion)

### 13.2 Shared pool / position weights (the 128 series, large raw files, trimming recommended)

- `LOCAL_EVIDENCE/moe-cache/global-pressure128-v2.summary.json` (1 059 328 B) — **fields that must be kept**: `config`, `checks` (especially `baseline_repeat_max_maxdiff`, `candidate_vs_reference_*`, and the presence of `noise_tolerance_used`), `latency`, `runs[].cache.counters`; `runs[].env_effective_relevant`, `vram_samples`, `rows_meta`, `tokens` may be deleted.
- `LOCAL_EVIDENCE/moe-cache/global-auto128.summary.json` (973 628 B), `layer-auto128.summary.json` (505 563 B), `prechange-cross-comparison.json` (1 000 981 B) — same trimming rule as above.
- `LOCAL_EVIDENCE/moe-cache/stride-memcheck.log` (the raw conclusion of the CUDA stride smoke: 24 cases / memcheck 0 errors)

### 13.3 Raw run logs inside the source project (take hashes as well if they are to be archived)

- `SOURCE_TREE/a400a-err.txt` (37 407 B), `SOURCE_TREE/a400b-err.txt` (37 415 B) — containing the three classes of raw lines `hot-set oracle`, `per-rank prediction accuracy`, `smoe per DECODE graph`; the most complete single-file evidence for this route.
- `SOURCE_TREE/L2048-err.txt` / `L6144-err.txt` / `T2048_1-err.txt` / `T6144_4-err.txt` (~34.7 KB each) — the 2048/6144 MiB × per-layer/per-tensor comparison.
- `SOURCE_TREE/8k-tbq-err.txt` (37 140 B) — the same format for SMoE `ahead=2` under the 8k/TBQ configuration.
- `LOCAL_MODELS/qwen38/traces/ablation-lazy-moe2-xt.log` (16 632 B), `ablation-lazy-moe4.log`/`moe6`/`moe8` (32.5 KB each) — the static table + XT stage (including `[ Prompt: … | Generation: 15.2/16.2/16.7 t/s ]`).
- `LOCAL_MODELS/qwen38/traces/moe-static90-8192.csv` / `-t2-` / `-t8-` (~33.7 KB each), `moe-xt-12288.csv` (33 730 B), `moe-xt-8192.csv` (60 077 B) — the static/XT coverage and hit columns (`pred_hits/pred_total`).
- `LOCAL_MODELS/qwen38/traces/build_expert_manifest.py` (4 774 B), `build_xt_manifest.py` (3 424 B), `analyze_cross_token.py` (4 920 B) — transition-manifest generation and hold-out replay (**the scripts themselves are the raw method record**).
- Manifest files (~3.1 MB each, optional): `moe-predict-v1.bin`, `moe-predict-static96.bin`, `moe-predict-xt.bin`; if only the evidence is to be kept and not the bytes, at least save the headers of the three (magic/`n_layers/n_experts/n_trans/n_static`).
- `LOCAL_MODELS/qwen38/traces/simulate_smoe_nk.py` (8 243 B), `direct_smoe_recall.py` (22 531 B), `ple-prefetch-probe.txt` (3 992 B, a cross-reference to the PLE stage, belonging to `01-host-and-devpart.md`).

### 13.4 Not recommended for archiving (size/reproducibility)

- `optimization400/fixed400*/*.logits.bin` (399 MB per run), `*stats*.csv` (423 KB per run).
- `optimization400/ggml-base.*.dll` (900 KB × 4): if evidence is to be kept, **only the sha256-to-file-name mapping needs to be kept** (`baseline=d2a4af54…`, `lifetime-baseline=ee22deb6…`, `rejected-all-frequency=ac46f065…`, `hot-backfill-candidate=e0d7a83f…`).
- `LOCAL_EVIDENCE/moe-cache/global-auto128.logits.bin` and other 129 MB-class logits.
