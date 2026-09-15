[中文](04-kv-tbq-and-nxq.md) · [English](04-kv-tbq-and-nxq.en.md)

# 04 KV cache quantization: TBQ3/TBQ4 (TQ3/TQ4) and NXQ/E8

This document assembles every attempt along the "KV cache quantization" line — its motivation, mechanism,
failure points, quantization choices, reasons for keeping/abandoning, and re-open conditions —
into a traceable experiment archive. **This document is a record only; it implements no optimization
and re-ran no model.**

Suggested reading order: `00-research-chronology.en.md` (main narrative) → this document (KV quantization branch) →
`sources/handoff.md` §6.29, §6.42–§6.45 (original handoff text).

---

## 0 Position of this chapter, status legend, and measurement conventions

### 0.1 Position in the overall narrative

Per the main narrative, KV quantization belongs to the **later stage** of the work: first the PLE cache, then the MoE/SMoE cache (static table + XT → Fate
(no usable predictive information can be squeezed out of the hidden layers) → pivot to shared-expert SMoE exploitation (user reports 99% hit rate on the teacher test) →
online implementation and engineering pitfalls (a high hit rate is in fact slower) → dual-gated adaptive transfer threshold → dev-path timing bug),
and only after that came the **TQ4 crash / TBQ diagnosis** and **NXQ/E8**.

- The starting point of this document is the passage "§6.29 cache parameters under 256k + TBQ4 KV"; the later handoff chapters are not treated as the starting point of the whole research line;
  the motivation and early conclusions of PLE/MoE/SMoE are governed by `00-research-chronology.en.md` and `sources/handoff.md`.
- **Hit-rate convention warning**: every hit% appearing in this document is the **MoE expert cache (MoE-CACHE) online hit rate**,
  which is **two different quantities** from the **99% hit rate on the user-reported SMoE teacher test** (different object, different protocol, different denominator).
  This document does not restate, does not deny, and does not conflate that 99% figure; it is retained as recalled by the user, labelled
  **"user recollection, protocol/denominator pending correspondence with the original record"**, and is not replaced or rebutted by any other experiment in this document (for example any recall@10 or online hit rate).

### 0.2 Status legend (used uniformly in this document)

| Marker | Meaning |
|---|---|
| **Released** | Exists in the native code history of the document parent version `0862af564` |
| **WIP branch only** | Exists only in commits of the source worktree branch `qwen4exp-tbq-devpart-wip` (`8a0bb3b7f`, `0d3d6c9d9`, `fa05f8637`, `17ca0de85`, etc.), **not** in master |
| **WIP working tree only** | Exists only in the source worktree's **uncommitted** dirty working tree (visible to `git status`, absent from HEAD `17ca0de85`) |
| **Design only** | Written only in plans/specs/PRDs, not implemented |
| **Retracted** | Done and measured, but the conclusion or implementation was explicitly invalidated; must not be used as quality evidence |
| **Still open** | Not located, not implemented, or the conclusion boundary is insufficient to settle the matter |

"Exists in the source code" ≠ "released": this document labels each item individually with which tier it belongs to.

### 0.3 Release baseline and source worktree

- Release starting point: document parent version `0862af564`.
  That commit **touches only `README.md` (+200 lines, documentation only)**, and its **code baseline = parent commit `7e01451b2`**
  (checked with `git show --stat 0862af564`). Every "released" mentioned in this document refers to this master history.
- Source worktree: `SOURCE_TREE`, HEAD `17ca0de85`, branch `qwen4exp-tbq-devpart-wip`,
  which additionally carries **a large amount of uncommitted change** (including all NXQ and E8 fake-quant code).
- Verified branch containment relations (checked in the release tree with `git merge-base --is-ancestor <c> HEAD`):
  `2f1a363c8` (TBQ KV quantization) → **in master**;
  `7e01451b2` → **in master**;
  `8a0bb3b7f`, `fa05f8637`, `17ca0de85` → **not in master**.
  That is: **the TBQ format itself is released, but the evaluation and its diagnosis of §6.42/§6.43 exist only in WIP.**
- NXQ: `ggml/src/ggml-nexusq.c`, `ggml/src/ggml-nexusq.h` are **untracked** in the source worktree;
  `GGML_TYPE_NXQ3_0/NXQ2_0` and the E8 fake-quant chain **do not exist** in HEAD (`git show HEAD:...` count checked as 0).
  **NXQ never entered master**, nor did it enter any commit.

### 0.4 Model and hardware base (information provided by the user)

| Item | Value | Source |
|---|---|---|
| CPU | AMD Ryzen 9 5950X (16 cores, Zen 3) | User-provided |
| Memory | DDR4-2666, 128 GB | User-provided |
| GPU | RTX A5000 Laptop 16 GB, PCIe 4.0 x8 | User-provided |
| Weights | `Qwen3.8-Flash-Next-UD-IQ3_XXS` (UD dynamic quantization, mixed format) | `sources/handoff.md` §6.35 |
| Context | `context_length = 262144` (256k) | GGUF metadata `qwen4exp.context_length` |
| Layer structure | `block_count = 48`, `full_attention_interval = 4`, `head_count = 24`, `head_count_kv = 2`, `key/value_length = 256` | GGUF metadata; `src/models/qwen4exp.cpp:893-901` |
| MoE | `expert_count = 512`, shared-expert FFN = 640 | GGUF metadata |

**KV size arithmetic (self-consistency verified)**: only the **12 layers** with `(il+1) % 4 == 0` are full-attention,
`n_head_kv = 2`, `head_dim = 256` ⇒ K+V = 12 × 2 × 256 × 2 = **12 288 elements/token**;
at 256k, f16 = 262144 × 12288 × 2 B = **6.0 GiB** (consistent with the table in `sources/handoff.md` §6.42).
The 256k sizes of the other kv types are reproducible with the same formula (q8_0 3.19 GiB, q4_0 1.69 GiB, tbq4_0 1.52 GiB, tbq3_0 1.15 GiB).

### 0.5 Numeric conventions (mixing protocols invalidates a comparison)

All comparisons in this document are constrained by the following **protocol boundaries**; any comparison that violates a boundary is treated as invalid:

1. **KLD protocol**: wikitext-2 `wiki.test.raw`, `-c 512 --chunks 8`, `-fa 1`, base = f16 KV
   (`ppl/base-f16.kld`, base PPL = 2.0120). Tool `tools-kv-matrix.py kld` (**WIP branch only**).
2. **PPL(8ch)**: the same 8-chunk protocol, PPL absolute values on the order of 2.0.
3. **PPL(1ch)**: only the first chunk is run (`--chunks 1`), PPL absolute values on the order of 2.3–2.7.
   **8ch and 1ch cannot be compared with each other**: the same f16 base gives 8ch = 2.0120, 1ch = 2.4566;
   the same `q8_0` gives 8ch = 2.0128, 1ch = 2.3458. The difference comes from corpus size and variance, not from the effect of the KV type.
4. **1ch with batch variants**: `ppl-nb-*` uses `-b 512 / n_seq=1` (the other 1ch runs use `-b 2048 / n_seq=4`).
   Comparable within a family, not comparable across families.
5. **t/s**: any t/s appearing in this document may only be compared within the same table.
   The handoff text §6.29 records the 256k capacity table as "256 token, greedy", but **the exact invocation parameters were not kept with the logs**
   (`sweep-vision-cache.py --tokens` defaults to 400, `run-256k-*.ps1` uses 32; neither tool group used 256),
   so this table draws conclusions only from quantities **independent of generation length** (slots, cache MiB, hit rate); t/s serves as directional reference only.
   The t/s of the KV type matrix is another configuration. It **must not** be cross-compared with retracted old results (20.3 tps, 32 tps) or
   the 19 t/s threshold of 400-token/6144 MiB/fullRAM.
6. **Rotation**: `attn_rot_k/v` is a llama.cpp-side pipeline feature (Hadamard rotation on cache write / inverse rotation after read),
   **not part of the TBQ format** (`ggml/src/ggml-turboq.c:1-8` explicitly states "No rotation happens here").
   Any "with rotation / rotation disabled" comparison must state `LLAMA_ATTN_ROT_DISABLE`.

---

## 1 Terminology correction: the symbols for TQ3/TQ4 in this repository, and how they differ from TQ1/TQ2

The user's "TQ3/TQ4" are actually the symbols **`tbq3_0` / `tbq4_0`** in this repository.
They and the upstream `TQ1_0` / `TQ2_0` are **two entirely different format families**, easily confused only because the names all start with TQ:

| Symbol | Enum value | Role | Bits per element | Block structure | Covered in this document |
|---|---|---|---|---|---|
| `GGML_TYPE_TQ1_0` | 34 | **Ternary weight** format (upstream BitNet lineage), used for weights | 1.6875 | `d(fp16) + qh[4] + qs[48]` = 54 B/256 | **No** (not on this chapter's line) |
| `GGML_TYPE_TQ2_0` | 35 | **Ternary weight** format, used for weights | 2.0625 | `d(fp16) + qs[64]` = 66 B/256 | **No** |
| `GGML_TYPE_TBQ3_0` | 43 | **KV cache** format (TurboQuant codebook, added in this repository) | 3.0625 | `qs[96] + d(fp16)` = 98 B/256 | Yes (R-KV-01/02/03) |
| `GGML_TYPE_TBQ4_0` | 44 | **KV cache** format (same as above, 4 bit) | 4.0625 | `qs[128] + d(fp16)` = 130 B/256 | Yes (R-KV-01/02/03) |
| `GGML_TYPE_NXQ3_0` | 46 | **KV cache** format (NexusQuant E8, added in this repository) | 3.1875 | `d(fp16) + qs[96] + qh[4]` = 102 B/256 | Yes (R-NXQ-*) |
| `GGML_TYPE_NXQ2_0` | 47 | Same as above, 2 bit | 2.1875 | `d(fp16) + qs[64] + qh[4]` = 70 B/256 | Yes (R-NXQ-*) |

- Definition sites: `ggml/include/ggml.h:424-425` (TQ1_0/TQ2_0), `:433-434` (TBQ3_0/TBQ4_0, **released**),
  `:436-437` (NXQ3_0/NXQ2_0, **WIP working tree only**).
- Block structure: `ggml/src/ggml-common.h:275-288` (tq1/tq2), `:291-305` (tbq3/tbq4), `:306-324` (nxq3/nxq2).
- Key differences (at the mechanism level, not the naming level):
  1. **Different purpose**: TQ1/TQ2 are **weight** quantization (ternary: -1/0/+1 times a block scale, `ggml/src/ggml-quants.c:2316-2336`);
     TBQ/NXQ are **KV cache** quantization, with a continuous value range and per-block codebook/lattice approximation.
  2. **Different quantization domain**: TQ1/TQ2 normalize weights directly by absolute value (`d = amax`), whereas TBQ/NXQ work in the **rotated KV domain**
     (rotation is handled by the KV pipeline); TBQ uses the block L2 norm, NXQ uses a per-head fp16 scale of `amax/(levels/2)`.
  3. **Quality metrics cannot be borrowed across families**: in `tests/test-quantize-fns.cpp`, TQ and TBQ go through different thresholds/different reference paths,
     and the two TBQ items are **currently failing items in WIP** (see §3.1).
- **Naming discipline**: from here on this document uses only the repository symbols `tbq3_0` / `tbq4_0` / `nxq3_0` / `nxq2_0`;
  any occurrence of "TQ3/TQ4" always means the user's `tbq3_0` / `tbq4_0`.

---

## 2 Uniform record fields for each line of work

Every line of work is recorded with the same set of fields; missing items are written as "none / unknown" and never omitted:

**Route ID** · **Status/version** · **Why it was attempted** · **Technical mechanism** · **Experimental conditions and evidence (source + chapter/line range + original log name)** ·
**Observations and conclusion boundaries (what can be said, what cannot)** · **Reason kept or abandoned** · **Open questions / re-open conditions**

---

## 3 The KV-TBQ line

### 3.1 R-KV-01: the tbq3_0 / tbq4_0 format and endpoint integration

- **Status/version**: **released** (commit `2f1a363c8`, within the history of master `0862af564`).
- **Why it was attempted**: at 256k, KV occupancy is the first constraint on "how many slots the MoE expert cache has left"
  (f16 = 6.0 GiB), so a KV type was needed that is more economical than `q4_0` and theoretically better in quality.
  The selection was motivated by TurboQuant (arXiv:2504.19874): random rotation + per-coordinate Lloyd-Max,
  with the KV paper claiming quality neutrality at 3.5 bit/channel.
- **Technical mechanism** (within the readable source range):
  - Block = 256 elements; `d` = block L2 norm (fp16); before quantization the data is scaled up by `sqrt(QK_K)` and
    divided back out during dequantization, so that the codebook faces approximately unit-variance data (`ggml/src/ggml-turboq.c:19-27,69-100,101-135`).
  - The codebook/boundaries are Lloyd-Max constants (3-bit endpoints ±2.152, 4-bit endpoints ±2.7326,
    boundaries at midpoints): `ggml/src/ggml-turboq-tables.h:5-40`.
  - Packing: 3 bit, 3 bytes per 8 elements, LSB-first; 4 bit, two elements per byte (nibble order: even index assigned,
    odd index OR-ed into the byte just written).
  - GPU side: `ggml/src/ggml-cuda/cpy-utils.cuh:191-235` (device-side F32→TBQ3/TBQ4 quantization),
    `ggml/src/ggml-cuda/dequantize.cuh:123-150` (the `d/16` convention used by KQ, `1/16 = 1/sqrt(256)`),
    `ggml/src/ggml-cuda/convert.cu:112-160` (block-level dequantization),
    `ggml/src/ggml-cuda/cpy.cu:311-361`, `set-rows.cu:269-290` (KV write path).
  - **The format itself contains no rotation** (`ggml/src/ggml-turboq.c:1-8`); rotation is gated by
    `attn_rot_k/v` in `llama-kv-cache.cpp` (`src/llama-kv-cache.cpp:323-360`: quantized KV and `head_dim % 64 == 0`).
- **Experimental conditions and evidence**:
  - Round-trip error (**WIP branch only**, `tests/test-quantize-perf.cpp --op roundtrip`, introduced by commit `17ca0de85`):
    `tbq4_0` mse/var = **0.00836** (the paper reports 0.009 for 4 bit), `tbq3_0` = **0.021** (the paper reports 0.03 for 3 bit).
    Log: no separate small log (the values come from the commit message of `17ca0de85`).
  - `test-quantize-fns` (WIP working tree): `tbq3_0` absolute quantization error **0.003214 / FAILED**,
    `tbq4_0` **0.002025 / FAILED**; the threshold is still the original value 0.002, and **the threshold was not relaxed**.
    Log: `/tmp/nxq-align-test.log` (small, failure lines only).
- **Observations and conclusion boundaries**:
  - Can say: the format's encode/decode, codebook, and nibble/bit order are **consistent on both the host and the device side** (the source conventions agree with the synthetic round-trip probe),
    and the error on that probe matches the paper's order of magnitude.
  - Cannot say:
    1. It **cannot** be inferred from this that "PPL 65 is not an error in the format definition itself". A round-trip probe on synthetic data can only show
       "this probe did not reveal an obvious packing/convention error"; it can **neither attribute nor rule out** a numerical format error on the real KV distribution,
       an endpoint implementation issue, or an attention-path difference; this work did **not** perform a controlled end-to-end comparison
       (for example, feeding the same TBQ4 encoding result into another already-validated attention path).
       The FA path is at present only a **suspect**, not an exclusive localization (see R-KV-03).
    2. It cannot be said that these two FAILs are unrelated to PPL 65 (the threshold 0.002 is a historical constant, not a criterion defined for TBQ);
       nor can "the round-trip mse matches the paper" be inferred as "usable end to end".
- **Reason kept**: the format is kept; under that historical protocol `tbq3_0` is the KV type with the most 256k slots (see R-KV-02).
- **Open questions / re-open conditions**: these two `test-quantize-fns` FAILs are recorded as a **known deviation** (documented on file is sufficient),
  and **"adjusting/relaxing that threshold" is not the fix target**; if tests are to be added, the target should be an **end-to-end quality criterion aimed at KV usage**
  (for example a KLD/PPL threshold against the f16 base), rather than accommodating a historical constant defined for a weight format.

### 3.2 R-KV-02: the 256k KV type matrix (speed / MoE slots / PPL / KLD)

- **Status/version**: **WIP branch only** (commit `fa05f8637`, including `tools-kv-summary.py`, `run-ppl-matrix.sh`;
  the KLD tool `tools-kv-matrix.py` was introduced by `0d3d6c9d9` and is likewise not in master).
  The binary used by the measurement scripts is `build-ple-trace-mrs/bin/`.
- **Why it was attempted**: the 256k measurements in §6.29 always used `tbq4_0`, but no "KV type × quality × slots" comparison had been done,
  so a matrix was needed to separate "speed difference" from "cache slot count difference".
- **Technical mechanism**: `llama-cli` (`-c 262144`, `-ngl 49 --cpu-moe`, `-fa 1`) measures speed and
  `[MOE-CACHE]` slots/hits; `llama-perplexity` measures PPL and KLD (base = the logits of f16 KV).
- **Experimental conditions and evidence** (same rig, same model, serial single instance):

| KV | bpw | 256k KV | MoE slots | gen t/s | PPL(8ch) | PPL(1ch) | Mean KLD | 99.9% KLD |
|---|---|---|---|---|---|---|---|---|
| `f16` | 16 | 6.0 GiB | **0** | no timing line in the log | 2.0120 | 2.4566 | baseline | — |
| `q8_0` | 8.5 | 3.19 GiB | 29 | 15.7 | 2.0128 | 2.3458 | 0.030848 | 1.202733 |
| `q4_0` | 4.5 | 1.69 GiB | 45 | 16.1 | 2.0234 | 2.5026 | 0.056886 | 1.988480 |
| `tbq4_0` | 4.06 | 1.52 GiB | 47 | 15.3 | **no result** | **63.7301** | **no result** | — |
| `tbq3_0` | 3.06 | 1.15 GiB | **51** | 16.2 | 2.0728 | 2.7151 | 0.116397 | 2.714735 |

  Slots and cache MiB (read directly from the same batch of logs): `kv-f16` 0 slots / 0.0 MiB, `kv-q8_0` 29 slots / 2771.5 MiB,
  `kv-q4_0` 45 slots / 4300.6 MiB, `kv-tbq4_0` 47 slots / 4491.7 MiB, `kv-tbq3_0` 51 slots / 4874.0 MiB.
  The online hit rate of `tbq3_0` is 78.9% (`sweep-kv.csv`, hits=226005 / misses=60585, 33.5 s, peak 14496 MiB).
  **Original logs**: `kv-{f16,q8_0,q4_0,tbq3_0,tbq4_0}-{out,err}.txt`, `sweep-kv.csv`,
  `ppl-matrix.log`, `ppl-{f16,q8_0,q4_0,tbq3_0}.log`, `ppl-tbq4_0.log`, `ppl-8ch-tbq4-retry.log`,
  `ppl-single-*.log`, `ppl-nb-{f16,tbq4_0}.log`, `kld-matrix.log`, `kld-tbq4_0.log`;
  for the handoff text see `sources/handoff.md` §6.42.
- **Observations and conclusion boundaries**:
  - **Can say**:
    1. The KV type has very little direct effect on generation speed (15.3–16.2 t/s within this table); the real difference is **how many slots it leaves for the MoE cache**
       (0 → 51 slots, online hit rate 0 → 78.9%). This conclusion **holds only within this table**.
    2. At 256k, `f16` **squeezes the expert cache entirely down to 0**: the log `kv-f16-err.txt` reads verbatim
       `clamp cache budget from 6144 MiB to 0 MiB`, `0 slots/layer, 0.0 MiB physical cache`,
       `budget is smaller than one persistent layer slot; cache remains empty`,
       and `prefetch_required=89550 / prefetch_ready=0` ⇒ **at 256k, f16 is not a "quality baseline"
       but "cache switched off"**. The quality baseline is provided by the standalone f16 run of `--kl-divergence-base` (PPL 2.0120),
       not by this 256k configuration.
    3. The Mean KLD of `tbq3_0`, 0.1164, is 2.0× that of `q4_0` and 3.8× that of `q8_0` (the ratios are computed from the numbers in the table).
       **This is only a KLD ratio**; it does not constitute a "per-bit merit" conclusion: the bpw accounting conventions differ between types,
       and this experiment did not control the number of bits (K/V bit widths, block scale, and whether FA is used all differ).
    4. `tbq4_0` **is unusable under the tested 512-context protocol** (**do not** extrapolate to other context lengths):
       8 chunks aborts outright (both `ppl-tbq4_0.log` and `ppl-8ch-tbq4-retry.log` stop after
       `calculating perplexity over 8 chunks` with **no result line at all**; the corresponding line in `kld-tbq4_0.log` is entirely `-`);
       under 1 chunk, **PPL = 63.7301** (`ppl-single-tbq4_0.log`, same protocol as the other 1ch rows in the main table:
       `-b 2048 / n_seq 4`). There is also a batch probe: `ppl-nb-tbq4_0.log` (`-b 512 / n_seq 1`) gives
       **65.0026** — both batch variants are broken, so it is **not** the batch that caused it; but the two 1ch families cannot be cross-compared (§0.5).
       **Exit code 5 comes from commit `fa05f8637` and the record in §6.42**; the existing logs contain **no** exit-code line,
       and the log-level evidence is "abort, i.e. no result line".
       Wording of the conclusion: **not recommended for deployment under these protocols**; other lengths such as 256k are **unverified** (consistent with §6.8).
    5. Short greedy generation with `llama-cli` **looks normal** for `tbq4_0` (`kv-tbq4_0-out.txt` outputs
       "The capital of France is Paris…") ⇒ eyeballing and short generation are **not sufficient** to find this defect; PPL/KLD must be used.
  - **Cannot say**:
    1. PPL(8ch) and PPL(1ch) cannot be compared in the same column, nor can the 1ch value 63.73 be used to infer 8ch behaviour;
       nor can the `-b 512/n_seq 1` probe (65.0026) be conflated with the `-b 2048/n_seq 4` 1ch rows (§0.5).
    2. The t/s of `tbq3_0` and the t/s of `tbq4_0` cannot be treated as a comparison on "the same compute path" —
       the two **do not take the same path** (see R-KV-03).
    3. It cannot be said that "4 bit is necessarily better than 3 bit" (this table gives a counterexample, but the cause of that counterexample is a defect, not the format).
    4. **This table must not be read as a production recommendation**: it is a comparison measured under the historical "256k capacity first" protocol with
       the cache configuration of the `tbq4_0` era; `tbq3_0` still has an FA gap (not accepted by CUDA FA, the source route is CPU FA) and a short-corpus (8×512) KLD cost.
- **Reasons kept/abandoned**:
  - Kept: under that **historical protocol**, `tbq3_0` is recorded as the **capacity-first candidate** (most slots; t/s comparable to the other types in the same table,
    but this table is insufficient to support a "fastest" conclusion; the cost is KLD 0.116 and not being accepted by CUDA FA); `q4_0` is recorded as the higher-fidelity control.
    **Not a current formal production recommendation** (the FA gap is unresolved, quality was measured only under a short-corpus protocol, and quality at 256k length is unknown).
  - Abandoned: the use of `tbq4_0` in the KV position (unusable under the **tested 512-context protocol**, hence not currently recommended for deployment;
    other context lengths unverified).
- **Open questions / re-open conditions**:
  - The `f16` row lacks a timing line ⇒ one supplementary measurement is needed to give the t/s of f16@256k (**currently unknown**).
  - The quality of `tbq3_0` is still worse than `q4_0` (0.116 vs 0.057); should QUALITY ever take priority, re-open `q4_0`.
  - **Release-surface risk (important)**: the **released** `run-256k-tbq.ps1` and `run-8k-tbq.ps1` use exactly `tbq4_0`,
    and the default KV of `sweep-vision-cache.py` is also `tbq4_0` (its docstring likewise says TBQ4).
    That is: **the defective type is still in master's script defaults**. Switching to `tbq3_0`/`q4_0` is a source-code change, which this document does not make.

### 3.3 R-KV-03: rotation and FlashAttention path diagnosis (`tbq3_0` missing from the whitelist / `tbq4_0` takes FA but has bad quality)

- **Status/version**: **WIP branch only** (the diagnostic text of commit `17ca0de85`; the fix was **not done**).
  That commit changed only `tests/test-quantize-perf.cpp` (adding `--op roundtrip`).
- **Why it was attempted**: R-KV-02 showed the asymmetry "3 bit fine, 4 bit crashes", so the three possibilities
  "format error", "rotation error", "FA path error" had to be separated.
- **Technical mechanism (FA kernel selection, read directly from source)**:
  - Whitelist: `ggml_cuda_fattn_kv_type_supported()` in `ggml/src/ggml-cuda/fattn.cu:339-357`
    lists `F32/F16/Q4_0/Q8_0/BF16/`**`TBQ4_0`** — **there is no `TBQ3_0`**.
    This state is the same in **released master** (checked in the release tree, `ggml/src/ggml-cuda/fattn.cu:339-357` agree),
    i.e. **the missing whitelist entry is a released defect**, and only the diagnosis itself is WIP.
  - Kernel instantiations: `FATTN_VEC_CASES_ALL_D` in `fattn.cu:254-330` instantiates only
    `F16/F16, Q4_0/Q4_0, Q8_0/Q8_0, BF16/BF16, TBQ4_0/TBQ4_0` in a build **without** `GGML_CUDA_FA_ALL_QUANTS`;
    the template instantiation file is `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tbq4_0-tbq4_0.cu`
    (D = 64/128/256). **The whole repository has no FA instantiation for tbq3** (checked in all three places —
    `fattn-vec.cuh`, `CMakeLists.txt`, `template-instances/` — and found empty).
  - Dangling code: `ggml/src/ggml-cuda/fattn-common.cuh:374-400` (`vec_dot_fattn_vec_KQ_tbq3_0`) and
    `:557-580` (`dequantize_V_tbq3_0`) **exist but are not instantiated**, so they will not be selected in the compiled artifact.
  - **The true shape of the fallback (read directly from the release-tree source)**: with `-fa 1` the graph **still builds `FLASH_ATTN_EXT` directly**
    (`src/llama-graph.cpp:2702-2719`: `use_flash_attn = cparams.flash_attn && kq_b == nullptr`,
    and only F32 is cast to F16); it does **not** switch to another attention operator because the KV type belongs to TBQ.
    That operator is **rejected** by the CUDA backend: `ggml/src/ggml-cuda/ggml-cuda.cu:5564-5565` hands support determination over to
    `ggml_cuda_flash_attn_ext_supported()`, which is equivalent to
    "`ggml_cuda_get_best_fattn_kernel() != BEST_FATTN_KERNEL_NONE`" (`fattn.cu:589-591`),
    and the whitelist has no `TBQ3_0` ⇒ it returns NONE.
    Meanwhile the **CPU backend accepts** it: `supports_op` in `ggml/src/ggml-cpu/ggml-cpu.cpp:481-484`
    is `default: return true` for operators not enumerated, and the CPU `FLASH_ATTN_EXT` implementation
    (`ggml/src/ggml-cpu/ops.cpp:8614/8852/9212`) consumes TBQ3 directly through **K's quantization traits** —
    `ops.cpp:8686-8691` takes `ggml_get_type_traits_cpu(k->type)->vec_dot`,
    and `ggml/src/ggml-cpu/ggml-cpu.c:239-242` has already registered
    `ggml_vec_dot_tbq3_0_q8_K` for `TBQ3_0`. The scheduler picks the backend on a "whoever can support it gets it" basis (`ggml/src/ggml-backend.cpp:1235-1250`).
    ⇒ The route implied by the current source is "**CUDA FA not supported, node falls to the CPU's `FLASH_ATTN_EXT`**",
    and **not** "GPU dequantizes and then takes a non-FA path".
  - **Evidence boundary**: which backend the historical binary **actually** landed on at runtime can only be confirmed with scheduling/operator
    placement logs, and the existing logs contain **no** such records ⇒ **unproven**. The presence of
    `dequantize_row_tbq3_0_cuda` (`537/598/662`) in `ggml/src/ggml-cuda/convert.cu` only shows "this dequantization entry point exists",
    and **cannot** serve as evidence that "it was called".
- **Experimental conditions and evidence**:

| Probe | Key command points | Result | Original log |
|---|---|---|---|
| Rotation on (default) | `-ctk/-ctv tbq4_0 -fa 1`, 1 chunk | PPL **63.7301** | `ppl-rot-tbq4_0-rot_on.log` (= `ppl-single-tbq4_0.log`) |
| Rotation off | `LLAMA_ATTN_ROT_DISABLE=1`, otherwise identical | PPL **73.3126** (worse) | `ppl-rot-tbq4_0-rot_off.log` |
| Rotation on | `tbq3_0`, 1 chunk | PPL **2.7151** | `ppl-rot-tbq3_0-rot_on.log` |
| Rotation off | `tbq3_0` + `LLAMA_ATTN_ROT_DISABLE=1` | PPL **2.7284** (same order, within noise) | `ppl-rot-tbq3_0-rot_off.log` |
| FA off | `tbq4_0` + `-fa 0` | **context creation fails outright**: `quantized V cache requires flash_attn to be enabled` | `ppl-nofa-tbq4.log` |

  Rotation gating and on-graph rotation: `src/llama-kv-cache.cpp:323-360`,
  `src/llama-graph.cpp:2967-2980` (Q/K rotation) and `:3000-3010` (V output inverse rotation).
- **Observations and conclusion boundaries**:
  - **Can say**:
    1. The quality gap between `tbq3_0` and `tbq4_0` in R-KV-02 **is not a comparison on the same route**:
       `tbq4_0` hits the CUDA FA whitelist and the vec instantiation, while `tbq3_0` is not accepted by CUDA FA
       (source route = CPU `FLASH_ATTN_EXT`, see above) ⇒ the asymmetry "3 bit fine / 4 bit crashes"
       **points the suspicion at the FA path**. **This is a suspicion, not an exclusive localization**: a round-trip error probe on synthetic data
       can only show "this probe did not reveal an obvious packing error"; it **cannot** rule out a numerical format error on the real KV distribution,
       an endpoint implementation issue, or other path differences; this work did **not** perform a controlled end-to-end comparison of
       "the same TBQ4 encoding result on another already-validated attention path".
    2. Rotation **is neither the sole cause nor the solution**: with rotation disabled `tbq4_0` is still broken (73.31 ≫ 2.4566),
       while disabling rotation makes `tbq3_0` slightly worse (2.7284 vs 2.7151). So "the interaction between rotation and FA" is not ruled out,
       but "turning rotation off fixes it" has been refuted.
    3. **The FA-off arm does not hold for any quantized V**: `llama_init_from_model` rejects it outright
       (a quantized V cache requires FA) ⇒ the comparison "FA vs non-FA for quantized KV" **cannot be constructed** in this repository,
       and one can only compare indirectly via "different KV types on the same FA path" or "different routes of different KV types".
    4. **The speed difference cannot be attributed directly**: that `tbq3_0` does not get the benefit of CUDA FA is **inferable at the source level**,
       but the 15.3–16.2 t/s spread within the R-KV-02 table **cannot** be used on its own to corroborate the route (placement logs are missing,
       and the configurations differ between types).
  - **Cannot say**:
    1. The root cause **cannot** be said to be localized. `17ca0de85` explicitly lists "auditing the FA-specific rotation / Q-side rotation and the attention-output inverse rotation"
       as the **next step**; this document likewise does not claim a root cause, nor does it **rule out** causes at the format/endpoint level.
    2. The 1-chunk numbers in this table cannot be compared across the board with the 8-chunk numbers of R-KV-02.
    3. "The format round-trip matches the paper" cannot be turned into an assertion that "the FA kernel uses the same set of conventions" — the code that has been read
       (`vec_dot_fattn_vec_KQ_tbq4_0` `fattn-common.cuh:334-372`, `dequantize_V_tbq4_0`
       `:532-556`, `dequantize_tbq4_0` `dequantize.cuh:123-133`) is **self-consistent**,
       but self-consistency is not the same as being validated end to end.
    4. The "CPU fallback" **cannot** be written as an event that happened: there is no log evidence for the actual runtime placement (**unproven**).
- **Reasons kept/abandoned**: the diagnostic conclusion is kept (asymmetry ⇒ suspicion on the FA path); the FA fix for `tbq3_0` was **not done**,
  because once the whitelist is fixed, `tbq3_0` would **enter** the more strongly suspected FA path, and the FA defect of `tbq4_0` must first be understood,
  otherwise a "usable `tbq3_0`" could be traded for "a broken FA version".
- **Open questions / re-open conditions** (all **still open**):
  1. The exact defect point of the `tbq4_0` FA path (not localized; candidate areas: FA-specific rotation handling, vec-kernel KQ/V dequantization,
     the interaction between the `d/16` and block-L2-norm conventions — **all are unverified guesses and must not be treated as conclusions**).
  2. Adding `TBQ3_0` to `ggml_cuda_fattn_kv_type_supported()` plus new tbq3 vec instantiations
     (template instantiation file + `CMakeLists.txt` + `FATTN_VEC_CASES_ALL_D`) — **not done**.
     The **precondition** for re-opening is that item 1 has a conclusion.
  3. Whether `tbq3_0` keeps the current source state (CPU `FLASH_ATTN_EXT`) or gets CUDA FA instantiations:
     this requires redoing the R-KV-02 comparison once item 1 has a conclusion and placement logging has been added.

### 3.4 R-KV-04: 256k capacity and cache budget configuration (§6.29)

- **Status/version**: the scripts are **released** (`run-256k-*.ps1`, `sweep-vision-cache.py`, `tools-run.py` are all in master);
  the measurement artifacts (`cases-auto*.txt`, `{tag}-out.txt` / `{tag}-err.txt`, `sweep-*.csv`, etc.) are
  untracked/gitignored in the source worktree and **are listed in the §9 archive candidate list** of this document (collected and indexed by the archival process).
- **Why it was attempted**: under 256k + vision encoder, how much VRAM should the MoE expert cache get? Does manual configuration miss the adaptive room?
- **Technical mechanism**: with `LLAMA_MOE_CACHE_MIB=auto`, `budget = vram_limit − device_used − guard`
  (the auto branch of `moe_cache_apply_vram_limit`); a manual positive value only gets **clamped down**, it never grows adaptively.
- **Experimental conditions and evidence** (256 token generation, greedy, `-ctg 256k`, KV **`tbq4_0`**, host mode, serial single instance):

| Configuration (budget source) | slots/layer | effective | online hit | gen t/s | Log |
|---|---|---|---|---|---|
| 8k, no vision (`auto`) | **80** | 7667 MiB | 76.3% | 17.4 | `auto-8k-novis-{out,err}.txt` |
| 256k, no vision (`auto`) | 37 | 3625 MiB | 65.4% | 15.8 | `auto-256k-novis-{out,err}.txt` |
| 256k, +vision encoder (**explicitly requested 8192**, clamped to 3881; **not** an `auto` budget) | 40 | 3881 MiB | 64.6% | 13.2 | `auto-256k-vis-{out,err}.txt` |
| 256k+vision, manual cap 2700 (two repeats) | 28 | 2700 MiB | 58.4 / 56.5 | 15.3 / 14.7 | `cap27-a-{out,err}.txt`, `cap27-b-{out,err}.txt` |

  The cases are defined by `cases-auto.txt` / `cases-auto2.txt` (tag = `auto-8k-novis`, `auto-256k-novis`,
  `auto-256k-vis`, `cap27-a|b`), and the results land in `{tag}-out.txt` / `{tag}-err.txt`.
  **The tag name is misleading**: the log of the `auto-256k-vis` run reads verbatim
  `clamp cache budget from 8192 MiB to 3881 MiB (device used=11734 MiB limit=15872 MiB guard=256 MiB)`,
  i.e. **an explicit high budget was requested and then clamped down**, not `LLAMA_MOE_CACHE_MIB=auto`; moreover it used guard=256,
  whereas `auto-256k-novis` used guard=512, so the two runs are not strictly the same parameters.
  For the handoff text see `sources/handoff.md` §6.29. **Be careful to distinguish the datasets**: `cases-vision256k.txt` +
  `sweep-vision.log` + `sweep-vision256k.csv` are a **separate single-factor sweep done later**
  (fixed `LLAMA_MOE_CACHE_MIB=6144`, baseline 27 slots / 57.4% hit, KV likewise `tbq4_0`),
  **not** the source of that §6.29 table; the launch scripts themselves leave no trace in the logs, so "256 token" can only be cited from the handoff text.
- **Observations and conclusion boundaries**:
  - **Can say (slots and hits only)**:
    1. Under the same 256k+vision configuration, the explicit high-budget sample leaves more slots than the manual cap 2700 (measured from the logs: 40 vs 28 = **+12 slots**),
       with a higher online hit rate (64.6% vs 58.4/56.5%). **Note**: the §6.29 text says "9 slots more" — 9 is exactly
       the difference between "256k without vision auto 37" and "cap 28", which is an **inconsistency in convention between the text and the logs**; this document follows the logs.
    2. **No speed direction was established**: the 13.2 t/s of that high-budget sample is **lower** than the cap's 15.3 / 14.7 t/s,
       the opposite of "faster"; the auto side has only a single sample while the cap side has two repeats ⇒ it **cannot** be written as "slightly better speed",
       nor can these samples prove that "`auto` gains on speed".
    3. Turning off the PLE-GPU cache (`LLAMA_PLE_GPU_CACHE_MIB=0`) is more worthwhile in the full-RAM scenario
       (the evidence is the 17.0 → 17.9 t/s in handoff text §6.29; this document found no separate small log, so it is marked
       **[handoff-text record, not independently verified]**).
    4. At 256k the run-to-run variance of t/s is large (13.2–17.9 for the same configuration, ≈35%): **a single measurement is insufficient to determine an "optimum"**,
       and the absolute values are unstable (suspected Laptop GPU throttling).
  - **Cannot say** (important):
    1. **256k is a "capacity configuration", not "validated 256k-corpus quality"**: this group of experiments measured only slots/hits/short-generation speed,
       and has **no** PPL/KLD at 256k length at all. All quality numbers come from the `-c 512 --chunks 8` or `--chunks 1`
       wikitext-2 protocol (§0.5). Treating 256k capacity numbers as quality evidence is an **invalid inference**.
    2. The KV of this group is `tbq4_0`, and that type is unusable under the **tested 512-context protocol** (R-KV-02).
       The conclusions therefore apply only to the **VRAM accounting and cache budget mechanism** (the mechanism is independent of the KV numerics);
       the t/s here **cannot** be treated as "the performance of a deployable configuration", nor can `tbq4_0` be recommended on this basis.
    3. "More slots" cannot be read as "faster": among these samples the high-budget sample actually has the lower t/s (13.2 vs 15.3/14.7).
    4. The hit-rate numbers are **unrelated** to the user-reported 99% SMoE teacher hit rate, and the two cannot be cited as evidence for each other.
- **Reason kept**: what is kept is the **mechanism-level** judgement — letting the budget adapt to the upper limit (instead of hand-filling a rather small number)
  leaves the expert cache more room, and the auto branch works as expected on `auto-8k-novis` / `auto-256k-novis`;
  what is **not** kept is the claim that "`auto` is necessarily better than a manual cap in speed".
- **Open questions / re-open conditions**: re-running the same 256k+vision experiment set with a **usable** KV type (`tbq3_0` or `q4_0`)
  (with repeats for both `auto` and manual cap) is required in order to confirm the optimum of the cache budget under the premise of "usable quality";
  at the same time the two conventions in the text, "+9 slots" and "slightly better speed", should be corrected.

---

## 4 The NXQ / E8 line (NexusQuant alignment)

**Overall status: everything exists only in the uncommitted working tree of the source worktree (`ggml/src/ggml-nexusq.c` etc. are untracked),
has not entered master, and has not entered any commit.**

**Reference object and source (accessibility)**: `nexusquant-kv` **0.6.3**, PyPI release page
<https://pypi.org/project/nexusquant-kv/0.6.3/>; the package metadata (`PKG-INFO` / `pyproject.toml`) declares
author Joao Marques, License Apache-2.0, `Project-URL: Repository = https://github.com/jagmarques/nexusquant`.
Every "reference behaviour" in this section is cited only from that **sdist's source and its README** (source worktree unpack path
`/tmp/nq/nexusquant_kv-0.6.3`); the original was **not installed or run on this machine**, and its Python/CUDA path was never executed;
so "reference" in this document = source-level reading, not a runtime comparison.

**Encoder convention (a limitation running through this whole section)**: although mode 1 / mode 2 in this repository replicate the reference's **chain shape**
(de-RoPE → Hadamard → quantization → inverse Hadamard → re-RoPE), **the quantization step itself is still our own restricted encoder**
(`ggml_cast(x, NXQ3_0/NXQ2_0)`: a narrower coordinate domain than the reference, an explicit 1-bit coset per 8 elements, and fixed-length packing).
These experiments are therefore **not** "the result of running the reference encoder on this machine", and cannot be used to draw conclusions about the quality of the reference method.

### 4.1 R-NXQ-01: endpoint integration and early experiments (including nkvo and the "no coset marker" format) — **retracted**

- **Status/version**: **retracted**. The early version has been superseded by the implementation of §6.43/§6.44; its data **must not** serve as quality evidence
  (`sources/handoff.md` §6.43 verbatim: "the NXQ round-trip error of the old no-coset-marker format and the nkvo experiment cannot serve as
  quality evidence for this implementation").
- **Why it was attempted**: after the 4.06 bpw `tbq4_0` proved unusable, a more economical KV compression scheme was needed that also "has a reference implementation to compare against"
  (E8 lattice quantization + whole-head Hadamard).
- **Technical mechanism and the pitfalls of that time** (readable from the original logs):
  1. **CUDA does not support SET_ROWS**: `nxq_ppl.log` verbatim
     `pre-allocated tensor (cache_k_l3 (reshaped) (view)) in a buffer (CUDA0) that cannot run the
     operation (SET_ROWS)` ⇒ the first version never ran at all.
  2. **The `-nkvo` path crashes**: `nxq_ppl_nkvo.log`, `nxq_base_f16_nkvo.log`, `nxq_base_f16_nkvo2.log`
     verbatim `GGML_ASSERT(id >= 0 && id < n_expert) failed`.
  3. **Different baseline conventions**: the base produced by `nxq_base.log` and `nxq_sanity_noenv.log` with `-nkvo` is
     `ppl/base-f16-nkvo.kld` (PPL **2.0347**), whereas the current KLD protocol uses `ppl/base-f16.kld`
     (PPL **2.0120**) ⇒ **the KLDs of the two bases are not comparable**.
  4. The early "no coset marker" format: the half-integer coset information of E8 was lost to rounding, equivalent to losing half the lattice resolution;
     the current implementation stores 1 bit of coset marker explicitly per 8 elements in `qh`
     (`ggml/src/ggml-common.h:306-324`, `ggml/src/ggml-nexusq.c:104-152`).
- **Observations and conclusion boundaries**: one can say "the early version is unusable and has been invalidated"; one **cannot** cite any of its PPL/round-trip errors to evaluate
  the NexusQuant method or the current implementation.
- **Re-open conditions**: it is not re-opened; its historical value lies only in showing that the two classes of pitfalls, "CUDA support surface" and "baseline conventions",
  must be written clearly in the protocol.

### 4.2 R-NXQ-02: encoder alignment — "project first, then choose" replaces "choose first, then saturate"

- **Status/version**: **WIP working tree only** (changed synchronously on both sides, `ggml/src/ggml-nexusq.c` and `ggml/src/ggml-cuda/cpy-utils.cuh`).
- **Why it was attempted**: the old approach chose candidates by "unprojected distance" and then saturated into the storable range, which caused **the largest coordinate to be consistently off by 1 more step**.
- **Technical mechanism**:
  - E8 nearest point = integer-coset candidate (even coordinate sum, Conway-Sloane correction) + half-integer-coset candidate
    (**deliberately relaxing the parity constraint, as the reference does**); each of the two is **first projected** into the range representable in this format
    (integer `[-maxq, maxq-1]`, half-integer `[-maxq+0.5, maxq-0.5]`) **and only then compared by distance**
    (`ggml/src/ggml-nexusq.c:38-97`, including the verbatim comments; GPU side `cpy-utils.cuh:255-306`).
  - Coordinate packing: 3 bit/2 bit LSB-first + `qh` with 1 coset bit per 8 elements (`ggml-nexusq.c:104-174`).
  - scale: per-head (one block = one 256-dimensional head) `amax/(levels/2)`, stored as fp16;
    **during quantization the fp16-rounded scale that will be stored is used** (avoiding different scales for encoding and decoding),
    with protection for `amax < FLT_MIN` and `sc == 0`, eliminating the divide-by-zero from a scale underflowing to 0
    (`ggml-nexusq.c:113-124`; see also R-NXQ-07).
- **Experimental conditions and evidence** (absolute error from `test-quantize-fns` quantization, and normalized MSE ratio against the reference;
  source `sources/handoff.md` §6.44; original logs `nxq-align-test.log`, `nxq-align-test-build.log`):

| Metric | Old (saturate after choosing) | New (choose after projecting) |
|---|---|---|
| Absolute quantization error, 3 bit | 0.002881 | **0.002367** |
| Absolute quantization error, 2 bit | 0.006117 | **0.004857** |
| Normalized MSE (Gaussian) 3b / 2b | 1.037 / 1.049 | **1.016 / 1.020** |
| Normalized MSE (uniform) | 1.257 / 1.475 | **1.095 / 1.135** |
| Normalized MSE (sine) | 5.15 / 8.02 | **1.83 / 2.40** |
| Coordinate agreement rate with the reference | 99.64% | 97.80% (**a deliberate decrease**) |

  Also: after recompilation the CPU kernel and the design-draft numpy replication are element-wise identical (6 test cases, 0 differences, §6.44).
- **Observations and conclusion boundaries**:
  - Can say: the new encoder is better on **every error** metric, and **the drop in coordinate agreement with the reference is expected** —
    it trades a deviation of 2 half-steps for a deviation of 1 half-step at the endpoint. **The agreement rate is not a quality metric; the MSE is.**
  - Cannot say: this is **not the same implementation** as the reference, only a **fixed-point approximation** of it (this format's coordinate domain is narrower than the reference's).
    The wording of §6.44, "a fixed-point approximation of the reference fake-quant result", is the accurate convention; this document follows it and **does not claim a full reproduction**.
- **Reason kept**: kept (better MSE + a consistent explanation).
- **Open questions**: 3 bit can only hold 8 integer codes (the reference's clamp domain has 9 values) — this is a
  structural difference caused by **format capacity**, not an encoder defect; if 3 bit is ever extended to "8 values + special endpoint encoding", this needs re-evaluation.

### 4.3 R-NXQ-03: V rotation changed to the whole head

- **Status/version**: **WIP working tree only** (`build_input_v_rot` now uses the same rule as K; for this model = 256).
- **Why it was attempted**: the reference also applies a **whole-head** Hadamard to V (`nexusquant/integrations/quantized_cache.py`
  `_e8_values`: `einsum("bhsd,de->bhse", v, H)`, where H is `hadamard_matrix(d)`, d = head_dim);
  the old implementation rotated only 64 dimensions.
- **Experimental conditions and evidence** (`tools-kv-matrix.py kld`, 8×512 chunks, against the f16 base, K3V2 = `nxq3_0/nxq2_0`):

| KV | V rot=64 (old) | V rot=256 (new, kept) |
|---|---|---|
| `q8_0/q8_0` | 0.030848 / 99.9% 1.203 | 0.033869 / 99.9% **0.913** |
| `nxq3_0/nxq2_0` | 0.266610 / 99.9% 6.917 | **0.256936 / 99.9% 4.700** |

  Logs: `/tmp/nxq-align-kld-v64.log` (old), `/tmp/nxq-align-kld.log` (new).
- **Observations and conclusion boundaries**:
  - Can say: NXQ mean KLD −3.6%, tail −32% ⇒ the new rule is kept; `q8_0@64 = 0.0308` agrees with the record in §6.42
    ⇒ the protocols are comparable (this serves as a cross-check that the protocol is "the same").
  - **Must also be recorded**: this rule **affects all quantized KV types**: for `q8_0` the mean KLD goes 0.0308 → 0.0339 (tail 1.203 → 0.913).
    Therefore the KLD of R-KV-02 and the KLD after R-NXQ-03 **are not on the same convention** (q8_0 changes from 0.0308 to 0.0339),
    and any cross-table citation must state whether it is the "V rot=64 era" or the "V rot=256 era".
- **Reason kept**: kept (it improves both metrics and agrees with the reference).
- **Open questions**: why a global rotation makes the tail of `q8_0` better while making its mean worse is **unknown** (no mechanism analysis was done).

### 4.4 R-NXQ-04: local control experiment for an fp16 fake-quant of the reference chain shape (`LLAMA_KV_E8_FAKEQUANT`)

- **Status/version**: **WIP working tree only** (the environment switch in `src/llama-impl.h`, the chain in `src/models/qwen4exp.cpp`,
  the gating in `src/llama-graph.cpp`, `build_inp_pos_neg()`).
- **Why it was attempted**: the implementation of R-NXQ-01~03 lacks the de-RoPE chain for K and the decode residual, and **cannot be deployed**
  by means of an fp16 runtime plus offline accounting (llama.cpp needs a real KV storage type). This experiment therefore does only two **local** things:
  ① **control "with or without de-RoPE"** on the same fake-quant chain; ② land the quantization result in **two materialized representations**
  (on-graph fake quant keeping the f16 cache vs packing into `nxq3_0/nxq2_0`). It is **not capable** of evaluating the reference method itself
  (the encoder is still our restricted implementation, see the start of §4).
- **Technical mechanism (a key semantic clarification)**: the reference's **runtime** cache (`E8QuantizedCache.update`)
  stores the **dequantized fp16** (fake quant): the chain "de-RoPE → Hadamard → E8 → inverse Hadamard → re-RoPE"
  is completed and then written back as fp16; `int8 + temporal differencing + zstd` appears only in
  the **offline measurement** path `compression_accounting.measure_compression()`
  (that function stores coordinates as `(lp*2).round().to(int8)`, using ×2 to preserve the half-integer coset), and **it is not the runtime format**.
  ⇒ "3 bit" is a nominal value; "runtime quality" and "offline byte rate" are **two different things**,
  and this repository's fixed-length KV storage type and the reference's offline accounting **are not on the same level**: the former is the
  KV type actually occupied by llama.cpp at runtime, the latter is a **side-channel measurement**. The two **cannot be called "substitutes" for each other**, and their byte rates cannot be aligned directly.
  - The chain shape on this machine (expressed with **existing operators** inside `models/qwen4exp.cpp::build_layer_attn`, cache kept at f16):
    ```
    mode 1: k: R(-p) → H → cast(NXQ3_0) → cast(F32) → H → R(p)
    mode 2: k:            H → cast(NXQ3_0) → cast(F32) → H
    v:                    H → cast(NXQ2_0) → cast(F32) → H
    ```
    `ggml_cast(x, NXQ3_0)` goes through **this repository's** quantizing cpy (block = one 256-dimensional head),
    and **not** the reference's Python E8 encoder.
  - When this mode is entered, `attn_rot_k/v` is forced on (solely to obtain the Hadamard matrix), and the model-side
    "rotate before write / inverse-rotate after read" is gated off according to mode + arch.
- **Experimental conditions and evidence** (same protocol: wikitext-2, 8×512 chunk, against the f16 base, `-c 512`, K3V2).
  The `±` comes from `tools/perplexity/perplexity.cpp:1770–1777,1948–1949`'s
  `sqrt((sum2/N - mean²)/(N-1))`: a **mean standard-error estimate** computed over the effective scoring positions,
  not the standard deviation of the 8 chunks, and not a confidence interval. That computation does not correct for token correlation;
  this experiment performed no paired significance or equivalence test across variants.

| Variant (the NXQ rows use the same local restricted encoder; q8_0 / f16 are separate controls) | Mean KLD | 99.9% KLD | PPL(Q)/PPL(base) |
|---|---|---|---|
| `q8_0` (packed, V rot=256 era) | 0.0339 | 0.913 | — |
| mode 1: K de-RoPE + on-graph fake quant (cache kept at f16) | 0.260734 ± 0.013948 | 5.612058 | 1.109624 ± 0.024911 |
| mode 2: K without de-RoPE + on-graph fake quant (cache kept at f16) | 0.256116 ± 0.014663 | 7.917815 | 1.106155 ± 0.024202 |
| packed `nxq3_0/nxq2_0` (real KV storage type, after R-NXQ-03) | 0.256936 | 4.700374 | — |
| packed `nxq2_0/nxq2_0` | 0.320371 | 6.559639 | — |
| `f16` baseline (mode 0) self-check | — | 0.000042 | — |

  Logs: `/tmp/e8-kld-all.log` (containing the mode 0 self-check 4.2e-5 and the mode 1 summary), `/tmp/e8-m1b.log` (mode 1),
  `/tmp/e8-m2b.log` (mode 2), `/tmp/e8-protect.log` (boundary protection),
  `/tmp/e8-m2-direct.log` (mode 2 direct connection failed: `GGML_ASSERT(buf != NULL && "tensor buffer not set")`).
- **Observations and conclusion boundaries**:
  - **Can say (only these two are controlled observations)**:
    1. **Adding K's de-RoPE chain** (mode 2 → mode 1) showed no lower mean KLD:
       `0.256116 → 0.260734`. The two means differ by about `0.0046`; their standard errors are not
       simply added, nor is the paired covariance ignored, and no statistical equivalence is claimed on this basis.
    2. For the local encoder, the on-graph fake-quant f16 cache and the packed `nxq3_0/nxq2_0` give
       means of `0.2561–0.2607` and `0.2569` respectively. These point values are close,
       but with no independent original-reference-encoder control, this cannot prove that the representation is costless or that the two implementations are equivalent.
  - **Cannot say** (this record explicitly tightens this):
    1. **Cannot** write "the hypothesis that this implementation is worse than the reference is refuted", "the three agree with the reference", or "it is not an implementation degradation" —
       modes 1/2 are **not the reference encoder** (restricted coordinate domain + explicit coset bits + fixed-length packing),
       and no comparison was ever made with "the output of stock NexusQuant on this machine"; this experiment **cannot** support any conclusion about
       the reference method's quality ceiling, or any "us vs reference" attribution.
    2. `KLD ≈ 0.26` **cannot** be stated as "a property of this method + configuration + protocol" — it can only be
       "an observed value of the **local restricted encoder** under this protocol".
    3. **No** "per-bit merit" conclusion can be drawn (the bpw accounting conventions differ between types, and this experiment did not control the number of bits).
    4. "De-RoPE is not worth doing" can only serve as an **engineering judgement under this protocol (no benefit observed)**,
       not as a conclusion that "the step is useless in the method" (for example, its behaviour for per-token generation or long contexts was not measured).
- **Reason kept**: kept as a **local control tool** (later changes can be A/B-tested in the same mode with clear variable control).
- **Open questions**: how much of the difference between the local restricted encoder and the reference encoder (coordinate domain of 8 values vs 9 levels, explicit coset bit,
  fixed length vs variable code rate) each contributes is **unknown**, and can only be decomposed with a genuine reference-encoder comparison path.
  Whether `KLD ≈ 0.26` has already reached the order-of-magnitude ceiling of this method is **unknown**.

### 4.5 R-NXQ-05: boundary-protection experiment (`LLAMA_KV_E8_PROTECT`) — the layer-count labels need correcting, the mapping semantics are unknown

- **Status/version**: **WIP working tree only** (`LLAMA_KV_E8_PROTECT=N` ⇒ the first and last N **block indices** each skip the round-trip;
  `src/llama-impl.h:74-78`, `src/models/qwen4exp.cpp:2042-2045`).
- **Why it was attempted**: the reference README requires `protect_boundary=2` for the Qwen family
  (verbatim: "Qwen-family models catastrophically fail without `protect_boundary=2`",
  "Boundary-protect first/last N layers at FP16 (**mandatory for Qwen-family**)",
  `/tmp/nq/nexusquant_kv-0.6.3/README.md` around lines 105, 125-128, 145).
  That reference parameter appears in the **eviction/compression pipeline** (`pipeline.py:1088-1092`, `huggingface.py:738`),
  taking `skip_layers.add(i)` and `skip_layers.add(n_layers - 1 - i)` over the layer list of `past_key_values` —
  i.e. it is **defined on the dense model's layer order in which every layer has KV**; `E8QuantizedCache` itself has no such parameter.
- **Experimental conditions and evidence** (mode 1, 8×512 chunk, against the f16 base; `/tmp/e8-protect.log`):

| `LLAMA_KV_E8_PROTECT` | "number of quantized layers" recorded in the handoff | Mean KLD | 99.9% KLD | PPL(Q)/PPL(base) |
|---|---|---|---|---|
| 0 | 48/48 | 0.260734 ± 0.013948 | 5.612058 | 1.109624 |
| 2 | **44/48** | 0.252590 ± 0.013904 | 5.998749 | 1.104169 |
| 4 | **40/48** | 0.240551 ± 0.013238 | 6.260358 | 1.088559 |

- **Observations and conclusion boundaries**:
  - **Corrected (fact)**: the handoff's "48/48, 44/48, 40/48" counts all blocks,
    whereas only 12 layers have KV; recomputed for this model it should be **12/12, 11/12, 10/12**
    (N=2 covers only `il=47`, N=4 covers `{3,47}`, see section 5).
  - **The data itself is valid**: this is a **restricted but real** measurement — `PROTECT=2` actually protected **1** layer that has KV,
    and `PROTECT=4` protected **2**; the KLD decreases monotonically (0.2607 → 0.2526 → 0.2406). The data is kept, not invalidated.
  - **Conclusions that must be retracted**:
    1. "Under the reference semantics, the first 2 / last 2 **attention layers** should be protected" — this is an **inference**.
       The reference parameter is defined on the dense model's layer order, and **the reference does not define how it should map
       onto a hybrid architecture (where some layers have no KV)**, nor does this repository implement a mapping rule ⇒ **the mapping is unknown**.
    2. "This experiment in fact did not test the configuration the reference requires / the protection window is wrong" — likewise beyond the evidence:
       it was a **restricted** experiment (1–2 layers with KV were protected), so it cannot be asserted that "no protection is done on this model" or that
       "the reference requirement is unmet"; one can only say that **the number of attention layers covered by the window is far less than 4**.
    3. "The benefit is linear", "each protected layer ≈ −0.005 KLD", "about 80% of layers need protection" — with two data points and a wrong denominator,
       **the extrapolation does not hold** (see section 5.3, which also retracts the earlier opposite conclusion of this document).
    4. "Boundary protection is not the solution" — **retracted**. The available data covers only the three points N=0/2/4, and the window mapping is undetermined,
       which is insufficient for any directional conclusion about this strategy.
  - **A design that can be proposed (a proposal, not a proven defect)**: if the protection window is to be interpretable on a hybrid architecture,
    it could be defined by **attention-layer ordinal** (taking the first N / last N of the 12-layer sequence with `(il+1) % 4 == 0`,
    for example N=2 ⇒ protect `il ∈ {3,7}` and `{43,47}`), while **also reporting the number of protected layers that have KV**.
    Whether that agrees with the reference semantics is **unknown**, and it can only be confirmed after aligning with the reference authors'/source's layer-order definition.
- **Reasons kept/abandoned**: **the data and implementation are kept**; only three conclusions are retracted — the layer-count labels, the linear extrapolation, and "boundary protection is not the solution".
- **Re-open conditions**: ① settle the mapping convention (or explicitly declare this experiment a custom variant under a "block index window");
  ② run each of N=0..4 (preferably including a variant by attention-layer ordinal) ≥1 time and report the number of protected layers;
  ③ to evaluate "how much boundary protection can bring", at least one endpoint of "protect all attention layers" (= no quantization) is needed for calibration.

### 4.6 R-NXQ-06: unaligned / unimplemented items (structural gaps, not bugs)

| Item | Reference behaviour | This implementation | Status | Evidence/source |
|---|---|---|---|---|
| K's quantization domain | **De-RoPE first** (`inverse_rope` → H → E8 → inverse H → `forward_rope`) | Quantization with RoPE (packed version); mode 1 expresses the reference's chain order (the encoder is still this machine's) | packed version has a **shape gap**; mode 1 is shape-aligned only | `quantized_cache.py::_e8_keys`; `sources/handoff.md` §6.44 |
| decode fp16 residual | Single-token writes (`seq < min_seq_for_quant=2`) keep fp16 (KIVI-style) | Everything is quantized | **not implemented** | `quantized_cache.py` lines 84, 146-149; llama.cpp's KV buffer has only a single type |
| Byte rate | The runtime cache is **fp16 fake quant**; `int8 + temporal differencing + zstd` is only the **offline measurement** of `measure_compression()` | A real KV storage type `nxq3_0/nxq2_0` (fixed length 3.1875 / 2.1875 bpw) | **different levels, cannot be called "substitutes" for each other** | `compression_accounting.py`; `ggml/src/ggml-common.h:306-324` |
| Boundary protection | `protect_boundary=2` (recommended for the Qwen family); defined over the **dense model's layer order** | A block-index window, not mapped to attention layers | **restricted experiment** (mapping unknown, see R-NXQ-05) | reference README; `pipeline.py:1088-1092`; `qwen4exp.cpp:2044-2045` |
| E8 coset | Half-integer coset parity constraint **deliberately relaxed**; offline accounting uses `(lp*2).round()` to preserve the .5 | 1 explicit bit per 8 elements to store the coset | different representation (we use 1 bit, the reference uses ×2 integer levels) | `e8_lattice.py:40-53`; `ggml-nexusq.c:104-152` |

- **Why "K de-RoPE" is not treated as a priority for now**: `/tmp/e8-m1b.log` vs `/tmp/e8-m2b.log`
  (0.2607 vs 0.2561) showed **no observed mean improvement** under this protocol (and no equivalence test was done),
  while implementing it would require a fused FA kernel that "reads packed NXQ + position table directly" (the current design is `NXQ → F16 → off-the-shelf F16 FA`,
  where `ggml_cast` materializes the whole layer's cache as F16; covering the entire cache with rope on the graph costs O(n_ctx)/token/layer,
  unacceptable at 256k). This is a **engineering-priority judgement**, not a conclusion that "the step is useless".
- **Why the decode residual does not affect the numbers in this document**: this protocol writes in chunks, so the residual path does not participate
  ⇒ the numbers of R-NXQ-04 are unaffected by it; but it **does affect the deployment quality of per-token generation**, and is an item that is **still open**.
- **Scope of applicability (a strict statement that must be cited together with any NXQ conclusion)**:
  the reference's own README states "Models with **4+ KV heads** are safe at K3V2 (the Qwen family needs
  `protect_boundary=2`)", and in the same table's Qwen row marks Qwen2.5-1.5B with **2 KV heads** as
  **catastrophic / not recommended**. **This model's `head_count_kv = 2`** falls within that warning range.
  But this is the **reference author's empirical scope statement**, not proof that "every 2-KV-head model is necessarily catastrophic":
  - The NXQ numbers in this document correspond only to the specific combination "2 KV head + head_dim 256 + 12/48 full-attention layers + local restricted encoder +
    this protocol", and **must not** be extrapolated;
  - **Nor may** a re-test conclusion be presupposed in the other direction (one can neither say "NXQ is unsuitable for this model", nor that "an N1/N2 re-test will necessarily improve things");
    the reference's results on 4+ KV head dense models (Mistral-7B K3V2 pb=0: +0.276% PPL, 2.625 bpe;
    Llama-3.1-8B NIAH 30/30) are **not comparable** with the numbers here, and can only serve as background risk information.

### 4.7 R-NXQ-07: endpoint correctness defects and accounting conventions (fixed / explicitly chosen and kept)

| Defect | Symptom / risk | Handling | Current evidence |
|---|---|---|---|
| The CUDA decode treats a 256-element block as a 32-element block | Device-side decoding misses elements ⇒ KV read back incomplete (symptom of the first version) | **Fixed** (recorded in handoff §6.43) | The current source `ggml/src/ggml-cuda/dequantize.cuh:154-171` uses `QK_K` (=256) as the block and takes the `iqs` / `iqs + QK_K/2` coordinates in pairs; the regression is covered by the NXQ CPY and SET_ROWS cases of `test-backend-ops` (`/tmp/nxq-tbo-cpy.log`, `/tmp/nxq-tbo-sr.log`, `2/2 backends passed`). **The old form is no longer visible in the current source**, so this item can only cite the handoff record and cannot be reconstructed from the source |
| Divide-by-zero from a scale underflowing to 0 | `amax` extremely small, or the scale becoming 0 after fp16 rounding | **Fixed** | `ggml/src/ggml-nexusq.c:113-124`: computed only when `amax >= FLT_MIN`, and **quantization and dequantization share the same fp16-rounded scale**, `inv = sc > 0 ? 1/sc : 0`. The CPU-side zero-block / extreme-value cases pass (`/tmp/nxq-align-test.log`) |
| Fixed-length bpw vs the reference's accounting | The reference stores fp16 at runtime; "3 bit" is a nominal value, whose **actual byte rate** is given by the offline `int8+delta+zstd` measurement | This repository provides a real KV storage type (fixed length 3.1875 / 2.1875 bpw) — the two are on **different levels**, not a "substitution" relation | `ggml/src/ggml-common.h:306-324`; reference `compression_accounting.py`. ⇒ any bpw comparison must first declare whether it is "runtime occupancy" or "offline measurement" |

- **Conclusion boundaries**: one can say "all three items have a corresponding handling or an explicit choice in the current implementation, and CPU/CUDA encoding consistency has tests";
  one **cannot** say "the byte rate is equivalent to or alignable with the reference" (the reference's measured accounting values were not recomputed in this repository, see §6.7).
  Also: the two TBQ FAILs in the existing `test-quantize-fns` (R-KV-01) should be treated as a **known deviation on file**;
  this document does **not** treat "adjusting that threshold" as the fix target — if tests are to be added in the future, the target should be an **end-to-end quality criterion for KV usage**,
  rather than relaxing an existing constant.

---

## 5 Key verification: can the quantized-layer count statistics and the linear extrapolation hold?

This section answers a question that was explicitly required to be verified: **the table in §6.45 writes the layer count as 48, but only 12 layers are full-attention.**
All conclusions below come from **reading the source and the GGUF metadata directly**, and do not cite the counts of §6.45.

### 5.1 Layers with KV = 12 (not 48)

- Source: `src/models/qwen4exp.cpp:893-901`:
  ```
  hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
  ```
  `is_recr(il) == true` ⇒ takes `build_layer_attn_linear` (recurrent/linear attention, **no KV cache**);
  otherwise it takes `build_layer_attn` (**has KV**). Dispatch site: `src/models/qwen4exp.cpp:1178-1183`.
- GGUF metadata (read directly from the first shard's header): `qwen4exp.block_count = 48`,
  `qwen4exp.full_attention_interval = 4`, **no** `attention.recurrent_layers` key
  (⇒ the default interval rule applies), **no** `nextn_predict_layers` key (⇒ `n_layer() = n_layer_all = 48`).
- Conclusion: the full-attention layers are `il ∈ {3, 7, 11, …, 47}`, **12 layers** in total;
  the remaining **36 layers** are linear attention (`ssm.*` parameters) and **are not allocated a KV cache**.
  This agrees with "only 12/48 layers of this model are full-attention" in the body of §6.42 — **only the protection table of §6.45 used 48 as the denominator**.
- Corroboration: the KV log line of `nxq-quality-nxq32.log`: `size = 8.06 MiB (512 cells, **12 layers**, 4/4 seqs)`.

### 5.2 Recomputing the protection window of R-NXQ-05 (correcting the layer-count labels)

`protected_layer = il < N || il >= hparams.n_layer() - N` (`qwen4exp.cpp:2044-2045`),
and the fake-quant chain exists only inside `build_layer_attn` ⇒ **only attention layers get quantized**:

| N | Protected blocks (`il`) | Of these, layers with **KV** | Attention layers actually quantized | What the handoff says | Mean KLD |
|---|---|---|---|---|---|
| 0 | — | 0 | **12 / 12** | 48/48 | 0.2607 |
| 2 | 0,1,46,47 | **{47} (1 layer)** | **11 / 12** | 44/48 | 0.2526 |
| 4 | 0,1,2,3,44,45,46,47 | **{3, 47} (2 layers)** | **10 / 12** | 40/48 | 0.2406 |

⇒ the "48/48, 44/48, 40/48" in the handoff table should be changed to "12/12, 11/12, 10/12".
**Note**: under this implementation, `LLAMA_KV_E8_PROTECT=2` protected only **1** layer that has KV (`il=47`),
and `=4` protected only **2** (`{3,47}`). As for "which layers the reference semantics should protect on this model" —
the reference's `protect_boundary` is defined over the **dense model's layer order** (`pipeline.py:1088-1092`),
and **the reference does not specify** how it applies to a hybrid architecture in which "some layers have no KV"; nor does this repository implement a mapping ⇒ **the mapping is unknown**,
so this experiment can only be described as a "restricted experiment under a block-index window", and cannot be asserted to deviate from the reference requirement.

### 5.3 Can the linear extrapolation hold: **extrapolation in either direction does not hold**

- The effective coverage of `LLAMA_KV_E8_PROTECT` on this model (section 5.2):
  N=0 → 0 layers with KV protected (12/12 quantized, KLD 0.2607);
  N=2 → **1** protected (11/12, 0.2526); N=4 → **2** protected (10/12, 0.2406).
- **The handoff's forward extrapolation does not hold**: the "≈ −0.005 KLD, ≈ −2% per protected layer" and "to push it down to the `q8_0`
  value of 0.034 would require protecting ~80% of the layers" of §6.45 — neither the denominator (48 instead of 12) nor the slope adds up,
  and the linear model of "a constant return per layer" contradicts the endpoint fact (protect all layers ⇒ no quantization ⇒ KLD=0):
  two points can only interpolate 0–2 layers and **cannot be extrapolated**.
- **This document's own earlier conclusion in the opposite direction is likewise retracted**: N=0/2/4 cannot yield "boundary protection is insufficient to bring NXQ up to the `q8_0` level",
  nor "boundary protection is not the solution" — the three available points cover far too few attention layers (≤2/12),
  and the window-to-hybrid-architecture mapping is undetermined (§4.5), so **any judgement of strength in either direction is beyond the available evidence**.
- The conclusion of this section therefore keeps only **one fact** and **one unknown**:
  - Fact: under the block-index window, protecting 1–2 layers that have KV lowers the mean KLD from 0.2607 to 0.2406;
  - Unknown: the shape and magnitude of the benefit curve once the window is correctly expanded by "attention-layer ordinal" (**to be measured**).

### 5.4 Boundaries of applicability (no re-test conclusion is presupposed)

- `n_head_kv = 2`. The empirical statement by the reference README's author lists "4+ KV heads" as the K3V2 safe zone,
  and marks Qwen2.5-1.5B with **2 KV heads** as catastrophic / not recommended.
  This model falls inside that **warning range**, which suggests the KLD of order `0.26` may be related to the architecture's KV head count.
- But this is an **empirical hint, not a proof**:
  1. It **does not constitute** a conclusion that "every 2-KV-head model is necessarily catastrophic";
  2. **Nor can** it be used to presuppose a re-test conclusion — one can neither conclude that "NXQ is unsuitable for this model and `q4_0`/`q8_0` should be used instead",
     nor that "after fixing the protection window the KLD will definitely improve markedly";
  3. The reference's numbers on 4+ KV head dense models (Mistral-7B +0.276% PPL, etc.) are **not comparable** with the numbers in this document,
     and can only serve as background.
- The R-NXQ rows of §7 are therefore all marked **re-test pending** rather than **conclusions**; any decision on "whether to abandon NXQ"
  should wait until after a re-test expanded by attention-layer ordinal (and, optionally, a genuine reference-encoder comparison path).

---

## 6 List of unknowns (items explicitly written as "unknown")

1. The exact defect point of the `tbq4_0` FA path. **Unknown**; all that is known is that "the synthetic round-trip probe revealed no obvious packing error,
   and rotation is not the sole cause", and that a **format/endpoint/attention-path cause can neither be attributed nor ruled out** yet.
2. The quality/speed of `tbq3_0` once it goes through FA. **Unknown** (the whitelist has not been extended, there is no tbq3 instantiation).
3. The generation speed of f16 KV under the 256k configuration. **Unknown** (`kv-f16-out.txt` has no timing line).
4. The mechanism by which the KLD of `q8_0` gets worse in the mean and better in the tail after V rot=64 → 256. **Unknown**.
5. How the reference's `protect_boundary` should map onto a hybrid architecture in which "some layers have no KV" — **the reference does not define this**;
   nor does this repository implement a mapping rule. Along with that, the shape and magnitude of its benefit curve (once expanded by attention-layer ordinal) are **unknown**.
6. The actual effect of the decode fp16 residual on per-token generation quality. **Unknown** (not implemented, not measured).
7. The reference's **offline accounting** (int8 + temporal differencing + zstd) measured byte rate on the same data — **not recomputed in this repository**
   (the reference README's 2.625 bpe is a **nominal value**, obtained from `(key_bits+value_bits)/2 + 0.125`);
   its relation to our runtime fixed-length type is "quantities on two levels", and **no directly alignable equation exists**.
8. PPL/KLD at 256k length (for any KV type). **Completely unknown** — for 256k only capacity and short-generation speed were measured.
9. How much of the difference between the local restricted encoder and the reference encoder (coordinate domain of 8 vs 9 levels, explicit coset bit, fixed-length unpack)
   each contributes — **unknown**; it can only be decomposed with a comparison path that can run the reference encoder.

---

## 7 Final decisions and re-open conditions (summary)

| Route ID | Decision | Basis | Re-open conditions |
|---|---|---|---|
| R-KV-01 | The format is kept; `tbq4_0` is unusable in the KV position; the two `test-quantize-fns` FAILs are recorded as a known deviation | The synthetic round-trip showed no anomaly; the end-to-end asymmetry (cause undetermined) | If tests are added, make a KV end-to-end quality criterion (**not** a relaxation of the existing threshold) |
| R-KV-02 | Under the historical protocol `tbq3_0` is recorded as the **capacity-first candidate** (51 slots / KLD 0.116), with `q4_0` for higher fidelity; `tbq4_0` **must not** be used (unusable under the tested 512-context protocol); **not a current production recommendation** | The slots and KLD table (8×512 protocol) | Add an f16@256k timing; re-run 256k with a usable KV; resolve the FA gap |
| R-KV-03 | **Defer fixing the whitelist**; first localize the `tbq4_0` FA defect | The asymmetry **points the suspicion** at the FA path (not exclusive; format/endpoint not ruled out) | Once the audit of "FA-specific rotation / Q-side rotation / output inverse rotation" has a conclusion, add the `TBQ3_0` whitelist entry and instantiations |
| R-KV-04 | Budget adapts to the upper limit (mechanism level) + turn PLE-GPU off; **256k is not quality evidence**; **no speed direction was established** | The slots and hit logs (the high-budget sample has the lower t/s) | Re-run after adding repeats for both `auto` and manual cap; correct the "+9 slots" and "slightly better speed" conventions |
| R-NXQ-01 | Retracted; the data must not be cited | nkvo/SET_ROWS/baseline convention problems | Not re-opened |
| R-NXQ-02 | Keep "choose after projecting" | Better on every MSE metric | If the 3-bit coordinate domain is widened |
| R-NXQ-03 | Keep "V rotation over the whole head" | NXQ improves in both mean and tail | The worsening mean of `q8_0` needs an explanation |
| R-NXQ-04 | Kept as a **local control tool**; **no** evaluation of the reference method's quality, and no claim of equivalence | Only the two degrees of freedom, RoPE and materialization, are observed in a controlled way | A genuine reference-encoder comparison path is needed to decompose the difference |
| R-NXQ-05 | Data kept (restricted experiment); the layer-count labels, the linear extrapolation and "boundary protection is not the solution" are **retracted**; the mapping semantics are **unknown** | The 12/48 layer count (section 5) + the reference layer-order definition | First settle the mapping convention, then re-run N=0..4 by attention-layer ordinal and report the number of protected layers |
| R-NXQ-06 | K de-RoPE is not a priority for now (no benefit observed, engineering judgement); the decode residual is **not implemented**; the byte rate and the reference accounting are on **different levels** | mode1 vs mode2 showed no observed improvement; the reference runtime is fp16 | If the goal shifts to per-token generation, re-open the residual |
| R-NXQ-07 | The endpoint defects (256/32 decode indexing, scale underflow) are fixed and kept | Current source + the CPY/SET_ROWS consistency tests | If alignment with the reference accounting is required, first add an offline-accounting recomputation |

---

## 8 Route ID × source chapter coverage list

| Route ID | Source chapter / commit | Coverage |
|---|---|---|
| R-KV-01 | `sources/handoff.md` §6.42 (opening passage); commits `2f1a363c8`; `17ca0de85` (round-trip error) | Format definition, codebook, packing, GPU endpoint, round-trip error |
| R-KV-02 | §6.42 (table + three conclusions + tool passage); commit `fa05f8637` | The 256k type matrix, f16 zero slots, tbq4 unusable, recommendation |
| R-KV-03 | The full text of the commit `17ca0de85` message; §6.43 (the contrasting statement that NXQ "is not added to the CUDA FA type whitelist" + the `attn_rot_k/v` measurements) | Whitelist/instantiations/fallback, rotation probes, FA-off rejection (the TBQ whitelist fact comes from reading the source directly, not from a citation) |
| R-KV-04 | The full text of §6.29 | 256k capacity and cache budget, auto vs cap, PLE-GPU, variance warning |
| R-NXQ-01 | §6.43 ("the old no-coset-marker format … cannot serve as evidence" + the fixed items); the early `/tmp/nxq_*` logs | Endpoint integration pitfalls, nkvo, baseline conventions |
| R-NXQ-02 | §6.43 (encoder description); §6.44 ("the two places changed this time", item 1 + evidence) | Choose after projecting, MSE, agreement rate |
| R-NXQ-03 | §6.44 (item 2 + the A/B table) | V rotation over the whole head, cross-type impact |
| R-NXQ-04 | §6.44 (reference-chain verification + unaligned items); §6.45 (question/implementation/result/conclusions 1–3, **restated under the restricted-encoder convention**) | Local control of the chain shape, no observed improvement from de-RoPE, comparison of materialized representations |
| R-NXQ-05 | §6.45 (boundary-protection passage + final judgement, **with the layer count/extrapolation retracted per section 5 of this document**) | The restricted protection-window experiment, correction of the layer-count labels, unknown mapping |
| R-NXQ-06 | §6.44 (two structural gaps); §6.45 (conclusion 4); reference sdist `quantized_cache.py`/`compression_accounting.py`/`pipeline.py`/`README.md` | K de-RoPE, decode residual, distinction of byte-rate levels, applicable model range (empirical) |
| R-NXQ-07 | §6.43 (two fixes: CUDA decode indexing, scale underflow); §6.43 (verification passage) | Endpoint correctness, test coverage, fixed-length bpw accounting conventions |

Adjacent items not included in this document (they belong to other chapters): §6.30–6.33 (SPLIT/SMoE timing), §6.34–6.41 (weight formats and AVX2 kernels).
**Note**: the three items in §6.45 — "quantized layer count 48/44/40", "−0.005 per layer", "protect 80% of layers" — **are not accepted by this document** (see section 5).

---

## 9 Precise small-log candidates recommended for preservation (archive candidate list)

The following entries are **archive candidates**: for the archival process to actually preserve and index; this document only lists the path, the size, and "which conclusion it supports".
The sizes are current measured values; a path beginning with `/tmp/` indicates a temporary directory outside the source worktree.
All of them are plain text, and a single file is enough to explain one conclusion.

### 9.1 Strongly recommended for preservation (each supports one conclusion above)

| Path | Size | Supports |
|---|---|---|
| `kv-tbq3_0-err.txt` / `-out.txt` | 35 KB / 2 KB | 51 slots / 4874.0 MiB / gen 16.2 t/s; DIRECT-VIEW line |
| `kv-tbq4_0-err.txt` / `-out.txt` | 35 KB / 2 KB | 47 slots / 4491.7 MiB / gen 15.3; short generation "looks normal" |
| `kv-f16-err.txt` | 3.4 KB | **`clamp cache budget … to 0 MiB` + 0 slots + `prefetch_ready=0`** (f16@256k turns the cache off) |
| `kv-q8_0-err.txt`, `kv-q4_0-err.txt` | ~35 KB ×2 | 29 slots / 45 slots |
| `sweep-kv.csv` | <1 KB | tbq3_0 online hit rate 78.9% (226005/60585) |
| `ppl-matrix.log` | <1 KB | 8ch PPL: f16 2.0120 / q8_0 2.0128 / q4_0 2.0234 / tbq3_0 2.0728 |
| `ppl-tbq4_0.log`, `ppl-8ch-tbq4-retry.log` | ~35 KB ×2 | tbq4 8ch **aborts, i.e. no result line** |
| `ppl-single-tbq4_0.log`, `ppl-nb-tbq4_0.log` | ~35 KB ×2 | 1ch PPL: `-b 2048/n_seq 4` = **63.7301** (main-table convention), `-b 512/n_seq 1` probe = **65.0026** (the two families cannot be cross-compared) |
| `ppl-single-{f16,q8_0,q4_0,tbq3_0}.log` | ~35 KB ×4 | 1ch baselines and controls |
| `ppl-rot-tbq4_0-rot_off.log` | ~35 KB | Still broken with rotation disabled (73.3126) |
| `ppl-nofa-tbq4.log` | <1 KB | **`quantized V cache requires flash_attn to be enabled`** |
| `kld-matrix.log`, `kld-tbq4_0.log` | <2 KB ×2 | KLD rows (q8_0 0.030848 / q4_0 0.056886 / tbq3_0 0.116397; tbq4 = `-`) |
| `auto-8k-novis-err.txt` | 35 KB | 80 slots / 7667 MiB / 76.3% / 17.4 t/s |
| `auto-256k-novis-err.txt` | 35 KB | 37 slots / 3625 MiB / 65.4% / 15.8 t/s |
| `auto-256k-vis-err.txt` | 35 KB | 40 slots / 3881 MiB / 64.6% / 13.2 t/s |
| `cap27-a-err.txt`, `cap27-b-err.txt` | 35 KB ×2 | Manual cap 2700 ⇒ 28 slots / 58.4% and 56.5% |
| `x-tiny-err.txt` | <5 KB | Minimal reproduction of "budget smaller than one layer slot ⇒ cache remains empty" |
| `/tmp/e8-m1b.log`, `/tmp/e8-m2b.log` | 20 KB ×2 | Reference chain 0.260734 vs without de-RoPE 0.256116 |
| `/tmp/e8-kld-all.log` | 1.5 KB | mode 0 self-check 4.2e-5 + mode 1 summary (**note: the mode 2 section in this file is `-`; the actual data is in `e8-m2b.log`**) |
| `/tmp/e8-protect.log` | <1 KB | The three KLD summaries for protect 0/2/4 (together with the layer-count correction of section 5) |
| `/tmp/nxq-align-kld.log`, `/tmp/nxq-align-kld-v64.log` | 1.3 KB / 0.9 KB | A/B of V rot=256 vs 64 |
| `/tmp/nxq-quality-q8.log`, `-nxq32.log`, `-nxq22.log` | 37 KB ×3 | 1ch PPL 2.3458 / 2.8378 / 2.7150 + KV buffer 25.50/8.06/6.56 MiB |
| `/tmp/nxq-align-test.log` | 0.7 KB | The two TBQ3/TBQ4 FAILs of `test-quantize-fns` |
| `/tmp/nxq-tbo-cpy.log`, `/tmp/nxq-tbo-sr.log` | 2 KB / 80 KB | CUDA consistency of CPY/SET_ROWS (`2/2 backends passed`) |

### 9.2 Recommended for preservation as "retracted / counterexample" (illustrating pitfalls, not participating in conclusions)

| Path | Size | Note |
|---|---|---|
| `/tmp/nxq_ppl.log` | 0.2 KB | The first version: CUDA has no SET_ROWS |
| `/tmp/nxq_ppl_nkvo.log`, `/tmp/nxq_base_f16_nkvo.log`, `/tmp/nxq_base_f16_nkvo2.log` | <1 KB ×3 | `-nkvo` crash |
| `/tmp/nxq_base.log`, `/tmp/nxq_sanity_noenv.log` | <1 KB ×2 | The `-nkvo` base PPL 2.0347 (**a different convention from the current base 2.0120**) |
| `/tmp/nxq_q80.log`, `/tmp/nxq_q40.log`, `/tmp/nxq_k3v2.log` | 3 KB / 3 KB / 400 KB | KL/Δp output of the early nkvo era, **retracted** |
| `/tmp/e8-m2-direct.log` | 0.6 KB | The `GGML_ASSERT` failure of the mode 2 direct connection |
| `ctx256k-tbq-err.txt` and other `ctx256k-*` | ~37 KB each | Historical records of 256k short generation under TBQ4 KV |

### 9.3 **Not** recommended for preservation

Build logs (`/tmp/e8-build*.log` 18–22 MB, `/tmp/nxq-*-build*.log` about 20 MB each)
are enormous and contain no conclusions; the archive only needs to record "build passed" and its timestamp.

---

## 10 Cross-references and disclaimer

- **Main index**: `README.en.md`; **methodology and correctness**: `05-correctness-and-methodology.en.md`;
  **main narrative**: `00-research-chronology.en.md`; adjacent chapters: `01-host-and-devpart.en.md`,
  `02-prediction-and-cache.en.md`, `03-weight-quantization-and-kernels.en.md`.
- **Historical text snapshots** (the § numbers cited in this document all refer to these files):
  `sources/handoff.md` (§6.29, §6.42–§6.45), `sources/smoe-nk-degradation-plan.md`,
  `sources/rebuild-spec.md`, `sources/moe-decode-perf-plan.md`, `sources/moe-cache-score-aware-prd.md`.
- **Release surface**: master `0862af564` (the TBQ format together with `run-*-tbq.ps1`, `sweep-vision-cache.py` are on the tree;
  **the TBQ evaluation, the TBQ diagnosis and NXQ/E8 are all off the tree**). WIP branch: `qwen4exp-tbq-devpart-wip`
  (`8a0bb3b7f`…`17ca0de85`). NXQ exists only in the uncommitted working tree.
- **External reference (accessible)**: `nexusquant-kv` **0.6.3** —
  PyPI <https://pypi.org/project/nexusquant-kv/0.6.3/>;
  the project repository (from the sdist `PKG-INFO`'s `Project-URL`) <https://github.com/jagmarques/nexusquant>;
  author Joao Marques, License Apache-2.0. Local unpack path `/tmp/nq/nexusquant_kv-0.6.3`.
- **Five disclaimers**:
  1. The NexusQuant-related experiments in this document are **not** the results of running the original implementation: the chain shape follows the reference, but **the encoder is a local restricted implementation**
     (narrower coordinate domain + explicit coset bits + fixed-length packing); it therefore **does not constitute a full reproduction**,
     and cannot be used to evaluate the quality ceiling of the reference method itself.
  2. `nexusquant-kv` 0.6.3 was **never installed or run on this machine** (no torch control run);
     every "reference behaviour" comes from a source-level reading of its sdist source and README.
  3. Reference parameters such as `protect_boundary` are defined over the dense model's layer order, and **the mapping to a hybrid architecture is undefined**;
     any comparison made with that parameter must state the mapping convention in the report.
  4. No t/s in this document **may** be mixed with the retracted old results (20.3 tps / 32 tps) or the 19 t/s threshold;
     the three conclusions of §6.45 — the layer-count labels, the linear extrapolation and "boundary protection is not the solution" — have been retracted/corrected by section 5 of this document,
     and this document prevails when cited.
  5. For 256k only **capacity and short generation** were measured; all quality numbers come from the wikitext-2 protocol of `-c 512 --chunks 8/1`,
     and **do not represent** the quality at 256k length.
