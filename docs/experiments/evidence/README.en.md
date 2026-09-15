[中文](README.md) · [English](README.en.md)

# Evidence index and source manifest

Back to the [complete archive](../README.en.md). This directory holds **portable evidence of experiments already performed**, not a new performance/quality experiment.

## 1. What is here, and what is not

- **New: [routing-small data package](../data/routing-small/README.en.md)**: 164 prompts, token records, collection manifest and small files from historical evaluations (about 0.70 MiB of body text), using a separate source manifest; not counted in the 125 logs below. The large hidden/router data is packaged as a 49.2 GB compressed archive, distributed via [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces), and does not enter Git/Git LFS.

- **125 original small logs**, with source files totaling 1,599,928 bytes. The complete text of the selected files is kept, with only encoding, path redaction and CRLF→LF processing; these are not excerpts of successful passages.
- [measurements.json](measurements.json): aggregated results for the original CLI, pressure128, fixed-history 400 steps and the old baseline400. Failures, exit codes, missing throughput and top-1 divergence are kept; the huge per-line arrays and full-vocabulary logits are removed. The source JSON itself is not all uploaded verbatim.
- [provenance.json](provenance.json): original source SHA-256, public copy SHA-256, transformation rules, 14 numeric sources, 125 logs and the missing material. The original hashes prove the identity of the material taken; they do not mean that the un-uploaded originals can be recovered from this directory.
- [publication-files.json](publication-files.json): final byte hashes of the archive files, excluding the manifest itself; does not include the repository-root README.
- [commit-notes.txt](commit-notes.txt): the original subject/body of 14 related commits. Commit messages are historical statements, not proof of correctness or performance.
- [deleted-logs-incident.json](deleted-logs-incident.json): record of the incident in which 30 old logs were mistakenly deleted by a wildcard. The maintainer later made clear that recovery was not needed; this directory has neither recovered nor fabricated them.
- 5 historical document snapshots, the maintainer's supplementary notes, and 6 read-only parse outputs. The 24-byte header check of the 1st GGUF shard is stored separately in provenance: tensor_count=0; the weight-type output comes from the 2nd/3rd shards, which contain tensors.

**Not uploaded**: model weights, tensor dumps, complete logits arrays, executables/DLL/PDB/dump, experiment source code, large build logs and complete scheduling debug logs. For the scheduler's original logs, only the parse results and the original hashes are published. The absence of the original large arrays means that readers cannot recompute all the statistics from this directory alone; the existing summaries and the retained divergence records can still be reviewed.

## 2. Identity, transformation and measurement rules

- `SOURCE_TREE`: the maintainer's experiment source workspace; `LOCAL_EVIDENCE`: the local temporary evidence root directory (which may contain `moe-cache/` under it); `LOCAL_MODELS`: the local model directory; `OTHER_SOURCE_TREES`: other local checkouts. These are not actual directories inside the repository.
- In file paths, the working point, run number and file name are kept; machine root paths are replaced. Historical snapshots get a status header added, and old conclusions are not silently corrected inside the snapshots.
- `original_sha256` corresponds to the original bytes before transformation; `archived_sha256` corresponds to the public UTF-8 copy. Decoding or newline changes will make the two differ; this is expected behavior.
- The archive's own `.gitattributes` pins `eol=lf`, to avoid a Windows checkout changing the evidence hashes. Standalone terminal carriage-return characters are preserved.
- For GGUF only the header/directory is read, not the tensor data, and **no SHA-256 of the whole GGUF was computed**; file name, size and metadata checks cannot be treated as complete model identity authentication.
- MB/MiB and GB/GiB are interpreted according to the specific definition used by each topic. `test-quantize-perf` originally displayed `GB/s` but actually divided by 1024³; the original output is kept, and the conversion correction is in chapter 03.
- Inherited fields such as `acceptance.status=pass` indicate only the runner's criteria at that time, and cannot override later correctness, comparability or failure conclusions. The discarded amplified noise tolerance is not used as an acceptance basis for this archive.

From the repository root you can check a public copy with PowerShell, then compare it with the `archived_sha256` in provenance:

```powershell
Get-FileHash -Algorithm SHA256 .\docs\experiments\evidence\logs\repo\ppl-single-tbq4_0.log
```

## 3. Historical documents and maintainer supplements

| Snapshot | Original location | Original line count |
|---|---|---:|
| [handoff.md](../sources/handoff.md) | `handoff.md` | 1703 |
| [smoe-nk-degradation-plan.md](../sources/smoe-nk-degradation-plan.md) | `docs/smoe-nk-degradation-plan.md` | 227 |
| [rebuild-spec.md](../sources/rebuild-spec.md) | `docs/rebuild-spec.md` | 546 |
| [moe-decode-perf-plan.md](../sources/moe-decode-perf-plan.md) | `docs/moe-decode-perf-plan.md` | 293 |
| [moe-cache-score-aware-prd.md](../sources/moe-cache-score-aware-prd.md) | `docs/moe-cache-score-aware-prd.md` | 286 |

[author-recollection.md](../sources/author-recollection.md) is facts and ordering that the maintainer added during this archiving, not an older version of the historical documents above. "PLE about 1G / 90%+", "SMoE teacher 99%" are kept as recalled; where the complete run protocol is missing, the space is explicitly left blank, and no other tier/metric is substituted for it.

## 4. Numeric JSON sources

The hashes and sizes of the following originals are in provenance; the single public numeric entry point is [measurements.json](measurements.json).

| Source ID | Category |
|---|---|
| `optimization400/lifetime-before.json` | `measurement-json` |
| `optimization400/lifetime-after.json` | `measurement-json` |
| `optimization400/final-r1.json` | `measurement-json` |
| `optimization400/final-r2.json` | `measurement-json` |
| `optimization400/baseline-paired-r2.json` | `measurement-json` |
| `optimization400/baseline-paired.json` | `measurement-json` |
| `optimization400/timeout-smoke.json` | `measurement-json` |
| `optimization400/lock-smoke.json` | `measurement-json` |
| `global-auto128.summary.json` | `measurement-json` |
| `layer-auto128.summary.json` | `measurement-json` |
| `global-pressure128-v2.summary.json` | `measurement-json` |
| `optimization400/fixed400/experiment.json` | `measurement-json` |
| `optimization400/fixed400-final/experiment.json` | `measurement-json` |
| `baseline400-6g.summary.json` | `measurement-json` |

Main reading notes: for CLI records, look at the requested step count, the exit code, whether Generation exists and the runtime binary hash at the same time; for fixed history, look at all 400 steps of `decode_all`, the steady384 steps are only a performance subset; pressure128 is not a substitute for the 400-step acceptance. That the CPU logic fixture has 15 PASS and the GPU stride probe's 24 groups show no errors also does not prove that the model as a whole is numerically consistent.

## 5. Per-file index of the original small logs

### Early PLE, static hot table and XT (13 files)

| Public file | Original source | Original bytes |
|---|---|---:|
| [ablation-lazy-moe2-xt.log](logs/early/ablation-lazy-moe2-xt.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe2-xt.log` | 16632 |
| [ablation-lazy-moe4.log](logs/early/ablation-lazy-moe4.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe4.log` | 32493 |
| [ablation-lazy-moe6.log](logs/early/ablation-lazy-moe6.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe6.log` | 32493 |
| [ablation-lazy-moe8-gpuple.log](logs/early/ablation-lazy-moe8-gpuple.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe8-gpuple.log` | 32896 |
| [ablation-lazy-moe8.log](logs/early/ablation-lazy-moe8.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe8.log` | 32494 |
| [ablation-ple-only.csv](logs/early/ablation-ple-only.csv) | `LOCAL_MODELS\qwen38\traces\ablation-ple-only.csv` | 3155 |
| [moe-static90-8192.csv](logs/early/moe-static90-8192.csv) | `LOCAL_MODELS\qwen38\traces\moe-static90-8192.csv` | 33748 |
| [moe-xt-12288.csv](logs/early/moe-xt-12288.csv) | `LOCAL_MODELS\qwen38\traces\moe-xt-12288.csv` | 33730 |
| [ple-cpu-l2-with-gpu-l1.csv](logs/early/ple-cpu-l2-with-gpu-l1.csv) | `LOCAL_MODELS\qwen38\traces\ple-cpu-l2-with-gpu-l1.csv` | 13446 |
| [ple-gpu-l1-rawpages-1g-async.csv](logs/early/ple-gpu-l1-rawpages-1g-async.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-l1-rawpages-1g-async.csv` | 13214 |
| [ple-gpu-l1-rawpages-1g.csv](logs/early/ple-gpu-l1-rawpages-1g.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-l1-rawpages-1g.csv` | 10465 |
| [ple-gpu-overlap-stats.csv](logs/early/ple-gpu-overlap-stats.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-overlap-stats.csv` | 5424 |
| [ple-lru-1g-no-prefetch.csv](logs/early/ple-lru-1g-no-prefetch.csv) | `LOCAL_MODELS\qwen38\traces\ple-lru-1g-no-prefetch.csv` | 340 |

### host/devpart, budget, timing, KV and quality raw output (78 files)

| Public file | Original source | Original bytes |
|---|---|---:|
| [L2048-err.txt](logs/repo/L2048-err.txt) | `SOURCE_TREE/L2048-err.txt` | 34698 |
| [L6144-err.txt](logs/repo/L6144-err.txt) | `SOURCE_TREE/L6144-err.txt` | 34694 |
| [a400a-err.txt](logs/repo/a400a-err.txt) | `SOURCE_TREE/a400a-err.txt` | 37407 |
| [a400b-err.txt](logs/repo/a400b-err.txt) | `SOURCE_TREE/a400b-err.txt` | 37415 |
| [auto-256k-novis-err.txt](logs/repo/auto-256k-novis-err.txt) | `SOURCE_TREE\auto-256k-novis-err.txt` | 35810 |
| [auto-256k-vis-err.txt](logs/repo/auto-256k-vis-err.txt) | `SOURCE_TREE\auto-256k-vis-err.txt` | 35837 |
| [auto-256k-vis-out.txt](logs/repo/auto-256k-vis-out.txt) | `SOURCE_TREE/auto-256k-vis-out.txt` | 2061 |
| [c1-3-err.txt](logs/repo/c1-3-err.txt) | `SOURCE_TREE\c1-3-err.txt` | 35396 |
| [cap27-a-err.txt](logs/repo/cap27-a-err.txt) | `SOURCE_TREE\cap27-a-err.txt` | 35726 |
| [cap27-a-out.txt](logs/repo/cap27-a-out.txt) | `SOURCE_TREE/cap27-a-out.txt` | 1870 |
| [cap27-b-out.txt](logs/repo/cap27-b-out.txt) | `SOURCE_TREE/cap27-b-out.txt` | 1970 |
| [cpunode-run.log](logs/repo/cpunode-run.log) | `SOURCE_TREE/cpunode-run.log` | 75057 |
| [cur-ref-err.txt](logs/repo/cur-ref-err.txt) | `SOURCE_TREE/cur-ref-err.txt` | 72910 |
| [cur-ref-out.txt](logs/repo/cur-ref-out.txt) | `SOURCE_TREE/cur-ref-out.txt` | 2166 |
| [dsh-G2-err.txt](logs/repo/dsh-G2-err.txt) | `SOURCE_TREE/dsh-G2-err.txt` | 4100 |
| [dsh-G2-out.txt](logs/repo/dsh-G2-out.txt) | `SOURCE_TREE/dsh-G2-out.txt` | 1932 |
| [dsh-r3-timing.txt](logs/repo/dsh-r3-timing.txt) | `SOURCE_TREE/dsh-r3-timing.txt` | 1103 |
| [dsh-r4-timing.txt](logs/repo/dsh-r4-timing.txt) | `SOURCE_TREE/dsh-r4-timing.txt` | 1378 |
| [dsh-r5-timing.txt](logs/repo/dsh-r5-timing.txt) | `SOURCE_TREE/dsh-r5-timing.txt` | 3037 |
| [dsh-r8.txt](logs/repo/dsh-r8.txt) | `SOURCE_TREE\dsh-r8.txt` | 772 |
| [dsh-r9.txt](logs/repo/dsh-r9.txt) | `SOURCE_TREE\dsh-r9.txt` | 656 |
| [fused-final.log](logs/repo/fused-final.log) | `SOURCE_TREE/fused-final.log` | 17375 |
| [guard-1-out.txt](logs/repo/guard-1-out.txt) | `SOURCE_TREE\guard-1-out.txt` | 1571 |
| [kld-matrix.log](logs/repo/kld-matrix.log) | `SOURCE_TREE\kld-matrix.log` | 2035 |
| [kld-tbq4_0.log](logs/repo/kld-tbq4_0.log) | `SOURCE_TREE\kld-tbq4_0.log` | 552 |
| [kv-f16-err.txt](logs/repo/kv-f16-err.txt) | `SOURCE_TREE\kv-f16-err.txt` | 3483 |
| [kv-f16-out.txt](logs/repo/kv-f16-out.txt) | `SOURCE_TREE\kv-f16-out.txt` | 1442 |
| [kv-q4_0-err.txt](logs/repo/kv-q4_0-err.txt) | `SOURCE_TREE\kv-q4_0-err.txt` | 35406 |
| [kv-q4_0-out.txt](logs/repo/kv-q4_0-out.txt) | `SOURCE_TREE\kv-q4_0-out.txt` | 1737 |
| [kv-q8_0-err.txt](logs/repo/kv-q8_0-err.txt) | `SOURCE_TREE\kv-q8_0-err.txt` | 35407 |
| [kv-q8_0-out.txt](logs/repo/kv-q8_0-out.txt) | `SOURCE_TREE\kv-q8_0-out.txt` | 1496 |
| [kv-tbq3_0-err.txt](logs/repo/kv-tbq3_0-err.txt) | `SOURCE_TREE\kv-tbq3_0-err.txt` | 35407 |
| [kv-tbq3_0-out.txt](logs/repo/kv-tbq3_0-out.txt) | `SOURCE_TREE\kv-tbq3_0-out.txt` | 1484 |
| [kv-tbq4_0-err.txt](logs/repo/kv-tbq4_0-err.txt) | `SOURCE_TREE\kv-tbq4_0-err.txt` | 35407 |
| [kv-tbq4_0-out.txt](logs/repo/kv-tbq4_0-out.txt) | `SOURCE_TREE\kv-tbq4_0-out.txt` | 1824 |
| [long-run.log](logs/repo/long-run.log) | `SOURCE_TREE/long-run.log` | 27109 |
| [phase-run.log](logs/repo/phase-run.log) | `SOURCE_TREE/phase-run.log` | 17615 |
| [ppl-8ch-tbq4-retry.log](logs/repo/ppl-8ch-tbq4-retry.log) | `SOURCE_TREE\ppl-8ch-tbq4-retry.log` | 1178 |
| [ppl-matrix.log](logs/repo/ppl-matrix.log) | `SOURCE_TREE\ppl-matrix.log` | 280 |
| [ppl-nb-tbq4_0.log](logs/repo/ppl-nb-tbq4_0.log) | `SOURCE_TREE\ppl-nb-tbq4_0.log` | 18047 |
| [ppl-nofa-tbq4.log](logs/repo/ppl-nofa-tbq4.log) | `SOURCE_TREE\ppl-nofa-tbq4.log` | 867 |
| [ppl-rot-tbq3_0-rot_off.log](logs/repo/ppl-rot-tbq3_0-rot_off.log) | `SOURCE_TREE\ppl-rot-tbq3_0-rot_off.log` | 18106 |
| [ppl-rot-tbq3_0-rot_on.log](logs/repo/ppl-rot-tbq3_0-rot_on.log) | `SOURCE_TREE\ppl-rot-tbq3_0-rot_on.log` | 17924 |
| [ppl-rot-tbq4_0-rot_off.log](logs/repo/ppl-rot-tbq4_0-rot_off.log) | `SOURCE_TREE\ppl-rot-tbq4_0-rot_off.log` | 18108 |
| [ppl-single-f16.log](logs/repo/ppl-single-f16.log) | `SOURCE_TREE\ppl-single-f16.log` | 17924 |
| [ppl-single-q4_0.log](logs/repo/ppl-single-q4_0.log) | `SOURCE_TREE\ppl-single-q4_0.log` | 17924 |
| [ppl-single-q8_0.log](logs/repo/ppl-single-q8_0.log) | `SOURCE_TREE\ppl-single-q8_0.log` | 17924 |
| [ppl-single-tbq3_0.log](logs/repo/ppl-single-tbq3_0.log) | `SOURCE_TREE\ppl-single-tbq3_0.log` | 17924 |
| [ppl-single-tbq4_0.log](logs/repo/ppl-single-tbq4_0.log) | `SOURCE_TREE\ppl-single-tbq4_0.log` | 17926 |
| [ppl-tbq4_0.log](logs/repo/ppl-tbq4_0.log) | `SOURCE_TREE\ppl-tbq4_0.log` | 671 |
| [sweep-2x2.log](logs/repo/sweep-2x2.log) | `SOURCE_TREE/sweep-2x2.log` | 180 |
| [sweep-ab.csv](logs/repo/sweep-ab.csv) | `SOURCE_TREE/sweep-ab.csv` | 281 |
| [sweep-ahead.csv](logs/repo/sweep-ahead.csv) | `SOURCE_TREE\sweep-ahead.csv` | 210 |
| [sweep-ahead2.csv](logs/repo/sweep-ahead2.csv) | `SOURCE_TREE/sweep-ahead2.csv` | 298 |
| [sweep-auto.csv](logs/repo/sweep-auto.csv) | `SOURCE_TREE/sweep-auto.csv` | 157 |
| [sweep-auto8k.csv](logs/repo/sweep-auto8k.csv) | `SOURCE_TREE/sweep-auto8k.csv` | 128 |
| [sweep-budget.csv](logs/repo/sweep-budget.csv) | `SOURCE_TREE/sweep-budget.csv` | 247 |
| [sweep-c1.csv](logs/repo/sweep-c1.csv) | `SOURCE_TREE/sweep-c1.csv` | 301 |
| [sweep-c1nb.csv](logs/repo/sweep-c1nb.csv) | `SOURCE_TREE/sweep-c1nb.csv` | 197 |
| [sweep-corrupt.csv](logs/repo/sweep-corrupt.csv) | `SOURCE_TREE\sweep-corrupt.csv` | 332 |
| [sweep-fix.csv](logs/repo/sweep-fix.csv) | `SOURCE_TREE\sweep-fix.csv` | 195 |
| [sweep-hd.csv](logs/repo/sweep-hd.csv) | `SOURCE_TREE/sweep-hd.csv` | 433 |
| [sweep-kv.csv](logs/repo/sweep-kv.csv) | `SOURCE_TREE\sweep-kv.csv` | 144 |
| [sweep-limit.csv](logs/repo/sweep-limit.csv) | `SOURCE_TREE/sweep-limit.csv` | 427 |
| [sweep-neutral.csv](logs/repo/sweep-neutral.csv) | `SOURCE_TREE/sweep-neutral.csv` | 203 |
| [sweep-press.csv](logs/repo/sweep-press.csv) | `SOURCE_TREE/sweep-press.csv` | 306 |
| [sweep-reg.csv](logs/repo/sweep-reg.csv) | `SOURCE_TREE\sweep-reg.csv` | 297 |
| [sweep-smoke.csv](logs/repo/sweep-smoke.csv) | `SOURCE_TREE\sweep-smoke.csv` | 124 |
| [sweep-t2.csv](logs/repo/sweep-t2.csv) | `SOURCE_TREE/sweep-t2.csv` | 196 |
| [sweep-tune.csv](logs/repo/sweep-tune.csv) | `SOURCE_TREE\sweep-tune.csv` | 195 |
| [sweep-u.csv](logs/repo/sweep-u.csv) | `SOURCE_TREE/sweep-u.csv` | 141 |
| [sweep-v.csv](logs/repo/sweep-v.csv) | `SOURCE_TREE\sweep-v.csv` | 196 |
| [sweep-x.csv](logs/repo/sweep-x.csv) | `SOURCE_TREE\sweep-x.csv` | 174 |
| [t-4096-err.txt](logs/repo/t-4096-err.txt) | `SOURCE_TREE\t-4096-err.txt` | 35283 |
| [t-4096-out.txt](logs/repo/t-4096-out.txt) | `SOURCE_TREE\t-4096-out.txt` | 1517 |
| [t-auto-err.txt](logs/repo/t-auto-err.txt) | `SOURCE_TREE\t-auto-err.txt` | 35365 |
| [t-auto-out.txt](logs/repo/t-auto-out.txt) | `SOURCE_TREE\t-auto-out.txt` | 1714 |
| [x-tiny-err.txt](logs/repo/x-tiny-err.txt) | `SOURCE_TREE\x-tiny-err.txt` | 3369 |

### Final 400-token CLI and failed runs (14 files)

| Public file | Original source | Original bytes |
|---|---|---:|
| [opt400-baseline-paired-err.txt](logs/cli/opt400-baseline-paired-err.txt) | `SOURCE_TREE\opt400-baseline-paired-err.txt` | 0 |
| [opt400-baseline-paired-out.txt](logs/cli/opt400-baseline-paired-out.txt) | `SOURCE_TREE\opt400-baseline-paired-out.txt` | 0 |
| [opt400-baseline-paired-r2-err.txt](logs/cli/opt400-baseline-paired-r2-err.txt) | `SOURCE_TREE\opt400-baseline-paired-r2-err.txt` | 35428 |
| [opt400-baseline-paired-r2-out.txt](logs/cli/opt400-baseline-paired-r2-out.txt) | `SOURCE_TREE\opt400-baseline-paired-r2-out.txt` | 1784 |
| [opt400-final-r1-err.txt](logs/cli/opt400-final-r1-err.txt) | `SOURCE_TREE\opt400-final-r1-err.txt` | 37179 |
| [opt400-final-r1-out.txt](logs/cli/opt400-final-r1-out.txt) | `SOURCE_TREE\opt400-final-r1-out.txt` | 2226 |
| [opt400-final-r2-err.txt](logs/cli/opt400-final-r2-err.txt) | `SOURCE_TREE\opt400-final-r2-err.txt` | 16699 |
| [opt400-final-r2-out.txt](logs/cli/opt400-final-r2-out.txt) | `SOURCE_TREE\opt400-final-r2-out.txt` | 2163 |
| [opt400-lifetime-after-err.txt](logs/cli/opt400-lifetime-after-err.txt) | `SOURCE_TREE\opt400-lifetime-after-err.txt` | 36524 |
| [opt400-lifetime-after-out.txt](logs/cli/opt400-lifetime-after-out.txt) | `SOURCE_TREE\opt400-lifetime-after-out.txt` | 1751 |
| [opt400-lifetime-before-err.txt](logs/cli/opt400-lifetime-before-err.txt) | `SOURCE_TREE\opt400-lifetime-before-err.txt` | 36518 |
| [opt400-lifetime-before-out.txt](logs/cli/opt400-lifetime-before-out.txt) | `SOURCE_TREE\opt400-lifetime-before-out.txt` | 1784 |
| [opt400-timeout-smoke-err.txt](logs/cli/opt400-timeout-smoke-err.txt) | `SOURCE_TREE\opt400-timeout-smoke-err.txt` | 0 |
| [opt400-timeout-smoke-out.txt](logs/cli/opt400-timeout-smoke-out.txt) | `SOURCE_TREE\opt400-timeout-smoke-out.txt` | 0 |

### Logic fixture and stride memcheck (4 files)

| Public file | Original source | Original bytes |
|---|---|---:|
| [logic-empty-before.log](logs/logic/logic-empty-before.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-empty-before.log` | 933 |
| [logic-final.log](logs/logic/logic-final.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-final.log` | 967 |
| [logic-round-robin-before.log](logs/logic/logic-round-robin-before.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-round-robin-before.log` | 671 |
| [stride-memcheck.log](logs/logic/stride-memcheck.log) | `LOCAL_EVIDENCE\moe-cache\stride-memcheck.log` | 64 |

### NXQ/E8/KV local quality diagnostics (16 files)

| Public file | Original source | Original bytes |
|---|---|---:|
| [e8-kld-all.log](logs/kv/e8-kld-all.log) | `LOCAL_EVIDENCE\e8-kld-all.log` | 1551 |
| [e8-m1b.log](logs/kv/e8-m1b.log) | `LOCAL_EVIDENCE\e8-m1b.log` | 20377 |
| [e8-m2-direct.log](logs/kv/e8-m2-direct.log) | `LOCAL_EVIDENCE\e8-m2-direct.log` | 640 |
| [e8-m2b.log](logs/kv/e8-m2b.log) | `LOCAL_EVIDENCE\e8-m2b.log` | 20377 |
| [e8-protect.log](logs/kv/e8-protect.log) | `LOCAL_EVIDENCE\e8-protect.log` | 660 |
| [nxq-align-kld-v64.log](logs/kv/nxq-align-kld-v64.log) | `LOCAL_EVIDENCE\nxq-align-kld-v64.log` | 894 |
| [nxq-align-kld.log](logs/kv/nxq-align-kld.log) | `LOCAL_EVIDENCE\nxq-align-kld.log` | 1309 |
| [nxq-align-test.log](logs/kv/nxq-align-test.log) | `LOCAL_EVIDENCE\nxq-align-test.log` | 718 |
| [nxq-quality-nxq22.log](logs/kv/nxq-quality-nxq22.log) | `LOCAL_EVIDENCE\nxq-quality-nxq22.log` | 37473 |
| [nxq-quality-nxq32.log](logs/kv/nxq-quality-nxq32.log) | `LOCAL_EVIDENCE\nxq-quality-nxq32.log` | 37472 |
| [nxq-quality-q8.log](logs/kv/nxq-quality-q8.log) | `LOCAL_EVIDENCE\nxq-quality-q8.log` | 37467 |
| [nxq-tbo-cpy.log](logs/kv/nxq-tbo-cpy.log) | `LOCAL_EVIDENCE\nxq-tbo-cpy.log` | 2070 |
| [nxq-tbo-sr.log](logs/kv/nxq-tbo-sr.log) | `LOCAL_EVIDENCE\nxq-tbo-sr.log` | 79658 |
| [nxq_base_f16_nkvo.log](logs/kv/nxq_base_f16_nkvo.log) | `LOCAL_EVIDENCE\nxq_base_f16_nkvo.log` | 552 |
| [nxq_ppl.log](logs/kv/nxq_ppl.log) | `LOCAL_EVIDENCE\nxq_ppl.log` | 179 |
| [nxq_ppl_nkvo.log](logs/kv/nxq_ppl_nkvo.log) | `LOCAL_EVIDENCE\nxq_ppl_nkvo.log` | 552 |

## 6. Read-only verification at archive time

These outputs parse existing material; they do not run the model, load weight tensors, or build programs. Script hashes, commands, input sizes and the obtainable input hashes are in provenance.

| Output | Scope |
|---|---|
| [gguf-shard2-types.txt](scans/gguf-shard2-types.txt) | GGUF header/tensor directory |
| [gguf-shard3-types.txt](scans/gguf-shard3-types.txt) | GGUF header/tensor directory |
| [gguf-shard2-geometry.txt](scans/gguf-shard2-geometry.txt) | GGUF header/tensor directory |
| [gguf-shard3-geometry.txt](scans/gguf-shard3-geometry.txt) | GGUF header/tensor directory |
| [scheduler-text.txt](scans/scheduler-text.txt) | backend summary of the existing scheduling logs |
| [scheduler-vision.txt](scans/scheduler-vision.txt) | backend summary of the existing scheduling logs |

The 2nd and 3rd shards list 468 and 756 tensors respectively, 1224 in total; the displayed GiB value on each line has already been rounded, so do not mistake a last-digit difference in the per-item displayed numbers for an original byte difference.

## 7. Missing evidence and parts that cannot be retro-inferred

- `SOURCE_TREE/dsh-r6-timing.txt`: `not-found-at-archive-time`.
- `SOURCE_TREE/dsh-r7-timing.txt`: `not-found-at-archive-time`.

In addition, the complete original protocol for teacher 99%, some early independent profiling output, the native fault stack of the latest exit failure, and the actual reproduction results of NXQ in the original project did not enter this archive. Some historical numbers exist only in notes or commit messages from that time, and the topics label them by evidence level; a pass must not be granted just because counterexamples are missing, and a newly run similar result must not be passed off as the original.

Warnings, old conclusions, failures and anomalies in the original logs are not equal to this archive's recommended settings. Please read the corrections and conclusion boundaries of the corresponding topic first.
