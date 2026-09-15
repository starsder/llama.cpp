[中文](README.md) · [English](README.en.md)

# Qwen3.8 Flash Next inference optimization research based on llama.cpp: NGRAM offload, MoE prefetching, and a record of pitfalls

## Results at a glance

In one sentence: **the 3-bit mixed-quantization MoE model fits entirely in 128 GB of main memory on a single machine, and expert prefetching plus a VRAM expert cache push 8k-context decoding to about 19 token/s** — while a pile of impressive-looking numbers got retracted and written up as failures.

- **Numbers you may look at (local observations, not an acceptance run of the release build)**: 400-token decoding at the conservative working point **19.2 / 19.3 token/s** (`-c 8192`, K/V `q8_0`, 6144 MiB cache budget, 64 slots per layer); a pinned weight buffer of **72.6 GiB with 0 lock failures**; a cache **access** hit rate of **69.7%** (access hits, not predictor accuracy).
- **Numbers you must not cite**: the historical 20.8 vs 9.2 (2.26× / about +126.1%) is only an arithmetic comparison of historical records, not a same-conditions measured net gain; the 20.3–22 token/s from before the routing-misalignment fix, the 28–32 token/s produced by devpart in a computationally wrong state, and candidate peaks that only cut transferred bytes without a stable end-to-end gain are all excluded. The reason for each is written up in the [correctness document](docs/experiments/05-correctness-and-methodology.en.md).
- **More worth reading than the scores**: **91 numbered route / diagnosis entries** recording, one by one, the motivation, the mechanism, the result, the reason for retraction, and the open questions — including conclusions I later overturned myself, a native crash that was never localized, and the measurement traps I walked into.

## Data and records (the most important part of this repository)

| What you want | Where to go |
|---|---|
| **All experiment documents (Chinese / English)** | [Archive index](docs/experiments/README.en.md) · [中文](docs/experiments/README.md) · [Read in research order](docs/experiments/00-research-chronology.en.md) |
| **Raw evidence and hash manifest** | [Evidence index](docs/experiments/evidence/README.en.md): 125 original small logs, 14 numeric sources, missing material, and the reading rules |
| **Aggregated result JSON** | [measurements.json](docs/experiments/evidence/measurements.json): CLI, pressure128, fixed-history 400 steps, old baseline400 — failures and exit codes retained |
| **Prompts, token histories, offline evaluation small files** | [Lightweight routing data package](docs/experiments/data/routing-small/README.en.md): 164 files, about 0.70 MiB of body text |
| **Raw hidden/router captures (49.2 GB)** | [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces) · [Format and batch notes](docs/experiments/data/hidden-routing/README.en.md) |
| **Provenance, redaction, per-file hashes** | [provenance.json](docs/experiments/evidence/provenance.json) · [publication-files.json](docs/experiments/evidence/publication-files.json) |

## About me, and why this repository exists

I am a **hobbyist beginner researcher**, and this is **not professional research**: no team, no review, no compute budget — just one 16 GB **modded card** (not a laptop; a modified card, because I could not afford anything better), and a wish to know whether a MoE model could still run a bit faster with that little VRAM.

So the material here **may well be imprecise**: too few repeats at the same working point, comparisons I never ran, conclusions I later overturned myself. I have tried to write the conventions, the counterexamples and the retractions into the documents, but **there are certainly still mistakes**. **Corrections, challenges to the conclusions and discussion are all welcome** — just open an issue; if you point at something that turns out to be fine, I will take that too.

I chose to **open up most of the research data**: apart from model weights, the large complete-logits arrays and the experiment binaries, the prompts, token records, aggregated results, original small logs and the 49.2 GB capture arrays are all downloadable. **I hope it helps someone** — even if it only saves you one pitfall, or tells you early that some direction does not work.

## Next step (in progress)

This repository started from an **SSD→main-memory PLE row cache**, but most of the later effort went into expert prefetching and caching on the **main memory→VRAM** side.

**What I am trying now: adapting the prefetch algorithm to main memory itself, to achieve efficient predictive loading from SSD to memory.** That is, letting "prediction" decide not only which experts go to VRAM, but also which weights/rows should be pulled from SSD into main memory ahead of time, so that decoding does not sit waiting on the disk in the critical path.

There is no conclusion from this yet, and no number worth citing; progress will go into the documents as usual, and so will failures.

This repository is [starsder/qwen3.8-flash-next-inference-research](https://github.com/starsder/qwen3.8-flash-next-inference-research), derived directly from [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp); the base inference engine comes from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and [ggml](https://github.com/ggml-org/ggml). This project focuses on **NGRAM/PLE offloading for Qwen3.8 Flash Next, MoE expert prefetching and caching, and mixed CPU/GPU execution**, and records optimization attempts, failed paths, and corrections to conclusions, rather than maintaining a general-purpose inference engine.

> **Compatibility warning: this is not a compatible replacement for general-purpose llama.cpp.** To study a specific model, this branch has deeply modified the loading, scheduling, caching, and execution paths, and may severely affect compatibility with existing models, backends, tools, and interfaces. Unverified upstream features should not be regarded as still working; if you need general model support or stable compatibility, use upstream llama.cpp. This repository retains its fork origin and acknowledgements, but is named as an independent research project to avoid confusion with upstream capabilities.

**Status: unstable, for research use, not a production-stable release.** The currently published conservative host code baseline is [`7e01451b2`](https://github.com/starsder/qwen3.8-flash-next-inference-research/commit/7e01451b2d7aab6a4ff58ecfdd2f3b13fcaeb0bf), which includes the host-partition routing-misalignment fix and protection for mixed-quantization weight addressing; it does not include the later repack/KV experiments or the cache-strategy candidates that have not yet been committed. "Conservative baseline" does not mean crash-free, free of computational errors, or that a comprehensive regression has already been completed.

> **Serious resource risk: switches, combinations, and concurrent runs outside the recommended parameters may exhaust the whole machine's main memory, commit limit, or VRAM, causing process crashes, an unresponsive system, or even a required reboot. 128 GB of memory does not mean it is safe. The recommended parameters are likewise not a safety guarantee. Save your other work first, and do not run this unattended on a machine that carries important tasks.**

## Documentation and data navigation

Development in this project starts from the **SSD→main-memory PLE row cache**, followed by static tables + XT, Fate, shared-expert SMoE, online caching and dual gating, dev-path timing, TBQ4, and NXQ. What is most worth preserving is why each attempt was made, why the direction was changed, and which impressive-looking results were later overturned.

### Research documentation: look at the process and conclusions first

- **[Complete archive index](docs/experiments/README.en.md)**: path coverage table, version boundaries, kept/abandoned/unfinished status.
- **[Read in development order](docs/experiments/00-research-chronology.en.md)**: includes the maintainer's supplementary PLE figure of about 1G/90%+ and SMoE teacher 99%, accounted separately from the online metrics.
- [PLE, host, devpart, prefill, MTP](docs/experiments/01-host-and-devpart.en.md) · [Prediction, caching, and dual gating](docs/experiments/02-prediction-and-cache.en.md).
- [Weight quantization selection and IQ/Q kernel differences](docs/experiments/03-weight-quantization-and-kernels.en.md) · [TBQ3/TBQ4, NXQ, and KV quality](docs/experiments/04-kv-tbq-and-nxq.en.md).
- [Correctness, measurement failures, and engineering incidents](docs/experiments/05-correctness-and-methodology.en.md) · **[Raw evidence and hash manifest](docs/experiments/evidence/README.en.md)**.

The topical documents together contain **91 numbered route/diagnostic entries**, mapped item by item to the **45 sections** of the original handoff; each entry records the motivation for the attempt, the technical mechanism, the result, the reason it was abandoned or kept, and the remaining questions. The number of entries does not represent the number of effective optimizations.

### Experimental data: then check the raw evidence

| What you want to look up | Direct entry point | Content and boundaries |
|---|---|---|
| Data overview and per-file guidance | **[Evidence index](docs/experiments/evidence/README.en.md)** | 125 raw small logs, 14 numeric-source records, missing material, and reading rules |
| Prompt, token history, and small offline-evaluation files | **[Routing lightweight data pack](docs/experiments/data/routing-small/README.en.md)** | 164 copies of raw small files, body text about 0.70 MiB; the batch relationship of the 51/20/8 prompts and the hashes of the originals |
| Batches and format of the raw hidden/router large data | **[Large-data directory description](docs/experiments/data/hidden-routing/README.en.md)** · **[HF Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces)** | 5 batches, 85 collection-request directories; the 49.2 GB archive is distributed by Hugging Face and does not go into Git/Git LFS; only metadata is placed in the repository |
| Throughput, hit rate, transfer, correctness, and failed runs | **[Aggregated results JSON](docs/experiments/evidence/measurements.json)** | CLI, pressure128, fixed-history 400 steps, and the old baseline400; not one set of results from the same protocol |
| Raw stdout/stderr and statistics logs | [Log directory](docs/experiments/evidence/logs/) | PLE/static tables/XT, host/devpart, KV/NXQ, the last CLI, and logical probes; choose files according to the evidence index |
| Weight types, geometry, and backend placement | [Read-only parsed output](docs/experiments/evidence/scans/) | GGUF header/tensor directory and parsing of existing scheduling logs; not newly run model experiments |
| Historical plans, old conclusions, and maintainer supplements | [Historical documents directory](docs/experiments/sources/) | 5 historical snapshots and supplementary records; retractions and corrections of old conclusions are in the topical documents |
| Provenance, sanitization rules, and file hashes | [Provenance manifest](docs/experiments/evidence/provenance.json) · [Public file hashes](docs/experiments/evidence/publication-files.json) | distinguishes SHA-256 of originals from that of public copies; the public file manifest does not include itself or the root README |
| Commit notes and the accidental log-deletion incident | [commit notes](docs/experiments/evidence/commit-notes.txt) · [incident record](docs/experiments/evidence/deleted-logs-incident.json) | retains the historical statements and the facts of the incident; they are not treated as proof of acceptance |

**Notes on using the data:** a comparison must verify the binary, prompts, context, KV, cache budget, and step count; a high hit rate or the runner's `pass` field does not mean numerically correct or consistently faster. The complete model, the large logits arrays, and the experimental binaries are not uploaded; missing logs are explicitly listed and were not fabricated.

**Large data is not distributed with the repository:** the raw hidden/router arrays are packed into a 49.2 GB archive, published on [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces); they do not enter Git or Git LFS, and the checksums are in `SHA256SUMS.txt` and `archive-receipt.json` in the same repository. The current lightweight pack does not contain these binary arrays.

The archive also collects research that was not published and not accepted, and **uploading documentation does not mean these implementations were merged**. The native code baseline is still `7e01451b2`; no model was re-run and no new optimization was enabled because of this archiving.

## 1. Test hardware and scope of applicability

| Item | Local machine configuration / test operating point |
|---|---|
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads; recommended inference thread count 16 |
| Main memory | DDR4-2666, 128 GB; DDR4-2666 means 2666 MT/s |
| GPU | NVIDIA RTX A5000 Laptop GPU, 16 GB VRAM |
| GPU link | PCIe 4.0 ×8 |
| System | Windows 11 x64; CUDA backend |
| Test model | the locally used `Qwen3.8-Flash-Next-UD-IQ3_XXS`, an Unsloth UD mixed-quantization GGUF, three shards |
| Recommended context / KV | `-c 8192`, both K/V are `q8_0` |
| Recommended loading mode | `--cpu-moe --no-mmap --lazy-mode off`, with the weights fully loaded into main memory |
| Recommended expert cache budget | `6144 MiB`; the budget does not equal the VRAM usage of the whole process |
| Long-run measurement | single request, greedy decoding, 400-token request, with `--ignore-eos` enabled |

The memory frequency, capacity, and PCIe link were provided by the user of the device; they are not hardware telemetry re-collected for this documentation update. The power limits, cooling, drivers, and background load of a Laptop GPU all affect the results, and the performance of a desktop A5000, PCIe ×16, other memory frequencies, or other models must not be extrapolated from them. The large model weights used for testing are not provided with this document; you must obtain them yourself and verify their source and license.

## 2. Performance baseline: how much was improved

### The record after the historical fix (not a strictly same-condition A/B)

After fixing the routing misalignment of the host split, the repository's [handoff.md §6.32](handoff.md) recorded the following results at an **8k context, q8_0 KV, auto cache budget (97 slots/layer)**:

| Historical configuration | Generation | Arithmetic comparison with the historical 9.2 figure |
|---|---:|---:|
| Cache-disabled path of the same fork | 9.2 token/s | 1.00× |
| host split, auto cache, `AHEAD=3` | 20.8 token/s | 2.26×, about **+126.1%** |
| host split, auto cache, `AHEAD=2` | 18.4 token/s | not to be conflated with the 6 GiB recommended operating point below |

The improvement in the historical entry is computed as `(20.8 / 9.2 - 1) × 100% = 126.1%`, i.e. about 2.26× throughput, not a 126% latency reduction. The reference comes from the cache-disabled path of this fork, and is **not a head-to-head ranking against the latest upstream llama.cpp, Unsloth, Fate, or HybriMoE**.

The historical record includes a statement that the Eiffel text regression passed, but no complete evidence chain was established that the prompts, generated token counts, and all run conditions were identical on both sides. Therefore **+126.1% is only an arithmetic comparison of numbers in the historical record, and this README does not accept it as a measured net improvement after strictly controlling the variables**, nor can it be used to claim lossless quality or consistent speedup. `auto/AHEAD=3` is not the conservative configuration recommended in this document; how much the currently recommended configuration improves things still requires a new same-condition comparison before it can be answered reliably.

### The currently recommended 6 GiB / 400-token operating point

Later local development builds used the conservative parameters below, and CLI runs that exited normally recorded **19.2 and 19.3 token/s**; a later re-test of the old baseline with the same parameters gave **18.4 token/s**. The observed run behind 19.3 recorded:

| Metric | Single observed value |
|---|---:|
| Request generation / per-graph record | 400 token (EOS ignored) / 403 graphs, the latter including non-decode graphs |
| Expert cache | 6144 MiB requested, 64 slots/layer, physical cache about 6116.3 MiB |
| Whole-card sampled VRAM peak | 12458 MiB, including other VRAM usage at the time; not a pure model allocation figure |
| Page-locked weight buffer | 72.6 GiB, with 0 lock failures reported in the log |
| Cache access hit rate | 69.7%, not predictor accuracy |

These are the maintainer's local records and **not a new round of acceptance for the `7e01451b2` published build**; the complete raw logs have not yet been published with this README, so this table alone cannot be used for independent review. This operating point was not re-run against a same-condition cache-disabled comparison, so **19.3 is not divided by the historical 9.2 to claim a net improvement for the current configuration**. 19 token/s is only the current internal screening floor on this machine; it is neither guaranteed to be reached every time, nor does reaching it mean the result is correct.

The following numbers are explicitly not counted as valid results:

- The host 20.3–22 token/s from the old documents, before the routing-misalignment fix; later records have withdrawn their eligibility as correct performance evidence.
- The roughly 28–32 token/s that devpart showed in a computationally incorrect state.
- Single peaks from later cache-strategy candidates, failed runs, and results that only reduced transfer bytes without a stable whole-step benefit.
- "Improvements" obtained by directly dividing between different prompts, cache budgets, PLE settings, contexts, and short-run warm-up states.

**The recommended parameters do not equal the historically highest-scoring parameters.** Later 400-token / 6 GiB runs with `AHEAD=3` have produced non-zero exits, so `AHEAD=2` is preferred on this machine. The available data cannot confirm that background software was the cause of the speed drop.

## 3. Recommended parameters: host split, not devpart

`host` means **`LLAMA_MOE_DEVPART=0` and `LLAMA_MOE_SPLIT=1`**; it is neither pure CPU nor split turned off. The experts actually selected by the model still need to be computed: resident experts are executed by the GPU, and the non-resident part is executed by the CPU.

| Parameter group | Recommended setting |
|---|---|
| Execution and cache layout | `DEVPART=0`, `SPLIT=1`, `DIRECT_READ=1`, `GLOBAL_POOL=0`, `WINDOW_LAYERS=0` |
| Capacity | `CACHE_MIB=6144`, `VRAM_LIMIT_MIB=15360`, `VRAM_GUARD_MIB=1024` |
| Eviction policy | `MRS=1`, `FIFO=0`, `EVICT_SCORE=0` |
| Prediction | `PREDICT_SMOE=1`, `SMOE_NONBLOCK=1`, `SMOE_AHEAD=2`, `PREDICT_TOPK=26` |
| Transfer and backfill | `PREFETCH=1`, `PREFETCH_JOIN=0`, `HOT_BACKFILL=8` |
| CPU part | `CPU_ASYNC=1`, `LLAMA_ARG_THREADS=16` |
| Other paths that are not enabled | `PREDICT_FATE=0`, `PREDICT_XT=0`, `INSERT_ON_MISS=0`, `FALLBACK_PREFETCH=0` |
| Main memory / PLE / KV | `PIN_WEIGHTS=1`, `LLAMA_PLE_CACHE_MIB=0`, `LLAMA_PLE_GPU_CACHE_MIB=0`, using ordinary q8_0 KV |

Every variable name in the table above that is not written out in full is prefixed with `LLAMA_MOE_`. `PREDICT_TOPK=26` is a prefetch-candidate parameter and is **not a change to the top-k of the model's actual routing**. MRS is the name carried over in this branch; `EVICT_SCORE=0` uses the actual routing frequency and is not the same as enabling the full gate-score strategy from the paper.

### Windows build and run example

Use an **x64 Developer PowerShell** with the Visual Studio C++ toolchain, CMake, Ninja, the CUDA Toolkit, and Python installed. See the [build documentation](docs/build.md) for detailed platform requirements. Build from this fork; do not treat the generic binaries on the upstream release page as a version that includes this branch's features:

```powershell
git clone --branch master https://github.com/starsder/qwen3.8-flash-next-inference-research.git
Set-Location qwen3.8-flash-next-inference-research
cmake -S . -B build-ple-trace-mrs -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-ple-trace-mrs --target llama-cli -j 16
```

Place the three GGUF shards in the same directory, and edit `MODEL` in [tools-run.py](tools-run.py) to point to the real `...-00001-of-00003.gguf`. The script still currently contains the absolute path of the maintainer's machine; do not copy that path directly, and do not download only the first shard.

Then run the following commands in a **new** Developer PowerShell, inside the repository root. The environment cleanup only affects the current terminal and its subsequent child processes, and does not modify the system's permanent environment:

```powershell
Get-ChildItem Env: |
    Where-Object { $_.Name -match '^(LLAMA_|GGML_)' } |
    ForEach-Object { Remove-Item -LiteralPath ("Env:" + $_.Name) }

$moeArgs = @(
    'LLAMA_MOE_DEVPART=0',
    'LLAMA_MOE_SPLIT=1',
    'LLAMA_MOE_GLOBAL_POOL=0',
    'LLAMA_MOE_WINDOW_LAYERS=0',
    'LLAMA_MOE_DIRECT_READ=1',
    'LLAMA_MOE_CACHE_MIB=6144',
    'LLAMA_MOE_VRAM_LIMIT_MIB=15360',
    'LLAMA_MOE_VRAM_GUARD_MIB=1024',
    'LLAMA_MOE_MRS=1',
    'LLAMA_MOE_FIFO=0',
    'LLAMA_MOE_EVICT_SCORE=0',
    'LLAMA_MOE_PREFETCH=1',
    'LLAMA_MOE_PREFETCH_JOIN=0',
    'LLAMA_MOE_PREDICT_SMOE=1',
    'LLAMA_MOE_PREDICT_FATE=0',
    'LLAMA_MOE_PREDICT_XT=0',
    'LLAMA_MOE_PREDICT_TOPK=26',
    'LLAMA_MOE_SMOE_NONBLOCK=1',
    'LLAMA_MOE_SMOE_AHEAD=2',
    'LLAMA_MOE_HOT_BACKFILL=8',
    'LLAMA_MOE_CPU_ASYNC=1',
    'LLAMA_MOE_INSERT_ON_MISS=0',
    'LLAMA_MOE_FALLBACK_PREFETCH=0',
    'LLAMA_MOE_PIN_WEIGHTS=1',
    'LLAMA_PLE_CACHE_MIB=0',
    'LLAMA_PLE_GPU_CACHE_MIB=0',
    'LLAMA_ARG_THREADS=16'
)
python .\tools-run.py --tag host400 --tokens 400 --ignore-eos `
    --min-free-mib 90000 --wait-mem 120 --timeout 900 @moeArgs
```

The script in this release **has no `--profile` or `--json` parameter**. The commands above explicitly override defaults of the old script such as the auto cache; do not use `--keep` to mix in another experimental environment. The script itself supplies `--cpu-moe --no-mmap --lazy-mode off -ngl 49 -c 8192`, q8_0 KV, greedy sampling, and single-turn run settings; these parameters only target the model operating point above and should not be generalized to all GGUF files.

Check `host400-out.txt`, `host400-err.txt`, `stats-host400.csv`, and the script's console report:

- Confirm that the host mode, effective cache capacity, actual thread count, and page-locking results match expectations.
- Confirm that the report shows the **model's real `exit=0`**, a final Generation line, and no anomalies; the old script returning 0 by itself is not enough to prove that the child process succeeded.
- 400 token completeness in the CLI depends mainly on the requested count, `--ignore-eos`, and a normal exit; the graph count is not the generated token count. A benchmark should save the complete command, environment, model shard checksums, output, and raw exit code.
- Runs that fail, time out, degrade in output, or lack statistics cannot be counted as successful performance. `--ignore-eos` is a long-run test setting and may make the model keep repeating output after a normal answer has ended; it is not recommended for everyday question answering.

## 4. Known instabilities and memory risks

1. **Only a single instance can run.** On this machine about 72.6 GiB of weight buffers are page-locked; two instances could exceed 145 GiB on this item alone. The repository does have a history of concurrent models exhausting the whole machine's memory and hanging. The script's `.tools-run.lock` only protects the same working directory and launch methods that honor that lock; another checkout, a bare CLI, or another service can bypass it.
2. **Keep the startup gate of at least 90000 MiB of free main memory.** It is not a mathematical upper bound on the total memory requirement, nor can it stop other programs from seizing memory after startup. Do not lower the gate, manually delete the lock of a live process, or restart without confirming that the old process has exited.
3. **Keep `--no-mmap --lazy-mode off` and the current page-locking approach.** This branch has historically seen severe memory pressure caused by the combination of mmap and weight page locking; do not mix them on your own. The 6 GiB expert cache is only part of the VRAM cost, and does not mean the model needs only 6 GiB of RAM or VRAM.
4. **`0xC0000005` can still occur during exit.** The profiler has an object-lifetime risk; the current recommendation keeps the script's timing with the `LLAMA_TOKEN_PROF=1` observation configuration, and this risk has not yet been fixed and verified. Even if the main output has already been produced, a non-zero exit should be treated as a failure, not as "the log was just lost and can be ignored".
5. **Treat everything outside the recommended scope as an unverified combination.** In particular, do not directly stack devpart, the shared pool, automatic cache growth, other look-ahead distances / online tuners, PLE caching, CPU-KV/QSA, TBQ KV, extremely long contexts, vision models, MTP/draft/speculative decoding, or multiple concurrent requests or models. They may change the memory footprint, graph shapes, and asynchronous timing, causing OOM, illegal access, or silent computational errors.
6. **A smaller cache is not automatically safer either.** Older versions had a problem where an extremely small budget caused zero slots and an incorrect degradation; changing the configuration cannot replace correctness verification.

The CPU/GPU numeric implementations for the same expert may produce floating-point differences, and free generation may also differ once cache placement changes. Therefore bit-by-bit equivalence, identical tokens for all tokens, or "lossless" is not promised. Conversely, a difference cannot be judged to be some new bug without being localized first. Identical text under forced token replay does not mean identical free generation.

For service deployment, unattended operation, or other important uses, it is advisable to first choose a verified upstream version that suits your own workload; do not treat the recommended configuration of this experimental branch as a safety certification. This update only organizes documentation; no model was run further, and no claim is made that these risks have been eliminated.

## 5. Techniques used and implementation boundaries

- **ggml/GGUF/quantized CPU and CUDA operators:** builds on the upstream inference and quantization foundation; the CPU uses the SIMD paths that are actually available, and the GPU uses CUDA. Unsloth UD is a mixed quantization, so you cannot assume that all weights have only the single type named in the file name.
- **Per-layer expert bundle cache:** manages the gate/up/down components of an expert under a common slot, and the residency check requires the necessary components to be complete; direct-read lets the GPU read from the cache slot, avoiding unnecessary intermediate movement.
- **CPU/GPU splitting and overlap:** execution placement is decided from the real router results, the CPU handles non-resident experts, asynchronous CPU workers partially overlap with the GPU portion, and they join at the consumption dependency. Predicted experts are not substituted for real experts, and missed computation is not treated as an optimization.
- **Side-graph prediction named SMoE in the source:** uses approximate intermediate states to generate prefetch candidates for future layers, together with `AHEAD=2` and non-blocking result consumption. This is the implementation name of this project, and no claim is made here to be a complete reproduction of any paper of the same name.
- **Actual usage frequency and hot backfill:** `EVICT_SCORE=0` is recommended, managing the cache according to actual routing usage information; `HOT_BACKFILL=8` and predictive prefetching come from different sources.
- **Transfer management:** uses CUDA page-locked main memory, side streams, events, and resident/in-flight deduplication, retaining the existing prefetch budget and feasibility constraints. Hitting a resident expert does not mean transferring it again; reducing a particular wait count or transferred bytes does not necessarily shorten the whole-step latency.
- **Targeted correctness fixes:** after updating the routing leaves in the host partition, the necessary copies are resubmitted, avoiding reads of the previous layer's routing; the offsets/strides of UD mixed-quantization weights are handled and protected accordingly. This is not a complete correctness proof for all backends, model shapes, or concurrency modes.

See [ggml-backend.cpp](ggml/src/ggml-backend.cpp), [llama-graph.cpp](src/llama-graph.cpp), the [cache design note](docs/moe-cache-score-aware-prd.md), and the [experiment ledger](handoff.md) for implementation and experiment descriptions. The ledger is a research record appended over time and contains conclusions and old parameters that were later rejected; when conflicts occur, it should be read together with the later corrections, and the highest speed in it cannot simply be picked out.

## 6. Sources, references, and open-source notices

### Code sources and technical references

| Source | This repository's relationship to it |
|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), [ggml](https://github.com/ggml-org/ggml) | Base inference framework, GGUF, quantization operators, and the various backends; retains the original authors' copyright, licenses, and contribution history |
| [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) | The direct upstream of this fork; model adaptations and related inherited code are governed by the Git history and the file notices, and existing Qwen model support is not claimed as original to this fork |
| [starsder/qwen3.8-flash-next-inference-research](https://github.com/starsder/qwen3.8-flash-next-inference-research) | The caching, scheduling, fixes, test tooling, and experiment records of this research project; it is not an official stable upstream release, and upstream compatibility is not guaranteed |
| [Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate](https://arxiv.org/abs/2502.12224) | Technical reference for cross-layer gate prediction / expert prefetching; the repository has a separate Fate path, but the recommended configuration explicitly sets `PREDICT_FATE=0` |
| [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](https://arxiv.org/abs/2504.05897), [the authors' code repository](https://github.com/PKU-SEC-Lab/HybriMoE) | Technical reference for hybrid scheduling, prefetching, and score-based caching; this is not a port of the whole HybriMoE/kTransformers stack or a reproduction of its performance |
| [Unsloth](https://github.com/unslothai/unsloth), [its Hugging Face organization](https://huggingface.co/unsloth) | The nominally stated source of the quantization files used for local testing; the specific weights must be governed by the model card, shard checksums, and license at the time of download |

The paper citations are technical attribution statements, and **do not mean that the papers, their accompanying code, the model weights, or the trademarks are relicensed by this repository**. The performance numbers here belong only to local experiments on this machine and cannot be cited as the experimental results of the papers or authors above. Project, model, and hardware names are used only for identification and attribution, and do not indicate any recognition, cooperation, or warranty for this fork by any upstream organization, model author, or NVIDIA.

### Licenses and distribution responsibilities

- The main project uses the root [MIT LICENSE](LICENSE), which retains `Copyright (c) 2023-2026 The ggml authors`. New modifications in this fork are provided under the repository's MIT license, and existing file-level or independent third-party notices are not changed by this README.
- MIT permits use, modification, and commercial distribution, but when distributing copies of the software or substantial portions of it, the copyright and complete license notice it requires must be retained. The instability warnings and deployment advice in this document are risk statements, and are **not additional license restrictions such as a "no commercial use" clause**.
- Third-party libraries, embedded code, components obtained at build time, and runtime libraries shipped with the package retain their respective license and NOTICE requirements; the top-level MIT notice alone cannot cover all dependencies. Please retain [AUTHORS](AUTHORS), and verify the notices in [licenses](licenses), [vendor](vendor), [gguf-py/LICENSE](gguf-py/LICENSE), the corresponding source files, and the build artifacts according to the version you actually distribute. For example, [cpp-httplib](vendor/cpp-httplib/LICENSE) is MIT and [xxHash](vendor/hash/xxhash/LICENSE) is BSD-2-Clause; the third-party acknowledgements in the upstream README below are also retained.
- The licenses of assets such as test models and tokenizers are handled separately from the software code. This document has not verified the complete license chain of that specific GGUF release, and **does not claim that it automatically falls under MIT, can be freely used commercially, or can be redistributed**; use, download, conversion, and redistribution should follow the terms applicable to the original model and the quantization publisher.
- The CUDA Toolkit, drivers, and other proprietary runtimes are used and distributed under their own terms, and do not automatically obtain redistribution authorization because this project is open source. Please install them yourself from legitimate sources, and do not unconditionally package third-party components.
- The software is provided under the license's **"AS IS"** terms, without warranties of merchantability, fitness for a particular purpose, or otherwise; liability limitations are governed by the original license and applicable law. This README is not proof that a legal audit of all dependencies has been completed, nor is it a substitute for formal legal advice.

If you find a missing copyright notice, source citation, or license conflict, please provide the specific file, version, and original source through [Issues](https://github.com/starsder/qwen3.8-flash-next-inference-research/issues) so that it can be checked and corrected. When submitting code, please state the borrowed source and license, and provide reproducible correctness/performance evidence; do not upload models, private logs, or credentials that you have no right to distribute.

---

## Upstream original README (retained)

The original upstream introduction, links, and acknowledgements are retained below. The release badges, installation entry points, and default download examples in it mainly point to upstream and **do not mean that they include the experimental features of this fork**; for this fork, please use the source build and parameter instructions above.

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
