# Standard Qwen4-Exp trace + local SMoE simulation (20 requests)

- Source: `LOCAL_MODELS\qwen38\traces\standard-smoe-20-20260910b`; standard checkout instrumentation only.
- Requests: 20; layers: 48; routed expert bundle estimate: 1.88 MiB/expert.
- Mode: `teacher_forced_local_smoe`. Each layer uses the standard run's true pre-FFN residual; this is not a cumulative counterfactual rollout.
- Pool: per-layer causal FIFO, updated after the current token from the true routed top-k (the standard trace exposes top-10); hit/recall metrics are evaluated against the top-8 subset.

## Aggregate result

| slots/layer | cache | events | local hidden cosine | relative L2 | FIFO pool hit@8 | ranked recall@8 | ranked recall@16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.41 GiB | 91849 | 0.98801 | 0.12306 | 42.39% | 23.01% | 42.39% |
| 22 | 1.94 GiB | 91849 | 0.98971 | 0.11048 | 49.34% | 20.69% | 38.12% |
| 32 | 2.82 GiB | 91849 | 0.99160 | 0.09520 | 57.71% | 18.20% | 33.32% |
| 44 | 3.88 GiB | 91849 | 0.99292 | 0.08423 | 63.32% | 16.65% | 30.30% |
| 64 | 5.64 GiB | 91849 | 0.99396 | 0.07399 | 68.58% | 15.12% | 27.20% |

## Same-layer cross-token route reuse

Adjacent-token top-8 overlap: **35.70%**; Jaccard: **24.54%** over 73649 comparisons.

## Interpretation

The local approximation remains directionally close because the shared expert and the resident routed contributions are measured from the standard run. The cache hit rate is the practical constraint: at the 2 GiB budget (22 slots/layer) this FIFO pool retains only the routed experts already in the layer-local window. These numbers do not establish that a cumulative SMoE rollout preserves later-layer routing; that requires rerunning the graph with the approximated residual fed into the next layer.

Artifacts: `overall.csv`, `per_layer.csv`, and `summary.json` in the summary directory.
