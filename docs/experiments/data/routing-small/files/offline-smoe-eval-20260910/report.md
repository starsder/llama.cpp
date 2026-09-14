# Offline SMoE trace evaluation

- Trace root: `LOCAL_MODELS\qwen38\traces\collect-run2`
- Loaded prompts: **51** / discovered **51**
- FIFO slot scans: `16, 32, 64`

## Current trace capability

- True hidden: `True`
- True router logits: `True`
- Attention input/output: `False`
- Per-expert contribution tensors: `False`

The counterfactual SMoE hidden cosine is **not reported** from this trace: attention tensors and per-expert output contributions are absent. The numbers below are causal FIFO candidate metrics, not a substitute for SMoE fidelity.

## Overall FIFO replay

| slots/layer | estimated cache | events | FIFO pool hit@8 | FIFO ranked recall@8 | FIFO ranked recall@16 | same-layer overlap@8 | Jaccard@8 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.41 GiB | 1401238 | 43.08% | 23.22% | 43.08% | 35.32% | 24.72% |
| 32 | 2.82 GiB | 1401238 | 58.47% | 18.08% | 33.02% | 35.32% | 24.72% |
| 64 | 5.64 GiB | 1401238 | 72.39% | 13.36% | 23.88% | 35.32% | 24.72% |

Cache estimate assumes 48 layers × 1.88 MiB per routed expert bundle; it excludes shared experts and allocator overhead. `pool hit@8` uses the entire resident window. `recall@8/@16` ranks the FIFO window newest-first for a comparable top-m report; FIFO itself has no score, so these are a conservative recency view, not an SMoE gate prediction.

## Required next trace fields

To complete the exact experiment, the collector must additionally dump, at the same graph/layer/token key:

1. attention input and output (or an equivalent replayable attention state);
2. each routed expert's weighted output before the MoE sum;
3. the shared-expert output and residual/norm ordering, if the approximate forward zeros routed experts;
4. a counterfactual hidden dump for each slot setting, or enough tensors to compute it offline.

The existing `hidden/` and `router/` files are still useful for the Fate/hidden/XT predictor baselines, but they cannot establish the SMoE approximation cosine by themselves.
