# Direct simulated next-layer gate recall

Existing standard traces only; no model rerun. Device=cuda; batches=1024. Approximate current-layer SMoE delta is added to the recorded next-layer attention residual, then exact HC-FFN mix and router weights are applied.

True-path router reconstruction: cosine=0.99999863, RMSE=0.018975, max_abs=0.179953 over 47 events.

| slots | recall@8 | recall@16 | recall@32 | residual recall@8 | residual recall@16 | residual recall@32 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 91.59% | 99.41% | 99.89% | 89.90% | 99.30% | 99.87% |
| 22 | 92.09% | 99.47% | 99.90% | 90.32% | 99.33% | 99.87% |
| 32 | 92.77% | 99.50% | 99.91% | 90.45% | 99.27% | 99.85% |
| 44 | 93.31% | 99.56% | 99.92% | 91.19% | 99.42% | 99.90% |
| 64 | 93.84% | 99.59% | 99.92% | 91.39% | 99.41% | 99.87% |

Residual recall denominator is only the true top-8 experts absent from the layer-local FIFO pool at prediction time; samples with an empty residual set are excluded.

This is a direct offline gate simulation with teacher-forced attention, not a cumulative end-to-end counterfactual rollout.
