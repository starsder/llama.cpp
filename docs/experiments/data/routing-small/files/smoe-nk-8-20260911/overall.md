# SMoE N+k prediction degradation

- trace root: `LOCAL_MODELS\qwen38\traces\standard-smoe-8-20260911`
- slots/layer: 22; k = 1..4

| variant | k | N+1 recall@10 | samples |
|---|---:|---:|---:|
| fifo | 1 | 68.29% | 56400 |
| fifo | 2 | 59.72% | 55200 |
| fifo | 3 | 55.47% | 54000 |
| fifo | 4 | 53.65% | 52800 |
| full | 1 | 68.53% | 56400 |
| full | 2 | 59.94% | 55200 |
| full | 3 | 55.70% | 54000 |
| full | 4 | 53.77% | 52800 |
| input_only | 1 | 67.90% | 56400 |
| input_only | 2 | 59.34% | 55200 |
| input_only | 3 | 55.22% | 54000 |
| input_only | 4 | 53.44% | 52800 |
| oracle | 1 | 100.00% | 56400 |
| oracle | 2 | 100.00% | 55200 |
| oracle | 3 | 100.00% | 54000 |
| oracle | 4 | 100.00% | 52800 |
| shared_only | 1 | 68.07% | 56400 |
| shared_only | 2 | 59.52% | 55200 |
| shared_only | 3 | 55.29% | 54000 |
| shared_only | 4 | 53.47% | 52800 |
