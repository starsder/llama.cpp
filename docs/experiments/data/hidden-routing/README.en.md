[中文](README.md) · [English](README.en.md)

# Qwen3.8 Flash Next hidden/router raw capture package

This is an offline research data package; it is not model weights, and it is not a release that can directly replace upstream llama.cpp. The raw arrays are distributed as a single compressed archive and do not go into Git or Git LFS; the download entry point is [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces).

## Main package contents

| Directory | Requests | Positioning |
|---|---:|---|
| collect-run1 | 3 | Early hidden/expert capture, no router directory |
| collect-run1b | 3 | Independent re-run of the same prompts, with router added; cannot be stitched together with run1 across runs |
| collect-run2 | 51 | Multi-domain hidden/router/expert capture |
| standard-smoe-20-20260910b | 20 | Stage tensors for attention, HC, gate and expert contributions, etc.; ffn_moe_input is not saved directly |
| standard-smoe-8-20260911 | 8 | Supplementary capture that adds ffn_moe_input |
| DATASET_INFO | — | This description and the metadata inventory |

The prompt relationship is 51 ⊃ 20 ⊃ 8; they cannot be split into 79 independent prompts for train/test; repeated runs of the same prompt should be grouped together. The earliest three prompts are a separate historical set; this table lists the number of capture directories, not the deduplicated prompt count of the whole package.

The prompts, token records, historical reports and source-file hashes have been placed in the [repository lightweight data package](https://github.com/starsder/qwen3.8-flash-next-inference-research/tree/master/docs/experiments/data/routing-small). Local raw directories are not renamed and their data is not modified; the archive excludes raw logs and miscellaneous TXT files to avoid carrying host paths, and the related sanitized reports are provided in the lightweight package. There are no model weights, no extracted gate/HC matrices, no GGUF, no NPY and no executables.

## Read format and alignment boundaries

### collect series

- hidden_meta.csv: `graph_id,sched_id,layer,ne0,ne1,ne2,nb1,nb2,type,file`. The hidden records in this batch are all F32 with ne0=2560; the logical shape is usually [2560,1,nt].
- experts.csv: `graph_id,split_id,layer,tensor,token_row,rank,expert_id,sched_id`.
- tokens.csv: `graph_id,ctx_id,pos,token_id`; pos is an index within the microbatch and cannot be used directly as a global context position.
- The router `rl_*.bin`/`rp_*.bin` files hold 512 F32 values, 2048 bytes per file; they cover only the single-token graphs accepted by the collector, not the complete prefill.
- Request identity is given by the directory. sched_id and ctx_id belong to the backend/context namespaces respectively and are not a global request_id.
- collect-run2 has 1,197,360 hidden metadata records and 14,348,677,120 logical hidden bytes; experts.csv totals 14,012,380 records. These are not counts of independent training samples.
- The 29,285 token records include non-decode records and cannot be called 29,285 generated tokens. Short sequences are kept as-is. Layer 47 prefill coverage differs and must be aligned in combination with the graph's output token selection; it must not be judged corrupt from shape alone.

### standard series

- tensors.csv contains `graph_id,sched_id,kind,layer,occurrence,ne0..ne3,nb0..nb3,type,bytes,file`.
- **The bin files are tightly packed by logical tensor row; the CSV's nb* retain the original tensor strides and are for auditing only, not the file strides.** For example, topk with logical shape [10,2] and F32-equally-wide int32 data is only 80 bytes even when the original nb1 is 2048; the file must not be read by skipping with the original nb1.
- topk is int32; floating-point tensors are read according to the recorded type. The same kind may have different shapes and occurrences, and the `_o` numbering must not be dropped at will.
- tokens.csv: `request_id,graph_id,round,pos,token_id,phase,batch_size`, which differs from the collect version schema.
- The 20 batch has 9,823 tensor records per request, 196,460 in total; the 8 batch has 11,647 per request, 93,176 in total. The 8 batch adds two ffn_moe_input shapes per layer, but selection should follow the actual metadata.

## Quality and scope of applicability

The hidden metadata and experts.csv fields of collect-run2 were scanned in full, and the first and last hidden size of each request and the single-token router samples were checked. The capture manifests of the 20/8 batches all have exit code 0. **The values of all arrays have not been verified one by one, and the model has not been re-run; archive verification is not equal to a certification of the numerical correctness of the experiment.** The collect series lacks a complete startup/exit-code list, so the full capture parameters or binary identity cannot be recovered from the old documents alone.

The known-truncated standard-smoe-20-20260910, the early smoke that suffered same-name overwrites and other development smokes are not mixed into the main package; the reasons are listed in inventory.json. The old 99%-level direct-recall report uses the recorded next-layer attention residual, not online early-prediction accuracy. The raw data comes from a specific model and experiment branch and is not guaranteed to apply to other Qwen versions or to upstream llama.cpp.

## Integrity and extraction

The archive file name is `qwen3.8-hidden-routing-research.7z` (49,235,820,003 bytes), and it is located in the same [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces) as `SHA256SUMS.txt` and `archive-receipt.json`. After downloading, verify the SHA-256 first, then run `7z t qwen3.8-hidden-routing-research.7z` with 7-Zip. Extract into a separate empty directory; do not overwrite the source worktree. There is a very large number of files, so reserve enough capacity and extraction time.

The Git repository only stores this description, metadata and lightweight small files. External distribution does not change the licensing requirements of the model or third-party materials, and the repository license should not be automatically extrapolated to other assets.
