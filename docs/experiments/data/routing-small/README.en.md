[中文](README.md) · [English](README.en.md)

# Lightweight Data Package for Routing Research

We first provide the original small files that are easy to download and verify: **164 files, totalling 735,256 bytes of public body text (about 0.70 MiB)**, plus this description and a source manifest. The large hidden/router binaries are not committed to Git and do not use Git LFS; they are instead distributed via the [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces).

Back to the [experiment archive](../../README.en.md) · [evidence overview](../../evidence/README.en.md).

## File entry points

| Content | Count | Entry |
|---|---:|---|
| Multi-domain prompt originals | 51 | [collect-prompts2](files/collect-prompts2/) |
| Originals of the earliest three prompts | 3 | [collect-prompts](files/collect-prompts/) |
| Token records for the 51 requests | 51 | [collect-run2](files/collect-run2/); each request directory contains tokens.csv |
| Token records for two independent runs of the same three prompts | 6 | [collect-run1](files/collect-run1/) · [collect-run1b](files/collect-run1b/) |
| Collection manifest and token records for the 20 requests | 21 | [standard-smoe-20-20260910b](files/standard-smoe-20-20260910b/) |
| Supplementary collection manifest and token records for the 8 requests | 9 | [standard-smoe-8-20260911](files/standard-smoe-8-20260911/) |
| Derived summaries of the stage tensors for the 20 requests | 5 | [standard summary](files/standard-smoe-20-20260910b-summary/) |
| Next-layer gate evaluation using teacher-forced attention | 4 | [direct recall](files/direct-smoe-recall-20-20260910b/) |
| N+k ablation | 2 | [N+k results](files/smoe-nk-8-20260911/) |
| FIFO offline evaluation and budget variants for the 51 requests | 10 | [eval](files/offline-smoe-eval-20260910/) · [budget](files/offline-smoe-budget-20260910/) |
| Historical output of the hidden prediction probe | 2 | [generalize-out](files/generalize-out.txt) · [probe4-out](files/probe4-out.txt) |
| Sources, byte counts and SHA-256 | — | [manifest.json](manifest.json) |

## What must not be mixed together

- **51 → 20 → 8 is a prompt subset relationship, not 79 independent prompts.** The 20 and the 8 come from a new round of collection, and hidden and router files from different runs must not be cross-joined. Different collections of the same prompt should likewise be placed in the same training/testing split.
- The token tables for `collect-*` are `graph_id,ctx_id,pos,token_id`; `standard-smoe-*` adds `request_id,round,phase,batch_size`. The `pos` of different collectors must not be treated directly as a global context position without verification.
- The collection graphs include non-decode graphs; the number of rows in a token table is not the number of generated tokens. Records shorter than the generation limit are preserved as-is, and are not padded out or passed off as equal-length sequences.
- The 99%-level recall@16 of `direct recall` used the recorded next-layer attention residual; it is a conditional offline simulation, not online-ahead-of-time prediction accuracy; nor was it taken as grounds to conclude that it is the teacher experiment the maintainer recounted.
- The FIFO report is not an implementation result of an SMoE gate predictor; the report itself declines to give complete counterfactual hidden reconstruction conclusions when attention/expert contributions are missing.
- The raw tensor set for the 20 requests does not directly store `ffn_moe_input`; only the supplementary collection for the 8 requests adds it. The current lightweight package only publishes the manifests and token records, and does not include these two batches of tensor data.

## Identity and distribution boundary

These are small-file copies of existing experiments; no model was re-run. The body text was only normalized to UTF-8/LF and had local machine root paths replaced; the results and wording of the old reports are retained, and the interpretation boundaries are given on this page and in the [prediction topic](../../02-prediction-and-cache.en.md). `manifest.json` lists the original and the public copy hashes separately.

Prompt content, input tokens and generated-token records are all retained, to facilitate verification of the experiments; this package is not a complete tokenizer or model snapshot. Model weights, extracted gate/HC weights, activation arrays, full-vocabulary logits, executables and local machine environment credentials are all not included. Do not automatically extrapolate this repository's license to the model or third-party data that are not provided.

**Large data download:** the 49.2 GB original compressed archive (`qwen3.8-hidden-routing-research.7z`) has been published on the [Hugging Face Dataset](https://huggingface.co/datasets/satsder/qwen3.8-flash-next-routing-traces), and the same repository provides `SHA256SUMS.txt` and `archive-receipt.json`; cloning the repository does not require downloading these arrays.
