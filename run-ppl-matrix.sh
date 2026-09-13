#!/bin/bash
cd /f/src/llama.cpp-unsloth-qwen4exp
export LLAMA_MOE_SPLIT=1 LLAMA_MOE_CACHE_MIB=auto LLAMA_MOE_PREDICT_SMOE=1 LLAMA_MOE_PREFETCH=1
export LLAMA_MOE_DIRECT_READ=1 LLAMA_MOE_MRS=1 LLAMA_MOE_VRAM_LIMIT_MIB=15667 LLAMA_MOE_VRAM_GUARD_MIB=512
export LLAMA_PLE_CACHE_MIB=0 LLAMA_PLE_GPU_CACHE_MIB=0 GGML_OP_OFFLOAD_MIN_BATCH=1
M="F:/models/qwen38/unsloth-iq3-xxs/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
F="ppl/wikitext-2-raw/wiki.test.raw"
for t in f16 q8_0 q4_0 tbq4_0 tbq3_0; do
  ./build-ple-trace-mrs/bin/llama-perplexity.exe -m "$M" -ngl 49 --cpu-moe -f "$F" -c 512 \
    --chunks 8 -fa 1 -ctk $t -ctv $t > ppl-$t.log 2>&1
  echo "ppl-$t done: $(grep -m1 'Final estimate' ppl-$t.log)"
done
./build-ple-trace-mrs/bin/llama-perplexity.exe -m "$M" -ngl 49 --cpu-moe -f "$F" -c 512 \
  --chunks 8 -fa 1 -ctk tbq4_0 -ctv tbq4_0 --kl-divergence-base ppl/base-f16.kld --kl-divergence > kld-tbq4_0.log 2>&1
echo "kld tbq4_0 done"
echo ALL_DONE
