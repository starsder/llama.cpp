$ErrorActionPreference = "Continue"
$base = @(
  "-m", "F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
  "-ngl", "49", "--cpu-moe", "--lazy-mode", "on",
  "-c", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
  "-p", "The capital of France is", "-n", "8", "--temp", "0", "-st",
  "--no-warmup", "--no-display-prompt"
)
function Set-Common {
  $env:LLAMA_MOE_CACHE_MIB="2048"; $env:LLAMA_MOE_PREDICT_SMOE="1"; $env:LLAMA_MOE_PREDICT_FATE="0"
  $env:LLAMA_MOE_PREDICT_TOPK="26"; $env:LLAMA_MOE_PREFETCH="1"; $env:LLAMA_MOE_SPLIT="1"
  $env:LLAMA_MOE_MRS="1"; $env:LLAMA_MOE_MRS_ALPHA="0.75"; $env:LLAMA_MOE_MRS_TOPP="20"
  $env:LLAMA_MOE_VRAM_LIMIT_MIB="15360"; $env:LLAMA_MOE_VRAM_GUARD_MIB="1024"
  $env:GGML_OP_OFFLOAD_MIN_BATCH="1"
  $env:LLAMA_PLE_CACHE_MIB="0"; $env:LLAMA_PLE_GPU_CACHE_MIB="0"
  Remove-Item Env:LLAMA_MOE_CACHE_TIMING -ErrorAction SilentlyContinue
  Remove-Item Env:LLAMA_TOKEN_PROF -ErrorAction SilentlyContinue
  Remove-Item Env:LLAMA_MOE_CACHE_STATS -ErrorAction SilentlyContinue
}

# R7: join=0, insert=0, direct_read=0 (isolate no-join)
Set-Common; $env:LLAMA_MOE_PREFETCH_JOIN="0"; $env:LLAMA_MOE_INSERT_ON_MISS="0"; $env:LLAMA_MOE_DIRECT_READ="0"
& build-ple-trace-mrs\bin\llama-cli.exe @base *> bz7.txt
Write-Output "R7 done"

# R8: join=1, insert=1, direct_read=1 (direct read with normal insert)
Set-Common; $env:LLAMA_MOE_PREFETCH_JOIN="1"; $env:LLAMA_MOE_INSERT_ON_MISS="1"; $env:LLAMA_MOE_DIRECT_READ="1"
& build-ple-trace-mrs\bin\llama-cli.exe @base *> bz8.txt
Write-Output "R8 done"

# R9: join=0, insert=1, direct_read=0 (no-join, normal config)
Set-Common; $env:LLAMA_MOE_PREFETCH_JOIN="0"; $env:LLAMA_MOE_INSERT_ON_MISS="1"; $env:LLAMA_MOE_DIRECT_READ="0"
& build-ple-trace-mrs\bin\llama-cli.exe @base *> bz9.txt
Write-Output "R9 done"
