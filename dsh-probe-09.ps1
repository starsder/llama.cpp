# Probe 06: decode/prefill-separated instrumentation.
#   F1: devpart OFF (correct output)   <- primary
#   F2: devpart ON
$model = 'F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf'
$exe   = 'build-ple-trace-mrs\bin\llama-cli.exe'
$flags = @('-ngl','49','--cpu-moe','--lazy-mode','off','--no-mmap','-c','8192',
           '--cache-type-k','q8_0','--cache-type-v','q8_0','-p','The capital of France is',
           '-n','32','--temp','0','-st','--no-warmup','--no-display-prompt')

function Set-Common {
    $env:LLAMA_MOE_CACHE_MIB='2048'; $env:LLAMA_MOE_PREDICT_SMOE='1'; $env:LLAMA_MOE_PREDICT_FATE='0'
    $env:LLAMA_MOE_PREDICT_TOPK='26'; $env:LLAMA_MOE_PREFETCH='1'; $env:LLAMA_MOE_PREFETCH_JOIN='0'
    $env:LLAMA_MOE_SPLIT='1'; $env:LLAMA_MOE_INSERT_ON_MISS='0'; $env:LLAMA_MOE_FALLBACK_PREFETCH='0'
    $env:LLAMA_MOE_DIRECT_READ='1'
    $env:LLAMA_MOE_MRS='1'; $env:LLAMA_MOE_MRS_ALPHA='0.75'; $env:LLAMA_MOE_MRS_TOPP='20'
    $env:LLAMA_MOE_VRAM_LIMIT_MIB='15360'; $env:LLAMA_MOE_VRAM_GUARD_MIB='1024'
    $env:LLAMA_MOE_CACHE_TIMING='1'; $env:LLAMA_TOKEN_PROF='1'
    $env:LLAMA_PLE_CACHE_MIB='2048'; $env:LLAMA_PLE_GPU_CACHE_MIB='1024'
    $env:GGML_OP_OFFLOAD_MIN_BATCH='1'
    Remove-Item Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_FUSION -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_SCHED_NO_SINGLE_COPY_EVENTS -ErrorAction SilentlyContinue
    Remove-Item Env:\LLAMA_MOE_DEVPART -ErrorAction SilentlyContinue
}

Write-Host "=== F1: devpart OFF (correct output) ==="
Set-Common
$env:LLAMA_MOE_CACHE_STATS='stats-I1-ref.csv'
& $exe -m $model @flags > dsh-I1-out.txt 2> dsh-I1-err.txt
Write-Host "F1 exit=$LASTEXITCODE"

Write-Host "=== F2: devpart ON ==="
Set-Common
$env:LLAMA_MOE_DEVPART='1'
$env:LLAMA_MOE_CACHE_STATS='stats-I2-devpart.csv'
& $exe -m $model @flags > dsh-I2-out.txt 2> dsh-I2-err.txt
Write-Host "F2 exit=$LASTEXITCODE"

Write-Host "=== probe 06 done ==="
