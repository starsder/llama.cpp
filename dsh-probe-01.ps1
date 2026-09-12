# Probe 01: separate the three variables on the CURRENT build (keep-alive views in tree).
# Writes only new files, never overwrites the historical *-out.txt evidence.
#
#   A: devpart ON  + PREFETCH ON   (the config the handoff calls "correct but 8.4")
#   B: devpart ON  + PREFETCH OFF  (the 13:31 config that was correct at 17.5)
#   C: devpart OFF                 (host path control, CUDA graphs left ON)
$model = 'F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf'
$exe   = 'build-ple-trace-mrs\bin\llama-cli.exe'
$flags = @('-ngl','49','--cpu-moe','--lazy-mode','off','--no-mmap','-c','8192',
           '--cache-type-k','q8_0','--cache-type-v','q8_0','-p','The capital of France is',
           '-n','32','--temp','0','-st','--no-warmup','--no-display-prompt')

function Set-Common {
    $env:LLAMA_MOE_CACHE_MIB='2048'; $env:LLAMA_MOE_PREDICT_SMOE='1'; $env:LLAMA_MOE_PREDICT_FATE='0'
    $env:LLAMA_MOE_PREDICT_TOPK='26'; $env:LLAMA_MOE_PREFETCH_JOIN='0'
    $env:LLAMA_MOE_SPLIT='1'; $env:LLAMA_MOE_INSERT_ON_MISS='0'; $env:LLAMA_MOE_FALLBACK_PREFETCH='0'
    $env:LLAMA_MOE_DIRECT_READ='1'
    $env:LLAMA_MOE_MRS='1'; $env:LLAMA_MOE_MRS_ALPHA='0.75'; $env:LLAMA_MOE_MRS_TOPP='20'
    $env:LLAMA_MOE_VRAM_LIMIT_MIB='15360'; $env:LLAMA_MOE_VRAM_GUARD_MIB='1024'
    $env:LLAMA_MOE_CACHE_TIMING='1'; $env:LLAMA_TOKEN_PROF='1'
    $env:LLAMA_PLE_CACHE_MIB='2048'; $env:LLAMA_PLE_GPU_CACHE_MIB='1024'
    $env:GGML_OP_OFFLOAD_MIN_BATCH='1'
    Remove-Item Env:\GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_FUSION -ErrorAction SilentlyContinue
}

Write-Host "=== A: devpart ON, PREFETCH ON ==="
Set-Common
$env:LLAMA_MOE_DEVPART='1'; $env:LLAMA_MOE_PREFETCH='1'
$env:LLAMA_MOE_CACHE_STATS='stats-A-devpart-pf.csv'
& $exe -m $model @flags > dsh-A-out.txt 2> dsh-A-err.txt

Write-Host "=== B: devpart ON, PREFETCH OFF ==="
Set-Common
$env:LLAMA_MOE_DEVPART='1'; $env:LLAMA_MOE_PREFETCH='0'
$env:LLAMA_MOE_CACHE_STATS='stats-B-devpart-nopf.csv'
& $exe -m $model @flags > dsh-B-out.txt 2> dsh-B-err.txt

Write-Host "=== C: devpart OFF (host path control) ==="
Set-Common
Remove-Item Env:\LLAMA_MOE_DEVPART -ErrorAction SilentlyContinue
$env:LLAMA_MOE_PREFETCH='1'
$env:LLAMA_MOE_CACHE_STATS='stats-C-ref.csv'
& $exe -m $model @flags > dsh-C-out.txt 2> dsh-C-err.txt

Write-Host "=== probe 01 done ==="
