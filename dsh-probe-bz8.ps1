# Reproduce the bz8 config (which once showed pred 99.6% / hit 98.1% / ready 100%)
# on the CURRENT binary, to decide: config difference vs source regression.
$model = 'F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf'
$exe   = 'build-ple-trace-mrs\bin\llama-cli.exe'

# exact flags from bisect3.ps1 (-n 8, --lazy-mode on)
$flags = @('-ngl','49','--cpu-moe','--lazy-mode','on','-c','8192',
           '--cache-type-k','q8_0','--cache-type-v','q8_0','-p','The capital of France is',
           '-n','8','--temp','0','-st','--no-warmup','--no-display-prompt')

Get-ChildItem Env: | Where-Object { $_.Name -like 'LLAMA_MOE*' -or $_.Name -like 'LLAMA_PLE*' -or $_.Name -like 'LLAMA_TOKEN*' } |
    ForEach-Object { Remove-Item "Env:\$($_.Name)" -ErrorAction SilentlyContinue }

# ---- bisect3 Set-Common ----
$env:LLAMA_MOE_CACHE_MIB='2048'; $env:LLAMA_MOE_PREDICT_SMOE='1'; $env:LLAMA_MOE_PREDICT_FATE='0'
$env:LLAMA_MOE_PREDICT_TOPK='26'; $env:LLAMA_MOE_PREFETCH='1'; $env:LLAMA_MOE_SPLIT='1'
$env:LLAMA_MOE_MRS='1'; $env:LLAMA_MOE_MRS_ALPHA='0.75'; $env:LLAMA_MOE_MRS_TOPP='20'
$env:LLAMA_MOE_VRAM_LIMIT_MIB='15360'; $env:LLAMA_MOE_VRAM_GUARD_MIB='1024'
$env:GGML_OP_OFFLOAD_MIN_BATCH='1'
$env:LLAMA_PLE_CACHE_MIB='0'; $env:LLAMA_PLE_GPU_CACHE_MIB='0'
# ---- R8 overrides ----
$env:LLAMA_MOE_PREFETCH_JOIN='1'; $env:LLAMA_MOE_INSERT_ON_MISS='1'; $env:LLAMA_MOE_DIRECT_READ='1'
$env:LLAMA_MOE_CACHE_STATS='stats-bz8-repro.csv'

& $exe -m $model @flags > bz8-repro-out.txt 2> bz8-repro-err.txt
Write-Host "bz8-repro exit=$LASTEXITCODE"
