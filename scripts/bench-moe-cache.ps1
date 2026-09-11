[CmdletBinding()]
param(
    [string] $BuildDir = "build-ple-trace-mrs",
    [string] $Model = "F:\models\qwen38\unsloth-iq3-xxs\UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf",
    [ValidateSet("LRU", "MRS", "FIFO")] [string] $Policy = "MRS",
    [ValidateSet("None", "Fate", "SMoE", "CrossLayer", "CrossToken")] [string] $PredictMode = "Fate",
    [int] $PredictTopK = 26,
    [double] $PredictMRSWeight = 0.0,
    [int] $WindowLayers = 0,
    [int] $FallbackPrefetch = 0,
    [int] $CacheMiB = 512,
    [int] $Tokens = 32,
    [int] $Context = 8192,
    [switch] $Prefetch,
    [switch] $GlobalPool,
    [switch] $SplitMoE,
    [switch] $DirectRead,
    [switch] $Timing,
    [string] $Prompt = "The capital of France is"
)

$ErrorActionPreference = "Stop"
$exe = (Resolve-Path (Join-Path $BuildDir "bin\llama-cli.exe")).Path

$env:LLAMA_MOE_CACHE_MIB = [string] $CacheMiB
$env:LLAMA_MOE_PREDICT = if ($PredictMode -eq "CrossLayer") {
    "F:\models\qwen38\traces\moe-predict-v1.bin"
} elseif ($PredictMode -eq "CrossToken") {
    "F:\models\qwen38\traces\moe-predict-xt.bin"
} else {
    $null
}
$env:LLAMA_MOE_PREDICT_FATE = if ($PredictMode -eq "Fate") { "1" } else { "0" }
$env:LLAMA_MOE_PREDICT_SMOE = if ($PredictMode -eq "SMoE") { "1" } else { "0" }
$env:LLAMA_MOE_PREDICT_TOPK = [string] $PredictTopK
$env:LLAMA_MOE_PREDICT_MRS_WEIGHT = [string] $PredictMRSWeight
$env:LLAMA_MOE_PREDICT_STATIC = "0"
$env:LLAMA_MOE_PIN_STATIC = "0"
$env:LLAMA_MOE_PREFETCH = if ($Prefetch) { "1" } else { "0" }
$env:LLAMA_MOE_SPLIT = if ($SplitMoE) { "1" } else { "0" }
$env:LLAMA_MOE_PREDICT_XT = if ($PredictMode -eq "CrossToken") { "1" } else { "0" }
$env:LLAMA_MOE_WINDOW_LAYERS = [string] $WindowLayers
$env:LLAMA_MOE_GLOBAL_POOL = if ($GlobalPool) { "1" } else { "0" }
$env:LLAMA_MOE_INSERT_ON_MISS = "1"
$env:LLAMA_MOE_FALLBACK_PREFETCH = [string] $FallbackPrefetch
$env:LLAMA_MOE_DIRECT_READ = if ($DirectRead) { "1" } else { "0" }
$env:LLAMA_MOE_FIFO = if ($Policy -eq "FIFO") { "1" } else { "0" }
$env:LLAMA_MOE_MRS = if ($Policy -eq "MRS") { "1" } else { "0" }
$env:LLAMA_MOE_MRS_ALPHA = "0.75"
$env:LLAMA_MOE_MRS_TOPP = "20"
$env:LLAMA_MOE_VRAM_LIMIT_MIB = "15360"
$env:LLAMA_MOE_VRAM_GUARD_MIB = "1024"
$env:LLAMA_MOE_CACHE_TIMING = if ($Timing) { "1" } else { "0" }
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"

$arguments = @(
    "-m", $Model,
    "-ngl", "49",
    "--cpu-moe",
    "--lazy-mode", "on",
    "-c", [string] $Context,
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "-p", $Prompt,
    "-n", [string] $Tokens,
    "--temp", "0",
    "-st",
    "--no-warmup",
    "--no-display-prompt"
)

$startInfo = [Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $exe
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
if ($null -ne $startInfo.ArgumentList) {
    foreach ($argument in $arguments) {
        [void] $startInfo.ArgumentList.Add($argument)
    }
} else {
    $startInfo.Arguments = [string]::Join(" ", ($arguments | ForEach-Object { '"' + $_ + '"' }))
}
$process = [Diagnostics.Process]::Start($startInfo)
$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
$peakMiB = 0
$samples = 0
while (-not $process.HasExited) {
    $sample = nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($sample -match "^\s*(\d+)") {
        $used = [int] $Matches[1]
        if ($used -gt $peakMiB) { $peakMiB = $used }
    }
    $samples++
    Start-Sleep -Milliseconds 250
}
$process.WaitForExit()

$stdout = $stdoutTask.Result
$stderr = $stderrTask.Result
$stdoutHash = ([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($stdout)) | ForEach-Object { $_.ToString("x2") }) -join ""
$modelStdout = $stdout -replace "\[ Prompt: [^\r\n]*\r?\n", ""
$modelStdoutHash = ([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($modelStdout)) | ForEach-Object { $_.ToString("x2") }) -join ""
$combined = $stdout + "`n" + $stderr
$perf = [regex]::Match($combined, "\[ Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+) t/s \]")
$summary = [regex]::Match($combined, "\[MOE-CACHE\] policy=(\w+) requested=(\d+) MiB effective=(\d+) MiB hits=(\d+) misses=(\d+) mrs_reads=(\d+) mrs_fallbacks=(\d+) mrs_updates=(\d+) mrs_victims=(\d+) protected_evictions=(\d+) prefetch_requests=(\d+) prefetch_experts=(\d+) prefetch_bytes=(\d+) window_layers=(\d+) window_recycles=(\d+) prefetch_required=(\d+) prefetch_predicted=(\d+) prefetch_ready=(\d+) fallback_prefetch=(\d+) fate=(\d+) fate_predictions=(\d+) fate_gate_inputs=(\d+) fate_gate_ms=([0-9.]+) smoe=(\d+) smoe_predictions=(\d+) smoe_logits=(\d+)")
$grouped = [regex]::Match($combined, "\[MOE-CACHE\] (\d+) weight tensors grouped into (\d+) (?:logical |persistent )?layer bundles, .*?, (\d+) (?:slots/layer|slots per layer), ([0-9.]+) MiB (?:physical cache|device total)")
$globalGrouped = [regex]::Match($combined, "\[MOE-CACHE\] (\d+) weight tensors grouped into one global layer-expert pool, (\d+) bytes/slot, (\d+) global slots, ([0-9.]+) MiB physical cache")
$timingMatch = [regex]::Match($combined, "\[MOE-CACHE\] timing per decode graph: (.*)")
$cudaError = $combined -match "CUDA error|GGML_ASSERT|Assertion failed|access violation"

[pscustomobject]@{
    policy = $Policy
    predict_mode = $PredictMode
    cache_mib = $CacheMiB
    prefetch = [bool] $Prefetch
    direct_read = [bool] $DirectRead
    global_pool = [bool] $GlobalPool
    split_moe = [bool] $SplitMoE
    timing_enabled = [bool] $Timing
    lazy_mode = $true
    cpu_moe = $true
    exit_code = $process.ExitCode
    stdout_sha256 = $stdoutHash
    model_stdout_sha256 = $modelStdoutHash
    peak_vram_mib = $peakMiB
    sample_count = $samples
    prompt_tps = if ($perf.Success) { [double] $perf.Groups[1].Value } else { $null }
    generation_tps = if ($perf.Success) { [double] $perf.Groups[2].Value } else { $null }
    effective_cache_mib = if ($summary.Success) { [int] $summary.Groups[3].Value } else { $null }
    hits = if ($summary.Success) { [int] $summary.Groups[4].Value } else { $null }
    misses = if ($summary.Success) { [int] $summary.Groups[5].Value } else { $null }
    mrs_reads = if ($summary.Success) { [int] $summary.Groups[6].Value } else { $null }
    mrs_fallbacks = if ($summary.Success) { [int] $summary.Groups[7].Value } else { $null }
    mrs_updates = if ($summary.Success) { [int] $summary.Groups[8].Value } else { $null }
    mrs_victims = if ($summary.Success) { [int] $summary.Groups[9].Value } else { $null }
    protected_evictions = if ($summary.Success) { [int] $summary.Groups[10].Value } else { $null }
    prefetch_requests = if ($summary.Success) { [int] $summary.Groups[11].Value } else { $null }
    prefetch_experts = if ($summary.Success) { [int] $summary.Groups[12].Value } else { $null }
    prefetch_bytes = if ($summary.Success) { [int64] $summary.Groups[13].Value } else { $null }
    window_layers = if ($summary.Success) { [int] $summary.Groups[14].Value } else { $null }
    window_recycles = if ($summary.Success) { [int] $summary.Groups[15].Value } else { $null }
    prefetch_required = if ($summary.Success) { [int] $summary.Groups[16].Value } else { $null }
    prefetch_predicted = if ($summary.Success) { [int] $summary.Groups[17].Value } else { $null }
    prefetch_ready = if ($summary.Success) { [int] $summary.Groups[18].Value } else { $null }
    fallback_prefetch = if ($summary.Success) { [int] $summary.Groups[19].Value } else { $null }
    fate_enabled = if ($summary.Success) { [bool] [int] $summary.Groups[20].Value } else { $null }
    fate_predictions = if ($summary.Success) { [int] $summary.Groups[21].Value } else { $null }
    fate_gate_inputs = if ($summary.Success) { [int] $summary.Groups[22].Value } else { $null }
    fate_gate_ms = if ($summary.Success) { [double] $summary.Groups[23].Value } else { $null }
    smoe_enabled = if ($summary.Success) { [bool] [int] $summary.Groups[24].Value } else { $null }
    smoe_predictions = if ($summary.Success) { [int] $summary.Groups[25].Value } else { $null }
    smoe_logits = if ($summary.Success) { [int] $summary.Groups[26].Value } else { $null }
    layer_bundles = if ($grouped.Success) { [int] $grouped.Groups[2].Value } else { $null }
    slots_per_layer = if ($grouped.Success) { [int] $grouped.Groups[3].Value } else { $null }
    global_slots = if ($globalGrouped.Success) { [int] $globalGrouped.Groups[3].Value } else { $null }
    cache_device_mib = if ($grouped.Success) { [double] $grouped.Groups[4].Value } elseif ($globalGrouped.Success) { [double] $globalGrouped.Groups[4].Value } else { $null }
    timing = if ($timingMatch.Success) { $timingMatch.Groups[1].Value.Trim() } else { $null }
    cuda_error = $cudaError
} | ConvertTo-Json -Compress
