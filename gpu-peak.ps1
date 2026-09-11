$m = 0; $s = 0
for ($i = 0; $i -lt 30; $i++) {
    $c = Get-Counter '\GPU Adapter Memory(luid_0x00000000_0x0001651f_phys_0)\Dedicated Usage','\GPU Adapter Memory(luid_0x00000000_0x0001651f_phys_0)\Shared Usage' -ErrorAction SilentlyContinue
    if ($c) {
        $d = ($c.CounterSamples[0].CookedValue)/1MB
        $h = ($c.CounterSamples[1].CookedValue)/1MB
        if ($d -gt $m) { $m = $d }
        if ($h -gt $s) { $s = $h }
    }
    Start-Sleep -Milliseconds 400
}
Write-Output ('peak_dedicated_MiB=' + [int]$m)
Write-Output ('peak_shared_MiB=' + [int]$s)
