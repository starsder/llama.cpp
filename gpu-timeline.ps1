for ($i = 0; $i -lt 35; $i++) {
    $c = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
    if ($c) {
        $line = ''
        foreach ($s in $c.CounterSamples) {
            $gpu = $s.Path -replace '.*\(luid_0x[0-9a-f]+_0x0001([0-9a-f]+)_phys_0\).*', '$1'
            $val = [int]($s.CookedValue/1MB)
            $line += "$gpu" + ':' + ($s.Path -replace '.*\\','') + '=' + $val + 'MB  '
        }
        Write-Output $line
    }
    Start-Sleep -Milliseconds 300
}
