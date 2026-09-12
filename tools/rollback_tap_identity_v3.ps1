[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param()
$ErrorActionPreference = 'Stop'
$branchRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checkpointRoot = Join-Path $branchRoot 'backup/tap-identity-v3'
$manifest = Get-Content -LiteralPath (Join-Path $checkpointRoot 'manifest.json') -Raw | ConvertFrom-Json
$restore = @()
foreach ($entry in $manifest.files) {
    if (-not $entry.baseline_sha256) { continue }
    $target = [IO.Path]::GetFullPath((Join-Path $branchRoot $entry.path))
    $source = [IO.Path]::GetFullPath((Join-Path $checkpointRoot $entry.path))
    if (-not $target.StartsWith($branchRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $source.StartsWith($checkpointRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Checkpoint path escaped the Double branch'
    }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $entry.baseline_sha256) {
        throw "Checkpoint checksum mismatch: $source"
    }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.candidate_sha256) {
        throw "Changed since validation; refusing to overwrite: $target"
    }
    $restore += [pscustomobject]@{ Source = $source; Target = $target }
}
foreach ($item in $restore) {
    if ($PSCmdlet.ShouldProcess($item.Target, 'Restore the four-FC pre-v3 source checkpoint')) {
        Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force
    }
}
Write-Output 'Only modified source/tool files are restored. New unused helpers, tests, recordings, logs and configuration are retained. Restart Double after an actual restore.'
