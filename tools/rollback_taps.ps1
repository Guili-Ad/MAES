[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param()
$ErrorActionPreference = 'Stop'
$branchRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checkpointRoot = Join-Path $branchRoot 'backup/tap-chord-v1'
$baseline = Get-Content -LiteralPath (Join-Path $checkpointRoot 'manifest.json') -Raw | ConvertFrom-Json
$candidate = Get-Content -LiteralPath (Join-Path $checkpointRoot 'candidate-manifest.json') -Raw | ConvertFrom-Json
$restore = @()
foreach ($entry in $candidate.files) {
    if ($entry.kind -ne 'modified') { continue }
    $original = $baseline.baseline | Where-Object { $_.path -eq $entry.path }
    if (-not $original) { throw "No baseline checksum for $($entry.path)" }
    $target = [IO.Path]::GetFullPath((Join-Path $branchRoot $entry.path))
    $source = [IO.Path]::GetFullPath((Join-Path (Join-Path $checkpointRoot 'baseline') $entry.path))
    if (-not $target.StartsWith($branchRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $source.StartsWith($checkpointRoot + '\baseline\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Checkpoint path escaped the Double branch'
    }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $original.sha256) { throw "Baseline changed: $source" }
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $entry.sha256) { throw "File changed after candidate validation; refusing to overwrite: $target" }
    $restore += [pscustomobject]@{ Source = $source; Target = $target }
}
foreach ($item in $restore) {
    if ($PSCmdlet.ShouldProcess($item.Target, 'Restore the pre-change tap/chord checkpoint')) {
        Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force
    }
}
Write-Output 'New helper modules, validation files, logs and user configuration are retained. Restart Double after an actual restore.'
