[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'High')]
param()
$ErrorActionPreference = 'Stop'
$branchRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$checkpointRoot = Join-Path $branchRoot 'backup/tap-identity-v4'
$manifest = Get-Content -LiteralPath (Join-Path $checkpointRoot 'manifest.json') -Raw | ConvertFrom-Json
function Get-CheckpointHash([string] $LiteralPath) {
    # Read-only .NET hashing also works under Windows PowerShell 5 -WhatIf,
    # where inherited WhatIfPreference can suppress Get-FileHash internals.
    $stream = [IO.File]::OpenRead($LiteralPath)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '') }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
$restore = @()
foreach ($entry in $manifest.files) {
    if (-not $entry.baseline_sha256) { continue }
    $target = [IO.Path]::GetFullPath((Join-Path $branchRoot $entry.path))
    $source = [IO.Path]::GetFullPath((Join-Path $checkpointRoot $entry.path))
    if (-not $target.StartsWith($branchRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $source.StartsWith($checkpointRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Checkpoint path escaped the Double branch'
    }
    if ((Get-CheckpointHash $source) -ne $entry.baseline_sha256) {
        throw "Checkpoint checksum mismatch: $source"
    }
    if ((Get-CheckpointHash $target) -ne $entry.candidate_sha256) {
        throw "Changed since validation; refusing to overwrite: $target"
    }
    $restore += [pscustomobject]@{ Source = $source; Target = $target }
}
foreach ($item in $restore) {
    if ($PSCmdlet.ShouldProcess($item.Target, 'Restore pre-v4 tap-identity-v3 candidate')) {
        Copy-Item -LiteralPath $item.Source -Destination $item.Target -Force
    }
}
Write-Output 'Only validated modified files are restored. New unused modules/tests, recordings, logs and configuration remain. Restart Double after an actual restore.'
