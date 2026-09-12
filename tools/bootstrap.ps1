param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RuntimeRoot = Join-Path $AppRoot "runtime"
$PythonRoot = Join-Path $RuntimeRoot "python"
$WheelRoot = Join-Path $RuntimeRoot "wheels"
$VendorRoot = Join-Path $AppRoot "vendor"
$FrameworkRoot = Join-Path $VendorRoot "maaframework"
$GuiRoot = Join-Path $VendorRoot "mfaavalonia"

function Get-Sha256Upper {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToUpperInvariant()
}

function Assert-Hash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path)) { throw "$Label is missing: $Path" }
    $actual = Get-Sha256Upper -Path $Path
    if ($actual -ne $Expected.ToUpperInvariant()) { throw "$Label hash mismatch: $Path" }
}

function Expand-ArchiveTo {
    param(
        [Parameter(Mandatory = $true)][string]$Archive,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if ($tar) {
        & $tar.Source -xf $Archive -C $Destination
        if ($LASTEXITCODE -eq 0) { return }
        Write-Warning "tar extraction failed with code $LASTEXITCODE; falling back to Expand-Archive"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}

function Enable-LongPaths {
    try {
        $key = "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem"
        $current = (Get-ItemProperty -Path $key -Name LongPathsEnabled -ErrorAction Stop).LongPathsEnabled
        if ($current -ne 1) {
            Set-ItemProperty -Path $key -Name LongPathsEnabled -Value 1 -ErrorAction Stop
            Write-Host "Enabled Windows long path support"
        }
    } catch {
        Write-Warning "Could not enable Windows long path support: $($_.Exception.Message)"
    }
}

Write-Host "MAES bootstrap"
Enable-LongPaths

$runtimeManifestPath = Join-Path $RuntimeRoot "manifest.json"
if (-not (Test-Path -LiteralPath $runtimeManifestPath)) { throw "runtime/manifest.json is missing" }
$runtimeManifest = Get-Content -LiteralPath $runtimeManifestPath -Raw | ConvertFrom-Json

$pythonArchive = Join-Path $RuntimeRoot "python-3.12.10-embed-amd64.zip"
Assert-Hash -Path $pythonArchive -Expected $runtimeManifest.python.archive_sha256 -Label "Python embeddable archive"

foreach ($package in $runtimeManifest.packages) {
    $wheelPath = Join-Path $WheelRoot $package.file
    Assert-Hash -Path $wheelPath -Expected $package.sha256 -Label "Wheel $($package.name)"
}

$vendorManifestPath = Join-Path $VendorRoot "manifest.json"
if (-not (Test-Path -LiteralPath $vendorManifestPath)) { throw "vendor/manifest.json is missing" }
$vendorManifest = Get-Content -LiteralPath $vendorManifestPath -Raw | ConvertFrom-Json
$guiArchive = Join-Path $VendorRoot $vendorManifest.gui.archive
Assert-Hash -Path $guiArchive -Expected $vendorManifest.gui.archive_sha256 -Label "MFAAvalonia archive"

$frameworkManifestPath = Join-Path $FrameworkRoot "manifest.json"
if (-not (Test-Path -LiteralPath $frameworkManifestPath)) { throw "vendor/maaframework/manifest.json is missing" }
$frameworkManifest = Get-Content -LiteralPath $frameworkManifestPath -Raw | ConvertFrom-Json
$missingFramework = @()
foreach ($record in $frameworkManifest.files) {
    $dllPath = Join-Path $FrameworkRoot $record.file
    if (-not (Test-Path -LiteralPath $dllPath)) {
        $missingFramework += $record.file
        continue
    }
    Assert-Hash -Path $dllPath -Expected $record.sha256 -Label "Framework file $($record.file)"
}
if ($missingFramework.Count -gt 0) {
    Write-Warning "Framework files missing from vendor (trying workspace bin fallback): $($missingFramework -join ', ')"
    $workspaceBin = Join-Path (Split-Path -Parent $AppRoot) "bin"
    foreach ($name in $missingFramework) {
        $source = Join-Path $workspaceBin $name
        if (-not (Test-Path -LiteralPath $source)) { throw "Framework file unavailable: $name" }
        Copy-Item -LiteralPath $source -Destination (Join-Path $FrameworkRoot $name) -Force
    }
}
$pluginsDir = Join-Path $FrameworkRoot "plugins"
if (-not (Test-Path -LiteralPath $pluginsDir)) {
    New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
}

$pythonExe = Join-Path $PythonRoot "python.exe"
if ($Force -or -not (Test-Path -LiteralPath $pythonExe)) {
    if (Test-Path -LiteralPath $PythonRoot) { Remove-Item -LiteralPath $PythonRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $PythonRoot -Force | Out-Null
    Write-Host "Extracting bundled Python runtime"
    Expand-ArchiveTo -Archive $pythonArchive -Destination $PythonRoot
}

$pthPath = Join-Path $PythonRoot "python312._pth"
$pthContent = "python312.zip`r`n.`r`n../..`r`nLib/site-packages`r`n`r`n# Uncomment to run site.main() automatically`r`nimport site`r`n"
Set-Content -LiteralPath $pthPath -Value $pthContent -Encoding ASCII -NoNewline

$sitePackages = Join-Path $PythonRoot "Lib\site-packages"
$packagesPresent = @("numpy", "maa", "MaaAgentBinary", "strenum") | ForEach-Object {
    Test-Path -LiteralPath (Join-Path $sitePackages $_)
} | Where-Object { -not $_ }
if ($Force -or $packagesPresent.Count -gt 0) {
    if (-not (Test-Path -LiteralPath $sitePackages)) {
        New-Item -ItemType Directory -Path $sitePackages -Force | Out-Null
    }
    Write-Host "Extracting Python wheels"
    foreach ($package in $runtimeManifest.packages) {
        $wheelPath = Join-Path $WheelRoot $package.file
        Expand-ArchiveTo -Archive $wheelPath -Destination $sitePackages
    }
}

$guiExe = Join-Path $GuiRoot "MFAAvalonia.exe"
if ($Force -or -not (Test-Path -LiteralPath $guiExe)) {
    if (Test-Path -LiteralPath $GuiRoot) { Remove-Item -LiteralPath $GuiRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $GuiRoot -Force | Out-Null
    Write-Host "Extracting MFAAvalonia host"
    Expand-ArchiveTo -Archive $guiArchive -Destination $GuiRoot
}

Write-Host "Verifying bundled runtime"
$probe = "import importlib.metadata,json,sys,numpy,maa;print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),'numpy':numpy.__version__,'maafw':importlib.metadata.version('maafw')}))"
$output = & $pythonExe -B -c $probe
if ($LASTEXITCODE -ne 0) { throw "Bundled Python smoke test failed" }
Write-Host "Runtime OK: $output"
Write-Host "Bootstrap complete"
