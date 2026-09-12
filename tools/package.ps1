param(
    [switch]$Release
)

$ErrorActionPreference = "Stop"
$AppRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DistRoot = Join-Path $AppRoot "dist"
$Target = Join-Path $DistRoot "MAES"

function Assert-PathInside {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $FullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $RootPrefix = $FullRoot + [System.IO.Path]::DirectorySeparatorChar
    if (-not $FullPath.StartsWith($RootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside $FullRoot`: $FullPath"
    }
    return $FullPath
}

function Remove-ItemWithRetry {
    param([Parameter(Mandatory = $true)][string]$Path)

    for ($Attempt = 1; $Attempt -le 5; $Attempt++) {
        try {
            Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            return
        }
        catch {
            if ($Attempt -eq 5) {
                throw
            }
            Start-Sleep -Milliseconds 250
        }
    }
}

function Remove-TreeInside {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $FullPath = Assert-PathInside -Path $Path -Root $Root
    if (Test-Path -LiteralPath $FullPath) {
        $TargetItem = Get-Item -LiteralPath $FullPath -Force
        if (-not $TargetItem.PSIsContainer) {
            Remove-ItemWithRetry -Path $FullPath
            return
        }

        $Files = @(
            Get-ChildItem -LiteralPath $FullPath -Recurse -Force -File -ErrorAction Stop
        )
        foreach ($File in $Files) {
            $FilePath = Assert-PathInside -Path $File.FullName -Root $FullPath
            Remove-ItemWithRetry -Path $FilePath
        }

        $Directories = @(
            Get-ChildItem -LiteralPath $FullPath -Recurse -Force -Directory -ErrorAction Stop |
                Sort-Object { $_.FullName.Length } -Descending
        )
        foreach ($Directory in $Directories) {
            $DirectoryPath = Assert-PathInside -Path $Directory.FullName -Root $FullPath
            Remove-ItemWithRetry -Path $DirectoryPath
        }
        Remove-ItemWithRetry -Path $FullPath
    }
}

function Remove-PythonCaches {
    param([Parameter(Mandatory = $true)][string]$Root)

    $FullRoot = [System.IO.Path]::GetFullPath($Root)
    $CacheDirectories = @(
        Get-ChildItem -LiteralPath $FullRoot -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending
    )
    foreach ($CacheDirectory in $CacheDirectories) {
        $CachePath = Assert-PathInside -Path $CacheDirectory.FullName -Root $FullRoot
        if (Test-Path -LiteralPath $CachePath) {
            Remove-Item -LiteralPath $CachePath -Recurse -Force
        }
    }

    $BytecodeFiles = @(
        Get-ChildItem -LiteralPath $FullRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Extension -in ".pyc", ".pyo" }
    )
    foreach ($BytecodeFile in $BytecodeFiles) {
        $BytecodePath = Assert-PathInside -Path $BytecodeFile.FullName -Root $FullRoot
        Remove-Item -LiteralPath $BytecodePath -Force
    }
}

function Remove-PackageRuntimeState {
    param([Parameter(Mandatory = $true)][string]$Root)

    foreach ($DirectoryName in @("backup", "config", "debug", "logs", "temp")) {
        Remove-TreeInside -Path (Join-Path $Root $DirectoryName) -Root $Root
    }
    $GeneratedSettings = Join-Path $Root "appsettings.json"
    if (Test-Path -LiteralPath $GeneratedSettings) {
        Remove-Item -LiteralPath (Assert-PathInside -Path $GeneratedSettings -Root $Root) -Force
    }
}

Assert-PathInside -Path $Target -Root $DistRoot | Out-Null

$ValidationRoot = Join-Path $AppRoot ".validation"
Remove-TreeInside -Path $ValidationRoot -Root $AppRoot
Remove-PythonCaches -Root $AppRoot

$env:PYTHONDONTWRITEBYTECODE = "1"

function Test-PythonNumpy {
    param([string[]]$Python)

    if (-not $Python -or $Python.Count -eq 0) {
        return $false
    }
    if ($Python.Count -eq 1) {
        & $Python[0] -B -c "import numpy" 2>$null | Out-Null
    }
    else {
        & $Python[0] @($Python[1..($Python.Count - 1)]) -B -c "import numpy" 2>$null | Out-Null
    }
    return ($LASTEXITCODE -eq 0)
}

function Get-TestPythonCommand {
    if ($env:MAES_TEST_PYTHON) {
        $Python = @(($env:MAES_TEST_PYTHON.Trim() -split "\s+") | Where-Object { $_ })
        if (Test-PythonNumpy -Python $Python) {
            return $Python
        }
        throw "MAES_TEST_PYTHON='$($env:MAES_TEST_PYTHON)' cannot import numpy"
    }
    $BundledPython = Join-Path $AppRoot "runtime\python\python.exe"
    if (Test-PythonNumpy -Python @($BundledPython)) {
        return @($BundledPython)
    }
    if (Test-PythonNumpy -Python @("py", "-3.13")) {
        return @("py", "-3.13")
    }
    if (Test-PythonNumpy -Python @("python")) {
        return @("python")
    }
    throw "No Python with NumPy was found. Install NumPy into Python 3.13 or set MAES_TEST_PYTHON."
}

function Invoke-TestPython {
    param([string[]]$Python, [string[]]$Arguments)

    if ($Python.Count -eq 1) {
        & $Python[0] @Arguments
    }
    else {
        & $Python[0] @($Python[1..($Python.Count - 1)]) @Arguments
    }
}

$TestPython = Get-TestPythonCommand
Write-Host "Using test interpreter: $($TestPython -join ' ')"

$CheckArgs = @((Join-Path $PSScriptRoot "check_project.py"), "--require-runtime")
if ($Release) {
    $CheckArgs += "--release"
}
Push-Location $AppRoot
try {
    Invoke-TestPython -Python $TestPython -Arguments (@("-B") + $CheckArgs) | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Project validation failed"
    }

    Invoke-TestPython -Python $TestPython -Arguments @("-B", (Join-Path $PSScriptRoot "run_tests.py")) | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Algorithm and project contract tests failed"
    }
}
finally {
    Pop-Location
}

Remove-TreeInside -Path $Target -Root $DistRoot
New-Item -ItemType Directory -Path $DistRoot -Force | Out-Null
New-Item -ItemType Directory -Path $Target | Out-Null

$GuiRoot = Join-Path $AppRoot "vendor\mfaavalonia"
if (-not (Test-Path -LiteralPath (Join-Path $GuiRoot "MFAAvalonia.exe"))) {
    throw "MFAAvalonia runtime is missing"
}
Copy-Item -Path (Join-Path $GuiRoot "*") -Destination $Target -Recurse -Force

$ProjectItems = @("interface.json", "LICENSE.md", "README.md", "THIRD_PARTY_NOTICES.md", "requirements.lock", "agent", "resource", "LICENSES")
foreach ($Item in $ProjectItems) {
    Copy-Item -LiteralPath (Join-Path $AppRoot $Item) -Destination $Target -Recurse
}

$RuntimeTarget = Join-Path $Target "runtime"
New-Item -ItemType Directory -Path $RuntimeTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $AppRoot "runtime\python") -Destination $RuntimeTarget -Recurse
Copy-Item -LiteralPath (Join-Path $AppRoot "runtime\manifest.json") -Destination $RuntimeTarget

$SitePackages = Join-Path $RuntimeTarget "python\Lib\site-packages"
Get-ChildItem -LiteralPath $SitePackages -Recurse -Force -File -Filter "*.whl" -ErrorAction SilentlyContinue |
    Remove-Item -Force
Remove-TreeInside -Path (Join-Path $SitePackages "bin") -Root $SitePackages

$NativeTarget = Join-Path $Target "runtimes\win-x64\native"
if (-not (Test-Path -LiteralPath $NativeTarget)) {
    throw "MFAAvalonia native runtime directory is missing"
}
$FrameworkRoot = Join-Path $AppRoot "vendor\maaframework"
if (-not (Test-Path -LiteralPath (Join-Path $FrameworkRoot "MaaFramework.dll"))) {
    throw "MaaFramework vendor directory is missing; run tools\bootstrap.ps1 first"
}
Copy-Item -Path (Join-Path $FrameworkRoot "*.dll") -Destination $NativeTarget -Force
New-Item -ItemType Directory -Path (Join-Path $NativeTarget "plugins") -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $FrameworkRoot "LICENSE.md") -Destination (Join-Path $Target "MaaFramework-LICENSE.md")
Copy-Item -LiteralPath (Join-Path $AppRoot "vendor\MFAAvalonia-LICENSE.txt") -Destination (Join-Path $Target "MFAAvalonia-LICENSE.txt")

Remove-PythonCaches -Root $Target
Remove-PackageRuntimeState -Root $Target
Copy-Item -LiteralPath (Join-Path $AppRoot "packaging\appsettings.json") -Destination (Join-Path $Target "appsettings.json")
$ConfigTarget = Join-Path $Target "config"
New-Item -ItemType Directory -Path $ConfigTarget | Out-Null
Copy-Item -LiteralPath (Join-Path $AppRoot "packaging\config.json") -Destination (Join-Path $ConfigTarget "config.json")

$TransientDirectories = @(
    Get-ChildItem -LiteralPath $Target -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in "__pycache__", "backup", "debug", "logs", "temp", "instances" }
)
$TransientBytecode = @(
    Get-ChildItem -LiteralPath $Target -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in ".pyc", ".pyo" }
)
if ($TransientDirectories.Count -gt 0 -or $TransientBytecode.Count -gt 0) {
    throw "Package contains transient runtime files"
}
$UnexpectedConfigFiles = @(
    Get-ChildItem -LiteralPath $ConfigTarget -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne (Join-Path $ConfigTarget "config.json") }
)
if ($UnexpectedConfigFiles.Count -gt 0) {
    throw "Package contains generated user configuration"
}

$SourceFrameworkHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $FrameworkRoot "MaaFramework.dll")).Hash
$PackagedFrameworkHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $NativeTarget "MaaFramework.dll")).Hash
if ($SourceFrameworkHash -ne $PackagedFrameworkHash) {
    throw "Packaged MaaFramework.dll does not match the validated vendor build"
}

Write-Host "Package created at $Target"
