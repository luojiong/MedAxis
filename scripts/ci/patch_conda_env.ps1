# MedAxis CI helper — patch a fresh conda-forge environment for the native
# build (mirrors the manual setup documented in the README).
# Usage: powershell -File scripts/ci/patch_conda_env.ps1 -EnvPath <conda env root>
param(
    [Parameter(Mandatory = $true)][string]$EnvPath
)

$ErrorActionPreference = "Stop"
$lib = Join-Path $EnvPath "Library"

Write-Host "==> Patching conda environment at $EnvPath"

# 1) vtk-base 9.6.2 ships no vtkIOFFMPEG module (conda packaging gap) but the
#    CMake config references it — create placeholders so configure succeeds.
$vtkCfg = Join-Path $lib "lib\cmake\vtk-9.6"
if (Test-Path $vtkCfg) {
    $ioffmpegLib = Join-Path $lib "lib\vtkIOFFMPEG-9.6.lib"
    $ioffmpegDll = Join-Path $lib "bin\vtkIOFFMPEG-9.6.dll"
    $ioffmpegPyd = Join-Path $EnvPath "Lib\site-packages\vtkmodules\vtkIOFFMPEG.cp312-win_amd64.pyd"
    if (-not (Test-Path $ioffmpegLib)) {
        Copy-Item (Join-Path $lib "lib\vtkIOMovie-9.6.lib") $ioffmpegLib
        Copy-Item (Join-Path $lib "bin\vtkIOMovie-9.6.dll") $ioffmpegDll
        New-Item -ItemType Directory -Force -Path (Split-Path $ioffmpegPyd) | Out-Null
        New-Item -ItemType File -Force -Path $ioffmpegPyd | Out-Null
        Write-Host "    patched vtkIOFFMPEG placeholder"
    }
}

# 2) conda-forge eigen installs its CMake config under share/eigen3 (lowercase)
#    which find_package(Eigen3) does not search — mirror it to lib/cmake/Eigen3.
$eigenSrc = Join-Path $lib "share\eigen3"
$eigenDst = Join-Path $lib "lib\cmake\Eigen3"
if ((Test-Path $eigenSrc) -and -not (Test-Path $eigenDst)) {
    Copy-Item -Recurse $eigenSrc $eigenDst
    Write-Host "    mirrored eigen cmake config to lib/cmake/Eigen3"
}

# 3) utf8cpp header (vtk-config dependency, not on conda-forge).
$utf8Header = Join-Path $lib "include\utf8cpp\utf8.h"
if (-not (Test-Path $utf8Header)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $utf8Header) | Out-Null
    Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nemtrif/utfcpp/master/source/utf8.h" `
        -OutFile $utf8Header
    Write-Host "    downloaded utf8cpp header"
}

Write-Host "==> conda environment patched"
