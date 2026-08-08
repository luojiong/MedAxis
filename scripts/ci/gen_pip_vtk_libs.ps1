# MedAxis CI helper — generate import libraries from the pip VTK wheel DLLs.
#
# The compiled medaxis_bridge links against VTK; linking the *pip* wheel's
# DLLs (vtkCommonCore-9.6.2.dll, ...) instead of the conda ones guarantees a
# single VTK runtime in the packaged app (conda names them -9.6, pip names
# them -9.6.2 — both loaded together would split VTK's static state and crash).
#
# Usage: powershell -File scripts/ci/gen_pip_vtk_libs.ps1
#        -SitePackages <path to .venv/Lib/site-packages> -OutDir <output dir>
param(
    [Parameter(Mandatory = $true)][string]$SitePackages,
    [Parameter(Mandatory = $true)][string]$OutDir
)

$ErrorActionPreference = "Stop"

# Locate dumpbin/lib from the Visual Studio installation.
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -latest -property installationPath
$vcTools = Get-ChildItem (Join-Path $vsPath "VC\Tools\MSVC") -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
$binDir = Join-Path $vcTools.FullName "bin\Hostx64\x64"
$dumpbin = Join-Path $binDir "dumpbin.exe"
$libexe = Join-Path $binDir "lib.exe"
if (-not (Test-Path $dumpbin)) { throw "dumpbin not found under $binDir" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$dlls = @("vtkCommonCore-9.6.2", "vtkCommonDataModel-9.6.2",
          "vtkCommonMath-9.6.2", "vtksys-9.6.2")

foreach ($dll in $dlls) {
    $dllPath = Join-Path $SitePackages "vtk.libs\$dll.dll"
    if (-not (Test-Path $dllPath)) { throw "missing $dllPath" }

    $def = Join-Path $OutDir "$dll.def"
    $lib = Join-Path $OutDir "$dll.lib"
    if (Test-Path $lib) { continue }

    # Export list -> .def (LIBRARY/EXPORTS + names).
    $exports = & $dumpbin /exports $dllPath
    $lines = $exports | Where-Object { $_ -match "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)" }
    $names = $lines | ForEach-Object { "    " + ($_ -replace '^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+).*$', '$1') }
    Set-Content -Path $def -Value "LIBRARY $dll.dll`nEXPORTS`n$($names -join "`n")"

    & $libexe /def:$def /out:$lib /machine:x64 /nologo | Out-Null
    Write-Host "    generated $lib"
}
Write-Host "==> pip vtk import libraries ready in $OutDir"
