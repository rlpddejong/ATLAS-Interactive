# One-command setup for ATLAS-Interactive.
# Installs a self-contained Python environment (no conda, no pre-installed Python required)
# and adds a desktop shortcut that opens the labeling tool with no console window.

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot
Set-Location $repoRoot

Write-Host '=== ATLAS-Interactive setup ===' -ForegroundColor Cyan

# 1. Ensure uv is available. uv is a small, self-contained Python package/project
#    manager: it can download and manage its own Python versions, so nothing needs
#    to be pre-installed on the machine.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing uv (Python environment manager)...'
    powershell -NoProfile -ExecutionPolicy Bypass -Command 'irm https://astral.sh/uv/install.ps1 | iex'
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# 2. Create a virtual environment pinned to a supported Python version.
Write-Host 'Creating virtual environment (Python 3.10)...'
uv venv --python 3.10 .venv

$venvPython = Join-Path $repoRoot '.venv\Scripts\python.exe'

# 3. Install PyTorch with CUDA support first (the plain PyPI build is CPU-only on
#    Windows). If this fails because it doesn't match your GPU driver, try cu118
#    or cu124 instead of cu121 below.
Write-Host 'Installing PyTorch with CUDA support...'
uv pip install --python $venvPython torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install ATLAS-Interactive and the rest of its dependencies.
Write-Host 'Installing ATLAS-Interactive...'
uv pip install --python $venvPython -e $repoRoot

# 5. Add a desktop shortcut that launches the GUI without a console window.
Write-Host 'Creating desktop shortcut...'
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop 'ATLAS-Interactive.lnk'
$pythonw = Join-Path $repoRoot '.venv\Scripts\pythonw.exe'

$wshell = New-Object -ComObject WScript.Shell
$shortcut = $wshell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '"' + (Join-Path $repoRoot 'gui.py') + '"'
$shortcut.WorkingDirectory = $repoRoot
$shortcut.Save()

Write-Host ''
Write-Host 'Done! A shortcut named "ATLAS-Interactive" was added to your desktop.' -ForegroundColor Green
Write-Host 'Double-click it to open the labeling tool.' -ForegroundColor Green
