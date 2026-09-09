@echo off
REM Double-click this file to install ATLAS-Interactive.
REM Sets up its own Python environment and adds a desktop shortcut - no
REM Python or conda needs to be installed beforehand.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

echo.
pause
