@echo off
echo ========================================
echo   Image Captioning - Start (background)
echo ========================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1"
pause
