@echo off
echo ========================================
echo   Image Captioning - Public Access
echo ========================================
echo.

REM Start ngrok tunnel
echo [1/2] Starting ngrok tunnel...
start "ngrok" "E:\tools\ngrok\ngrok.exe" http 5000 --domain snowfall-idealize-uncombed.ngrok-free.dev

REM Wait for ngrok to initialize
timeout /t 3 /nobreak >nul

REM Start Flask app
echo [2/2] Starting Flask app...
echo.
python app.py

pause
