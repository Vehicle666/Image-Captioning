$ErrorActionPreference = "Continue"
$webapp = $PSScriptRoot
$py     = Join-Path (Split-Path $webapp -Parent) ".venv\Scripts\python.exe"
$ngrok  = "E:\tools\ngrok\ngrok.exe"
$domain = "snowfall-idealize-uncombed.ngrok-free.dev"
$localUrl = "http://127.0.0.1:5000"

Write-Host "[1/4] Stopping old instances..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "[2/4] Starting Flask + ngrok in background..." -ForegroundColor Yellow
Start-Process -FilePath $py -ArgumentList "app.py" -WorkingDirectory $webapp -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $webapp "stdout.log") `
    -RedirectStandardError  (Join-Path $webapp "stderr.log")
Start-Process -FilePath $ngrok -ArgumentList "http 5000 --domain $domain" -WindowStyle Hidden

Write-Host "[3/4] Waiting for the web app to be ready (up to 90s)..." -ForegroundColor Yellow
$ok = $false
foreach ($i in 1..45) {
    try {
        $r = Invoke-WebRequest -Uri $localUrl -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch {
        Start-Sleep -Seconds 2
    }
}

if ($ok) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  [OK] WEB IS RUNNING (background mode)"  -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Local : $localUrl"
    Write-Host "  Public: http://$domain"
    Write-Host "  Logs  : stderr.log / stdout.log in this folder"
    Write-Host "========================================" -ForegroundColor Green
    Start-Process $localUrl
} else {
    Write-Host ""
    Write-Host "[FAIL] Web did not respond after 90s." -ForegroundColor Red
    Write-Host "Last lines of stderr.log:" -ForegroundColor Red
    Get-Content (Join-Path $webapp "stderr.log") -Tail 15 -ErrorAction SilentlyContinue
}
