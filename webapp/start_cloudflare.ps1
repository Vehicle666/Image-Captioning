$ErrorActionPreference = "Continue"
$webapp = $PSScriptRoot
$py     = Join-Path (Split-Path $webapp -Parent) ".venv\Scripts\python.exe"
$cf     = "C:\Program Files (x86)\cloudflared\cloudflared.exe"
$localUrl = "http://127.0.0.1:5000"
$cfOut = Join-Path $webapp "cf_out.log"
$cfErr = Join-Path $webapp "cf_err.log"

param(
    [switch]$NoBrowser
)

Write-Host "[1/4] Stopping old instances..." -ForegroundColor Yellow
Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "[2/4] Starting Flask app..." -ForegroundColor Yellow
$env:STUDY_TAG = "mobilenet_v3_clip"
$env:USE_V3 = "1"
$env:USE_CLIP = "1"
Start-Process -FilePath $py -ArgumentList "app.py" -WorkingDirectory $webapp -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $webapp "stdout.log") `
    -RedirectStandardError  (Join-Path $webapp "stderr.log")

Write-Host "[3/4] Starting Cloudflare Tunnel..." -ForegroundColor Yellow
Start-Process -FilePath $cf -ArgumentList "tunnel --url $localUrl --no-autoupdate" -WindowStyle Hidden `
    -RedirectStandardOutput $cfOut -RedirectStandardError $cfErr

Write-Host "[4/4] Waiting for tunnel URL..." -ForegroundColor Yellow
$url = ""
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $log = Get-Content $cfErr -Raw -ErrorAction Stop
        $m = [regex]::Match($log, "https://[a-z0-9-]+\.trycloudflare\.com")
        if ($m.Success) { $url = $m.Value; break }
    } catch {}
}

if ($url) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  [OK] WEB IS RUNNING"  -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  Local : $localUrl"
    Write-Host "  Public: $url"
    Write-Host "========================================" -ForegroundColor Green
    Set-Content -Path (Join-Path $webapp "public_url.txt") -Value $url
    if (-not $NoBrowser) { Start-Process $localUrl }
} else {
    Write-Host "[FAIL] Could not get tunnel URL. Last lines of cf_err.log:" -ForegroundColor Red
    Get-Content $cfErr -Tail 15 -ErrorAction SilentlyContinue
}