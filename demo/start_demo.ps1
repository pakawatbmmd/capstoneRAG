# One-click demo: Nova chat UI + hybrid_llm on this laptop's GPU, fully offline, warmed up, browser opened.
#   Double-click demo\start_demo.bat   (or: powershell -ExecutionPolicy Bypass -File demo\start_demo.ps1)
#   Default: also opens a public https://*.trycloudflare.com link to this laptop (Cloudflare quick tunnel).
#   -LocalOnly keeps it on http://127.0.0.1 only.
param(
    [string]$Python = "$env:USERPROFILE\.venvs\capstone-rag\Scripts\python.exe",
    [string]$Cloudflared = "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe",
    [int]$Port = 8793,
    [int]$TimeoutSeconds = 300,
    [switch]$LocalOnly
)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Url = "http://127.0.0.1:$Port"
$WarmQuestion = "How often should an adult at high risk of pressure ulcers be repositioned?"

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "   OK  $text" -ForegroundColor Green }
function Warn($text) { Write-Host "   !!  $text" -ForegroundColor Yellow }
function Fail($text) { Write-Host "   XX  $text" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# Offline: load every model from the local Hugging Face cache, never from the internet.
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

Step "Pre-flight checks"
if (-not (Test-Path $Python)) { Fail "Python environment not found: $Python" }
Ok "Python environment"
if (-not (Test-Path "$Root\store\nice\chunks.jsonl")) { Fail "Guideline store missing: run python examples\ingest.py --name nice Guidelines\*.pdf" }
Ok "Guideline store (store\nice)"
$gpu = & $Python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
if (-not $gpu) { Fail "No CUDA GPU visible to PyTorch: the LLM pipeline needs it" }
Ok "GPU: $gpu"
$models = & $Python -c @"
from huggingface_hub import try_to_load_from_cache
names = ['BAAI/bge-m3', 'BAAI/bge-reranker-v2-m3', 'Qwen/Qwen3-4B-Instruct-2507', 'cross-encoder/nli-deberta-v3-base']
print(','.join(n for n in names if not isinstance(try_to_load_from_cache(n, 'config.json'), str)))
"@
if ($models) { Fail "Models not in the local cache (needs internet once): $models" }
Ok "All 4 models cached locally (offline mode)"
$battery = Get-CimInstance Win32_Battery -ErrorAction SilentlyContinue
if ($battery -and $battery.BatteryStatus -ne 2) { Warn "Laptop is on battery: plug in the charger, or the GPU will throttle" } else { Ok "On mains power" }

Step "Starting the app"
$running = $false
try { Invoke-RestMethod "$Url/api/systems" -TimeoutSec 2 | Out-Null; $running = $true } catch { }
if ($running) {
    Ok "Already running at $Url"
} else {
    # Normal window + High priority: Windows throttles minimized background processes, and token-by-token
    # generation is CPU-bound, so a minimized server answered 1.9x slower (52.7 s vs 28.4 s, same question).
    $server = Start-Process -FilePath $Python -ArgumentList "-m", "web.app", "--port", $Port -WorkingDirectory $Root -WindowStyle Normal -PassThru
    Start-Sleep -Seconds 2
    try { $server.PriorityClass = "High" } catch { Warn "Could not raise the server priority" }
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ($true) {
        try { Invoke-RestMethod "$Url/api/systems" -TimeoutSec 2 | Out-Null; break } catch { }
        if ((Get-Date) -gt $deadline) { Fail "App did not start within $TimeoutSeconds s (see the minimized server window)" }
        Write-Host "   ... loading models" -NoNewline; Start-Sleep -Seconds 5; Write-Host "`r" -NoNewline
    }
    Ok "Models loaded"
}
$systems = Invoke-RestMethod "$Url/api/systems"  # Windows PowerShell returns the JSON array unenumerated
$default = ($systems | ForEach-Object { $_ } | Where-Object { $_.default -eq $true } | Select-Object -First 1).name
if (-not $default) { Fail "The app reported no default system" }
Ok "Default system: $default"

Step "Warm-up: answer the UI's example questions once (cached, so clicking an example is instant)"
$html = Invoke-WebRequest "$Url/" -UseBasicParsing
$examples = [regex]::Matches($html.Content, 'data-question="([^"]+)"') | ForEach-Object { [Net.WebUtility]::HtmlDecode($_.Groups[1].Value) }
foreach ($q in @($WarmQuestion) + $examples | Select-Object -Unique) {
    $body = @{ question = $q; system = $default } | ConvertTo-Json
    $answer = Invoke-RestMethod "$Url/api/ask" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 300
    $state = if ($answer.abstained) { "refused ($($answer.blocked_by))" } else { "answered" }
    Ok ("{0,6:N1} s  {1}: {2}" -f $answer.seconds, $state, $q)
}

if ($LocalOnly) {
    Step "Ready (local only)"
    Start-Process $Url
    Write-Host "   Demo running at $Url. Close the server window to stop it.`n" -ForegroundColor Green
    exit 0
}

Step "Public link (Cloudflare quick tunnel)"
if (-not (Test-Path $Cloudflared)) { Fail "cloudflared not found: winget install Cloudflare.cloudflared" }
$log = Join-Path $env:TEMP "capstone-rag-tunnel.log"
Remove-Item $log -ErrorAction SilentlyContinue
Get-Process cloudflared -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Process -FilePath $Cloudflared -ArgumentList "tunnel", "--url", $Url, "--no-autoupdate", "--logfile", $log -WindowStyle Normal
$public = $null
$deadline = (Get-Date).AddSeconds(90)
while (-not $public) {
    if (Test-Path $log) { $public = ([regex]::Match((Get-Content $log -Raw), 'https://[a-z0-9-]+\.trycloudflare\.com')).Value }
    if ((Get-Date) -gt $deadline) { Fail "No tunnel link after 90 s (check internet); the local demo still works at $Url" }
    if (-not $public) { Start-Sleep -Seconds 2 }
}
$reachable = $false
$deadline = (Get-Date).AddSeconds(90)
while (-not $reachable -and (Get-Date) -lt $deadline) {
    try { Invoke-RestMethod "$public/api/systems" -TimeoutSec 10 | Out-Null; $reachable = $true } catch { Start-Sleep -Seconds 3 }
}
if ($reachable) { Ok "Public link answers through the tunnel" } else { Warn "Link created but not reachable yet; give it a minute" }
try { Set-Clipboard -Value $public } catch { }

Step "Ready"
Start-Process $public
Write-Host ""
Write-Host "   SHARE THIS LINK:  $public" -ForegroundColor Black -BackgroundColor Green
Write-Host "   (copied to clipboard; works while this laptop runs the server and tunnel windows)" -ForegroundColor Green
Write-Host "   Local: $Url. To stop: close the server window and the cloudflared window.`n" -ForegroundColor Green
