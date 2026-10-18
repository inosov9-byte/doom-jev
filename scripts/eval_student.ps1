# Serve a trained student with the fast server, run jev-local live on a seed range, stop the server.
#   .\scripts\eval_student.ps1 -Adapter C:\tmp\jev-doom\models\v4-4b-d1\best -Out runs\eval-d1-best
#   .\scripts\eval_student.ps1 -Adapter ... -Map E1M2 -First 1 -Last 10 -Video
#   .\scripts\eval_student.ps1 -Adapter ... -Map E1M8 -First 1 -Last 10 -Timeout 240 -Loadout nominal
param(
    [Parameter(Mandatory)][string]$Adapter,
    [Parameter(Mandatory)][string]$Out,
    [string]$Model = "Qwen/Qwen3.5-4B",
    [string]$Map = "E1M1",
    [int]$First = 26,
    [int]$Last = 45,
    [int]$Timeout = 120,
    [switch]$Video,
    [string]$Loadout = "",
    [string]$Root = "C:\tmp\jev-doom"
)
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
$env:Path = "$Root\bin;$env:Path"
$env:PYTHONDONTWRITEBYTECODE = "1"; $env:TEMP = "$Root\cache\tmp"; $env:TMP = $env:TEMP
$own = Get-NetTCPConnection -LocalPort 8012 -State Listen -ErrorAction SilentlyContinue
if ($own) { throw "port 8012 is in use (pid $($own.OwningProcess)); stop that server first" }
$log = "$Root\logs\serve-eval.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
$srv = Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$repo\scripts\serve_fast.ps1`" -Model $Model -Adapter $Adapter" `
    -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err"
try {
    $ready = $false
    for ($i = 0; $i -lt 300; $i++) {
        try { Invoke-WebRequest http://127.0.0.1:8012/health -UseBasicParsing -TimeoutSec 2 | Out-Null; $ready = $true; break }
        catch { if ($srv.HasExited) { break }; Start-Sleep 1 }
    }
    if (-not $ready) { throw "server did not start; see $log.err" }
    $a = @("-m", "jevdoom.compare", "--arms", "jev-local", "--map", $Map, "--seeds") + ($First..$Last) + @("--timeout", $Timeout, "--out", $Out)
    if (-not $Video) { $a += "--no-video" }
    if ($Loadout) { $a += @("--loadout", $Loadout) }
    & "$Root\venv\Scripts\python.exe" @a
} finally {
    $own = Get-NetTCPConnection -LocalPort 8012 -State Listen -ErrorAction SilentlyContinue
    if ($own) { Stop-Process -Id $own.OwningProcess -Force -Confirm:$false }
    if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -Confirm:$false }
}
