# Rebuild the student training set with the current perception text (v4 = perception v2).
# CPU only; runs the three situation collectors and the expert / lagged-expert games in parallel.
#   situations   teleport anchors + random rollouts (collect.py): general, heading-aligned, route tail
#   expert-traj  instant rules expert, seeds 6-25 (states the expert itself reaches)
#   lag-traj     rules expert at 0.2 s and 0.35 s per decision, seeds 6-15 (states a slow student reaches)
# Labels come from expert.py in dataset.py, so no API calls.
param([string]$Root = "C:\tmp\jev-doom", [string]$Tag = "v4")
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "$Root\venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"; $env:TEMP = "$Root\cache\tmp"; $env:TMP = $env:TEMP
$logs = "$Root\logs\build-$Tag"; New-Item -ItemType Directory -Force $logs | Out-Null
Set-Location $repo

$jobs = @(
    @("situations-$Tag",         "-m jevdoom.collect --out data\situations-$Tag.jsonl --anchors 800 --seed 11 --prefix s"),
    @("situations-$Tag-aligned", "-m jevdoom.collect --out data\situations-$Tag-aligned.jsonl --anchors 250 --seed 12 --prefix b --heading-sd 20 --kinds 1,0,0"),
    @("situations-$Tag-tail",    "-m jevdoom.collect --out data\situations-$Tag-tail.jsonl --anchors 200 --seed 13 --prefix t --route-from 0.6"),
    @("expert-traj-$Tag",        "-m jevdoom.compare --arms rules --seeds $(6..25 -join ' ') --no-video --out data\expert-traj-$Tag"),
    @("lag-traj-$Tag",           "-m jevdoom.compare --arms rules-lag-200 rules-lag-350 --seeds $(6..15 -join ' ') --no-video --out data\lag-traj-$Tag")
)
$procs = foreach ($j in $jobs) {
    Start-Process $py -ArgumentList $j[1] -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput "$logs\$($j[0]).log" -RedirectStandardError "$logs\$($j[0]).err"
}
$procs | Wait-Process
for ($i = 0; $i -lt $jobs.Count; $i++) { Write-Output "$($jobs[$i][0]) exit $($procs[$i].ExitCode)" }
