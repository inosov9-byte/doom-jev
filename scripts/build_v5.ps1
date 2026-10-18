# Relabel every perception-v2 state collected so far with the current expert (26 commands) into
#   data\rfdt-v5-base.jsonl     expert / lagged-expert games and teleport situations, E1M1 + E1M2
#   data\rfdt-v5-student.jsonl  states the student reached in its own games (DAgger); weight these up
# Run from this checkout so expert.py is the version under test; -Data points at the folder with the states.
param([string]$Data = "$HOME\repos\jev-doom", [string]$Root = "C:\tmp\jev-doom")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONDONTWRITEBYTECODE = "1"; $env:TEMP = "$Root\cache\tmp"; $env:TMP = $env:TEMP
$py = "$Root\venv\Scripts\python.exe"
function decisions($dirs) { $dirs | ForEach-Object { Get-ChildItem "$Data\$_" -Recurse -Filter decisions.jsonl } | ForEach-Object FullName }

$base = @("$Data\data\situations-v4.jsonl", "$Data\data\situations-v4-aligned.jsonl", "$Data\data\situations-v4-tail.jsonl") +
        (decisions @("data\expert-traj-v4", "data\lag-traj-v4", "data\expert-traj-e1m2", "data\lag-traj-e1m2"))
$student = decisions @("data\dagger-v4a", "runs\compare-pc-v4", "runs\compare-pc-v4-reliability", "runs\eval-d1-best",
                       "runs\eval-d1-e1m2-transfer", "runs\eval-joint-a-e1m1", "runs\eval-joint-a-e1m2")
& $py -m jevdoom.dataset --expert @base --dedupe --out "$Data\data\rfdt-v5-base.jsonl"
& $py -m jevdoom.dataset --expert @student --dedupe --out "$Data\data\rfdt-v5-student.jsonl"
