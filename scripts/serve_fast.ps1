# Start the lean local Jev server (compact prompt, one forward pass) for the jev-local arm.
#   .\scripts\serve_fast.ps1                                                  # zero-shot Qwen3.5-4B
#   .\scripts\serve_fast.ps1 -Adapter C:\tmp\jev-doom\models\v4-4b\best       # trained student
# Same /v1/classifier API as scripts\serve.ps1, so play.py and compare.py need no change.
param(
    [string]$Model = "Qwen/Qwen3.5-4B",
    [string]$Adapter = "",
    [int]$Port = 8012,
    [string]$Root = "C:\tmp\jev-doom"
)
$env:HF_HOME = "$Root\cache\hf"
$env:TEMP = "$Root\cache\tmp"; $env:TMP = "$Root\cache\tmp"
$env:PYTHONDONTWRITEBYTECODE = "1"
$a = @("-m", "jevdoom.fast_server", "--model", $Model, "--port", $Port)
if ($Adapter) { $a += @("--adapter", $Adapter) }
Set-Location (Split-Path $PSScriptRoot -Parent)
& "$Root\venv-rocm\Scripts\python.exe" @a
