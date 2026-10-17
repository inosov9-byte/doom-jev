# Start a local Simple Jev classifier server on the AMD GPU.
#   .\scripts\serve.ps1                                   # zero-shot Qwen3.5-0.8B
#   .\scripts\serve.ps1 -Model C:\tmp\jev-doom\models\jev-doom-v1   # fine-tuned student
# First request after start is slow (~20 s): GPU kernels compile once.
# RFDT trains with the baseline prompt policy, so the server must use it too.
param(
    [string]$Model = "Qwen/Qwen3.5-0.8B",
    [int]$Port = 8012,
    [string]$Device = "cuda",
    [string]$DType = "bfloat16",
    [string]$Root = "C:\tmp\jev-doom"
)
$env:HF_HOME = "$Root\cache\hf"
$env:TEMP = "$Root\cache\tmp"; $env:TMP = "$Root\cache\tmp"
$env:PYTHONDONTWRITEBYTECODE = "1"
& "$Root\venv-rocm\Scripts\python.exe" -m hf_server --model $Model --device $Device --dtype $DType --port $Port --served-model-name student --classifier-prompt-policy baseline
