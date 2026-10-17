# One-time setup. Everything heavy lives outside OneDrive under -Root (default C:\tmp\jev-doom).
#   venv       game side: ViZDoom, numpy, OpenCV          (Python 3.13)
#   venv-rocm  Jev side: Simple Jev server + RFDT on AMD ROCm PyTorch (Python 3.12)
#   wad        shareware DOOM1.WAD (MD5 checked)
#   cache      uv + Hugging Face caches
param(
    [string]$Root = "C:\tmp\jev-doom",
    [string]$SimpleJevCommit = "31c8d7a",
    [switch]$CpuTorch    # use CPU PyTorch instead of ROCm (no AMD GPU)
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force "$Root\wad", "$Root\cache\uv", "$Root\cache\hf", "$Root\cache\tmp" | Out-Null
$env:UV_CACHE_DIR = "$Root\cache\uv"

# Game venv
if (-not (Test-Path "$Root\venv")) { uv venv -p 3.13 "$Root\venv" -q }
uv pip install -p "$Root\venv\Scripts\python.exe" -q vizdoom numpy opencv-python

# Shareware WAD, verified against the id Software v1.9 shareware MD5
$wad = "$Root\wad\doom1.wad"
if (-not (Test-Path $wad)) {
    Invoke-WebRequest "https://github.com/nneonneo/universal-doom/raw/main/DOOM1.WAD" -OutFile $wad
}
$md5 = (Get-FileHash $wad -Algorithm MD5).Hash.ToLower()
if ($md5 -ne "f0cefca49926d00903cf57551d901abe") { throw "DOOM1.WAD checksum mismatch: $md5" }

# Simple Jev at a pinned commit
$src = "$Root\src-simple-jev"
if (-not (Test-Path $src)) { git clone -q https://github.com/featherless-ai/simple-jev.git $src }
git -C $src fetch -q --depth 50 origin
git -C $src checkout -q $SimpleJevCommit

# Jev venv: ROCm PyTorch for Radeon 890M (gfx1150) on Windows, AMD ROCm 7.2.1 wheels
if (-not (Test-Path "$Root\venv-rocm")) { uv venv -p 3.12 "$Root\venv-rocm" -q }
$py = "$Root\venv-rocm\Scripts\python.exe"
if ($CpuTorch) {
    uv pip install -p $py -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    $R = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
    uv pip install -p $py -q "$R/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$R/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" `
        "$R/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$R/rocm-7.2.1.tar.gz" `
        "$R/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$R/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
}
uv pip install -p $py -q -e "$src\hf-server"
uv pip install -p $py -q -r "$src\RFDT\requirements.txt"
& $py -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available())"
Write-Host "Setup done in $Root"
