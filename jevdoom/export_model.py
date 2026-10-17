"""Merge a trained LoRA adapter into a standalone model the Simple Jev server can load.

RFDT's export.py saves model + tokenizer only. Qwen3.5 loads through AutoProcessor,
so the base model's (video_)preprocessor_config.json are copied in as well.

usage: python -m jevdoom.export_model --adapter models/lora-v2 --output C:\\tmp\\jev-doom\\models\\jev-doom-v2
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rfdt", default=r"C:\tmp\jev-doom\src-simple-jev\RFDT")
    args = ap.parse_args(argv)
    subprocess.run([sys.executable, str(Path(args.rfdt) / "export.py"), "--model", args.base, "--adapter", args.adapter,
                    "--output", args.output, "--dtype", "bfloat16"], check=True)
    snap = Path(snapshot_download(args.base, allow_patterns=["*preprocessor_config.json"]))
    for f in snap.glob("*preprocessor_config.json"):
        shutil.copy2(f, Path(args.output) / f.name)
        print(f"copied {f.name}")


if __name__ == "__main__":
    main()
