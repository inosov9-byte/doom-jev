"""Run Simple Jev's RFDT trainer with two local adjustments, without editing the checkout.

1. Windows ROCm PyTorch ships without torch.distributed. accelerate and transformers import
   torch.distributed.tensor.DTensor only for isinstance checks, so a placeholder class
   (never instantiated) lets them run.
2. RFDT computes vocabulary logits at every prompt position and then uses only the last
   real one. This version asks the model for logits at those positions only
   (`logits_to_keep`), which removes most of the 248k-wide output projection work.
   The loss is unchanged.
3. --eval-every N turns on evaluation every N optimizer steps (RFDT hard-codes no evaluation
   during training), so progress shows up while training runs. Use a small --validation file
   for this; evaluate the full held-out set after export with evaluate.py.

usage: python -m jevdoom.train_rfdt --rfdt C:\\tmp\\jev-doom\\src-simple-jev\\RFDT <RFDT train.py args>
"""
from __future__ import annotations

import os
import sys
import time
import types

import torch


def _stub_distributed():
    if torch.distributed.is_available():
        return
    stub = types.ModuleType("torch.distributed.tensor")

    class DTensor(torch.Tensor):
        pass

    stub.DTensor = DTensor
    sys.modules["torch.distributed.tensor"] = stub
    torch.distributed.tensor = stub


def decision_logits_last_only(model, inputs):
    positions = inputs["attention_mask"].sum(dim=1) - 1
    keep, row_index = torch.unique(positions, return_inverse=True)   # few distinct end positions per batch
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    use_cache=False, logits_to_keep=keep)
    final = outputs.logits[torch.arange(len(positions), device=positions.device), row_index]
    selected = final.gather(1, inputs["allowed_ids"]).float()
    return selected.masked_fill(~inputs["allowed_mask"], -1e9)


def main():
    argv = sys.argv[1:]
    if argv[:1] == ["--rfdt"]:
        rfdt_dir, argv = argv[1], argv[2:]
    else:
        rfdt_dir = r"C:\tmp\jev-doom\src-simple-jev\RFDT"
    full = "--full-logits" in argv
    argv = [a for a in argv if a != "--full-logits"]
    eval_every = 0
    if "--eval-every" in argv:
        i = argv.index("--eval-every")
        eval_every = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    _stub_distributed()
    sys.path.insert(0, rfdt_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(rfdt_dir), "hf-server"))
    import train  # RFDT/train.py

    if not full:
        train.decision_logits = decision_logits_last_only
    if eval_every:
        original = train.TrainingArguments

        def with_eval(*a, **kw):
            kw.update(eval_strategy="steps", eval_steps=eval_every)
            return original(*a, **kw)

        train.TrainingArguments = with_eval
    sys.argv = [os.path.join(rfdt_dir, "train.py")] + argv
    t0 = time.time()
    try:
        train.main()
    finally:
        if torch.cuda.is_available():
            print(f"[train_rfdt] wall={time.time() - t0:.1f}s peak_alloc={torch.cuda.max_memory_allocated() / 2**30:.2f}GiB",
                  flush=True)


if __name__ == "__main__":
    main()
