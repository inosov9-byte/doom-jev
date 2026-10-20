"""LoRA-train a local Jev student on the compact prompt (fastjev.py).

Input: RFDT-format records from dataset.py ({"request", "targets": {"cmd": {"answer"|"probabilities"}}, "group_id"}).
Loss: cross-entropy over the option letters' logits at the answer position (what the server scores).
Split: whole group_ids held out (hash), so neighbouring frames of one trajectory never straddle the split.
Progress: held-out agreement and loss every --eval-every steps, printed, appended to metrics.jsonl,
and rendered to --progress (a small Markdown file) if given. The best adapter is kept in <out>/best.

usage: python -m jevdoom.train_fast --data data/rfdt-v4.jsonl --model Qwen/Qwen3.5-4B --out C:\\tmp\\jev-doom\\models\\v4-4b --steps 1500
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .fastjev import label_token_ids, prompt_text, request_parts


def held_out(group_id: str, frac: float) -> bool:
    return int(hashlib.md5(group_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def encode(tok, rec):
    state, instr, options = request_parts(rec["request"])
    ids = tok(prompt_text(state, instr, options), add_special_tokens=False).input_ids
    t = rec["targets"]["cmd"]
    if "answer" in t:
        target = [1.0 if o == t["answer"] else 0.0 for o in options]
    else:
        target = [float(t["probabilities"][o]) for o in options]
    return {"ids": ids, "target": target, "options": options, "group": rec.get("group_id") or rec["id"]}


def batches(rows, bs, rng):
    idx = list(range(len(rows)))
    while True:
        rng.shuffle(idx)
        for i in range(0, len(idx) - bs + 1, bs):
            yield [rows[j] for j in idx[i:i + bs]]


def collate(batch, pad_id, device):
    L = max(len(r["ids"]) for r in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, r in enumerate(batch):                  # right padding: causal attention never sees the pads
        ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
        mask[i, :len(r["ids"])] = 1
    last = mask.sum(1) - 1
    target = torch.tensor([r["target"] for r in batch])
    return ids.to(device), mask.to(device), last.to(device), target.to(device)


def option_logits(model, ids, mask, last, label_ids):
    """Hidden state at each row's answer position, projected onto the option letters only."""
    base = model.get_base_model()
    h = base.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    h = h[torch.arange(len(last), device=h.device), last]
    w = base.lm_head.weight[label_ids]
    return (h @ w.T).float()


@torch.no_grad()
def evaluate(model, rows, label_ids, pad_id, device, bs):
    model.eval()
    agree, loss, conf = 0, 0.0, collections.Counter()
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        ids, mask, last, target = collate(b, pad_id, device)
        lg = option_logits(model, ids, mask, last, label_ids)
        loss += float(-(target * lg.log_softmax(-1)).sum())
        pred, gold = lg.argmax(-1).tolist(), target.argmax(-1).tolist()
        for r, p, g in zip(b, pred, gold):
            agree += p == g
            conf[(r["options"][g], r["options"][p])] += 1
    model.train()
    return agree / len(rows), loss / len(rows), conf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--init-adapter", help="start from this LoRA adapter instead of a fresh one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", help="Markdown progress file (e.g. in OneDrive) rewritten at each evaluation")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    recs = [json.loads(l) for p in args.data for l in open(p, encoding="utf-8") if l.strip()]
    rows = [encode(tok, r) for r in recs]
    val = [r for r in rows if held_out(r["group"], args.val_frac)]
    train = [r for r in rows if not held_out(r["group"], args.val_frac)]
    if len({tuple(r["options"]) for r in rows}) > 1:     # e.g. records built before the weapon commands (26 options)
        raise SystemExit("records have different option lists: rebuild them with jevdoom.dataset from the states")
    label_ids = torch.tensor(label_token_ids(tok, len(rows[0]["options"])), device=device)
    lens = sorted(len(r["ids"]) for r in rows)
    print(f"{len(train)} train / {len(val)} held-out records, prompt tokens median {lens[len(lens) // 2]} max {lens[-1]}", flush=True)
    majority = collections.Counter(max(range(len(r["target"])), key=r["target"].__getitem__) for r in val).most_common(1)[0][1] / len(val)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if args.init_adapter:          # continue from an earlier student (DAgger rounds)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) *
                                              0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    meta = {"model": args.model, "data": args.data, "train": len(train), "val": len(val), "majority_baseline": round(majority, 3),
            **{k: getattr(args, k) for k in ("steps", "batch", "lr", "lora_rank")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    metrics = open(out / "metrics.jsonl", "a", encoding="utf-8")
    history, best = [], -1.0

    def checkpoint(step, train_loss, t0):
        nonlocal best
        acc, vloss, conf = evaluate(model, val, label_ids, pad_id, device, args.batch * 2)
        m = {"step": step, "train_loss": round(train_loss, 4), "val_loss": round(vloss, 4), "val_agreement": round(acc, 4),
             "elapsed_s": round(time.time() - t0, 1), "time": time.strftime("%H:%M:%S"),
             "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        history.append(m)
        metrics.write(json.dumps(m) + "\n"); metrics.flush()
        rate = f"{(m['elapsed_s'] - history[-2]['elapsed_s']) / (step - history[-2]['step']):.2f} s/step incl. eval" if len(history) > 1 else ""
        print(f"step {step:5d}  train loss {train_loss:.3f}  held-out loss {vloss:.3f}  agreement {acc:.3f}  "
              f"({m['elapsed_s']:.0f} s, {rate}, peak {m['peak_gb']} GB)", flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")
        if args.progress:
            write_progress(Path(args.progress), meta, history, conf, best)

    t0 = time.time()
    model.train()
    run_loss, n = 0.0, 0
    checkpoint(0, float("nan"), t0)
    gen = batches(train, args.batch, rng)
    for step in range(1, args.steps + 1):
        ids, mask, last, target = collate(next(gen), pad_id, device)
        lg = option_logits(model, ids, mask, last, label_ids)
        loss = -(target * lg.log_softmax(-1)).sum(-1).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        run_loss += loss.item(); n += 1
        if step % args.eval_every == 0 or step == args.steps:
            checkpoint(step, run_loss / n, t0)
            run_loss, n = 0.0, 0
    model.save_pretrained(out / "last"); tok.save_pretrained(out / "last")
    print(f"done; best held-out agreement {best:.3f} -> {out / 'best'}", flush=True)


def write_progress(path: Path, meta, history, conf, best):
    rows = "\n".join(f"| {h['step']} | {h['train_loss']} | {h['val_loss']} | {h['val_agreement']:.1%} | {h['elapsed_s']:.0f} | {h['time']} |"
                     for h in history)
    per = collections.defaultdict(lambda: [0, 0])
    for (g, p), c in conf.items():
        per[g][1] += c
        if g == p:
            per[g][0] += c
    cls = "\n".join(f"| {g} | {n} | {ok / n:.0%} |" for g, (ok, n) in sorted(per.items(), key=lambda x: -x[1][1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Training: {meta['model']}\n\n{meta['train']} train / {meta['val']} held-out decisions; "
        f"always-the-most-common-command baseline {meta['majority_baseline']:.1%}. "
        f"Steps {meta['steps']}, batch {meta['batch']}, lr {meta['lr']}, LoRA rank {meta['lora_rank']}. "
        f"Best held-out agreement so far: **{best:.1%}**.\n\n"
        f"| step | train loss | held-out loss | agreement with expert | seconds | time |\n|---|---|---|---|---|---|\n{rows}\n\n"
        f"## Latest check, per expert command\n\n| expert chose | n | student matched |\n|---|---|---|\n{cls}\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
