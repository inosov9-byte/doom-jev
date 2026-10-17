"""Convert teacher labels into RFDT training records (student prompt).

Two sources:
  --labels    hosted-teacher label files (soft targets: the teacher's probabilities)
  --expert    any jsonl with a "state" field (situations, or decisions.jsonl from runs);
              labelled here by the rule-based expert (hard targets)

usage: python -m jevdoom.dataset --expert data/situations-v3*.jsonl data/expert-traj/*/decisions.jsonl --out data/rfdt-v3.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import COMMAND_IDS
from .jev import student_request


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=[])
    ap.add_argument("--expert", nargs="+", default=[], help="state files to label with expert.py")
    ap.add_argument("--max-per-file", type=int, default=0, help="subsample each expert file to at most N rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-prompt", action="store_true", help="train on the teacher-style prompt instead of the short one")
    ap.add_argument("--min-prob", type=float, default=1e-4, help="floor for soft targets before renormalising")
    args = ap.parse_args(argv)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in args.labels:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                p = {k: max(float(r["probabilities"].get(k, 0.0)), args.min_prob) for k in COMMAND_IDS}
                s = sum(p.values())
                p = {k: v / s for k, v in p.items()}
                rec = {"id": r["id"], "group_id": r["group_id"],
                       "request": student_request(r["state"], full_prompt=args.full_prompt),
                       "targets": {"cmd": {"probabilities": p}},
                       "provenance": {"teacher": r["teacher"], "teacher_choice": r["choice"]}}
                f.write(json.dumps(rec) + "\n")
                n += 1
        from .expert import decide
        import random
        rng = random.Random(0)
        for path in args.expert:
            rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
            if args.max_per_file and len(rows) > args.max_per_file:
                rows = rng.sample(rows, args.max_per_file)
            for i, r in enumerate(rows):
                choice = decide(r["state"])
                rid = r.get("id") or f"{Path(path).parent.name}-{i:05d}"
                rec = {"id": rid, "group_id": r.get("group_id") or rid,
                       "request": student_request(r["state"], full_prompt=args.full_prompt),
                       "targets": {"cmd": {"answer": choice}},
                       "provenance": {"teacher": "expert.py", "source": str(path)}}
                f.write(json.dumps(rec) + "\n")
                n += 1
    print(f"{n} records -> {args.out}")


if __name__ == "__main__":
    main()
