"""Convert teacher labels into RFDT training records (student prompt, soft targets).

usage: python -m jevdoom.dataset --labels data/labels-v1.jsonl --out data/rfdt-v1.jsonl
"""
from __future__ import annotations

import argparse
import json

from .commands import COMMAND_IDS
from .jev import student_request


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, nargs="+")
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
    print(f"{n} records -> {args.out}")


if __name__ == "__main__":
    main()
