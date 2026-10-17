"""Compare a local Jev endpoint against the teacher labels on held-out situations.

Metrics: top-1 agreement with the teacher, mean teacher probability of the student's pick
("would the teacher accept this move"), cross-entropy against the teacher distribution,
per-command confusion, and latency.

usage: python -m jevdoom.evaluate --labels data/labels-v1.jsonl --ids data/rfdt/validation.jsonl --url http://127.0.0.1:8012
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics

from .commands import COMMAND_IDS
from .jev import JevClient, student_request


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, nargs="+")
    ap.add_argument("--ids", help="RFDT jsonl whose ids select the held-out subset (e.g. validation.jsonl)")
    ap.add_argument("--url", required=True)
    ap.add_argument("--full-prompt", action="store_true", help="send the teacher-style prompt (zero-shot baseline)")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", help="write per-item results here")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for p in args.labels for l in open(p, encoding="utf-8")]
    if args.ids:
        keep = {json.loads(l)["id"] for l in open(args.ids, encoding="utf-8")}
        rows = [r for r in rows if r["id"] in keep]
    rows = rows[: args.limit]
    client = JevClient(args.url, timeout=60)
    agree, accept, ce, lat = [], [], [], []
    confusion = collections.Counter()
    out = open(args.out, "w", encoding="utf-8") if args.out else None
    client.classify(student_request(rows[0]["state"], full_prompt=args.full_prompt))   # warm-up
    for r in rows:
        ans, dt = client.classify(student_request(r["state"], full_prompt=args.full_prompt))
        sp, tp = ans["probabilities"], r["probabilities"]
        agree.append(ans["choice"] == r["choice"])
        accept.append(tp.get(ans["choice"], 0.0))
        ce.append(-sum(tp[k] * math.log(max(sp.get(k, 0.0), 1e-9)) for k in COMMAND_IDS))
        lat.append(dt)
        confusion[(r["choice"], ans["choice"])] += 1
        if out:
            out.write(json.dumps({"id": r["id"], "teacher": r["choice"], "student": ans["choice"],
                                  "student_conf": ans["confidence"], "latency": dt}) + "\n")
    print(f"n={len(rows)}  top1 agreement {sum(agree) / len(agree):.3f}  teacher-prob of student pick "
          f"{statistics.mean(accept):.3f}  cross-entropy {statistics.mean(ce):.3f}  "
          f"latency median {statistics.median(lat):.3f}s p90 {sorted(lat)[int(0.9 * len(lat))]:.3f}s")
    by_teacher = collections.Counter(r["choice"] for r in rows)
    for c in COMMAND_IDS:
        if by_teacher[c]:
            print(f"  teacher={c:15s} n={by_teacher[c]:4d}  student matched {confusion[(c, c)] / by_teacher[c]:.2f}")


if __name__ == "__main__":
    main()
