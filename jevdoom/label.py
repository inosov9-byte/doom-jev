"""Label recorded situations with the hosted teacher (Qwen3.6-35B-A3B Jev classifier).

Resumable: ids already in the output are skipped. Serial requests with a minimum gap
between starts, to stay inside the free demo's 2 requests/second limit.

usage: python -m jevdoom.label --situations data/situations-v1.jsonl --out data/labels-v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .jev import HOSTED_URL, TEACHER_MODEL, JevClient, teacher_request


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--situations", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=HOSTED_URL)
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument("--min-gap", type=float, default=0.55)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [json.loads(l) for p in args.situations for l in open(p, encoding="utf-8") if l.strip()]
    todo = [r for r in todo if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already labelled, {len(todo)} to go", flush=True)

    client = JevClient(args.url, timeout=30)
    last_start = 0.0
    lat, errors = [], 0
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            wait = args.min_gap - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)
            last_start = time.monotonic()
            try:
                ans, dt = client.classify_retry(teacher_request(r["state"], args.model))
            except Exception as e:                       # keep going; failed ids are retried on the next run
                errors += 1
                print(f"{r['id']}: {type(e).__name__} {getattr(e, 'code', '')} {str(e)[:120]}", flush=True)
                if errors > 20 and errors > i // 5:
                    raise SystemExit("too many errors, stopping")
                continue
            lat.append(dt)
            f.write(json.dumps({"id": r["id"], "group_id": r["group_id"], "state": r["state"], "meta": r.get("meta", {}),
                                "teacher": args.model, "choice": ans["choice"], "confidence": ans["confidence"],
                                "probabilities": ans["probabilities"], "input_tokens": ans.get("input_tokens"),
                                "latency": round(dt, 3)}) + "\n")
            f.flush()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(todo)} labelled, median latency {sorted(lat)[len(lat) // 2]:.2f}s, errors {errors}", flush=True)
    print(f"done, errors {errors}")


if __name__ == "__main__":
    main()
