"""Run the Jev vs Haiku experiment (EXPERIMENT.md) and write one results table.

Live runs are interleaved by seed (seed 1 for every arm, then seed 2, ...) so slow drifts
in hosted latency hit all arms alike. Every run keeps its own folder (video, decisions,
summary); this script only aggregates.

usage:
  python -m jevdoom.compare --arms jev-local jev-hosted haiku --seeds 1 2 3 4 5
  python -m jevdoom.compare --probe --arms jev-local haiku
  python -m jevdoom.compare --dry-run --arms jev-local jev-hosted jev-hosted-gemma haiku haiku-think --seeds 1 2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import urllib.request
from pathlib import Path

from .engines import ARMS, make_engine
from .jev import LOCAL_URL
from .play import git_commit, run
from .probe import make_cases, score_decide


def preflight(arms: list[str], local_url: str, dry_run: bool) -> list[str]:
    """Return problems that would make an arm fail before spending any time on runs."""
    if dry_run:
        return []
    problems = []
    if "jev-local" in arms:
        try:
            with urllib.request.urlopen(local_url.rstrip("/") + "/health", timeout=5) as r:
                if b"ready" not in r.read():
                    problems.append(f"jev-local: server at {local_url} is not ready")
        except OSError as e:
            problems.append(f"jev-local: no server at {local_url} ({e}); start scripts/serve.ps1")
    if any(a.startswith("haiku") for a in arms):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            problems.append("haiku: anthropic SDK not installed in this venv")
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE") or Path.home().joinpath(".config", "anthropic").exists()):
            problems.append("haiku: no Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)")
    return problems


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 2) if xs else None


def aggregate(results: list[dict], arms: list[str]) -> list[dict]:
    rows = []
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        if not rs:
            continue
        rows.append({
            "arm": arm, "runs": len(rs),
            "exit_rate": round(sum(r["outcome"] == "exit" for r in rs) / len(rs), 2),
            "deaths": sum(r["outcome"] == "died" for r in rs),
            "progress_median_pct": med([r["route_progress_pct"] for r in rs]),
            "progress_best_pct": max(r["route_progress_pct"] for r in rs),
            "survival_median_s": med([r["game_seconds"] for r in rs]),
            "kills_median": med([r["kills"] for r in rs]),
            "damage_median": med([r["damage_taken"] for r in rs]),
            "latency_median_s": med([r["latency_median"] for r in rs]),
            "latency_p90_s": med([r["latency_p90"] for r in rs]),
            "decisions_per_s": med([r["decisions_per_s"] for r in rs]),
            "errors": sum(r["errors"] for r in rs),
            "cost_usd_total": round(sum(r["cost_usd"] for r in rs), 4),
        })
    return rows


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_no runs_\n"
    cols = list(rows[0])
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        out += "| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |\n"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--probe", action="store_true", help="offline 68-situation probe per arm instead of live runs")
    ap.add_argument("--dry-run", action="store_true", help="mock engines with each arm's latency profile; free")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--local-url", default=LOCAL_URL)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args(argv)

    problems = preflight(args.arms, args.local_url, args.dry_run)
    if problems:
        raise SystemExit("cannot start:\n  " + "\n  ".join(problems))

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "probe" if args.probe else "live"
    out_dir = Path(args.runs) / (f"compare-{stamp}-{kind}" + ("-dryrun" if args.dry_run else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"kind": kind, "arms": args.arms, "seeds": args.seeds, "dry_run": args.dry_run,
            "timeout_s": args.timeout, "commit": git_commit(), "started": stamp}

    if args.probe:
        cases = make_cases()
        rows = [{"arm": "always-forward (floor)", **{k: v for k, v in score_decide(lambda s: "forward", 0, cases).items()
                                                      if k != "rows"}}]
        for arm in args.arms:
            eng = make_engine(arm, local_url=args.local_url, dry_run=args.dry_run)
            r = score_decide(lambda st: eng.decide(st).choice, eng.min_gap, cases)
            rows.append({"arm": arm, **{k: v for k, v in r.items() if k != "rows"}})
            print(f"{arm:18s} probe score {r['score']:.2f}  wrong-direction {r['wrong_direction_turns']}", flush=True)
        flat = [{"arm": r["arm"], "score": r["score"], "wrong_direction_turns": r["wrong_direction_turns"],
                 **{k: v for k, v in r["by_kind"].items()}} for r in rows]
        (out_dir / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        (out_dir / "results.md").write_text(f"# Probe: Jev vs Haiku\n\n`{json.dumps(meta)}`\n\n" + md_table(flat))
        print(md_table(flat))
        print(f"written: {out_dir}")
        return

    results = []
    for seed in args.seeds:
        for arm in args.arms:
            print(f"--- seed {seed} arm {arm}", flush=True)
            s = run(arm, seed, local_url=args.local_url, timeout=args.timeout, window=args.window,
                    record_video=not args.no_video, runs_dir=str(out_dir), dry_run=args.dry_run)
            results.append(s)
            print(f"    {s['outcome']:8s} progress {s['route_progress_pct']:5.1f}%  {s['game_seconds']:5.1f}s  "
                  f"kills {s['kills']}  latency {s['latency_median']}s  cost ${s['cost_usd']}", flush=True)
            (out_dir / "results.json").write_text(json.dumps({"meta": meta, "runs": results}, indent=2))
    agg = aggregate(results, args.arms)
    per_run = [{k: r[k] for k in ("arm", "seed", "outcome", "route_progress_pct", "game_seconds", "kills",
                                  "damage_taken", "latency_median", "decisions_per_s", "errors", "cost_usd")}
               for r in results]
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, "aggregate": agg, "runs": results}, indent=2))
    (out_dir / "results.md").write_text(
        f"# Live runs: Jev vs Haiku\n\n`{json.dumps(meta)}`\n\n## Per arm (medians over seeds)\n\n"
        + md_table(agg) + "\n## Every run\n\n" + md_table(per_run))
    print(md_table(agg))
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
