"""End-to-end episode: play maps in order with one arm and seed, carrying health, armor, weapons
and ammo from each exit into the next map (loadout.py). The campaign stops at the first map that
is not exited.

    python -m jevdoom.campaign --arm rules --maps E1M1 E1M2 E1M3 --seed 26 --out runs\\campaign-rules
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engines import ARMS
from .jev import LOCAL_URL
from .loadout import load as load_loadout
from .play import run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="rules", choices=ARMS)
    ap.add_argument("--maps", nargs="+", default=["E1M1", "E1M2"])
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--loadout", help="state entering the first map: pistol (default), nominal, or a file")
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--local-url", default=LOCAL_URL)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-needs", action="store_true", help="no supply subgoals during the maps (needs.py)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, legs = load_loadout(args.loadout, args.maps[0]), []
    for mp in args.maps:
        print(f"--- {mp} seed {args.seed} carrying {lo or 'pistol start'}", flush=True)
        s = run(args.arm, args.seed, map_name=mp, local_url=args.local_url, timeout=args.timeout,
                record_video=not args.no_video, runs_dir=str(out / mp), loadout=lo,
                needs=not args.no_needs, carry=not args.no_needs and mp != args.maps[-1])
        legs.append(s)
        print(f"    {s['outcome']:8s} progress {s['route_progress_pct']:5.1f}%  {s['game_seconds']:5.1f}s  "
              f"kills {s['kills']}  end {s['end_loadout']}", flush=True)
        (out / "campaign.json").write_text(json.dumps({"arm": args.arm, "seed": args.seed, "legs": legs}, indent=2))
        if s["outcome"] != "exit":
            break
        lo = s["end_loadout"]
    done = sum(s["outcome"] == "exit" for s in legs)
    print(f"cleared {done}/{len(args.maps)} maps, {sum(s['game_seconds'] for s in legs):.1f} s game time")


if __name__ == "__main__":
    main()
