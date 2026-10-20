"""Record E1M1 situations for labelling, with no bot in the loop.

Each anchor teleports the player (cheat warp, god mode) to a sampled spot and heading,
records the situation, then plays a few random commands and records after each one.
Anchor types: along the exit route, near a live monster, anywhere walkable.
A later DAgger pass records situations reached by the student itself (see play.py --record).

usage: python -m jevdoom.collect --out data/situations.jsonl --anchors 800
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .commands import COMMAND_IDS
from .executor import Executor
from .env import blank_action, make_game, observe, teleport
from .perception import Perception
from .wadmap import NavGrid, exit_target, load_map

ROLLOUT_WEIGHTS = {"forward": 4, "forward_short": 1, "fire_strafe_left": 0.5, "fire_strafe_right": 0.5, "back": 0.5, "strafe_left": 1, "strafe_right": 1,
                   "turn_left_5": 0.6, "turn_left_15": 1, "turn_left_30": 1, "turn_left_45": 0.8, "turn_left_90": 0.5,
                   "turn_right_5": 0.6, "turn_right_15": 1, "turn_right_30": 1, "turn_right_45": 0.8, "turn_right_90": 0.5,
                   "turn_around": 0.3, "fire": 1, "use": 0.5,
                   **{f"turn_{s}_{a}_fire": 0.3 for s in ("left", "right") for a in (5, 15, 30)},
                   "weapon_shotgun": 0.1, "weapon_chaingun": 0.1, "weapon_pistol": 0.1}


def free_cells(grid: NavGrid):
    door = set(grid.m.door_sectors)
    return [(r, c) for r in range(grid.h) for c in range(grid.w)
            if not grid.blocked[r, c] and int(grid.sector[r, c]) not in door]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=r"C:\tmp\jev-doom\wad\doom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=800)
    ap.add_argument("--rollout", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="s")
    ap.add_argument("--route-from", type=float, default=0.0, help="sample route anchors from this fraction of the route onward")
    ap.add_argument("--heading-sd", type=float, default=None, help="route anchors: Gaussian heading noise in degrees (default: mixed)")
    ap.add_argument("--kinds", default="0.65,0.25,0.10", help="weights for route,monster,random anchors")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    m = load_map(args.wad, args.map)
    grid = NavGrid(m)
    stand, _, _ = exit_target(m)
    start = next((t[0], t[1]) for t in m.things if t[3] == 1)
    route = grid.plan(start, stand)
    cells = free_cells(grid)
    free_set = set(cells)
    perc = Perception(grid)
    exe = Executor()
    names = list(ROLLOUT_WEIGHTS)
    weights = [ROLLOUT_WEIGHTS[n] for n in names]
    assert set(names) == set(COMMAND_IDS)

    g = make_game(args.wad, args.map, args.skill, visible=False, cheats=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    tic = 0
    with out.open("w", encoding="utf-8") as f:
        for a in range(args.anchors):
            if a % 60 == 0:                                   # fresh monsters every 60 anchors
                g.new_episode()
                for _ in range(10):
                    g.make_action(blank_action())
                g.send_game_command("god")
                if rng.random() < 0.4:
                    g.send_game_command("give shotgun")
                    g.send_game_command("give shells")
            obs = observe(g, tic)
            kind = rng.choices(["route", "monster", "random"], [float(w) for w in args.kinds.split(",")])[0]
            live = [mo for mo in obs.monsters]
            if kind == "monster" and not live:
                kind = "route"
            if kind == "route":
                i = rng.randrange(int(args.route_from * (len(route) - 1)), len(route) - 1)
                (x0, y0), (x1, y1) = route[i], route[min(i + 3, len(route) - 1)]
                heading = math.degrees(math.atan2(y1 - y0, x1 - x0))
                x, y = x0 + rng.uniform(-32, 32), y0 + rng.uniform(-32, 32)
                r = rng.random()
                if args.heading_sd is not None:
                    r = -1.0
                angle = heading + rng.gauss(0, args.heading_sd) if r < 0 else heading + (rng.gauss(0, 25) if r < 0.5 else rng.gauss(0, 80) if r < 0.8 else rng.uniform(-180, 180))
            elif kind == "monster":
                mo = rng.choice(live)
                near = [p for p in route if 150 < math.dist(p, (mo.x, mo.y)) < 900]
                if not near:
                    near = [grid.to_world(*c) for c in rng.sample(cells, 200) if 150 < math.dist(grid.to_world(*c), (mo.x, mo.y)) < 900] or [route[0]]
                x, y = rng.choice(near)
                angle = math.degrees(math.atan2(mo.y - y, mo.x - x)) + rng.gauss(0, 40)
            else:
                x, y = grid.to_world(*rng.choice(cells))
                angle = rng.uniform(0, 360)
            if grid.to_cell(x, y) not in free_set:
                x, y = grid.to_world(*grid.nearest_free(*grid.to_cell(x, y)) or grid.to_cell(*route[0]))
            teleport(g, x, y, angle % 360)
            for _ in range(2):
                g.make_action(blank_action())
            perc.reset()
            if kind == "route" and args.heading_sd is not None:
                # face the compass target itself, plus small noise
                o = observe(g, tic)
                b = perc.route_bearing(o)
                if b is not None:
                    teleport(g, o.x, o.y, (o.angle + b + rng.gauss(0, args.heading_sd)) % 360)
                    g.make_action(blank_action())
                perc.reset()
            hp = rng.randint(12, 100) if rng.random() < 0.5 else None
            last = None
            for k in range(args.rollout + 1):
                obs = observe(g, tic)
                if obs is None:
                    break
                state = perc.describe(obs, last_cmd=last, health_override=hp)
                rec = {"id": f"{args.prefix}{a:05d}-{k}", "group_id": f"{args.prefix}{a:05d}", "state": state,
                       "meta": {"kind": kind, "x": round(obs.x), "y": round(obs.y), "angle": round(obs.angle),
                                "health_override": hp, "last_cmd": last}}
                f.write(json.dumps(rec) + "\n")
                n_written += 1
                if k == args.rollout:
                    break
                last = rng.choices(names, weights)[0]
                exe.start(last)
                while exe.busy:
                    g.make_action(exe.step(obs))
                    tic += 1
                    if g.is_episode_finished():
                        break
                    obs = observe(g, tic)
                if g.is_episode_finished():
                    break
            if a % 50 == 0:
                print(f"anchor {a}/{args.anchors} written {n_written}", flush=True)
    g.close()
    print(f"done: {n_written} situations -> {out}")


if __name__ == "__main__":
    main()
