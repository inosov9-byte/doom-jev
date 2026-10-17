"""Level progression: which keys and triggers must come before the exit, and in what order.

The planner assumes ordinary doors open and lifts move. Locked doors need their key; doors
opened by a switch, a walk-over line or a shot need that trigger first. A breadth-first
search over world states (keys held, trigger tags fired) finds the shortest sequence of
subgoals after which the exit is reachable.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass

from .wadmap import DOOR_SPECIALS, MapData, NavGrid, exit_target

KEY_DOORS = {26: "blue", 32: "blue", 27: "yellow", 34: "yellow", 28: "red", 33: "red"}
KEY_THINGS = {5: "blue", 40: "blue", 6: "yellow", 39: "yellow", 13: "red", 38: "red"}
KEY_NAMES = {"blue": "blue keycard", "yellow": "yellow keycard", "red": "red keycard"}
# tagged-door triggers -> how they fire: S = use (switch), W = walk over, G = shoot
DOOR_TRIGGERS = {2: "W", 4: "W", 29: "S", 46: "G", 61: "S", 63: "S", 86: "W", 90: "W", 103: "S",
                 105: "W", 106: "W", 108: "W", 109: "W", 111: "S", 112: "S", 114: "S", 115: "S"}
LIFT_TRIGGERS = {10: "W", 21: "S", 62: "S", 88: "W", 120: "W", 121: "W", 122: "S", 123: "S"}


@dataclass
class Subgoal:
    kind: str                         # key | switch | walk | shoot | exit
    label: str                        # text shown to the engine, e.g. "red keycard", "switch"
    pos: tuple[float, float]          # where to stand / go
    target: tuple[float, float]       # what to face (switch midpoint, key position)
    tag: int = 0
    color: str = ""


def world(m: MapData, keys: frozenset, opened: frozenset) -> MapData:
    doors, lifts = set(), set()
    for l in m.lines:
        if l.special in DOOR_SPECIALS and l.back != -1:
            if l.special in KEY_DOORS and KEY_DOORS[l.special] not in keys:
                continue
            doors.add(l.back)
        if l.special in LIFT_TRIGGERS and l.tag:
            lifts |= {i for i, t in enumerate(m.sector_tag) if t == l.tag}
    for tag in opened:
        doors |= {i for i, t in enumerate(m.sector_tag) if t == tag}
    return dataclasses.replace(m, door_sectors=doors, lift_sectors=lifts)


def _stand_points(line, dist=40.0):
    (x1, y1), (x2, y2) = line.v1, line.v2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    nx, ny = (y2 - y1), -(x2 - x1)
    n = math.hypot(nx, ny) or 1.0
    nx, ny = nx / n, ny / n
    return [(mx + nx * dist, my + ny * dist), (mx - nx * dist, my - ny * dist)], (mx, my)


def solve(m: MapData, start: tuple[float, float], max_depth: int = 5, verbose: bool = False) -> list[Subgoal] | None:
    stand, mid, _ = exit_target(m)
    exit_goal = Subgoal("exit", "exit", stand, mid)
    keys_avail = [(KEY_THINGS[t[3]], (float(t[0]), float(t[1]))) for t in m.things if t[3] in KEY_THINGS]
    triggers = {}
    for l in m.lines:
        if l.special in DOOR_TRIGGERS and l.tag and l.tag not in triggers:
            triggers[l.tag] = l
    grids: dict = {}

    def grid(keys, opened):
        k = (keys, opened)
        if k not in grids:
            grids[k] = NavGrid(world(m, keys, opened))
        return grids[k]

    q = deque([(frozenset(), frozenset(), start, [])])
    seen = set()
    while q:
        keys, opened, pos, steps = q.popleft()
        if (keys, opened) in seen:
            continue
        seen.add((keys, opened))
        g = grid(keys, opened)
        if g.plan(pos, stand):
            return steps + [exit_goal]
        if len(steps) >= max_depth:
            continue
        for color, kpos in keys_avail:
            if color not in keys and g.plan(pos, kpos):
                q.append((keys | {color}, opened, kpos,
                          steps + [Subgoal("key", KEY_NAMES[color], kpos, kpos, color=color)]))
        for tag, l in triggers.items():
            if tag in opened:
                continue
            kind = {"S": "switch", "W": "walk", "G": "shoot"}[DOOR_TRIGGERS[l.special]]
            pts, mid_l = _stand_points(l, 40.0 if kind != "shoot" else 200.0)
            for p in pts:
                if g.sector_at(*p) >= 0 and g.plan(pos, p):
                    label = {"switch": "switch", "walk": "trigger line", "shoot": "switch to shoot"}[kind]
                    q.append((keys, opened | {tag}, p, steps + [Subgoal(kind, label, p, mid_l, tag=tag)]))
                    break
        if verbose:
            print(f"  state keys={sorted(keys)} opened={sorted(opened)} expanded, queue {len(q)}")
    return None


CACHE = r"C:\tmp\jev-doom\cache\progression"


def load_or_solve(m: MapData, start: tuple[float, float]) -> list[Subgoal]:
    """Progression for a map, cached on disk (solving E1M2 takes ~30 s)."""
    import json
    import os
    path = os.path.join(CACHE, f"{m.name}.json")
    if os.path.exists(path):
        return [Subgoal(**{**g, "pos": tuple(g["pos"]), "target": tuple(g["target"])})
                for g in json.load(open(path))]
    goals = solve(m, start)
    if goals is None:
        raise RuntimeError(f"no progression found for {m.name}")
    os.makedirs(CACHE, exist_ok=True)
    json.dump([dataclasses.asdict(g) for g in goals], open(path, "w"))
    return goals


def stage_grids(m: MapData, goals: list[Subgoal]) -> list[NavGrid]:
    """Planner grid for each stage: the world as it is before that subgoal."""
    grids, keys, opened = [], frozenset(), frozenset()
    for g in goals:
        grids.append(NavGrid(world(m, keys, opened)))
        if g.kind == "key":
            keys = keys | {g.color}
        elif g.kind in ("switch", "walk", "shoot"):
            opened = opened | {g.tag}
    return grids
