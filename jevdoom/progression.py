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
# Any other tagged trigger moves floors/ceilings (raise floor, stairs, ...): once fired, its sectors
# are treated like lifts (passable whatever their height). Lights, exits and teleports are excluded.
S_TYPES = {7, 9, 14, 15, 18, 20, 21, 23, 29, 41, 42, 49, 50, 55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
           99, 101, 102, 103, 111, 112, 113, 114, 115, 116, 122, 123, 127, 131, 132, 133, 134, 135, 136, 137, 140}
G_TYPES = {24, 46, 47}
NOT_MOVERS = {11, 51, 52, 124, 197, 198, 12, 13, 17, 35, 79, 80, 81, 104, 138, 139, 39, 97, 125, 126, 48}


def how(special: int) -> str:
    return "S" if special in S_TYPES else "G" if special in G_TYPES else "W"


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
    door_tags = {l.tag for l in m.lines if l.special in DOOR_TRIGGERS}
    for tag in opened:
        secs = {i for i, t in enumerate(m.sector_tag) if t == tag}
        if tag in door_tags:
            doors |= secs
        else:
            lifts |= secs
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
        if l.tag and l.special and l.special not in NOT_MOVERS and l.special not in LIFT_TRIGGERS                 and l.tag not in triggers:
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
        area = int(g.reachable(pos).sum())
        for color, kpos in keys_avail:
            if color not in keys and g.plan(pos, kpos):
                q.append((keys | {color}, opened, kpos,
                          steps + [Subgoal("key", KEY_NAMES[color], kpos, kpos, color=color)]))
        for tag, l in triggers.items():
            if tag in opened:
                continue
            kind = {"S": "switch", "W": "walk", "G": "shoot"}[how(l.special)]
            pts, mid_l = _stand_points(l, 40.0 if kind != "shoot" else 200.0)
            for p in pts:
                if g.sector_at(*p) >= 0 and g.plan(pos, p):
                    # only worth it if firing the trigger lets the player reach more of the map
                    if int(grid(keys, opened | {tag}).reachable(p).sum()) <= area:
                        break
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


DETOUR_WEAPONS = {2001: "shotgun"}   # thing type -> weapon worth a short detour (E1M2: the expert died
                                     # in pistol-only imp fights next to a shotgun 152 units off its route)
MAX_DETOUR = 1000.0                  # extra route units; E1M2 shotgun +961, ~2 s of running


def with_weapon_detours(m: MapData, goals: list[Subgoal], start: tuple[float, float],
                        max_detour: float = MAX_DETOUR) -> list[Subgoal]:
    """Insert an 'item' subgoal for each detour weapon whose cheapest detour from a stage's route
    is within max_detour. The stage is done once the weapon is held (perception.py)."""
    from .wadmap import path_length
    goals = list(goals)
    for t in m.things:
        name = DETOUR_WEAPONS.get(t[3])
        if not name:
            continue
        pos = (float(t[0]), float(t[1]))
        grids, here, best = stage_grids(m, goals), start, None
        for i, (g, gr) in enumerate(zip(goals, grids)):
            base, a, b = gr.plan(here, g.pos), gr.plan(here, pos), gr.plan(pos, g.pos)
            if base and a and b:
                extra = path_length(a) + path_length(b) - path_length(base)
                if extra <= max_detour and (best is None or extra < best[1]):
                    best = (i, extra)
            here = g.pos
        if best is not None:
            goals.insert(best[0], Subgoal("item", name, pos, pos))
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
