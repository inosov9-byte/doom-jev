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

from .wadmap import DOOR_SPECIALS, SKILL_FLAG, MapData, NavGrid, exit_target

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


def world(m: MapData, keys: frozenset, opened: frozenset, strict_lifts: bool = False) -> MapData:
    """strict_lifts: a lift moved by a trigger counts as passable only once its tag has fired (detour checks:
    E1M8 starts on a lift raised by the switch in front of the player)."""
    doors, lifts = set(), set()
    for l in m.lines:
        if l.special in DOOR_SPECIALS and l.back != -1:
            if l.special in KEY_DOORS and KEY_DOORS[l.special] not in keys:
                continue
            doors.add(l.back)
        manual = l.special in LIFT_TRIGGERS and any(sd >= 0 and m.sector_tag[sd] == l.tag for sd in (l.front, l.back))
        if l.special in LIFT_TRIGGERS and l.tag and (not strict_lifts or manual or l.tag in opened):
            lifts |= {i for i, t in enumerate(m.sector_tag) if t == l.tag}
    door_tags = {l.tag for l in m.lines if l.special in DOOR_TRIGGERS}
    stair_step = {l.tag: STAIR_SPECIALS[l.special] for l in m.lines if l.special in STAIR_SPECIALS and l.tag}
    lower_tags = {l.tag for l in m.lines if l.special in LOWER_TO_LOWEST and l.tag}
    lower_tags |= {b[2] for name, b in BOSS_TRIGGERS.items() if name == m.name}
    floor = m.floor
    for tag in opened:
        secs = {i for i, t in enumerate(m.sector_tag) if t == tag}
        if tag in stair_step:
            floor = _build_stairs(m, secs, stair_step[tag], floor.copy() if floor is m.floor else floor)
        elif tag in lower_tags:      # floor drops to the lowest neighbouring floor (E1M8's tag 666 wall)
            floor = floor.copy() if floor is m.floor else floor
            for s in secs:
                nb = [l.front if l.back == s else l.back for l in m.lines if s in (l.front, l.back)]
                nb = [n for n in nb if n not in (-1, s)]
                if nb:
                    floor[s] = min(m.floor[n] for n in nb)
            lifts |= secs
        elif tag in door_tags:
            doors |= secs
        else:
            lifts |= secs
    return dataclasses.replace(m, door_sectors=doors, lift_sectors=lifts, floor=floor)


STAIR_SPECIALS = {7: 8, 8: 8, 100: 16, 127: 16}   # stair builders -> step height (100, 127: turbo 16)
LOWER_TO_LOWEST = {23, 37, 38, 60, 82, 84}         # floor lowers to the lowest neighbouring floor


def _build_stairs(m: MapData, start: set[int], step: float, floor: np.ndarray) -> np.ndarray:
    """Floors after a stair builder fires (Doom's EV_BuildStairs): each tagged sector rises by one step;
    the next step is the sector behind a two-sided line whose front is the current step, with the same
    floor texture; it rises one more step. E1M3's climb to the exit is such a staircase; the planner
    used to raise only the tagged first step, so the 88-unit ledge looked unclimbable."""
    if not m.floor_flat:
        return floor
    done = set()
    for s in start:
        cur, h = s, m.floor[s] + step
        while cur is not None and cur not in done:
            done.add(cur)
            floor[cur] = h
            nxt = next((l.back for l in m.lines
                        if l.front == cur and l.back >= 0 and l.back not in done
                        and m.floor_flat[l.back] == m.floor_flat[cur]), None)
            cur, h = nxt, h + step
    return floor


def _stand_points(line, dist=40.0):
    (x1, y1), (x2, y2) = line.v1, line.v2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    nx, ny = (y2 - y1), -(x2 - x1)
    n = math.hypot(nx, ny) or 1.0
    nx, ny = nx / n, ny / n
    return [(mx + nx * dist, my + ny * dist), (mx - nx * dist, my - ny * dist)], (mx, my)


BOSS_TRIGGERS = {"E1M8": (3003, "baron of hell", 666)}   # map -> (boss thing type, label, tag opened when all die)


def _boss_spots(m: MapData, g: NavGrid, pos) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """A reachable spot near a boss (the boss itself if its cell is reachable), and the boss position."""
    boss = BOSS_TRIGGERS.get(m.name)
    if not boss:
        return None
    for t in m.things:
        if t[3] != boss[0] or (len(t) > 4 and (not t[4] & SKILL_FLAG[3] or t[4] & 16)):
            continue
        bp = (float(t[0]), float(t[1]))
        for r in (0, 128, 256):
            for k in range(8 if r else 1):
                p = (bp[0] + r * math.cos(k * math.pi / 4), bp[1] + r * math.sin(k * math.pi / 4))
                if g.sector_at(*p) >= 0 and g.plan(pos, p):
                    return p, bp
    return None


def solve(m: MapData, start: tuple[float, float], max_depth: int = 5, verbose: bool = False) -> list[Subgoal] | None:
    stand, mid, _ = exit_target(m)
    exit_goal = Subgoal("exit", "exit" if m.exit_lines else "teleporter", stand, mid)
    boss = BOSS_TRIGGERS.get(m.name)
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
        if boss and boss[2] not in opened:           # kill the bosses: their deaths open the boss tag
            spot = _boss_spots(m, g, pos)
            if spot and int(grid(keys, opened | {boss[2]}).reachable(spot[0]).sum()) > area:
                q.append((keys, opened | {boss[2]}, spot[0],
                          steps + [Subgoal("kill", boss[1], spot[0], spot[1], tag=boss[2])]))
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
                        max_detour: float = MAX_DETOUR, held: set[str] = frozenset()) -> list[Subgoal]:
    """Insert an 'item' subgoal for each detour weapon whose cheapest detour from a stage's route
    is within max_detour. The stage is done once the weapon is held (perception.py). Weapons already
    held (carried from the previous level) and second copies get no detour."""
    from .wadmap import path_length
    goals = list(goals)
    for t in m.things:
        name = DETOUR_WEAPONS.get(t[3])
        if not name or name in held or (len(t) > 4 and (not t[4] & SKILL_FLAG[3] or t[4] & 16)):
            continue          # absent at skill 3 or multiplayer only (E1M3: the expert circled a missing shotgun)
        pos = (float(t[0]), float(t[1]))
        grids, here, best = stage_grids(m, goals, strict_lifts=True), start, None
        for i, (g, gr) in enumerate(zip(goals, grids)):
            base, a, b = gr.plan(here, g.pos), gr.plan(here, pos), gr.plan(pos, g.pos)
            if base and a and b:
                extra = path_length(a) + path_length(b) - path_length(base)
                if extra <= max_detour and (best is None or extra < best[1]):
                    best = (i, extra)
            here = g.pos
        if best is not None:
            goals.insert(best[0], Subgoal("item", name, pos, pos))
            held = set(held) | {name}
    return goals


def stage_grids(m: MapData, goals: list[Subgoal], strict_lifts: bool = False) -> list[NavGrid]:
    """Planner grid for each stage: the world as it is before that subgoal."""
    grids, keys, opened = [], frozenset(), frozenset()
    for g in goals:
        grids.append(NavGrid(world(m, keys, opened, strict_lifts)))
        if g.kind == "key":
            keys = keys | {g.color}
        elif g.kind in ("switch", "walk", "shoot"):
            opened = opened | {g.tag}
    return grids
