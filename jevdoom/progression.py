"""Level progression: which keys and triggers must come before the exit, and in what order.

The planner assumes ordinary doors open and lifts move. Locked doors need their key; doors
opened by a switch, a walk-over line or a shot need that trigger first. A breadth-first
search over world states (keys held, trigger tags fired) finds the shortest sequence of
subgoals after which the exit is reachable.
"""
from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
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
    need: str = ""                    # optional campaign resource stage (with_resource_detours): "<kind>#<n>"


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
    floor_moves = {l.tag: l.special for l in m.lines if l.special in FLOOR_MOVES and l.tag}
    floor, special, moved = m.floor, m.special, set()
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
        elif tag in floor_moves:
            floor = floor.copy() if floor is m.floor else floor
            special = special.copy() if special is m.special else special
            for s in secs:
                floor[s], special[s] = _moved_floor(m, s, floor_moves[tag], floor, special[s])
            lifts |= secs              # boardable while it moves (E1M9: the exit bridge rises under the player)
            moved |= secs
        else:
            lifts |= secs
    return dataclasses.replace(m, door_sectors=doors, lift_sectors=lifts, floor=floor, special=special,
                               moved_sectors=moved, keys=frozenset(keys), opened=frozenset(opened))


STAIR_SPECIALS = {7: 8, 8: 8, 100: 16, 127: 16}   # stair builders -> step height (100, 127: turbo 16)
LOWER_TO_LOWEST = {23, 37, 38, 60, 82, 84}         # floor lowers to the lowest neighbouring floor
# Floor movers whose end height is known (p_floor.c, p_plats.c): rule, offset. The planner treats a fired
# one as a lift (it can be boarded while it moves) at its end height; perception drops the lift once the
# live floor has settled there. As a lift for good, E1M5's raised bridge (tag 6) and the nukage pool beside
# it read as one area, so falling off was a drop with a way back.
FLOOR_MOVES = {**dict.fromkeys((18, 69, 119, 128, 20, 22, 47, 68, 95), ("next", 0)),
               **dict.fromkeys((5, 91, 101, 64, 24), ("ceil", 0)), **dict.fromkeys((55, 56, 65, 94), ("ceil", -8)),
               **dict.fromkeys((19, 45, 83, 102), ("high", 0)), **dict.fromkeys((36, 70, 71, 98), ("high", 8)),
               **dict.fromkeys((58, 92, 15, 66, 59, 93), ("up", 24)), 14: ("up", 32), 67: ("up", 32), 140: ("up", 512)}
CLEARS_SPECIAL = {20, 22, 47, 68, 95}               # raise to next floor and change: the damage special goes


def _moved_floor(m: MapData, s: int, line_special: int, floor: np.ndarray, special: int) -> tuple[float, int]:
    rule, off = FLOOR_MOVES[line_special]
    nb = {l.front if l.back == s else l.back for l in m.lines if s in (l.front, l.back)} - {-1, s}
    if rule == "up" or not nb:
        h = floor[s] + (off if rule == "up" else 0)
    elif rule == "next":
        h = min((floor[n] for n in nb if floor[n] > floor[s]), default=floor[s])
    elif rule == "ceil":
        h = min(min(m.ceil[n] for n in nb), m.ceil[s]) + off
    else:
        h = max(floor[n] for n in nb) + off
    return h, (0 if line_special in CLEARS_SPECIAL else special)


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


def _pickup_spot(g: NavGrid, reach, kpos) -> tuple[float, float] | None:
    """A reachable cell from which an item on an unclimbable ledge is still picked up. Doom touches an
    item when the bounding boxes overlap (|dx|, |dy| < player radius 16 + item radius 20) and the item
    is at most 56 above / 8 below the player (E1M7: the blue key lies on a lip 32 above the floor)."""
    r, c = g.to_cell(*kpos)
    kf, best = g.floor[r, c], None
    for rr in range(max(r - 2, 0), min(r + 3, g.h)):
        for cc in range(max(c - 2, 0), min(c + 3, g.w)):
            x, y = g.to_world(rr, cc)
            if reach[rr, cc] and max(abs(x - kpos[0]), abs(y - kpos[1])) < 32 and -8 <= kf - g.floor[rr, cc] <= 56:
                d = math.dist((x, y), kpos)
                if best is None or d < best[0]:
                    best = (d, (x, y))
    return best and best[1]


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


def solve(m: MapData, start: tuple[float, float], max_depth: int = 7, verbose: bool = False) -> list[Subgoal] | None:
    stand, mid, _ = exit_target(m)
    exit_goal = Subgoal("exit", "exit" if m.exit_lines else "teleporter", stand, mid)
    boss = BOSS_TRIGGERS.get(m.name)
    keys_avail = [(KEY_THINGS[t[3]], (float(t[0]), float(t[1]))) for t in m.things if t[3] in KEY_THINGS]
    triggers: dict[int, list] = {}      # tag -> every line firing it (E1M4: tag 1 has a W1 box around the
    for l in m.lines:                   # blue key behind the doors and reachable WR lines in front of them)
        if l.tag and l.special and l.special not in NOT_MOVERS and l.special not in LIFT_TRIGGERS:
            triggers.setdefault(l.tag, []).append(l)
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
        reach = g.reachable(pos)
        area = int(reach.sum())
        if boss and boss[2] not in opened:           # kill the bosses: their deaths open the boss tag
            spot = _boss_spots(m, g, pos)
            if spot and int(grid(keys, opened | {boss[2]}).reachable(spot[0]).sum()) > area:
                q.append((keys, opened | {boss[2]}, spot[0],
                          steps + [Subgoal("kill", boss[1], spot[0], spot[1], tag=boss[2])]))
        for color, kpos in keys_avail:
            spot = None if color in keys else kpos if g.plan(pos, kpos) else _pickup_spot(g, reach, kpos)
            if spot:
                q.append((keys | {color}, opened, spot,
                          steps + [Subgoal("key", KEY_NAMES[color], spot, kpos, color=color)]))
        for tag, lines in triggers.items():
            if tag in opened:
                continue
            for l in lines:
                kind = {"S": "switch", "W": "walk", "G": "shoot"}[how(l.special)]
                pts, mid_l = _stand_points(l, 40.0 if kind != "shoot" else 200.0)
                p = next((p for p in pts if g.sector_at(*p) >= 0 and g.plan(pos, p)), None)
                if p is None:
                    continue
                # only worth it if firing the trigger lets the player reach more of the map
                if int(grid(keys, opened | {tag}).reachable(p).sum()) > area:
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


DETOUR_WEAPONS = {2001: "shotgun", 2002: "chaingun"}   # thing type -> weapon worth a short detour (E1M2: the
                                     # expert died in pistol-only imp fights next to a shotgun 152 units off its route)
MAX_DETOUR = 1000.0                  # extra route units; E1M2 shotgun +961, ~2 s of running
# A chaingun only matters once the shells are gone: worth a short detour only (E1M4: +169 beside the blue
# key room, where the expert ran out of shells and died to imps with 140 bullets for the pistol; E1M1 +801).
WEAPON_DETOUR = {"chaingun": 400.0}


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
        grids, here, best = stage_grids(m, goals, strict_lifts=True, avoid_barrels=False), start, None
        for i, (g, gr) in enumerate(zip(goals, grids)):
            base, a, b = gr.plan(here, g.pos), gr.plan(here, pos), gr.plan(pos, g.pos)
            if base and a and b:
                extra = path_length(a) + path_length(b) - path_length(base)
                if extra <= min(max_detour, WEAPON_DETOUR.get(name, max_detour)) and (best is None or extra < best[1]):
                    best = (i, extra)
            here = g.pos
        if best is not None:
            goals.insert(best[0], Subgoal("item", name, pos, pos))
            held = set(held) | {name}
    return goals


# Supplies swept before a boss (E1M8: three boxes of shells, armor, a medikit and a soulsphere lie
# off the route; the expert met two Barons with 23 shells and died having dealt ~700 of 2000).
SUPPLIES = {2049: "box of shells", 2018: "green armor", 2012: "medikit", 2013: "soulsphere"}
MAX_SUPPLY_DETOUR = 1500.0


# Without a boss: boxes of shells on cheap detours, when a shotgun is carried in (E1M4 starts with 15 shells;
# the expert ran dry by 25-35 s and died to imps, a box of shells 241 units off the first stretch). Tried and
# dropped: single shells (+4; E1M3's opening detour shifted the walkway fight and lost 2 of 4 seeds) and
# health or armor (E1M5: more time in fights, earlier deaths). Also tried (2026-10-19, campaign runs): an armor
# detour up to +800 when a map is entered with under 30 armor; E1M2 fell from 9/9 to 6/9 (rules) for 89 armor
# carried out. Health and ammo are handled on the move instead (needs.py).
CHEAP_SUPPLIES = {2049: ("box of shells", 600.0)}


def with_supply_detours(m: MapData, goals: list[Subgoal], start: tuple[float, float],
                        max_detour: float = MAX_SUPPLY_DETOUR, held: set[str] = frozenset(),
                        resources: bool = False) -> list[Subgoal]:
    """Before the first kill subgoal, visit every supply whose detour from that stage's route is
    within max_detour, in nearest-neighbour order. The visit is done on arrival (perception.py).
    Without a kill subgoal: boxes of shells on cheap detours, or with resources=True (a campaign map
    another map follows) the full resource plan (with_resource_detours)."""
    from .wadmap import path_length
    k = next((i for i, g in enumerate(goals) if g.kind == "kill"), None)
    if k is None:
        if resources:
            return with_resource_detours(m, goals, start, held=held)
        return _cheap_supplies(m, goals, start) if "shotgun" in held else goals
    gr = stage_grids(m, goals, strict_lifts=True, avoid_barrels=False)[k]
    here = goals[k - 1].pos if k else start
    base = gr.plan(here, goals[k].pos)
    if not base:
        return goals
    base_len, todo = path_length(base), []
    for t in m.things:
        name = SUPPLIES.get(t[3])
        if not name or (len(t) > 4 and (not t[4] & SKILL_FLAG[3] or t[4] & 16)):
            continue
        pos = (float(t[0]), float(t[1]))
        a, b = gr.plan(here, pos), gr.plan(pos, goals[k].pos)
        if a and b and path_length(a) + path_length(b) - base_len <= max_detour:
            todo.append(Subgoal("item", name, pos, pos))
    order, cur = [], here
    while todo:
        nxt = min(todo, key=lambda g: math.dist(cur, g.pos))
        todo.remove(nxt)
        order.append(nxt)
        cur = nxt.pos
    return goals[:k] + order + goals[k:]


def _cheap_supplies(m: MapData, goals: list[Subgoal], start: tuple[float, float]) -> list[Subgoal]:
    """Each CHEAP_SUPPLIES item goes before the stage whose route it is the cheapest detour from, if within
    its limit; items of one stage in nearest-neighbour order from the stage start."""
    from .wadmap import path_length
    grids = stage_grids(m, goals, strict_lifts=True, avoid_barrels=False)
    starts = [start] + [g.pos for g in goals[:-1]]
    bases = [gr.plan(a, g.pos) for gr, a, g in zip(grids, starts, goals)]
    per_stage: dict[int, list[Subgoal]] = {}
    for t in m.things:
        name, limit = CHEAP_SUPPLIES.get(t[3], (None, 0.0))
        if not name or (len(t) > 4 and (not t[4] & SKILL_FLAG[3] or t[4] & 16)):
            continue
        pos, best = (float(t[0]), float(t[1])), None
        for i, (gr, a, g, base) in enumerate(zip(grids, starts, goals, bases)):
            if not base or math.dist(a, pos) + math.dist(pos, g.pos) - path_length(base) > limit:
                continue                  # straight-line bound first: A* per item and stage is slow
            r, c = gr.to_cell(*pos)       # not beside a ledge (E1M3: shells on the toxic walkway among barrels;
            if gr.near_drop[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3].any():    # two seeds of four died there)
                continue
            p1, p2 = gr.plan(a, pos), gr.plan(pos, g.pos)
            if p1 and p2:
                extra = path_length(p1) + path_length(p2) - path_length(base)
                if extra <= limit and (best is None or extra < best[1]):
                    best = (i, extra)
        if best:
            per_stage.setdefault(best[0], []).append(Subgoal("item", name, pos, pos))
    out = []
    for i, (g, a) in enumerate(zip(goals, starts)):
        todo, cur = per_stage.get(i, []), a
        while todo:
            nxt = min(todo, key=lambda s: math.dist(cur, s.pos))
            todo.remove(nxt)
            out.append(nxt)
            cur = nxt.pos
        out.append(g)
    return out


# Campaign resource plan: what the next map needs, worth a detour in proportion to its value. thing type ->
# (label, longest detour in route units, need). ~280 units/s: 2000 units is ~7 s of running. Medikits and
# Medikits and stimpacks are left to needs.py: whether one is needed is known only once the damage is taken
# (a planned medikit stage was judged at the stage start, at full health, and skipped). Single shells, clips too.
RESOURCES = {2013: ("soulsphere", 2000.0, "health"), 2019: ("blue armor", 1800.0, "armor200"),
             2018: ("green armor", 1200.0, "armor100"), 2002: ("chaingun", 1500.0, "weapon"),
             2003: ("rocket launcher", 1200.0, "weapon"), 8: ("backpack", 1200.0, "ammo"),
             2049: ("box of shells", 600.0, "shells"), 2048: ("box of bullets", 400.0, "bullets")}
ENABLED = {2013, 2019, 2002, 2003, 8}        # worth a key or a trigger on the way (a door opened for them)
ENABLER_COST = 300.0                         # an extra key or trigger stage costs this much on top of the walk
WEAPON_THING = {2001: "shotgun", 2002: "chaingun", 2003: "rocket launcher"}
RESOURCE_CACHE = os.environ.get("JEV_RESOURCE_CACHE", r"C:\tmp\jev-doom\cache\resources")
MONSTER_THINGS = {3004, 9, 3001, 3002, 58, 3006, 3005, 3003}
MON_NEAR, MON_CLOSER = 400.0, 250.0          # a monster this close to the detour, and this much closer than to the route
MON_COST, MON_MAX = 250.0, 3                 # adds this much to the detour; this many rule the item out
RESOURCE_VERSION = 7


def _on_skill(t) -> bool:
    return len(t) < 5 or bool(t[4] & SKILL_FLAG[3] and not t[4] & 16)


def with_resource_detours(m: MapData, goals: list[Subgoal], start: tuple[float, float],
                          held: set[str] = frozenset()) -> list[Subgoal]:
    """Optional 'item' stages for what the next map needs (RESOURCES): each item goes before the stage it is
    cheapest from (armor: the earliest stage within its limit, so later fights are fought with it on),
    with at most one key or trigger stage to open its room for the valuable ones (ENABLED). Items beside
    a ledge, on a damaging floor, or whose detour crosses one are left out. Stages carry need='<kind>#<n>';
    needs.py skips a stage (and its enabler) when it is no longer needed at play time. Cached on disk."""
    import json
    held = set(held) | {g.label for g in goals if g.kind == "item" and g.label in WEAPON_THING.values()}
    key = f"{m.name}-{'-'.join(sorted(w.replace(' ', '') for w in held)) or 'none'}-v{RESOURCE_VERSION}"
    path = os.path.join(RESOURCE_CACHE, key + ".json")
    if os.path.exists(path):
        try:
            return [Subgoal(**{**g, "pos": tuple(g["pos"]), "target": tuple(g["target"])}) for g in json.load(open(path))]
        except (OSError, ValueError):
            pass
    out = _plan_resources(m, goals, start, held)
    os.makedirs(RESOURCE_CACHE, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    json.dump([dataclasses.asdict(g) for g in out], open(tmp, "w"))
    os.replace(tmp, path)
    return out


def _plan_resources(m: MapData, goals: list[Subgoal], start: tuple[float, float], held: set[str]) -> list[Subgoal]:
    from .wadmap import DAMAGE_SECTORS, path_length
    states, keys, opened = [], frozenset(), frozenset()
    for g in goals:
        states.append((keys, opened))
        if g.kind == "key":
            keys = keys | {g.color}
        elif g.kind in ("switch", "walk", "shoot"):
            opened = opened | {g.tag}
    cache: dict = {}

    def grid(k, o):
        if (k, o) not in cache:
            cache[(k, o)] = NavGrid(world(m, k, o, strict_lifts=True), avoid_barrels=False)
        return cache[(k, o)]

    starts = [start] + [g.pos for g in goals[:-1]]
    grids = [grid(*s) for s in states]
    bases = [gr.plan(a, g.pos) for gr, a, g in zip(grids, starts, goals)]
    reach = [gr.reachable(a) for gr, a in zip(grids, starts)]
    taken = {g.pos for g in goals if g.kind == "item"}
    keys_avail = [(KEY_THINGS[t[3]], (float(t[0]), float(t[1]))) for t in m.things if t[3] in KEY_THINGS]
    triggers: dict[int, list] = {}
    for l in m.lines:
        if l.tag and l.special and l.special not in NOT_MOVERS:
            triggers.setdefault(l.tag, []).append(l)
    enablers: dict[int, list] = {}           # stage -> [(chain of subgoals, grid after it, reach from its end)]

    def openers(gr, k, o, rmask):
        """(subgoal, keys, opened) for each key or trigger reachable in rmask."""
        out = []
        for color, kpos in keys_avail:
            if color not in k:
                spot = kpos if rmask[gr.to_cell(*kpos)] else _pickup_spot(gr, rmask, kpos)
                if spot:
                    out.append((Subgoal("key", KEY_NAMES[color], spot, kpos, color=color), k | {color}, o))
        for tag, lines in triggers.items():
            if tag in o:
                continue
            for l in lines:
                kind = {"S": "switch", "W": "walk", "G": "shoot"}[how(l.special)]
                pts, mid_l = _stand_points(l, 40.0 if kind != "shoot" else 200.0)
                p = next((p for p in pts if gr.sector_at(*p) >= 0 and rmask[gr.to_cell(*p)]), None)
                if p is not None:
                    label = {"switch": "switch", "walk": "trigger line", "shoot": "switch to shoot"}[kind]
                    out.append((Subgoal(kind, label, p, mid_l, tag=tag), k, o | {tag}))
                    break
        return out

    def stage_enablers(i):
        """One opener, or two when the second one's key or line only comes within reach after the first
        (E1M3: a switch opens the way to a walk-over line that lowers the lift up to the soulsphere)."""
        if i not in enablers:
            out = []
            first = openers(grids[i], *states[i], reach[i])
            seen = {(sg.kind, sg.color, sg.tag) for sg, _, _ in first}
            for sg, k2, o2 in first:
                g2 = grid(k2, o2)
                r2 = g2.reachable(sg.pos)
                out.append(([(sg, grids[i])], g2, r2))
                for sg2, k3, o3 in openers(g2, k2, o2, r2):
                    if (sg2.kind, sg2.color, sg2.tag) in seen:
                        continue
                    g3 = grid(k3, o3)
                    out.append(([(sg, grids[i]), (sg2, g2)], g3, g3.reachable(sg2.pos)))
            enablers[i] = out
        return enablers[i]

    mons = np.array([(t[0], t[1]) for t in m.things if t[3] in MONSTER_THINGS and _on_skill(t)], float).reshape(-1, 2)

    def guards(base, *paths) -> int:
        """Monsters the detour walks up to (within MON_NEAR) and passes MON_CLOSER units closer than the stage
        route does: fights the plan would add (E1M4: the blue armor room's shotgun guys killed 3 of 9 runs;
        straight-line distances, so a monster behind a wall from the route still counts as off it)."""
        if not len(mons):
            return 0
        def dist(ps):
            pts = np.array([q for p in ps for q in p[::2]], float)
            return np.sqrt(((mons[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).min(1)
        d_det, d_base = dist(paths), dist([base])
        return int(((d_det < MON_NEAR) & (d_base - d_det > MON_CLOSER)).sum())

    def hazard(gr, base, *paths):
        """The detour crosses a damaging floor the stage route itself does not (E1M4: the first stretch
        crosses nukage anyway; the box of shells beside the start was dropped for it)."""
        bad = lambda ps: {s for p in ps for q in p if (s := gr.sector_at(*q)) >= 0 and int(gr.m.special[s]) in DAMAGE_SECTORS}
        return bool(bad(paths) - bad([base]))

    picks = []                                # (stage, extra, [subgoals]) per item
    for t in m.things:
        spec = RESOURCES.get(t[3])
        pos = (float(t[0]), float(t[1]))
        if not spec or not _on_skill(t) or pos in taken:
            continue
        label, limit, need = spec
        if ((t[3] in WEAPON_THING and WEAPON_THING[t[3]] in held) or (t[3] == 2049 and "shotgun" not in held)
                or (t[3] == 2048 and "chaingun" not in held)):
            continue
        r, c = grids[-1].to_cell(*pos)
        s = grids[-1].sector_at(*pos)
        if s < 0 or int(m.special[s]) in DAMAGE_SECTORS or grids[-1].near_drop[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3].any():
            continue
        cands = []
        for i, (gr, a, g, base) in enumerate(zip(grids, starts, goals, bases)):
            if not base:
                continue
            bl = path_length(base)
            if reach[i][gr.to_cell(*pos)]:
                if math.dist(a, pos) + math.dist(pos, g.pos) - bl > limit:
                    continue
                p1, p2 = gr.plan(a, pos), gr.plan(pos, g.pos)
                if p1 and p2 and not hazard(gr, base, p1, p2) and (n := guards(base, p1, p2)) < MON_MAX:
                    cands.append((i, path_length(p1) + path_length(p2) - bl + MON_COST * n, []))
            elif t[3] in ENABLED:
                for chain, g2, r2 in stage_enablers(i):
                    pts = [a] + [sg.pos for sg, _ in chain] + [pos, g.pos]
                    extra_cost = ENABLER_COST * len(chain)
                    if (not r2[g2.to_cell(*pos)]
                            or sum(math.dist(pts[j], pts[j + 1]) for j in range(len(pts) - 1)) - bl + extra_cost > limit):
                        continue
                    legs = [gx.plan(pts[j], pts[j + 1]) for j, (_, gx) in enumerate(chain)]
                    legs += [g2.plan(chain[-1][0].pos, pos), g2.plan(pos, g.pos)]
                    if all(legs) and not hazard(g2, base, *legs) and (n := guards(base, *legs)) < MON_MAX:
                        cands.append((i, sum(map(path_length, legs)) - bl + extra_cost + MON_COST * n,
                                      [sg for sg, _ in chain]))
        cands = [c for c in cands if c[1] <= limit]
        if not cands:
            continue
        # Armor: the earliest stage within the limit (fights after it are fought with armor on); else cheapest.
        best = min(cands, key=lambda c: (c[0], c[1])) if need.startswith("armor") else min(cands, key=lambda c: c[1])
        picks.append((best[0], best[1], best[2] + [Subgoal("item", label, pos, pos)], need))
        if os.environ.get("JEV_RESOURCE_DEBUG"):
            print(f"  {m.name} {label} {pos} stage {best[0]} cost {best[1]:.0f} via {[g.label for g in best[2]]}")
    # one enabler per (stage, key/tag): two items behind the same door share it
    per_stage: dict[int, list] = {}
    for n, (i, extra, sgs, need) in enumerate(sorted(picks, key=lambda p: (p[0], p[1]))):
        per_stage.setdefault(i, []).append((sgs, f"{need}#{n}"))
    out = []
    for i, (g, a) in enumerate(zip(goals, starts)):
        todo, cur, done_en = per_stage.get(i, []), a, set()
        while todo:                           # nearest first, by the first subgoal of each pick
            nxt = min(todo, key=lambda p: math.dist(cur, p[0][0].pos))
            todo.remove(nxt)
            for sg in nxt[0]:
                ek = (sg.kind, sg.color, sg.tag)
                if sg.kind != "item" and ek in done_en:
                    continue
                done_en.add(ek)
                out.append(dataclasses.replace(sg, need=nxt[1]))
                cur = sg.pos
        out.append(g)
    return _verified(m, out, start, goals)


def _verified(m: MapData, out: list[Subgoal], start, goals: list[Subgoal]) -> list[Subgoal]:
    """Every stage of the new plan still has a route (an enabler can raise a floor across a later route);
    otherwise drop the enabler-gated picks, then everything."""
    def ok(gl):
        here = start
        for g, gr in zip(gl, stage_grids(m, gl, strict_lifts=True, avoid_barrels=False)):
            if not gr.plan(here, g.pos):
                return False
            here = g.pos
        return True
    if ok(out):
        return out
    gated = {g.need for g in out if g.need and g.kind != "item"}
    plain = [g for g in out if g.need not in gated]
    return plain if gated and ok(plain) else goals


def trigger_route(g: NavGrid, a: tuple[float, float], b: tuple[float, float], max_expand: int = 200_000):
    """(route, index of the point past the line) for a route a -> b that first crosses a walk-over line
    opening doors shut in g, or None. After a fall the
    way out can run through such a door (E1M5: from the nukage pool below the bridge, a manual door, a
    corridor, then a WR line that opens the door back to the start); a door that opened for a few seconds
    can have shut again (E1M4: the blue key room's doors, opened from outside by a WR line, closed behind
    the expert; the WR box around the key opens them for good)."""
    from .wadmap import path_length
    m, reach, best = g.m, g.reachable(a), None
    inside = lambda p: 0 <= (rc := g.to_cell(*p))[0] < g.h and 0 <= rc[1] < g.w and reach[rc]
    for tag in {l.tag for l in m.lines if DOOR_TRIGGERS.get(l.special) == "W" and l.tag}:
        secs = {i for i, t in enumerate(m.sector_tag) if t == tag}
        if secs <= m.door_sectors:
            continue
        cands = [(l, [p for p in _stand_points(l, 24.0)[0] if inside(p)])
                 for l in m.lines if l.tag == tag and DOOR_TRIGGERS.get(l.special) == "W"]
        cands = [(l, pts) for l, pts in cands if pts]
        if not cands:
            continue
        l, pts = min(cands, key=lambda c: min(math.dist(a, p) for p in c[1]))
        heads = [h for h in (g.plan(a, p, max_expand) for p in pts) if h]
        if not heads:
            continue
        head = max(heads, key=path_length)          # the far side: the walk crosses the line
        g2 = NavGrid(dataclasses.replace(m, door_sectors=m.door_sectors | secs, opened=m.opened | {tag}))
        tail = g2.plan(head[-1], b, max_expand)
        if tail and (best is None or path_length(head) + path_length(tail) < best[0]):
            best = (path_length(head) + path_length(tail), head + tail[1:], len(head) - 1)
    return best and best[1:]


def stage_grids(m: MapData, goals: list[Subgoal], strict_lifts: bool = False,
                avoid_barrels: bool = True) -> list[NavGrid]:
    """Planner grid for each stage: the world as it is before that subgoal. Detour checks (weapons, supplies)
    pass avoid_barrels=False: their limits are in plain path lengths."""
    grids, keys, opened = [], frozenset(), frozenset()
    for g in goals:
        grids.append(NavGrid(world(m, keys, opened, strict_lifts), avoid_barrels=avoid_barrels))
        if g.kind == "key":
            keys = keys | {g.color}
        elif g.kind in ("switch", "walk", "shoot"):
            opened = opened | {g.tag}
    return grids
