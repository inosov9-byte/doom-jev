"""Turns a game observation into the short text Jev reads.

Perception only: distances, bearings and facts. No advice and no recommended action.
Sources: ViZDoom game variables, the labels buffer (what is on screen), live sector
heights, and the WAD geometry for raycasts and the route compass.
"""
from __future__ import annotations

import math

import numpy as np

from .env import MONSTER_NAMES, PICKUP_NAMES, WEAPON_NAMES, Obs
from .progression import DOOR_TRIGGERS, KEY_DOORS, KEY_THINGS, LIFT_TRIGGERS, Subgoal
from .wadmap import DAMAGE_SECTORS, DOOR_SPECIALS, EXIT_SPECIALS, MAX_STEP, PLAYER_HEIGHT, NavGrid, exit_target, path_length

RAYS = [("ahead", 0), ("30 left", 30), ("60 left", 60), ("30 right", -30), ("60 right", -60), ("behind", 180)]
MAX_RAY = 1500
USE_RANGE = 64           # Doom use reach in map units


class Perception:
    def __init__(self, grid: NavGrid, compass: bool = True, goals: list[Subgoal] | None = None,
                 stage_grids: list[NavGrid] | None = None):
        """goals/stage_grids: level progression (progression.py). Without them the only goal is the
        exit and the text is exactly the single-level E1M1 text."""
        self.grid = grid
        self.m = grid.m
        self.compass = compass
        L = self.m.lines
        self.ax = np.array([l.v1[0] for l in L], float)
        self.ay = np.array([l.v1[1] for l in L], float)
        self.bx = np.array([l.v2[0] for l in L], float)
        self.by = np.array([l.v2[1] for l in L], float)
        self.front = np.array([l.front for l in L])
        self.back = np.array([l.back for l in L])
        self.solid = np.array([l.back == -1 or bool(l.flags & 1) for l in L])
        self.special = np.array([l.special for l in L])
        self.stand, self.exit_mid, _ = exit_target(self.m)
        self.goals = goals or [Subgoal("exit", "exit", self.stand, self.exit_mid)]
        self.stage_grids = stage_grids or [grid] * len(self.goals)
        self.stage = 0
        self.map_keys = {KEY_THINGS[t[3]] for t in self.m.things if t[3] in KEY_THINGS}
        S = {sp for sp, how in {**DOOR_TRIGGERS, **LIFT_TRIGGERS}.items() if how == "S"}
        G = {sp for sp, how in DOOR_TRIGGERS.items() if how == "G"}
        self.switch_kind = {i: ("switch" if l.special in S else "switch to shoot")
                            for i, l in enumerate(L) if l.special in S | G}
        self.locked = {i: KEY_DOORS[l.special] for i, l in enumerate(L) if l.special in KEY_DOORS}
        self.route: list | None = None
        self.idx = 0
        self.prev: Obs | None = None
        self.prev_cmd: str | None = None
        self.door_ceil_prev: dict[int, float] = {}   # door sector -> ceiling height at the previous description

    # -- raycasting -----------------------------------------------------------
    def ray(self, obs: Obs, rel_deg: float) -> tuple[float, str, float | None]:
        """Distance to first blocking thing along a ray, what it is, and first nukage distance."""
        th = math.radians(obs.angle + rel_deg)
        dx, dy = math.cos(th), math.sin(th)
        ex, ey = self.bx - self.ax, self.by - self.ay
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.ax - obs.x) * ey - (self.ay - obs.y) * ex) / den
            u = ((self.ax - obs.x) * dy - (self.ay - obs.y) * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0.5) & (t < MAX_RAY) & (u >= 0) & (u <= 1)
        nukage = None
        for i in np.argsort(np.where(hit, t, np.inf))[: int(hit.sum())]:
            d = float(t[i])
            if self.special[i] in EXIT_SPECIALS:
                return d, "exit switch", nukage
            if i in self.switch_kind and self.solid[i]:
                return d, self.switch_kind[i], nukage
            if i in self.locked and self.locked[i] not in self.keys_held(obs):
                return d, f"door (locked, needs {self.locked[i]} key)", nukage
            if self.solid[i]:
                return d, "wall", nukage
            # den < 0: the ray goes from the line's right (front) side to its left (back) side
            f, b = int(self.front[i]), int(self.back[i])
            cur, nxt = (f, b) if den[i] < 0 else (b, f)
            cf, cc = obs.heights.get(cur, (self.m.floor[cur], self.m.ceil[cur]))
            nf, nc = obs.heights.get(nxt, (self.m.floor[nxt], self.m.ceil[nxt]))
            if min(cc, nc) - max(cf, nf) < PLAYER_HEIGHT:
                if self.special[i] in DOOR_SPECIALS or nxt in self.m.door_sectors:
                    door = nxt if nxt in self.m.door_sectors else cur
                    before = self.door_ceil_prev.get(door)
                    now = obs.heights.get(door, (0, 0))[1]
                    state = "closed" if before is None or abs(now - before) < 1 else ("opening" if now > before else "closing")
                    if state == "opening":            # Doom doors rise 2 units per tic
                        eta = max(0.0, (self.grid.ceil_s[door] - now) / 2 / 35)
                        state = f"opening, fully open in about {max(0.5, round(eta * 2) / 2):g} s"
                    return d, f"door ({state})", nukage
                return d, "wall", nukage
            if nf - cf > MAX_STEP:
                return d, "ledge too high to climb", nukage
            if nukage is None and self.m.special[nxt] in DAMAGE_SECTORS:
                nukage = d
        return float(MAX_RAY), "open", nukage

    # -- route compass ----------------------------------------------------------
    def keys_held(self, obs: Obs) -> set[str]:
        return self.map_keys - obs.keys_present

    @property
    def goal(self) -> Subgoal:
        return self.goals[min(self.stage, len(self.goals) - 1)]

    def _advance(self, obs: Obs) -> None:
        """Move to the next subgoal once the current one is done (key held, tagged sector moved)."""
        while self.stage < len(self.goals) - 1:
            g = self.goal
            if g.kind == "key":
                done = g.color in self.keys_held(obs)
            else:
                done = any(abs(obs.heights.get(i, (f, c))[0] - f) > 4 or abs(obs.heights.get(i, (f, c))[1] - c) > 4
                           for i, (f, c, t) in enumerate(zip(self.m.floor, self.m.ceil, self.m.sector_tag))
                           if t == g.tag)
            if not done:
                return
            self.stage += 1
            self.route, self.idx = None, 0

    def route_bearing(self, obs: Obs) -> float | None:
        """Bearing the compass would report (positive = left), or None without a route."""
        t = self._route_target(obs)
        return None if t is None else obs.bearing_to(*t[0])

    def _route_hint(self, obs: Obs) -> str | None:
        t = self._route_target(obs)
        if t is None:
            return None
        target, left = t
        g = self.goal
        if left < 120:     # at the end of the route: point at the goal itself
            d = obs.dist_to(*g.target)
            name = "exit switch" if g.kind == "exit" else g.label
            return f"{name} {d:.0f} away, {_side(obs.bearing_to(*g.target))}"
        return f"{_side(obs.bearing_to(*target))}, {left:.0f} units left"

    def _route_target(self, obs: Obs):
        pos = (obs.x, obs.y)
        if self.route:
            lo, hi = self.idx, min(len(self.route), self.idx + 60)
            best = min(range(lo, hi), key=lambda i: math.dist(pos, self.route[i]))
            self.idx = best
            off = math.dist(pos, self.route[best])
        if not self.route or off > 96:
            self.route = self.stage_grids[min(self.stage, len(self.goals) - 1)].plan(pos, self.goal.pos)
            self.idx = 0
            if not self.route:
                return None
        path = self.route
        target = path[min(self.idx + 1, len(path) - 1)]
        for j in range(min(self.idx + 14, len(path) - 1), self.idx, -2):
            if self.grid.walkable_line(pos, path[j]):
                target = path[j]
                break
        left = math.dist(pos, path[self.idx]) + path_length(path[self.idx:])
        if left < 120:
            target = self.goal.target
        return target, left

    # -- text -----------------------------------------------------------------
    def describe(self, obs: Obs, last_cmd: str | None = None, health_override: float | None = None) -> str:
        hp = obs.health if health_override is None else health_override
        weapon = WEAPON_NAMES.get(obs.weapon, "fist")
        self._advance(obs)
        parts = [f"Health {hp:.0f}, armor {obs.armor:.0f}. Weapon: {weapon}; "
                 f"{obs.bullets:.0f} bullets, {obs.shells:.0f} shells."]
        held = sorted(self.keys_held(obs))
        if held:
            parts[0] += f" Keys: {', '.join(held)}."

        vis = sorted((m for m in obs.monsters if m.visible), key=lambda m: obs.dist_to(m.x, m.y))[:3]
        if vis:
            enemies = []
            for m in vis:
                d, b = obs.dist_to(m.x, m.y), obs.bearing_to(m.x, m.y)
                tol = max(3.0, math.degrees(math.atan2(16, max(d, 1))))
                aim = "in the crosshair" if abs(b) <= tol else _side(b, ahead=0.5)   # exact angle for aiming
                enemies.append(f"{MONSTER_NAMES.get(m.name, m.name)} {d:.0f} away, {aim}")
            parts.append("Enemies in view: " + "; ".join(enemies) + ".")
        else:
            parts.append("Enemies in view: none.")

        items = sorted((p for p in obs.pickups if p.visible), key=lambda p: obs.dist_to(p.x, p.y))[:2]
        if items:
            parts.append("Pickups in view: " + "; ".join(
                f"{PICKUP_NAMES[p.name]} {obs.dist_to(p.x, p.y):.0f} away, {_side(obs.bearing_to(p.x, p.y))}" for p in items) + ".")

        rays, hazard = [], None
        for name, rel in RAYS:
            d, kind, nuk = self.ray(obs, rel)
            if kind in ("door (closed)", "exit switch") and d <= USE_RANGE:
                kind += ", within use reach"
            rays.append(f"{name} {d:.0f} {kind}" if kind != "open" else f"{name} open")
            if nuk is not None and rel in (0, 30, -30) and (hazard is None or nuk < hazard[0]):
                hazard = (nuk, name)
        parts.append("Obstacles: " + "; ".join(rays) + ".")
        if hazard:
            parts.append(f"Toxic nukage floor {hazard[0]:.0f} units {hazard[1]}.")

        if self.compass:
            hint = self._route_hint(obs)
            if hint:
                parts.append(f"Route to {self.goal.label}: {hint}.")

        if self.prev is not None and obs.damage_taken > self.prev.damage_taken and not vis:
            # Doom's status-bar face turns toward the damage source; give the same cue in words:
            # the nearest monster with a clear line of sight.
            src = None
            for mo in sorted(obs.monsters, key=lambda mo: obs.dist_to(mo.x, mo.y)):
                b = obs.bearing_to(mo.x, mo.y)
                if self.ray(obs, b)[0] >= obs.dist_to(mo.x, mo.y) - 24:
                    src = b
                    break
            if src is not None:
                parts.append(f"Being shot from out of view, {_side(src)}.")

        if self.prev is not None and last_cmd:
            moved = math.hypot(obs.x - self.prev.x, obs.y - self.prev.y)
            dmg = obs.damage_taken - self.prev.damage_taken
            note = f"Last command: {last_cmd}, moved {moved:.0f} units"
            if last_cmd in ("forward", "back", "strafe_left", "strafe_right") and moved < 10:
                note += " (blocked)"
            parts.append(note + (f", took {dmg:.0f} damage." if dmg > 0 else "."))
        self.prev = obs
        self.door_ceil_prev = {sec: obs.heights[sec][1] for sec in self.m.door_sectors if sec in obs.heights}
        return " ".join(parts)

    def reset(self) -> None:
        self.route, self.idx, self.prev = None, 0, None
        self.door_ceil_prev = {}
        self.stage = 0


def _side(b: float, ahead: float = 7.5) -> str:
    """Angle in words. Within `ahead` degrees counts as straight ahead (default: half a 15 degree turn,
    fine for walking; enemies use an exact angle so the 5 degree turns can aim)."""
    if abs(b) < ahead:
        return "straight ahead"
    if abs(b) > 165:
        return "behind"
    return f"{abs(b):.0f} deg {'left' if b > 0 else 'right'}"
