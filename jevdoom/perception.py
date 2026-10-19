"""Turns a game observation into the short text Jev reads.

Perception only: distances, bearings and facts. No advice and no recommended action.
Sources: ViZDoom game variables, the labels buffer (what is on screen), live sector
heights, and the WAD geometry for raycasts and the route compass.
"""
from __future__ import annotations

import dataclasses
import os
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .env import DECOR_NAMES, MONSTER_NAMES, SOLID_DECOR, PICKUP_NAMES, WEAPON_NAMES, Obs
from .progression import DOOR_TRIGGERS, KEY_DOORS, KEY_THINGS, LIFT_TRIGGERS, Subgoal, trigger_route
from .wadmap import DAMAGE_SECTORS, DOOR_SPECIALS, EXIT_SPECIALS, MAX_STEP, PLAYER_HEIGHT, NavGrid, exit_target, path_length

RAYS = [("ahead", 0), ("30 left", 30), ("60 left", 60), ("30 right", -30), ("60 right", -60), ("behind", 180)]
MAX_RAY = 1500
USE_RANGE = 64           # Doom use reach in map units
BODY_RADIUS = 16         # Doom player radius: a move is blocked by whatever either shoulder meets
LINE_OF_FIRE = os.environ.get("JEVDOOM_LINE_OF_FIRE", "1") == "1"
SHOTGUN_REFIRE = 37      # tics between shotgun shots (Doom)
HISTORY = 4              # commands listed in "Recent commands"
# A drop the player cannot climb back out of toward the current goal (E1M3: the toxic ring around the
# walkway; fights and dodges pushed the expert off the edge on every seed, and it died in the pit).
DEAD_DROP = "drop with no way back"


class Perception:
    def __init__(self, grid: NavGrid, compass: bool = True, goals: list[Subgoal] | None = None,
                 stage_grids: list[NavGrid] | None = None, stage_routes: list | None = None):
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
        self.lift_walls = {i for i, l in enumerate(L) if LIFT_TRIGGERS.get(l.special) == "S" and l.back != -1}
        self.lift_sectors_all = {sd for i in self.lift_walls for sd in (L[i].front, L[i].back)}
        lift_tags = {l.tag for l in L if l.special in LIFT_TRIGGERS and l.tag}
        self.lift_floors = {sd for sd, t in enumerate(self.m.sector_tag) if t in lift_tags}
        self.lift_floor_prev: dict[int, float] = {}  # lift sector -> floor at the previous description
        self.route: list | None = None
        self.via: int | None = None                    # route index just past a trigger line it must cross
        self.idx = 0
        self.prev: Obs | None = None
        self.prev_cmd: str | None = None
        self.history: list[tuple[str, float]] = []   # (command, units moved) for recent decisions
        self.door_ceil_prev: dict[int, float] = {}   # door sector -> ceiling height at the previous description
        self.doors_seen_open: set[int] = set()
        self.live_grids: dict[int, NavGrid] = {}     # stage -> grid rebuilt from live sector heights
        self._stand_cache: tuple = (None, 0.0)
        self._reach: dict[tuple, tuple] = {}           # (id(grid), goal) -> (grid, cells the goal is reachable from)
        self.at_goal = False                          # close to the goal with a clear walk to it
        self.live_check_tic = -10**9
        self.plan_fail_tic = -10**9
        self.chase_tic = -10**9
        # Path planning never runs inside describe(): A* took up to 0.7 s per decision on E1M8. Each
        # stage starts on a route planned before the game (stage_routes); replans run on this worker
        # and are swapped in when done, while decisions keep using the last route.
        self.stage_routes = stage_routes or []
        self._planner = ThreadPoolExecutor(max_workers=1)
        self._job = None                              # (future, kind, stage)
        self.replan = False
        if self.stage_routes and self.stage_routes[0]:
            self.route = list(self.stage_routes[0])

    # -- raycasting -----------------------------------------------------------
    def body_ray(self, obs: Obs, rel_deg: float) -> tuple[float, str, float | None]:
        """Clearance for the player's whole body in a direction: the centre ray plus two rays
        through the shoulders. A thin centre ray reports open space while a shoulder is already
        against a corner, and the player cannot move. Doors, switches and the exit keep the
        centre reading, since use acts on what is straight ahead."""
        c = self.ray(obs, rel_deg)
        if c[1] not in ("wall", "open", DEAD_DROP) and not c[1].startswith("ledge") and c[0] <= USE_RANGE * 2:
            return c
        best, side = c, None
        for off, name in ((BODY_RADIUS, "left"), (-BODY_RADIUS, "right")):
            r = self.ray(obs, rel_deg, offset=off)
            if r[0] < best[0] - 8:
                best, side = r, name
        if side is None:
            return c
        return best[0], f"{best[1]} at the {side} shoulder", c[2]

    def ray(self, obs: Obs, rel_deg: float, offset: float = 0.0) -> tuple[float, str, float | None]:
        d, kind, nuk, thing = self._ray(obs, rel_deg, offset)
        if thing is not None and thing[0] < d:
            return thing[0], thing[1], (nuk if nuk is not None and nuk < thing[0] else None)
        return d, kind, nuk

    def _ray(self, obs: Obs, rel_deg: float, offset: float = 0.0):
        """Distance to first blocking thing along a ray, what it is, and first nukage distance.
        offset: start the ray this many units to the left of the player (negative = right)."""
        th = math.radians(obs.angle + rel_deg)
        dx, dy = math.cos(th), math.sin(th)
        ox, oy = obs.x - dy * offset, obs.y + dx * offset
        ex, ey = self.bx - self.ax, self.by - self.ay
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.ax - ox) * ey - (self.ay - oy) * ex) / den
            u = ((self.ax - ox) * dy - (self.ay - oy) * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0.5) & (t < MAX_RAY) & (u >= 0) & (u <= 1)
        nukage = None
        # solid decorations as circles; Doom collides square boxes, a circle is close enough here
        thing = None
        for so in obs.solids:
            r = SOLID_DECOR[so.name]
            fx, fy = so.x - ox, so.y - oy
            along = fx * dx + fy * dy
            perp2 = fx * fx + fy * fy - along * along
            if along > 0 and perp2 < r * r:
                d = along - math.sqrt(r * r - perp2)
                if thing is None or d < thing[0]:
                    thing = (max(d, 0.0), DECOR_NAMES.get(so.name, "pillar"))
        for i in np.argsort(np.where(hit, t, np.inf))[: int(hit.sum())]:
            d = float(t[i])
            if self.special[i] in EXIT_SPECIALS:
                return d, "exit switch", nukage, thing
            if i in self.switch_kind and self.solid[i]:
                return d, self.switch_kind[i], nukage, thing
            if i in self.locked and self.locked[i] not in self.keys_held(obs):
                return d, f"door (locked, needs {self.locked[i]} key)", nukage, thing
            if self.solid[i]:
                return d, "wall", nukage, thing
            # den < 0: the ray goes from the line's right (front) side to its left (back) side
            f, b = int(self.front[i]), int(self.back[i])
            cur, nxt = (f, b) if den[i] < 0 else (b, f)
            cf, cc = obs.heights.get(cur, (self.m.floor[cur], self.m.ceil[cur]))
            nf, nc = obs.heights.get(nxt, (self.m.floor[nxt], self.m.ceil[nxt]))
            if d < BODY_RADIUS + 8:           # within the player's box: it stands on the highest floor it touches
                cf = max(cf, self._stand_floor(obs))
            if min(cc, nc) - max(cf, nf) < PLAYER_HEIGHT:
                if self.special[i] in DOOR_SPECIALS or nxt in self.m.door_sectors:
                    door = nxt if nxt in self.m.door_sectors else cur
                    before = self.door_ceil_prev.get(door)
                    now = obs.heights.get(door, (0, 0))[1]
                    state = "closed" if before is None or abs(now - before) < 1 else ("opening" if now > before else "closing")
                    if state == "opening":            # Doom doors rise 2 units per tic
                        eta = max(0.0, (self.grid.ceil_s[door] - now) / 2 / 35)
                        state = f"opening, fully open in about {max(0.5, round(eta * 2) / 2):g} s"
                    return d, f"door ({state})", nukage, thing
                if i in self.lift_walls:              # the raised lift itself (E1M8: special 62 before the Barons)
                    return d, self._lift_state(obs, nxt), nukage, thing
                return d, "wall", nukage, thing
            if nf - cf > MAX_STEP:
                # A lift under a high ceiling reads as a ledge, not a wall (E1M5: the lift at the top of the
                # stairs to the teleport room; the expert slid along its face for 20 s without pressing use).
                # Only when the lift is the far side: E1M2's exit lift has a switch face toward the step above.
                if i in self.lift_walls and self.m.sector_tag[nxt] == self.m.lines[i].tag:
                    return d, self._lift_state(obs, nxt), nukage, thing
                return d, "ledge too high to climb", nukage, thing
            if cf - nf > MAX_STEP and self._dead_end(ox + dx * (d + 12), oy + dy * (d + 12)):
                return d, DEAD_DROP, nukage, thing
            if nukage is None and self.m.special[nxt] in DAMAGE_SECTORS:
                nukage = d
        return float(MAX_RAY), "open", nukage, thing

    def _lift_state(self, obs: Obs, lift: int) -> str:
        before = self.lift_floor_prev.get(lift)
        moving = before is not None and obs.heights.get(lift, (before, 0))[0] < before - 0.5
        return "lift (lowering)" if moving else "lift (raised)"

    def _dead_end(self, x: float, y: float) -> bool:
        """True when (x, y) lies where the current goal cannot be reached from (reverse reachability
        on the stage grid; the mask is computed once per grid)."""
        stage = min(self.stage, len(self.goals) - 1)
        grid = self.live_grids.get(stage) or self.stage_grids[stage]
        key = (id(grid), self.goal.pos)
        mask = self._reach.get(key)
        if mask is None or mask[0] is not grid:
            mask = self._reach[key] = (grid, grid.can_reach(self.goal.pos))
        r, c = grid.to_cell(x, y)
        if not (0 <= r < grid.h and 0 <= c < grid.w) or grid.blocked[r, c]:
            return False
        return not mask[1][r, c]

    def _stand_floor(self, obs: Obs) -> float:
        """Floor the player stands on: Doom holds it at the highest floor under any corner of its box, so
        a player whose centre hangs over a drop still stands on the edge (E1M3: the centre over the toxic
        pit made the walkway read as a ledge too high to climb, and the expert froze on the edge)."""
        if self._stand_cache[0] == (obs.tic, obs.x, obs.y):
            return self._stand_cache[1]
        best = -1e9
        for ox, oy in ((0, 0), (BODY_RADIUS, BODY_RADIUS), (BODY_RADIUS, -BODY_RADIUS),
                       (-BODY_RADIUS, BODY_RADIUS), (-BODY_RADIUS, -BODY_RADIUS)):
            sec = self.grid.sector_at(obs.x + ox, obs.y + oy)
            if sec >= 0:
                best = max(best, obs.heights.get(sec, (self.m.floor[sec], 0))[0])
        self._stand_cache = ((obs.tic, obs.x, obs.y), best)
        return best

    def _in_pit(self, obs: Obs, x: float, y: float) -> bool:
        """A dead-end spot below the player's floor (not just somewhere shut off by a door)."""
        floor, _ = self._live_heights(obs)
        ps, ts = self.grid.sector_at(obs.x, obs.y), self.grid.sector_at(x, y)
        return ps >= 0 and ts >= 0 and floor[ps] - floor[ts] > MAX_STEP and self._dead_end(x, y)

    # -- cover ------------------------------------------------------------------
    COVER_FROM = ("BaronOfHell",)            # monsters a cover spot is reported for
    # Two or more of these with a line of fire: a spot out of sight of all of them is reported too.
    # Gunmen cannot be dodged; in E1M4's opening room four of them took 71 health in five seconds.
    SHOOTERS = ("Zombieman", "ShotgunGuy", "DoomImp")
    SHOOTER_RANGE = 800
    HITSCAN = ("Zombieman", "ShotgunGuy")
    EYE, CHEST = 41.0, 32.0                  # Doom player eye height; a monster's mid height

    def _live_heights(self, obs: Obs):
        f, c = self.m.floor.copy(), self.m.ceil.copy()
        for i, (fl, ce) in obs.heights.items():
            if i < len(f):
                f[i], c[i] = fl, ce
        return f, c

    def _sight_clear(self, a, az, b, bz, floor, ceil) -> bool:
        """Line of sight from (a, az) to (b, bz) through map lines with live heights: one-sided
        lines block, two-sided lines block when the sight line passes outside their opening."""
        dx, dy = b[0] - a[0], b[1] - a[1]
        ex, ey = self.bx - self.ax, self.by - self.ay
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((self.ax - a[0]) * ey - (self.ay - a[1]) * ex) / den
            u = ((self.ax - a[0]) * dy - (self.ay - a[1]) * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
        if (hit & self.solid).any():
            return False
        idx = np.flatnonzero(hit)
        if not len(idx):
            return True
        fr, bk = self.front[idx], self.back[idx]
        lo = np.maximum(floor[fr], floor[bk])
        hi = np.minimum(ceil[fr], ceil[bk])
        z = az + (bz - az) * t[idx]
        return bool(((z >= lo) & (z <= hi)).all())

    def cover(self, obs: Obs, threat, reach: int = 256) -> tuple[float, float] | None:
        """Nearest spot (distance, bearing) the player can walk to straight that the threat (one monster
        or a list) cannot see: behind a pillar, a wall corner or below a ledge (E1M8's Barons)."""
        grid = self._stage_grid(obs)
        floor, ceil = self._live_heights(obs)
        threats = threat if isinstance(threat, (list, tuple)) else [threat]
        tz = []
        for t in threats:
            tsec = grid.sector_at(t.x, t.y)
            tz.append(floor[tsec] if tsec >= 0 else 0.0)
        best = None
        for d in range(48, reach + 1, 32):
            for k in range(12):
                th = math.radians(obs.angle + 30 * k)
                p = (obs.x + d * math.cos(th), obs.y + d * math.sin(th))
                sec = grid.sector_at(*p)
                if sec < 0 or not grid.walkable_line((obs.x, obs.y), p):
                    continue
                pz = floor[sec] + self.EYE
                if any(self._sight_clear(p, pz, (t.x, t.y), z + h, floor, ceil)
                       for t, z in zip(threats, tz) for h in (self.CHEST, 60.0)):
                    continue
                best = (float(d), obs.bearing_to(*p))
                break
            if best:
                return best
        return None

    def escape(self, obs: Obs, threats) -> tuple[float, float] | None:
        """Where to back off to from bosses: of the spots 160-320 units away the player can walk to
        straight, the best on distance from the nearest threat plus openness (free directions around
        it). Distance alone led into the narrowing tips of E1M8's star room, where the Barons clawed
        the expert against the walls."""
        grid = self._stage_grid(obs)
        best = None
        for d in (320, 240, 160):
            for k in range(16):
                th = math.radians(obs.angle + 22.5 * k)
                p = (obs.x + d * math.cos(th), obs.y + d * math.sin(th))
                if grid.sector_at(*p) < 0 or not grid.walkable_line((obs.x, obs.y), p):
                    continue
                if any(_seg_point_dist((obs.x, obs.y), p, (t.x, t.y)) < 64 for t in threats):
                    continue                          # the way out runs into a Baron's body (blocked strafes)
                gap = min(math.dist(p, (t.x, t.y)) for t in threats)
                open_dirs = sum(grid.walkable_line(p, (p[0] + 160 * math.cos(math.radians(45 * j)),
                                                       p[1] + 160 * math.sin(math.radians(45 * j))))
                                for j in range(8))
                score = min(gap, 350.0) + 80.0 * open_dirs       # past 350 units distance matters less than room
                if best is None or score > best[0]:
                    best = (score, float(d), obs.bearing_to(*p))
        return None if best is None else (best[1], best[2])

    BARREL_NEAR = 200         # report the nearest barrel this close (blast reach 128 + 16 along each axis, more diagonally)
    DODGE_HORIZON = 1.2      # seconds ahead a projectile is worth reporting
    DODGE_MISS = 44.0         # closest approach that still hits: player radius 16 + fireball ~8-16, margin

    def incoming(self, obs: Obs) -> tuple[float, str] | None:
        """The soonest projectile whose straight path passes within DODGE_MISS of the player: seconds
        to closest approach and the sidestep that moves away from its path."""
        best = None
        for _name, x, y, vx, vy in obs.projectiles:
            rx, ry = x - obs.x, y - obs.y
            v2 = vx * vx + vy * vy
            if v2 < 1:
                continue
            t = -(rx * vx + ry * vy) / v2                       # tics to closest approach
            if t <= 0 or t > self.DODGE_HORIZON * 35:
                continue
            mx, my = rx + vx * t, ry + vy * t                    # projectile relative to player then
            if math.hypot(mx, my) > self.DODGE_MISS:
                continue
            if math.hypot(mx, my) < 4:                           # dead on: any perpendicular
                mx, my = -vy, vx
            th = math.radians(obs.angle + 90)                    # strafe_left direction
            side = "right" if mx * math.cos(th) + my * math.sin(th) > 0 else "left"
            if best is None or t < best[0]:
                best = (t / 35, side)
        return best

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
            elif g.kind == "item" and g.label in ("shotgun", "chaingun"):
                done = {"shotgun": obs.has_shotgun, "chaingun": obs.has_chaingun}[g.label]
            elif g.kind == "item":                    # a supply: done on arrival (not taken at full health)
                done = obs.dist_to(*g.pos) < 40
            elif g.kind == "kill":
                done = not self._targets(obs)
            else:
                done = any(abs(obs.heights.get(i, (f, c))[0] - f) > 4 or abs(obs.heights.get(i, (f, c))[1] - c) > 4
                           for i, (f, c, t) in enumerate(zip(self.stage_grids[0].m.floor, self.stage_grids[0].m.ceil, self.m.sector_tag))
                           if t == g.tag)
            if not done:
                return
            self.stage += 1
            r = self.stage_routes[self.stage] if self.stage < len(self.stage_routes) else None
            self.route, self.idx, self.via = (list(r) if r else None), 0, None

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
        if self.at_goal:   # at the end of the route with a clear walk: point at the goal itself
            d = obs.dist_to(*target)            # target is the goal (live position for a kill subgoal)
            name = "exit switch" if g.kind == "exit" and g.label == "exit" else g.label
            return f"{name} {d:.0f} away, {_side(obs.bearing_to(*target))}"
        return f"{_side(obs.bearing_to(*target))}, {left:.0f} units left"

    LIVE_CHECK_TICS = 35      # compare live sector heights with the route grid at most once a second
    LIVE_DELTA = 24.0         # a floor or ceiling this far from the grid's value means the map changed

    def _stage_grid(self, obs: Obs) -> NavGrid:
        """The current stage's planner grid, rebuilt from live sector heights when the map has changed
        since the grid was built: trap doors that shut, floors raised or lowered by triggers (E1M6, E1M9:
        the WAD's initial heights routed through a door that closes during play). Manual doors and lifts
        are left to the grid, which already treats them as passable."""
        stage = min(self.stage, len(self.goals) - 1)
        base = self.live_grids.get(stage, self.stage_grids[stage])
        if obs.tic - self.live_check_tic < self.LIVE_CHECK_TICS or not obs.heights:
            return base
        self.live_check_tic = obs.tic
        wm = self.stage_grids[stage].m
        skip = wm.door_sectors | wm.lift_sectors
        changed = [i for i, (f, c) in obs.heights.items()
                   if i not in skip and i < len(base.m.floor)
                   and (abs(f - base.m.floor[i]) > self.LIVE_DELTA or abs(c - base.m.ceil[i]) > self.LIVE_DELTA)]
        # A floor moved by a trigger is a lift while it moves; settled at its end height it is plain floor again.
        settled = {i for i in base.m.moved_sectors & base.m.lift_sectors
                   if i in obs.heights and abs(obs.heights[i][0] - wm.floor[i]) < 1}
        # A tagged door seen open that has shut again and stays shut: closed until something reopens it.
        reclosed = {i for i in wm.door_sectors & self.doors_seen_open
                    if self.m.sector_tag[i] and i in obs.heights and obs.heights[i][1] - obs.heights[i][0] < PLAYER_HEIGHT
                    and abs(obs.heights[i][1] - self.door_ceil_prev.get(i, -1e9)) < 1}
        if not changed and not settled and reclosed == wm.door_sectors - base.m.door_sectors:
            return base
        floor, ceil = wm.floor.copy(), wm.ceil.copy()
        for i, (f, c) in obs.heights.items():
            if i not in skip and i < len(floor):
                floor[i], ceil[i] = f, c
        lifts = wm.lift_sectors - settled - (wm.moved_sectors - base.m.lift_sectors)
        self.live_grids[stage] = NavGrid(dataclasses.replace(wm, floor=floor, ceil=ceil, lift_sectors=lifts,
                                                             door_sectors=wm.door_sectors - reclosed))
        self.replan = True                            # keep the old route until the new one is ready
        return self.live_grids[stage]

    KILL_TARGETS = {"baron of hell": "BaronOfHell"}

    def _targets(self, obs: Obs) -> list:
        """Living monsters a kill subgoal asks for (live positions)."""
        name = self.KILL_TARGETS.get(self.goal.label)
        return [mo for mo in obs.monsters if mo.name == name] if name else []

    def _goal_pos(self, obs: Obs) -> tuple[float, float]:
        """Where the compass aims at the end of the route: the goal's target, or for a kill subgoal the
        nearest living target. The route itself always runs to the goal's stand point: E1M8's Barons
        stand in alcoves the grid cannot enter, and planning to them failed on every decision."""
        if self.goal.kind == "kill":
            t = self._targets(obs)
            if t:
                mo = min(t, key=lambda mo: obs.dist_to(mo.x, mo.y))
                return (mo.x, mo.y)
        return self.goal.target

    REPLAN_TICS = 35          # after a failed plan, try again at most once a second

    def _submit(self, kind: str, grid: NavGrid, a, b, max_expand: int = 200_000) -> None:
        if self._job is None:
            fn = _plan_or_trigger if kind == "route" else grid.plan
            self._job = (self._planner.submit(fn, grid, a, b, max_expand) if kind == "route"
                         else self._planner.submit(fn, a, b, max_expand), kind, self.stage)

    def _collect(self, obs: Obs) -> None:
        if self._job is None or not self._job[0].done():
            return
        fut, kind, stage = self._job
        self._job = None
        res = fut.result()
        if stage != self.stage:
            return                                    # planned for a finished stage
        via = None
        if kind == "route":
            res, via = res
        if res:
            self.route, self.idx, self.via = res, 0, via
        elif kind == "route":
            self.plan_fail_tic = obs.tic

    def _route_target(self, obs: Obs):
        pos = (obs.x, obs.y)
        grid = self._stage_grid(obs)
        self._collect(obs)
        gpos = self.goal.pos
        if self.goal.kind == "kill":
            # Chase the live target when the grid reaches it (Barons leave their alcoves and roam);
            # otherwise the stand spot. Bounded search, at most once a second, in the background.
            live = self._goal_pos(obs)
            if self.route and math.dist(self.route[-1], live) > 128 and obs.tic - self.chase_tic >= self.REPLAN_TICS:
                self.chase_tic = obs.tic
                self._submit("chase", grid, pos, live, 12_000)
            if self.route:
                gpos = self.route[-1]
        off = 0.0
        if self.via is not None and self.idx >= self.via:
            self.via = None
        # Until the player is past a trigger line on the route, neither tracking nor shortcuts skip the
        # crossing (the way back runs right beside the way there).
        cap = len(self.route) if self.route and self.via is None else (self.via + 1 if self.route else 0)
        if self.route:
            lo, hi = self.idx, min(cap, self.idx + 60)
            best = min(range(lo, hi), key=lambda i: math.dist(pos, self.route[i]))
            self.idx = best
            off = math.dist(pos, self.route[best])
            # The nearest route point lies in a pit the player is above, not in: a route planned from the pit
            # (E1M5: the way out of the pool runs under the bridge; back on the bridge, the compass pointed
            # down into the pool).
            if off <= 96 and self._dead_end(*self.route[best]) and not self._dead_end(*pos):
                off = math.inf
        if (not self.route or off > 96 or self.replan) and obs.tic - self.plan_fail_tic >= self.REPLAN_TICS:
            if self._job is None:
                self.replan = False
            self._submit("route", grid, pos, gpos)
        if not self.route:
            return None
        path = self.route
        target = path[min(self.idx + 1, len(path) - 1)]
        for j in range(min(self.idx + 14, cap - 1), self.idx, -2):
            # Current stage: later stages may open doors still shut now. Clear of drop edges: a straight
            # line hugging an edge took the expert off E1M3's walkway at the bends.
            if grid.walkable_line(pos, path[j], keep_off_drops=True):
                target = path[j]
                break
        left = math.dist(pos, path[self.idx]) + path_length(path[self.idx:])
        # Point at the goal itself only with a clear walk to its standing spot (E1M2 exit: within 120
        # units the direct line to the switch ran into a wall corner and the expert pushed into it).
        self.at_goal = left < 120 and grid.walkable_line(pos, gpos) and self._before_switch(pos)
        if self.at_goal:
            target = self._goal_pos(obs)
        return target, left

    def _before_switch(self, pos) -> bool:
        """For a switch or exit switch: the player stands in front of its line, where use reaches it. Beside
        the line the switch lies straight ahead along the wall and use hits nothing (E1M8: the stair switch
        after the Barons; the expert pressed use from its side for 45 s)."""
        g = self.goal
        if g.kind not in ("switch", "exit") or g.pos == g.target:
            return True
        line = next((l for l in self.m.lines if math.dist(((l.v1[0] + l.v2[0]) / 2, (l.v1[1] + l.v2[1]) / 2), g.target) < 1), None)
        if line is None:
            return True
        (x1, y1), (x2, y2) = line.v1, line.v2
        dx, dy = x2 - x1, y2 - y1
        n2 = dx * dx + dy * dy or 1.0
        t = ((pos[0] - x1) * dx + (pos[1] - y1) * dy) / n2
        side = ((pos[0] - x1) * dy - (pos[1] - y1) * dx) * ((g.pos[0] - x1) * dy - (g.pos[1] - y1) * dx)
        return 0.0 <= t <= 1.0 and side > 0

    # -- text -----------------------------------------------------------------
    def describe(self, obs: Obs, last_cmd: str | None = None, health_override: float | None = None) -> str:
        hp = obs.health if health_override is None else health_override
        weapon = WEAPON_NAMES.get(obs.weapon, "fist")
        self._advance(obs)
        parts = [f"Health {hp:.0f}, armor {obs.armor:.0f}. Weapon: {weapon}; "
                 f"{obs.bullets:.0f} bullets, {obs.shells:.0f} shells"
                 + (f", {obs.rockets:.0f} rockets." if obs.rockets > 0 else ".")]
        if obs.weapon == 3:       # the shotgun fires once per 37 tics whatever is pressed: say when it is ready
            wait = max(0, SHOTGUN_REFIRE - (obs.tic - obs.shot_tic)) / 35
            parts[0] += " Shotgun ready." if wait < 0.05 else f" Shotgun reloading, ready in {wait:.1f} s."
        held = sorted(self.keys_held(obs))
        if held:
            parts[0] += f" Keys: {', '.join(held)}."

        vis = sorted((m for m in obs.monsters if m.visible), key=lambda m: obs.dist_to(m.x, m.y))[:3]
        shooters = []
        if vis:
            enemies = []
            floor, ceil = self._live_heights(obs)
            psec = self.grid.sector_at(obs.x, obs.y)
            eye = (floor[psec] if psec >= 0 else 0.0) + self.EYE
            for m in vis:
                d, b = obs.dist_to(m.x, m.y), obs.bearing_to(m.x, m.y)
                msec = self.grid.sector_at(m.x, m.y)
                # Seen is not shootable: a Baron on a ledge above the pit showed its legs, and every
                # shot hit the ledge face (E1M8). Line of fire: eye to the monster's chest.
                # Doom aims hitscan at whatever part of a monster is visible: a gunman or imp behind a step
                # with only its head showing is still hit (E1M4's stairs: an imp "out of the line of fire"
                # threw fireballs at the expert, which ignored it). The Baron keeps the chest-only test.
                heights = (self.CHEST,) if m.name == "BaronOfHell" else (self.CHEST, 48.0)
                if LINE_OF_FIRE and msec >= 0 and not any(
                        self._sight_clear((obs.x, obs.y), eye, (m.x, m.y), floor[msec] + h, floor, ceil) for h in heights):
                    enemies.append(f"{MONSTER_NAMES.get(m.name, m.name)} {d:.0f} away, {_side(b, ahead=0.5)}, out of the line of fire")
                    continue
                r = 24 if m.name == "BaronOfHell" else 16      # Baron's radius: wide enough to shoot, not re-aim (E1M8)
                tol = max(3.0, math.degrees(math.atan2(r, max(d, 1))))
                aim = "in the crosshair" if abs(b) <= tol else _side(b, ahead=0.5)   # exact angle for aiming
                if m.name in self.SHOOTERS and d <= self.SHOOTER_RANGE:
                    shooters.append(m)
                enemies.append(f"{MONSTER_NAMES.get(m.name, m.name)} {d:.0f} away, {aim}")
            parts.append("Enemies in view: " + "; ".join(enemies) + ".")
        else:
            parts.append("Enemies in view: none.")

        items = sorted((p for p in obs.pickups if p.visible), key=lambda p: obs.dist_to(p.x, p.y))[:2]
        if items:
            parts.append("Pickups in view: " + "; ".join(
                f"{PICKUP_NAMES[p.name]} {obs.dist_to(p.x, p.y):.0f} away, {_side(obs.bearing_to(p.x, p.y))}"
                + (", in a pit with no way back" if self._in_pit(obs, p.x, p.y) else "") for p in items) + ".")

        inc = self.incoming(obs)
        if inc:
            parts.append(f"Fireball incoming in {inc[0]:.1f} s, sidestep {inc[1]}.")
        bosses = [mo for mo in vis if mo.name in self.COVER_FROM]
        if bosses:
            cv = self.cover(obs, bosses[0])
            name = MONSTER_NAMES.get(bosses[0].name, "boss")
            parts.append(f"Cover from {name}: {cv[0]:.0f} units, {_side(cv[1])}." if cv else f"Cover from {name}: none near.")
            es = self.escape(obs, bosses)
            if es:
                parts.append(f"Escape from {name}: {es[0]:.0f} units open, {_side(es[1])}.")
        elif len(shooters) >= 2 or any(m.name in self.HITSCAN for m in shooters):
            # One gunman is reported too: hitscan cannot be dodged, only its line of sight broken (E1M3, E1M5:
            # most damage was taken standing in a shotgun guy's view while the shotgun reloaded).
            n = len(shooters)
            cv = self.cover(obs, shooters, reach=320)
            what = f"{n} shooters" if n > 1 else "1 shooter"
            parts.append(f"Cover from {what}: {cv[0]:.0f} units, {_side(cv[1])}." if cv
                         else f"Cover from {what}: none near.")

        rays, hazard = [], None
        for name, rel in RAYS:
            d, kind, nuk = self.body_ray(obs, rel)
            if kind in ("door (closed)", "exit switch", "lift (raised)") and d <= USE_RANGE:
                kind += ", within use reach"
            rays.append(f"{name} {d:.0f} {kind}" if kind != "open" else f"{name} open")
            if nuk is not None and rel in (0, 30, -30) and (hazard is None or nuk < hazard[0]):
                hazard = (nuk, name)
        parts.append("Obstacles: " + "; ".join(rays) + ".")
        # Strafes move sideways, where no obstacle ray looks: report a dead-end drop there too.
        side_drops = [f"{d:.0f} units {name}" for name, rel in (("90 left", 90), ("90 right", -90))
                      for d, kind, _ in [self.ray(obs, rel)] if kind == DEAD_DROP and d < 300]
        if side_drops:
            parts.append("Drop with no way back " + ", ".join(side_drops) + ".")
        # A barrel within its blast radius (128) of the player: any stray shot that hits it hurts the player
        # (E1M5, E1M6: gunmen's bullets set off barrels beside the expert, 68 health gone at once).
        barrels = [(obs.dist_to(so.x, so.y), so) for so in obs.solids if so.name == "ExplosiveBarrel"]
        if barrels:
            d, so = min(barrels, key=lambda b: b[0])
            if d < self.BARREL_NEAR:
                parts.append(f"Barrel {d:.0f} units away, {_side(obs.bearing_to(so.x, so.y))}.")
        psec = self.grid.sector_at(obs.x, obs.y)
        if psec in self.lift_floors:
            parts.append("Standing on a lift.")
        if psec >= 0 and self.m.special[psec] in DAMAGE_SECTORS:
            parts.append("Standing on toxic nukage.")       # the floor hurts every second spent on it
        elif hazard:
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
            # A short step moves ~80 units; none at all means something is in the way (E1M3: 25 short steps
            # into a wall at the exit room, standing in a shotgun guy's fire).
            blocked = (last_cmd in ("forward", "back", "strafe_left", "strafe_right") and moved < 10
                       or last_cmd == "forward_short" and moved < 3)
            self.history = (self.history + [(last_cmd + (" (blocked)" if blocked else ""), moved)])[-HISTORY:]
            if len(self.history) >= 2:
                parts.append("Recent commands, oldest first: " + ", ".join(c for c, _ in self.history)
                             + f"; moved {sum(m for _, m in self.history):.0f} units in total.")
            note = f"Last command: {last_cmd}, moved {moved:.0f} units"
            if blocked:
                note += " (blocked)"
            parts.append(note + (f", took {dmg:.0f} damage." if dmg > 0 else "."))
        self.prev = obs
        self.door_ceil_prev = {sec: obs.heights[sec][1] for sec in self.m.door_sectors if sec in obs.heights}
        self.doors_seen_open |= {sec for sec, (f, c) in obs.heights.items()
                                 if sec in self.m.door_sectors and c - f >= PLAYER_HEIGHT}
        self.lift_floor_prev = {sec: obs.heights[sec][0] for sec in self.lift_sectors_all if sec in obs.heights}
        return " ".join(parts)

    def reset(self) -> None:
        self.route, self.idx, self.prev, self.via = None, 0, None, None
        self.door_ceil_prev = {}
        self.doors_seen_open = set()
        self.lift_floor_prev = {}
        self.history = []
        self.stage = 0
        self.live_grids, self.live_check_tic, self.plan_fail_tic = {}, -10**9, -10**9
        self._job = None


def _plan_or_trigger(grid: NavGrid, a, b, max_expand: int):
    """A route on the stage grid, or else one over a walk-over line that opens a shut door (progression.py)."""
    path = grid.plan(a, b, max_expand)
    return (path, None) if path is not None else (trigger_route(grid, a, b, max_expand) or (None, None))


def _side(b: float, ahead: float = 7.5) -> str:
    """Angle in words. Within `ahead` degrees counts as straight ahead (default: half a 15 degree turn,
    fine for walking; enemies use an exact angle so the 5 degree turns can aim)."""
    if abs(b) < ahead:
        return "straight ahead"
    if abs(b) > 165:
        return "behind"
    return f"{abs(b):.0f} deg {'left' if b > 0 else 'right'}"


def _seg_point_dist(a, b, p) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / max(dx * dx + dy * dy, 1e-9)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)
