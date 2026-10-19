"""Supply needs during a map: a temporary "Route to <item>" subgoal when a resource runs low.

The planner's subgoals are fixed before the game; what the player carries is not. When health or
the ammo for the weapons held drops below a threshold and a matching item still lies in the map on
a short walk from the player, perception swaps the compass target to that item until it is picked up
(or gone, or the detour takes too long), then restores the stage route. In a campaign (carry=True)
the exit stage also tops up: whatever is left at the exit is what the next map starts with.

Facts and thresholds only; the teacher follows the compass as usual. Path searches run on a worker
thread, never inside describe().
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

from .env import Obs
from .progression import Subgoal
from .wadmap import DAMAGE_SECTORS, path_length

HEALTH = {"Stimpack": 10, "Medikit": 25}
ARMOR = {"GreenArmor": 100, "BlueArmor": 200}
SHELLS = {"Shell": 4, "ShellBox": 20, "Shotgun": 8}
BULLETS = {"Clip": 10, "ClipBox": 50, "Chaingun": 20}
LABEL = {"Stimpack": "stimpack", "Medikit": "medikit", "GreenArmor": "green armor", "BlueArmor": "blue armor",
         "Shell": "shells", "ShellBox": "box of shells", "Shotgun": "shotgun", "Clip": "ammo clip",
         "ClipBox": "box of bullets", "Chaingun": "chaingun"}

RETRY_TICS = 175          # an item too far by path is looked at again after 5 s
READY_TICS = 105          # a planned detour stays valid for 3 s
CHECK_TICS = 17           # look for a need at most twice a second
TOPUP_WALK = 450.0        # walk to a top-up item of modest worth (medikit at 85 health, box of bullets)
TOPUP_WALK_FAR = 700.0    # medikit with 25+ health missing, box of shells under 30 shells
TOPUP_ARMOR_WALK = 1000.0 # armor for the next map
TOPUP_CLEAR = 700         # a top-up item with a living monster this close is left alone (E1M1 nukage room imps)
MONSTER_CLEAR = 350       # skip items with a living monster this close (a detour into an uncleared room)
TOPUP_LEFT = 2500.0       # route units left to the exit when the top-up starts (was 1200: E1M3 seed 52
                          # left at 46 health past two medikits ~850 units off the last stretch)
SPEED = 280.0             # units per second the teacher covers on a route, for the detour deadline

# Planned resource stages (progression.with_resource_detours, Subgoal.need): still worth the walk? A stage
# that is not (armor already on, health already full) or that has run past its deadline is skipped, with
# the key or trigger stages planned for it.
STAGE_NEEDED = {"green armor": lambda o: o.armor < 60, "blue armor": lambda o: o.armor < 150,
                "soulsphere": lambda o: o.health < 180, "medikit": lambda o: o.health <= 80,
                "stimpack": lambda o: o.health <= 90, "box of shells": lambda o: o.shells <= 30,
                "box of bullets": lambda o: o.bullets <= 150, "shotgun": lambda o: not o.has_shotgun,
                "chaingun": lambda o: not o.has_chaingun, "rocket launcher": lambda o: not o.has_launcher}
STAGE_SLACK = 8.0         # seconds on top of twice the stage route's running time before a planned stage is dropped


def wanted(obs: Obs, exit_stage: bool, carry: bool) -> list[tuple[set[str], float, bool]]:
    """(item names, max walk in units, top-up?) per resource that is low now. Mid-map: health below 40, or
    out of ammo for every weapon. Campaign exit stretch: top up what the next map needs, walks in proportion
    to the item's worth (a medikit with 40 health missing is worth 700 units, a stimpack 300).
    Not clips on the way: the pistol is a fallback, and E1M1's clip detours walked the teacher into the
    barrels beside the nukage room (wedged, then killed by an imp, 2 of 20 runs)."""
    out = []
    shells = obs.shells if obs.has_shotgun else 0
    if obs.health < 40:
        out.append((set(HEALTH), 450.0 if obs.health < 25 else 350.0, False))
    if obs.has_shotgun and shells < 6:     # shotgun nearly dry: shells on a short walk (E1M5 deaths out of shells)
        out.append(({"Shell", "ShellBox"}, 300.0, False))
    if shells < 2 and obs.bullets < 15:
        out.append(((set(SHELLS) if obs.has_shotgun else {"Shotgun"}) | set(BULLETS), 400.0, False))
    if carry and exit_stage:
        if obs.health <= 75:
            out.append(({"Medikit"}, TOPUP_WALK_FAR, True))
        if obs.health <= 85:
            out.append(({"Medikit"}, TOPUP_WALK, True))
        if obs.health <= 90:
            out.append(({"Stimpack"}, 300.0, True))
        if obs.armor < 50:
            out.append(({"GreenArmor", "BlueArmor"}, TOPUP_ARMOR_WALK, True))
        elif obs.armor < 120:
            out.append(({"BlueArmor"}, TOPUP_ARMOR_WALK, True))
        if obs.has_shotgun and obs.shells < 30:
            out.append(({"ShellBox"}, TOPUP_WALK_FAR, True))
            out.append(({"Shell"}, 250.0, True))
        if obs.has_chaingun and obs.bullets < 100:
            out.append(({"ClipBox"}, TOPUP_WALK, True))
    return out


class Needs:
    def __init__(self, carry: bool = False, max_expand: int = 15_000):
        self.carry, self.max_expand = carry, max_expand
        self.goal: Subgoal | None = None
        self.item_id: int | None = None
        self.deadline = 0
        self.saved = None                     # stage route (route, idx, via) to restore afterwards
        self.check_tic = -10**9
        self.skip: dict[int, int] = {}        # item id -> tic until which it is not considered
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._job = None                      # (future, thing, limit, stage)
        self.ready = None                     # (thing, path, tic, stage) planned, waiting for the fight to end
        self.log: list[tuple] = []            # (tic, event, item) for summaries
        self.opt = (-1, 0, 0)                 # (planned optional stage, tic by which it must be done, last tic seen)
        self._after: list[float] | None = None  # route units of the stages after each stage

    def step(self, perc, obs: Obs) -> bool:
        """Start, keep or finish a need. True while a need owns the compass."""
        if self.goal is not None:
            alive = any(p.id == self.item_id for p in obs.pickups)
            if alive and obs.dist_to(*self.goal.pos) >= 40 and obs.tic < self.deadline:
                return True
            if alive:                             # timed out, or reached and not taken (not needed)
                self.skip[self.item_id] = 10**9
            self.log.append((obs.tic, "taken" if not alive else "timeout" if obs.tic >= self.deadline else "arrived",
                             self.goal.label))
            self._finish(perc)
            return False
        self._skip_optional(perc, obs)
        stage = perc.stage
        if self._after is None:
            lens = [path_length(r) if r else 0.0 for r in perc.stage_routes] or [0.0]
            self._after = [sum(lens[i + 1:]) for i in range(len(lens))]
        # top-up window: the last TOPUP_LEFT route units before the exit, over every stage left (a one-stage
        # map like E1M1 is all exit stage). Not during a planned resource stage: its route is a detour already.
        exit_stage = (perc.route is not None and not perc.goals[min(stage, len(perc.goals) - 1)].need
                      and path_length(perc.route[perc.idx:]) + self._after[min(stage, len(self._after) - 1)] <= TOPUP_LEFT)
        fighting = any(m.visible for m in obs.monsters)
        if self._job is not None:
            fut, thing, limit, jstage = self._job
            if not fut.done():
                return False
            self._job = None
            path = fut.result()
            if (jstage == stage and path and path_length(path) <= limit and math.dist(path[-1], (thing.x, thing.y)) < 40
                    and not self._hazard(perc, path)):
                self.ready = (thing, path, obs.tic, jstage)
            else:
                self.skip[thing.id] = obs.tic + RETRY_TICS     # too far from here: maybe later
        if self.ready is not None:
            # A planned detour starts once no enemy is in view (planned during a fight, it is ready the moment
            # the fight ends: E1M2's exit room fight took 86 -> 58 health right beside the exit switch).
            thing, path, t0, rstage = self.ready
            if (obs.tic - t0 > READY_TICS or rstage != stage or math.dist(path[0], (obs.x, obs.y)) > 128
                    or not any(p.id == thing.id for p in obs.pickups)):
                self.ready = None
            elif not fighting:
                self.ready = None
                self._start(perc, obs, thing, path)
                return True
        if obs.tic - self.check_tic < CHECK_TICS or self.ready is not None:
            return False
        self.check_tic = obs.tic
        best = None
        for names, limit, topup in wanted(obs, exit_stage, self.carry):
            clear = TOPUP_CLEAR if topup else MONSTER_CLEAR
            for p in obs.pickups:
                if p.name not in names or self.skip.get(p.id, -1) > obs.tic:
                    continue
                d = obs.dist_to(p.x, p.y)
                if d > limit or any(math.hypot(m.x - p.x, m.y - p.y) < clear for m in obs.monsters):
                    continue
                if best is None or d < best[0]:
                    best = (d, p, limit)
        if best is None:
            return False
        _, thing, limit = best
        grid = perc._stage_grid(obs)
        r, c = grid.to_cell(thing.x, thing.y)
        if (not (0 <= r < grid.h and 0 <= c < grid.w) or grid.near_drop[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3].any()
                or perc._in_pit(obs, thing.x, thing.y)):
            self.skip[thing.id] = 10**9           # beside a ledge or down a pit (E1M3's toxic walkway)
            return False
        self._job = (self._pool.submit(grid.plan, (obs.x, obs.y), (thing.x, thing.y), self.max_expand), thing, limit, stage)
        return False

    def _skip_optional(self, perc, obs: Obs) -> None:
        """Drop the current planned resource stage (and the rest of its group) when the item is gone, no
        longer needed, or the stage has run past its deadline."""
        while perc.stage < len(perc.goals) - 1 and perc.goals[perc.stage].need:
            need, g = perc.goals[perc.stage].need, perc.goals[perc.stage]
            if self.opt[0] != perc.stage:
                # the stage route starts where the previous stage ended, which may have been skipped
                r = perc.stage_routes[perc.stage] if perc.stage < len(perc.stage_routes) else None
                walk = max(path_length(r) if r else 0.0, 1.4 * obs.dist_to(*g.pos))
                self.opt = (perc.stage, obs.tic + int(35 * (2.0 * walk / SPEED + STAGE_SLACK)), obs.tic)
            elif any(m.visible for m in obs.monsters):     # fighting: the clock waits
                self.opt = (self.opt[0], self.opt[1] + obs.tic - self.opt[2], obs.tic)
            else:
                self.opt = (self.opt[0], self.opt[1], obs.tic)
            item = next((g for g in perc.goals[perc.stage:] if g.need == need and g.kind == "item"), None)
            names = {k for k, v in LABEL.items() if item is not None and v == item.label}
            gone = bool(names) and not any(p.name in names and math.hypot(p.x - item.pos[0], p.y - item.pos[1]) < 32
                                           for p in obs.pickups)
            needed = item is not None and not gone and STAGE_NEEDED.get(item.label, lambda o: True)(obs)
            if needed and obs.tic < self.opt[1]:
                return
            why = "gone" if gone else "late" if needed else "not needed"
            while perc.stage < len(perc.goals) - 1 and perc.goals[perc.stage].need == need:
                perc.stage += 1
            self.log.append((obs.tic, f"skip ({why})", item.label if item else need))
            r = perc.stage_routes[perc.stage] if perc.stage < len(perc.stage_routes) else None
            perc.route, perc.idx, perc.via, perc._job, perc.replan = (list(r) if r else None), 0, None, None, True

    @staticmethod
    def _hazard(perc, path) -> bool:
        grid = perc.grid
        secs = {grid.sector_at(*p) for p in path} - {-1}
        return any(int(grid.m.special[s]) in DAMAGE_SECTORS for s in secs)     # over a damaging floor

    def _start(self, perc, obs: Obs, thing, path) -> None:
        self.goal = Subgoal("item", LABEL[thing.name], (thing.x, thing.y), (thing.x, thing.y))
        self.item_id = thing.id
        self.deadline = obs.tic + int(35 * (2.0 * path_length(path) / SPEED + 4.0))
        self.saved = (perc.route, perc.idx, perc.via)
        perc.route, perc.idx, perc.via, perc._job = list(path), 0, None, None
        self.log.append((obs.tic, "start", self.goal.label))

    def _finish(self, perc) -> None:
        self.goal, self.item_id = None, None
        if self.saved is not None:
            perc.route, perc.idx, perc.via = self.saved
        perc._job, perc.replan, self.saved = None, True, None
