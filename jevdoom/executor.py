"""Plays one Jev command as a fixed macro.

The executor does not choose anything: it replays the macro exactly, and a new command
replaces the running one (moves last about one decision interval, so they rarely end first). When a macro ends before the next command arrives, the
player stands still.
"""
from __future__ import annotations

from .commands import FIRE_TICS, MOVE_TICS, SHORT_MOVE_TICS, SWITCH_TICS, TURN_DEG, TURN_FIRE_DEG, USE_TICS, WEAPON_CMD
from .env import B, BI, Obs, blank_action

import os

MAX_TURN_PER_TIC = 30.0
SHOT_WAIT = 40          # a fire command holds the trigger until its shot is out, at most this many tics
# A move/fire/use counts as applied after this many tics; the next decision's snapshot is taken then.
# Lower = overlap: the snapshot goes out while the previous command is still running, so the answer
# lands sooner (JEVDOOM_SETTLE_TICS, experiment 2026-10-18; 8 was the fix for the laptop's
# oscillation bug, which came from snapshots taken before a command had any effect).
SETTLE_TICS = int(os.environ.get("JEVDOOM_SETTLE_TICS", "5"))   # 5 adopted 2026-10-18: teacher at 0.1 s on E1M2 7/10 vs 5/10 at 8, 4/10 at 3
RAISE_TICS = 18         # Doom raises a weapon for ~15-19 tics after the slot changes (measured: first shot 15-19 tics later)
MOVE_BUTTON = {"forward": B.MOVE_FORWARD, "forward_short": B.MOVE_FORWARD, "back": B.MOVE_BACKWARD,
               "strafe_left": B.MOVE_LEFT, "strafe_right": B.MOVE_RIGHT}


class Executor:
    def __init__(self):
        self.cmd: str | None = None
        self.left = 0
        self.elapsed = 0
        self.turn_left_deg = 0.0
        self.ammo0 = None          # ammo when a fire command started; a drop means the shot went out
        self.shot = False
        self.want: int | None = None    # weapon Jev chose (Doom slot); None until the first tic adopts the best one
        self.target = 0                 # weapon the executor is pressing for this tic
        self.owned: tuple | None = None
        self.held = 0
        self.raising = 0                # tics left until the weapon in hand is up

    def start(self, cmd: str) -> None:
        self.cmd = cmd
        self.elapsed = 0
        self.ammo0, self.shot = None, False
        self.turn_left_deg = float(TURN_DEG.get(cmd, TURN_FIRE_DEG.get(cmd, 0)))
        if cmd in TURN_FIRE_DEG:
            self.left = SHOT_WAIT + 1                 # one tic to turn (at most 30 degrees), then fire
        elif cmd in TURN_DEG:
            self.left = int(abs(self.turn_left_deg) // MAX_TURN_PER_TIC) + 2
        elif cmd == "forward_short":
            self.left = SHORT_MOVE_TICS
        elif cmd in MOVE_BUTTON:
            self.left = MOVE_TICS
        elif cmd.startswith("fire_strafe"):
            self.left = MOVE_TICS
        elif cmd == "fire":
            self.left = SHOT_WAIT
        elif cmd in WEAPON_CMD:
            self.left = SWITCH_TICS
        else:
            self.left = USE_TICS

    @property
    def busy(self) -> bool:
        return self.left > 0

    @property
    def settled(self) -> bool:
        """True once the command's effect shows in the game: a turn is complete, or a
        move/fire/use has run SETTLE_TICS tics. The next decision should look after this."""
        if self.cmd is None or self.left <= 0:
            return True
        if self.cmd in WEAPON_CMD:      # done once the chosen (or fallback) weapon is up
            return self.held == self.target and self.raising == 0 and self.elapsed >= 1
        if self.cmd in TURN_DEG:
            return abs(self.turn_left_deg) < 1e-6 and self.elapsed >= 1
        if self.cmd in TURN_FIRE_DEG or self.cmd == "fire":
            # Hold the trigger until the shot is out: a fire replaced after 6 tics fell inside the
            # shotgun's 37-tic reload half the time (E1M8: 5 shells in 10 s against Barons).
            fist = self.ammo0 is not None and self.ammo0[0] == 1          # no ammo to watch
            return (abs(self.turn_left_deg) < 1e-6 and (self.shot or (fist and self.elapsed >= SETTLE_TICS))
                    or self.elapsed >= SHOT_WAIT)
        return self.elapsed >= SETTLE_TICS

    @staticmethod
    def _ammo(obs: Obs) -> float:
        return {3: obs.shells, 5: obs.rockets}.get(obs.weapon, obs.bullets if obs.weapon in (2, 4) else 0)

    def step(self, obs: Obs) -> list[float]:
        a = blank_action()
        if self.cmd in WEAPON_CMD and self.left > 0 and self._has(obs, WEAPON_CMD[self.cmd]):
            self.want = WEAPON_CMD[self.cmd]          # a weapon not carried or without ammo: the command does nothing
        self._weapon(obs, a)
        if self.ammo0 is None:
            self.ammo0 = (obs.weapon, self._ammo(obs))
        elif not self.shot and (obs.weapon != self.ammo0[0] or self._ammo(obs) < self.ammo0[1]):
            self.shot = obs.weapon == self.ammo0[0]
            self.ammo0 = (obs.weapon, self._ammo(obs))
        if self.left <= 0 or self.cmd is None:
            return a
        self.left -= 1
        self.elapsed += 1
        c = self.cmd
        if c in TURN_FIRE_DEG:
            if abs(self.turn_left_deg) > 1e-6:
                a[BI[B.TURN_LEFT_RIGHT_DELTA]] = -self.turn_left_deg
                self.turn_left_deg = 0.0
            else:
                a[BI[B.ATTACK]] = 1
        elif c in TURN_DEG:
            if abs(self.turn_left_deg) > 1e-6:
                d = max(-MAX_TURN_PER_TIC, min(MAX_TURN_PER_TIC, self.turn_left_deg))
                a[BI[B.TURN_LEFT_RIGHT_DELTA]] = -d       # positive delta turns right
                self.turn_left_deg -= d
        elif c in MOVE_BUTTON:
            a[BI[MOVE_BUTTON[c]]] = 1
            a[BI[B.SPEED]] = c != "forward_short"     # short step walks, for precision
        elif c == "fire":
            a[BI[B.ATTACK]] = 1
        elif c.startswith("fire_strafe"):
            a[BI[B.ATTACK]] = 1
            a[BI[B.MOVE_LEFT if c.endswith("left") else B.MOVE_RIGHT]] = 1
            a[BI[B.SPEED]] = 1
        elif c == "use":
            a[BI[B.USE]] = 1 if self.left % 2 == 1 else 0  # press/release so Doom registers it
        return a

    def switch_to(self, obs: Obs) -> tuple[int, float]:
        """(weapon being brought up, seconds until it can fire), or (0, 0) when the weapon in hand is ready.
        For perception: a switch lowers the old weapon (after a shotgun reload) and raises the new one."""
        if self.target and obs.weapon != self.target:
            return self.target, (RAISE_TICS + 16) / 35
        if self.raising and obs.weapon == self.held:
            return obs.weapon, self.raising / 35
        return 0, 0.0

    @staticmethod
    def _has(obs: Obs, w: int) -> bool:
        """Owns weapon w and has ammo for it."""
        return {1: True, 2: obs.bullets > 0, 3: obs.has_shotgun and obs.shells > 0,
                4: obs.has_chaingun and obs.bullets > 0, 5: obs.has_launcher and obs.rockets > 0}[w]

    def _weapon(self, obs: Obs, a) -> None:
        # Jev chooses the weapon (weapon_* commands). The executor keeps that choice, with three fixed rules:
        # - out of ammo (or never owned): the best weapon with ammo, shotgun > chaingun > pistol > fist, and that
        #   becomes the choice (as when Doom switches on an empty gun; no switch back when ammo turns up);
        # - a newly picked-up shotgun or chaingun becomes the choice (Doom switches to a new weapon on pickup);
        # - the rocket launcher only against a Baron while every visible enemy is far enough away that the blast
        #   cannot reach the player (switch to it at 300+ units, keep it down to 200), never with nothing in view;
        #   afterwards back to the choice. Barons only: a rocket bursts on the first thing it touches, and against
        #   imps and gunmen in cluttered rooms it went off on a pillar, a door edge or a barrel beside the player
        #   (E1M7's start: 4 of 16 runs died at 37-60 health the moment a rocket left; E1M6: 6 deaths).
        # Until 2026-10-20 the choice itself was a fixed rule (always the best weapon with ammo).
        if obs.weapon != self.held:
            self.held, self.raising = obs.weapon, RAISE_TICS if self.held else 0
        elif self.raising:
            self.raising -= 1
        owned = (obs.has_shotgun, obs.has_chaingun)
        if self.owned is not None:
            if owned[0] and not self.owned[0]:
                self.want = 3
            elif owned[1] and not self.owned[1]:
                self.want = 4
        self.owned = owned
        if self.want is None or not self._has(obs, self.want):
            self.want = next(w for w in (3, 4, 2, 1) if self._has(obs, w))
        near = min((obs.dist_to(m.x, m.y) for m in obs.monsters if m.visible), default=None)
        baron = any(m.visible and m.name == "BaronOfHell" for m in obs.monsters)
        safe = baron and near is not None and near >= (200 if obs.weapon == 5 else 300)
        self.target = 5 if self._has(obs, 5) and safe else self.want
        if obs.weapon != self.target:
            a[BI[getattr(B, f"SELECT_WEAPON{self.target}")]] = 1
