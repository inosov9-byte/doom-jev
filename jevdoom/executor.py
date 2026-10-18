"""Plays one Jev command as a fixed macro.

The executor does not choose anything: it replays the macro exactly, and a new command
replaces the running one (moves last about one decision interval, so they rarely end first). When a macro ends before the next command arrives, the
player stands still.
"""
from __future__ import annotations

from .commands import FIRE_TICS, MOVE_TICS, SHORT_MOVE_TICS, TURN_DEG, TURN_FIRE_DEG, USE_TICS
from .env import B, BI, Obs, blank_action

import os

MAX_TURN_PER_TIC = 30.0
# A move/fire/use counts as applied after this many tics; the next decision's snapshot is taken then.
# Lower = overlap: the snapshot goes out while the previous command is still running, so the answer
# lands sooner (JEVDOOM_SETTLE_TICS, experiment 2026-10-18; 8 was the fix for the laptop's
# oscillation bug, which came from snapshots taken before a command had any effect).
SETTLE_TICS = int(os.environ.get("JEVDOOM_SETTLE_TICS", "5"))   # 5 adopted 2026-10-18: teacher at 0.1 s on E1M2 7/10 vs 5/10 at 8, 4/10 at 3
MOVE_BUTTON = {"forward": B.MOVE_FORWARD, "forward_short": B.MOVE_FORWARD, "back": B.MOVE_BACKWARD,
               "strafe_left": B.MOVE_LEFT, "strafe_right": B.MOVE_RIGHT}


class Executor:
    def __init__(self):
        self.cmd: str | None = None
        self.left = 0
        self.elapsed = 0
        self.turn_left_deg = 0.0

    def start(self, cmd: str) -> None:
        self.cmd = cmd
        self.elapsed = 0
        self.turn_left_deg = float(TURN_DEG.get(cmd, TURN_FIRE_DEG.get(cmd, 0)))
        if cmd in TURN_FIRE_DEG:
            self.left = FIRE_TICS + 1                 # one tic to turn (at most 30 degrees), then fire
        elif cmd in TURN_DEG:
            self.left = int(abs(self.turn_left_deg) // MAX_TURN_PER_TIC) + 2
        elif cmd == "forward_short":
            self.left = SHORT_MOVE_TICS
        elif cmd in MOVE_BUTTON:
            self.left = MOVE_TICS
        elif cmd.startswith("fire_strafe"):
            self.left = MOVE_TICS
        elif cmd == "fire":
            self.left = FIRE_TICS
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
        if self.cmd in TURN_DEG:
            return abs(self.turn_left_deg) < 1e-6 and self.elapsed >= 1
        if self.cmd in TURN_FIRE_DEG:      # turned and the first shot is out: look again at once
            return abs(self.turn_left_deg) < 1e-6 and self.elapsed >= 2
        return self.elapsed >= SETTLE_TICS

    def step(self, obs: Obs) -> list[float]:
        a = blank_action()
        self._weapon(obs, a)
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

    @staticmethod
    def _weapon(obs: Obs, a) -> None:
        # A fixed rule, not a decision: the best weapon with ammo. The rocket launcher only while
        # every visible enemy is far enough away that the blast cannot reach the player
        # (switch to it at 300+ units, keep it down to 200), never with nothing in view.
        near = min((obs.dist_to(m.x, m.y) for m in obs.monsters if m.visible), default=None)
        safe = near is not None and near >= (200 if obs.weapon == 5 else 300)
        if obs.has_launcher and obs.rockets > 0 and safe:
            want = 5
        elif obs.has_shotgun and obs.shells > 0:
            want = 3
        elif obs.has_chaingun and obs.bullets > 0:
            want = 4
        elif obs.bullets > 0:
            want = 2
        else:
            want = 1
        if obs.weapon != want:
            a[BI[getattr(B, f"SELECT_WEAPON{want}")]] = 1
