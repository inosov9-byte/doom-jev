"""Plays one Jev command as a short, fixed macro (about 0.3 s of game time).

The executor does not choose anything: it replays the macro exactly, then the player
stands still until Jev's next command arrives.
"""
from __future__ import annotations

from .env import B, BI, Obs, blank_action

TICS = 10        # length of every macro in game tics (35 tics = 1 s)
TURN_DEG = {"turn_left": 15, "turn_right": -15, "turn_left_big": 60, "turn_right_big": -60}
MOVE = {"forward": B.MOVE_FORWARD, "back": B.MOVE_BACKWARD,
        "strafe_left": B.MOVE_LEFT, "strafe_right": B.MOVE_RIGHT}


class Executor:
    """Plays one command over TICS tics. Weapon auto-switch is the only non-command input."""

    def __init__(self):
        self.cmd: str | None = None
        self.left = 0
        self.turn_left_deg = 0.0

    def start(self, cmd: str) -> None:
        self.cmd = cmd
        self.left = TICS
        self.turn_left_deg = float(TURN_DEG.get(cmd, 0))

    @property
    def busy(self) -> bool:
        return self.left > 0

    def step(self, obs: Obs) -> list[float]:
        a = blank_action()
        self._weapon(obs, a)
        if self.left <= 0 or self.cmd is None:
            return a
        self.left -= 1
        c = self.cmd
        if c in TURN_DEG:
            per_tic = TURN_DEG[c] / 3                     # complete the turn in the first 3 tics
            if abs(self.turn_left_deg) > 1e-6:
                d = per_tic if abs(per_tic) <= abs(self.turn_left_deg) else self.turn_left_deg
                a[BI[B.TURN_LEFT_RIGHT_DELTA]] = -d       # positive delta turns right
                self.turn_left_deg -= d
        elif c in MOVE:
            a[BI[MOVE[c]]] = 1
            a[BI[B.SPEED]] = 1
        elif c == "fire":
            a[BI[B.ATTACK]] = 1
        elif c == "use":
            a[BI[B.USE]] = 1 if self.left % 2 == 1 else 0  # press/release so Doom registers it
        return a

    @staticmethod
    def _weapon(obs: Obs, a) -> None:
        # Shotgun when it has shells, else pistol, else fist: a fixed rule, not a decision.
        if obs.has_shotgun and obs.shells > 0:
            want = 3
        elif obs.bullets > 0:
            want = 2
        else:
            want = 1
        if obs.weapon != want:
            a[BI[{1: B.SELECT_WEAPON1, 2: B.SELECT_WEAPON2, 3: B.SELECT_WEAPON3}[want]]] = 1
