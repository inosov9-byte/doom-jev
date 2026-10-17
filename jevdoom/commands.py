"""Jev's command set. Labels and the descriptions the teacher model sees.

Jev picks labels and cannot output numbers, so turn amounts live in the label
(15/45/90 degrees each side, or 180). Turns are exact: Doom turning has no momentum.
Moves have one length, about one decision interval (20 tics, 0.57 s). A new command
cancels whatever is still running, so the player keeps moving between decisions.
Distances below were measured in E1M1 with run held, including the slide after release.

No game imports here, so training code can use it without ViZDoom.
"""
from __future__ import annotations

# id -> description shown to the teacher model (None: the id says it all)
COMMANDS: dict[str, str | None] = {
    "forward": "run forward about 300 units",
    "back": "back up about 300 units",
    "strafe_left": "sidestep left about 250 units",
    "strafe_right": "sidestep right about 250 units",
    "turn_left_15": None,
    "turn_left_45": None,
    "turn_left_90": None,
    "turn_right_15": None,
    "turn_right_45": None,
    "turn_right_90": None,
    "turn_around": "turn 180 degrees",
    "fire": "shoot at whatever is in the crosshair",
    "use": "open a door or press a switch within use reach straight ahead",
}
COMMAND_IDS = list(COMMANDS)

# Execution parameters (degrees: positive = left; tics at 35 per second)
TURN_DEG = {"turn_left_15": 15, "turn_left_45": 45, "turn_left_90": 90,
            "turn_right_15": -15, "turn_right_45": -45, "turn_right_90": -90, "turn_around": 180}
MOVE_TICS = 20
FIRE_TICS = 10
USE_TICS = 6
