"""Jev's command set. Labels and the descriptions the teacher model sees.

No game imports here, so training code can use it without ViZDoom.
"""
from __future__ import annotations


# id -> description shown to the teacher model
COMMANDS: dict[str, str] = {
    "forward": "run forward about 100 units",
    "back": "step backward about 100 units",
    "turn_left": "turn left 15 degrees",
    "turn_right": "turn right 15 degrees",
    "turn_left_big": "turn left 60 degrees",
    "turn_right_big": "turn right 60 degrees",
    "strafe_left": "sidestep left about 100 units",
    "strafe_right": "sidestep right about 100 units",
    "fire": "shoot straight ahead at whatever is in the crosshair",
    "use": "press use: opens a door or flips a switch directly in front, within 60 units",
}
COMMAND_IDS = list(COMMANDS)
