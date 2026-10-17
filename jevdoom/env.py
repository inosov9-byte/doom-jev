"""ViZDoom wrapper: game setup, compact per-tic observation, teleport helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import vizdoom as vzd

B = vzd.Button
V = vzd.GameVariable

BUTTONS = [B.ATTACK, B.USE, B.MOVE_FORWARD, B.MOVE_BACKWARD, B.MOVE_LEFT, B.MOVE_RIGHT,
           B.SPEED, B.TURN_LEFT_RIGHT_DELTA, B.SELECT_WEAPON1, B.SELECT_WEAPON2, B.SELECT_WEAPON3]
BI = {b: i for i, b in enumerate(BUTTONS)}

VARS = [V.POSITION_X, V.POSITION_Y, V.POSITION_Z, V.ANGLE, V.HEALTH, V.ARMOR, V.DEAD,
        V.SELECTED_WEAPON, V.AMMO2, V.AMMO3, V.WEAPON2, V.WEAPON3,
        V.KILLCOUNT, V.DAMAGE_TAKEN]

MONSTER_NAMES = {"Zombieman": "zombieman", "ShotgunGuy": "shotgun guy", "DoomImp": "imp",
                 "Demon": "demon", "Spectre": "spectre", "LostSoul": "lost soul",
                 "Cacodemon": "cacodemon", "BaronOfHell": "baron"}
PICKUP_NAMES = {"Stimpack": "stimpack", "Medikit": "medikit", "HealthBonus": "health bonus",
                "ArmorBonus": "armor bonus", "GreenArmor": "green armor", "BlueArmor": "blue armor",
                "Clip": "ammo clip", "ClipBox": "box of bullets", "Shell": "shells", "ShellBox": "box of shells",
                "Shotgun": "shotgun", "Chaingun": "chaingun",
                "RedCard": "red keycard", "BlueCard": "blue keycard", "YellowCard": "yellow keycard",
                "RedSkull": "red skull key", "BlueSkull": "blue skull key", "YellowSkull": "yellow skull key"}
KEY_OBJECTS = {"RedCard": "red", "RedSkull": "red", "BlueCard": "blue", "BlueSkull": "blue",
               "YellowCard": "yellow", "YellowSkull": "yellow"}
WEAPON_NAMES = {1: "fist", 2: "pistol", 3: "shotgun"}


@dataclass
class Thing:
    id: int
    name: str
    category: str
    x: float
    y: float
    visible: bool = False


@dataclass
class Obs:
    tic: int
    x: float
    y: float
    angle: float
    health: float
    armor: float
    dead: bool
    weapon: int
    bullets: float
    shells: float
    has_shotgun: bool
    kills: int
    damage_taken: float
    sector: int = -1
    monsters: list[Thing] = field(default_factory=list)   # alive monsters; visible flag from labels buffer
    pickups: list[Thing] = field(default_factory=list)
    heights: dict[int, tuple[float, float]] = field(default_factory=dict)  # live floor/ceiling per sector
    keys_present: set[str] = field(default_factory=set)   # key colours still lying in the map
    screen: np.ndarray | None = None

    def bearing_to(self, tx: float, ty: float) -> float:
        """Signed degrees from view direction to target. Positive = target is to the left."""
        b = math.degrees(math.atan2(ty - self.y, tx - self.x))
        return (b - self.angle + 180) % 360 - 180

    def dist_to(self, tx: float, ty: float) -> float:
        return math.hypot(tx - self.x, ty - self.y)


def make_game(wad: str, map_name: str = "E1M1", skill: int = 3, visible: bool = True,
              resolution: str = "RES_640X480", cheats: bool = False, timeout_s: int = 900,
              sound: bool = False, seed: int | None = None) -> vzd.DoomGame:
    g = vzd.DoomGame()
    g.set_doom_game_path(wad)
    g.set_doom_map(map_name)
    g.set_doom_skill(skill)
    g.set_window_visible(visible)
    g.set_mode(vzd.Mode.PLAYER)
    g.set_screen_resolution(getattr(vzd.ScreenResolution, resolution))
    g.set_screen_format(vzd.ScreenFormat.BGR24)
    g.set_render_hud(True)
    g.set_render_crosshair(True)
    g.set_labels_buffer_enabled(True)
    g.set_objects_info_enabled(True)
    g.set_sectors_info_enabled(True)
    g.set_available_buttons(BUTTONS)
    g.set_button_max_value(B.TURN_LEFT_RIGHT_DELTA, 180)
    g.set_available_game_variables(VARS)
    g.set_episode_timeout(timeout_s * 35)
    g.set_sound_enabled(sound)
    if cheats:
        g.add_game_args("+sv_cheats 1")
    if seed is not None:
        g.set_seed(seed)                  # fixes the game's RNG (monster behaviour, damage rolls)
    g.init()
    return g


def observe(g: vzd.DoomGame, tic: int) -> Obs | None:
    s = g.get_state()
    if s is None:
        return None
    v = s.game_variables
    obs = Obs(tic=tic, x=v[0], y=v[1], angle=v[3], health=v[4], armor=v[5], dead=bool(v[6]),
              weapon=int(v[7]), bullets=v[8], shells=v[9], has_shotgun=bool(v[11]),
              kills=int(v[12]), damage_taken=v[13], screen=s.screen_buffer)
    visible = {l.object_id for l in s.labels}
    for o in s.objects:
        if o.category == "Self":
            obs.sector = o.sector_id
        elif o.category == "Monster":
            obs.monsters.append(Thing(o.id, o.name, o.category, o.position_x, o.position_y, o.id in visible))
        elif o.name in PICKUP_NAMES:
            if o.name in KEY_OBJECTS:
                obs.keys_present.add(KEY_OBJECTS[o.name])
            obs.pickups.append(Thing(o.id, o.name, o.category, o.position_x, o.position_y, o.id in visible))
    obs.heights = {sec.id: (sec.floor_height, sec.ceiling_height) for sec in s.sectors}
    return obs


def blank_action() -> list[float]:
    return [0.0] * len(BUTTONS)


def teleport(g: vzd.DoomGame, x: float, y: float, angle: float) -> None:
    """Cheat-warp the player and rotate to an absolute angle (needs cheats=True)."""
    g.send_game_command(f"warp {x:.0f} {y:.0f}")
    g.make_action(blank_action(), 1)
    for _ in range(6):
        cur = g.get_state().game_variables[3]
        err = (angle - cur + 180) % 360 - 180
        if abs(err) < 1:
            break
        a = blank_action()
        a[BI[B.TURN_LEFT_RIGHT_DELTA]] = -err      # positive delta turns right
        g.make_action(a, 1)
