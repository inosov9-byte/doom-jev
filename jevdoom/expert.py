"""Rule-based expert: reads the same state text as Jev and Haiku and picks one of the 20 commands.

Used as the teacher for distillation (instant, free, consistent labels for any state,
including states the student reaches on its own) and as a reference arm in live runs.
It never runs inside a Jev or Haiku arm.

Priorities: act on the current goal up close (exit or other switch, key, trigger line) > fight close enemies (aim, shoot, dodge when hit; skipped
near the exit or at low health, where running is safer) >
face an attacker shooting from out of view > pick up ammo when out / health when low >
open a closed door on the way > turn toward the route > get unstuck > walk forward.
"""
from __future__ import annotations

import re

STEPS = (5, 15, 30, 45, 90)
USE_REACH = 64
ENGAGE = 250          # fight only close enemies; otherwise keep running the route (lag-robust: a moving player is hard to hit)
FIST = 90
AMMO_ITEMS = ("ammo clip", "box of bullets", "shells", "box of shells", "shotgun")
HEALTH_ITEMS = ("stimpack", "medikit")      # a health bonus (+1) is not worth a detour
MELEE = ("demon", "spectre")   # bite only: keep them from reaching the player
MELEE_ENGAGE = 450             # demons are fought from this far: they come to the player anyway
BOSSES = ("baron",)            # fought at any range when in view
BOSS_NEAR, BOSS_FAR = 330, 520
AIM_LEAD = 0.35                # seconds before the shotgun is ready: stop dodging and aim
RELOAD_DODGE = 0.2             # seconds of shotgun reload worth a sidestep instead of a fire  # Baron kiting band
COVER_HEALTH = 70             # below this, a hit from a boss sends the expert behind cover
BARREL_BLAST = 160             # barrel explosion radius is 128; never shoot one closer than this
KILL_GOALS = ("baron of hell",)
DEAD_DROP = "drop with no way back"
# How far a move can carry the player, slide included (a run builds momentum that slides ~60-150 units
# after the next command replaces it), and the drop rays it must keep clear of: a strafe goes 90
# degrees sideways, which the 60 degree ray sees at a slant.
# A run also keeps sliding the old way after the next turn (Doom momentum): near a drop on either side
# the expert walks (E1M3 seed 1: ran along the walkway, turned at the bend, slid into the pool).
MOVE_REACH = {"forward": (("ahead", 150), ("30 left", 120), ("30 right", 120), ("60 left", 90), ("60 right", 90),
                          ("90 left", 70), ("90 right", 70)), "forward_short": (("ahead", 90),), "back": (("behind", 200),),
              **{f"{pre}strafe_{side}": ((f"90 {side}", 180), (f"60 {side}", 200))
                 for pre in ("", "fire_") for side in ("left", "right")}}


def _angle(text: str) -> float | None:
    """'20 deg left' -> +20, '20 deg right' -> -20, 'straight ahead' -> 0, 'behind' -> 180."""
    text = text.strip()
    if text.startswith("straight ahead") or text.startswith("in the crosshair"):
        return 0.0
    if text.startswith("behind"):
        return 180.0
    m = re.match(r"(\d+) deg (left|right)", text)
    if not m:
        return None
    return float(m.group(1)) * (1 if m.group(2) == "left" else -1)


def parse_state(s: str) -> dict:
    f: dict = {"enemies": [], "pickups": [], "rays": {}, "route": None, "route_left": None, "exit_near": None,
               "attacker": None, "blocked": False, "last": None, "damage": 0, "health": 100,
               "bullets": 0, "shells": 0, "weapon": "pistol"}
    m = re.search(r"Health (\d+), armor (\d+)", s)
    f["health"], f["armor"] = (int(m.group(1)), int(m.group(2))) if m else (100, 0)
    m = re.search(r"Weapon: (\w+); (\d+) bullets, (\d+) shells", s)
    if m:
        f["weapon"], f["bullets"], f["shells"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"Enemies in view: ([^.]*)\.", s)
    if m and m.group(1) != "none":
        for part in m.group(1).split(";"):
            e = re.match(r"\s*([a-z ]+?) (\d+) away, (.+)", part)
            if e and "out of the line of fire" not in e.group(3):     # cannot be shot from here: not a target
                f["enemies"].append({"kind": e.group(1), "dist": int(e.group(2)), "angle": _angle(e.group(3)),
                                     "crosshair": e.group(3).startswith("in the crosshair")})
    m = re.search(r"Pickups in view: ([^.]*)\.", s)
    if m:
        for part in m.group(1).split(";"):
            p = re.match(r"\s*([a-z ]+?) (\d+) away, (.+)", part)
            if p:
                if "in a pit with no way back" not in p.group(3):     # cannot be fetched and survived
                    f["pickups"].append({"name": p.group(1), "dist": int(p.group(2)), "angle": _angle(p.group(3))})
    m = re.search(r"Obstacles: ([^.]*(?:\.\d[^.]*)*)\.", s)
    if m:
        for part in m.group(1).split(";"):
            r = re.match(r"\s*(ahead|behind|\d+ left|\d+ right) (open|(\d+) (.+))", part)
            if r:
                f["rays"][r.group(1)] = (1500.0, "open") if r.group(2) == "open" else (float(r.group(3)), r.group(4))
    f["drops"] = {k: d for k, (d, kind) in f["rays"].items() if kind.startswith(DEAD_DROP)}
    for d, side in re.findall(r"(\d+) units (90 left|90 right)", (re.search(r"Drop with no way back ([^.]*)\.", s) or [""])[0]):
        f["drops"][side] = float(d)
    m = re.search(r"Shotgun reloading, ready in ([\d.]+) s", s)
    f["reload"] = float(m.group(1)) if m else 0.0
    m = re.search(r"Fireball incoming in ([\d.]+) s, sidestep (left|right)\.", s)
    f["incoming"] = (float(m.group(1)), m.group(2)) if m else None
    m = re.search(r"Escape from [a-z ]+: (\d+) units open, ([^.]+)\.", s)
    f["escape"] = (float(m.group(1)), _angle(m.group(2))) if m else None
    m = re.search(r"Cover from [a-z ]+: (\d+) units, ([^.]+)\.", s)
    f["cover"] = (float(m.group(1)), _angle(m.group(2))) if m else None
    f["goal"] = "exit"
    m = re.search(r"Route to ([a-z ]+): ([a-z ]+?) (\d+) away, ([^.]+)\.", s)
    if m:
        f["goal"] = m.group(1)
        f["exit_near"] = (float(m.group(3)), _angle(m.group(4)))
    else:
        m = re.search(r"Route to ([a-z ]+): ([^,]+), (\d+) units left", s)
        if m:
            f["goal"] = m.group(1)
            f["route"], f["route_left"] = _angle(m.group(2)), float(m.group(3))
    m = re.search(r"Toxic nukage floor \d+ units (ahead|30 left|30 right)", s)
    f["nukage"] = m.group(1) if m else None
    m = re.search(r"Being shot from out of view, ([^.]+)\.", s)
    if m:
        f["attacker"] = _angle(m.group(1))
    m = re.search(r"Recent commands, oldest first: ([^;]*);", s)
    f["recent"] = [c.strip().split(" ")[0] for c in m.group(1).split(",")] if m else []
    m = re.search(r"Last command: (\w+), moved (\d+) units( \(blocked\))?(, took (\d+) damage)?", s)
    if m:
        f["last"], f["blocked"] = m.group(1), bool(m.group(3))
        f["damage"] = int(m.group(5)) if m.group(5) else 0
    return f


_last_turn_angle: list[float | None] = [None]     # angle behind the most recent turn decision (for soft labels)


def turn_toward(angle: float) -> str:
    """Nearest available turn for a signed angle (positive = left)."""
    _last_turn_angle[0] = angle
    if abs(angle) > 135:
        return "turn_around"
    side = "left" if angle > 0 else "right"
    step = min(STEPS, key=lambda st: abs(abs(angle) - st))
    return f"turn_{side}_{step}"


def _dodge(f: dict) -> str:
    """Sidestep while the weapon reloads: keep going the same way while there is room (a projectile
    is dodged by leaving its line, not by stepping back and forth), else the open side."""
    last = f["last"] or ""
    for side in ("left", "right"):
        if last.endswith("strafe_" + side) and f["rays"].get(f"60 {side}", (0.0, ""))[0] > 96:
            return "strafe_" + side
    return _open_side(f, "strafe_")


def _open_side(f: dict, prefix: str = "") -> str:
    """Side with more room, never the side the text reports nukage on."""
    if f.get("nukage") == "30 left":
        return prefix + "right"
    if f.get("nukage") == "30 right":
        return prefix + "left"
    left = min(f["rays"].get("30 left", (0.0, ""))[0], f["rays"].get("60 left", (0.0, ""))[0])
    right = min(f["rays"].get("30 right", (0.0, ""))[0], f["rays"].get("60 right", (0.0, ""))[0])
    return prefix + ("left" if left >= right else "right")


RAY_ANGLES = {"ahead": 0, "30 left": 30, "60 left": 60, "30 right": -30, "60 right": -60}


def _cut_off(f: dict, p: dict) -> bool:
    """The ray nearest the pickup's bearing meets a wall, ledge or drop well before it: no straight walk
    (E1M3: a stimpack across a bend of the walkway; the expert stepped to the drop and back 20 times)."""
    ray = min(RAY_ANGLES, key=lambda k: abs(RAY_ANGLES[k] - p["angle"]))
    if abs(RAY_ANGLES[ray] - p["angle"]) > 15:
        return False
    d, kind = f["rays"].get(ray, (1500.0, "open"))
    return kind != "open" and not kind.startswith(("door", "barrel")) and d < p["dist"] - 24


def _go_to(angle: float, dist: float) -> str:
    if abs(angle) >= 7.5:
        return turn_toward(angle)
    return "forward_short" if dist < 160 else "forward"


def _decide(f: dict) -> str:
    ahead = f["rays"].get("ahead", (1500.0, "open"))
    has_ammo = f["bullets"] > 0 or f["shells"] > 0

    goal = f["goal"]
    # 1. the current goal up close: exit switch / switch (use), switch to shoot (fire), key or trigger line (walk on)
    if ahead[1].startswith("exit switch") and ahead[0] <= USE_REACH:
        return "use"
    if goal == "switch" and ahead[1] == "switch" and ahead[0] <= USE_REACH:
        return "use"
    if f["exit_near"] is not None and goal not in KILL_GOALS:     # a kill target is fought, not walked onto
        d, a = f["exit_near"]
        if a is not None and abs(a) >= (2.5 if goal == "switch to shoot" else 7.5):
            return turn_toward(a)
        if goal == "switch to shoot":
            return "fire"
        if goal in ("exit", "switch"):
            return "use" if d <= USE_REACH else ("forward_short" if d < 200 else "forward")
        return "forward_short" if d < 200 else "forward"     # key or trigger line: walk onto it

    # 1b. a fireball on course to hit: sidestep off its line first (E1M8: 585 damage taken from Barons
    #     at 300+ units, all fireballs). Blocked that way: the other side.
    if f["incoming"] is not None:
        side = f["incoming"][1]
        other = "right" if side == "left" else "left"
        if f["rays"].get(f"60 {side}", (1500.0, ""))[0] < 40 and f["rays"].get(f"60 {other}", (0.0, ""))[0] >= 40:
            side = other
        return "strafe_" + side

    # 2. enemies within range (fist range when out of ammo). Near the exit or at low health,
    #    run instead: the switch ends the level, and a fight at 0.7 s per decision is a bad trade.
    #    E1M2: running past an imp and two zombiemen into the exit room killed the expert from behind
    #    on most failed seeds, so the near-exit sprint only applies with at most one enemy within 300.
    crowd = sum(1 for e in f["enemies"] if e["dist"] <= 300) >= 2
    sprint = (f["route_left"] is not None and f["route_left"] < 700 and not crowd) or f["health"] < 30
    reach = (100 if sprint else ENGAGE) if has_ammo else FIST
    # Demons close the gap fast and only bite: shoot them on the way in rather than walking into them
    # (E1M8's opening: the expert walked into the barrel field to meet the pack).
    melee_reach = MELEE_ENGAGE if has_ammo and not sprint else 0
    threats = [e for e in f["enemies"] if e["angle"] is not None
               and (e["dist"] <= reach or (e["kind"] in BOSSES and has_ammo)
                    or (e["kind"] in MELEE and e["dist"] <= melee_reach))]
    if threats:
        # Shotgun guys first among similar distances: one blast takes 9-45 health (E1M4, E1M5 openings).
        t = min(threats, key=lambda e: e["dist"] * (0.6 if e["kind"] == "shotgun guy" else 1.0))
        # A barrel in the firing cone inside its blast radius takes the shot, shotgun spread included
        # (E1M8: one blast, 100 -> 10 health; a pellet clipped a barrel 41 units off, 100 -> 35).
        cone = min((d for k, (d, kind) in f["rays"].items() if k in ("ahead", "30 left", "30 right")
                    and kind.startswith("barrel")), default=1500.0)
        if cone < BARREL_BLAST:
            move = "back" if f["rays"].get("behind", (0.0, ""))[0] > 120 else _open_side(f, "strafe_")
            if not _unsafe(f, move):
                return move
            threats = []            # boxed in by drops beside barrels (E1M3's walkway): hold fire, keep walking
    if threats:
        t = min(threats, key=lambda e: e["dist"] * (0.6 if e["kind"] == "shotgun guy" else 1.0))
        # A barrel beside enemies, outside its blast: shoot the barrel (E1M8's opening: demon packs, pistol only).
        # Not with a second barrel near the player: blasts chain (E1M8 seed 3: 100 -> 35 in one shot).
        near_barrel = min((d for k, (d, kind) in f["rays"].items() if k != "ahead" and kind.startswith("barrel")),
                          default=1500.0)
        if ahead[1].startswith("barrel") and BARREL_BLAST <= ahead[0] <= 450 and near_barrel >= 200 and has_ammo and any(
                e["angle"] is not None and abs(e["angle"]) <= 12 and abs(e["dist"] - ahead[0]) < 110 for e in f["enemies"]):
            return "fire"
        # Hurt in a boss fight: break line of sight behind the reported cover, then peek again
        # (the compass leads back toward the boss once it is out of view).
        if (t["kind"] in BOSSES and f["cover"] is not None
                and ((f["damage"] > 0 and f["health"] < COVER_HEALTH) or f["health"] < 40)):
            d, a = f["cover"]
            if a is not None and abs(a) >= 30:
                return turn_toward(a)
            return "forward" if d > 120 else "forward_short"
        if t["kind"] in BOSSES:
            # Baron (E1M8): every hit the expert took came within 250 units (claws, fireballs too close
            # to dodge), none beyond. Kite: back off inside 300, strafe-fire at 300-500, close in beyond.
            if t["dist"] < BOSS_NEAR:
                # Too close: retreat while reloading, but a ready shotgun fires (point-blank is full
                # damage; cornered against a wall the expert strafed under two Barons with 20 shells left).
                if f["reload"] <= RELOAD_DODGE:
                    if t["crosshair"]:
                        return "fire"
                    turn = turn_toward(t["angle"])
                    return turn + "_fire" if turn.endswith(("_5", "_15", "_30")) else turn
                # Reloading: back off facing them (running backwards is as fast as forwards, a Baron walks
                # at half that). Against a wall, strafe toward the open side the escape hint names; never
                # turn away (the expert spun between escape turns and "shot from out of view" turns).
                # The escape hint weighs openness: E1M8's star tips are dead ends. Move toward it without
                # turning: back when it lies behind, strafe when it lies to a side.
                a = f["escape"][1] if f["escape"] is not None else None
                last = f["last"] or ""
                if f["blocked"] and last.startswith(("strafe_", "back")):   # that way is shut: another one
                    if last != "back" and f["rays"].get("behind", (0.0, ""))[0] > 60:
                        return "back"
                    return "strafe_right" if last.endswith("left") else "strafe_left"
                if a is None:
                    return "back" if f["rays"].get("behind", (0.0, ""))[0] > 120 else _dodge(f)
                if abs(a) >= 120:
                    return "back"
                return "strafe_left" if a > 0 else "strafe_right"
            if t["crosshair"] and t["dist"] > BOSS_FAR:
                return "forward_short"
            # Dodge while the shotgun reloads; from AIM_LEAD s before it is ready, stand and aim, and fire
            # only with the Baron in the crosshair. Turn-and-fire from a strafe aimed at a stale angle
            # (100 ms decisions: 23-48 damage per shell against 41-72 for the instant teacher).
            if f["reload"] > AIM_LEAD:
                return _dodge(f) if abs(t["angle"]) < 30 else turn_toward(t["angle"])
            if t["crosshair"]:
                return "fire"
            return turn_toward(t["angle"])
        if t["crosshair"] and f["reload"] > RELOAD_DODGE and t["dist"] > 110:
            return _dodge(f)                               # shotgun reloading: move instead of a wasted fire
        # Within claw reach and reloading: back off facing them instead of standing to wait for the shot
        # (E1M3's blue key trap: three imps clawed the expert 46 -> 16 while it waited out each reload).
        if f["reload"] > RELOAD_DODGE and t["dist"] <= 110 and t["kind"] not in MELEE                 and f["rays"].get("behind", (0.0, ""))[0] > 120 and not (f["blocked"] and f["last"] == "back"):
            return "back"
        if t["crosshair"]:
            if f["damage"] > 0 and "fire" in (f["last"] or ""):
                return _open_side(f, "fire_strafe_")       # getting hit while shooting: dodge and keep firing
            if t["kind"] in MELEE and t["dist"] < 110:
                return _open_side(f, "fire_strafe_")       # demon about to bite (E1M6): sidestep while firing
            return "fire"
        turn = turn_toward(t["angle"])
        # Small aim correction: aim and shoot in one decision. Not while the shotgun reloads: turn-and-fire
        # holds the trigger until the shot is out, and the player stood still for up to a second in
        # E1M4's opening crossfire (zombiemen and shotgun guys, 71 -> 41 in one such wait).
        # Nor with a barrel near the line to the target: the turn would put it in the cone (E1M3: a
        # turn-and-fire past a barrel 20 units off killed the expert at 58 health).
        barrel_near_aim = any(kind.startswith("barrel") and d < BARREL_BLAST and abs(RAY_ANGLES[k] - t["angle"]) <= 30
                              for k, (d, kind) in f["rays"].items() if k in RAY_ANGLES)
        if turn.endswith(("_5", "_15", "_30")) and not (t["kind"] in BOSSES and t["dist"] > 500)                 and f["reload"] <= RELOAD_DODGE and not barrel_near_aim:
            return turn + "_fire"
        return turn

    # 3. shot from out of view: face the attacker if we can fight back
    if f["attacker"] is not None and has_ammo and not sprint:
        return turn_toward(f["attacker"])

    # 4. supplies: ammo when out, health when low
    low_shells = f["weapon"] == "shotgun" and f["shells"] < 8      # Barons take ~15 shotgun blasts each
    # Health before ammo when low (E1M3: at 3 health the expert walked past a medikit 108 units away
    # to fetch shells). Below 60 a near medikit is worth the steps: carried health decides the next level.
    health = [p for p in f["pickups"] if p["name"] in HEALTH_ITEMS and p["angle"] is not None
              and p["dist"] < (600 if f["health"] < 40 else 300) and f["health"] < 60 and not _cut_off(f, p)]
    ammo = [p for p in f["pickups"] if p["name"] in AMMO_ITEMS and p["angle"] is not None and p["dist"] < 600
            and (not has_ammo or low_shells) and not _cut_off(f, p)]
    items = health if health and (f["health"] < 40 or not ammo) else ammo
    if items:
        p = min(items, key=lambda p: p["dist"])
        return _go_to(p["angle"], p["dist"])

    route = f["route"]
    # 4b. a barrel on the way, outside its blast: shoot it before walking up to it. E1M3's walkway runs
    #     between barrels and the toxic pit; every expert that stood beside them died in one blast when a
    #     fireball or pellet set them off (57-58 health gone at once, three seeds).
    near_barrel = min((d for k, (d, kind) in f["rays"].items() if k != "ahead" and kind.startswith("barrel")),
                      default=1500.0)
    if (ahead[1].startswith("barrel") and BARREL_BLAST + 20 <= ahead[0] <= 400 and near_barrel >= 200 and has_ammo
            and route is not None and abs(route) < 20):
        return "fire"
    # 5. doors on the way
    # A door still closed right after "use" does not open by hand (E1M9: a sector raised by a trigger
    # elsewhere; the expert pressed use 60 times while the route went 33 deg right).
    if (ahead[1].startswith("door (closed)") and ahead[0] <= USE_REACH and (route is None or abs(route) <= 60)
            and f["last"] != "use"):
        return "use"
    if ahead[1].startswith("door (opening"):
        return "forward_short"

    # 6. face the route
    if route is not None and abs(route) >= 7.5:
        return turn_toward(route)

    # 7. a lift: the route goes up a ledge too high to climb right in front: press use to lower it.
    #    A lift wall (manual lift line, E1M8 before the Barons) is used once, then walked onto as it lowers.
    if ahead[1].startswith("ledge too high") and ahead[0] <= USE_REACH and f["blocked"]:
        return "use"
    if ahead[1].startswith("lift (raised)") and ahead[0] <= USE_REACH:
        return "use"
    if ahead[1].startswith("lift (lowering)"):
        return "forward_short"                    # step on as it comes down

    # 8. unstick: sidestep toward the open side; if the sidestep was blocked too, back off
    if f["blocked"]:
        if f["last"] in ("strafe_left", "strafe_right"):
            return "back"
        return _open_side(f, "strafe_")
    if ahead[1] != "open" and ahead[0] < 40:
        return _open_side(f, "strafe_")

    return "forward" if ahead[0] > 200 else "forward_short"


def _unsafe(f: dict, cmd: str) -> bool:
    """The move would carry the player over a drop with no way back."""
    return any(f["drops"].get(ray, 1e9) < reach for ray, reach in MOVE_REACH.get(cmd, ()))


def _flip_flop(f: dict) -> bool:
    """The last commands alternate between turning left and right: rules are undoing each other."""
    turns = [c.split("_")[1] for c in f["recent"][-3:] if c.startswith("turn_") and c != "turn_around"]
    return len(turns) == 3 == len(f["recent"][-3:]) and turns[0] == turns[2] != turns[1]


def decide(state: str) -> str:
    f = parse_state(state)
    cmd = _decide(f)
    # Turning back and forth beside a drop (E1M3 seed 1: 1.5 s of left-right turns at the pool's corner
    # while being shot): break it with a short step where there is room.
    if cmd.startswith("turn_") and not cmd.endswith("_fire") and _flip_flop(f):
        ahead = f["rays"].get("ahead", (1500.0, "open"))
        if ahead[0] > 40 and not _unsafe(f, "forward_short"):
            return "forward_short"
    if not _unsafe(f, cmd):
        return cmd
    # E1M3's walkway is ~100 units wide between toxic pits: a 250-unit strafe or a full run off the
    # bend fell in on every seed. Swap the move for the nearest one that stays on the floor.
    side = "left" if cmd.endswith("left") else "right"
    other = "right" if side == "left" else "left"
    last = f["last"] or ""
    room = lambda ray: f["rays"].get(ray, (1500.0, ""))[0]
    options = []
    if cmd.startswith(("strafe_", "fire_strafe_")) and room(f"60 {other}") > 40 and not (f["blocked"] and last.endswith(other)):
        options.append(cmd.replace(side, other))
    if cmd.startswith("fire_strafe_"):
        options.append("fire")
    if cmd == "forward":
        options.append("forward_short")
    if cmd != "back" and room("behind") > 30 and not (f["blocked"] and last == "back"):
        options.append("back")
    if cmd != "forward_short" and room("ahead") > 40 and not (f["blocked"] and last.startswith("forward")):
        options.append("forward_short")
    for o in options:
        if not _unsafe(f, o):
            return o
    # Boxed in between walls and the drop: turn 30 degrees away from the nearer drop, so the next short
    # step leads off the edge at a slant (a strafe would carry the player 200 units).
    near = {sd: min(f["drops"].get(f"{a} {sd}", 1e9) for a in (30, 60, 90)) for sd in ("left", "right")}
    if near["left"] == near["right"]:
        return "turn_" + _open_side(f) + "_30"
    return "turn_" + ("left" if near["left"] > near["right"] else "right") + "_30"


def decide_soft(state: str, smoothing: float = 0.05, temperature: float = 6.0) -> dict[str, float]:
    """The expert's choice as a distribution: turns share probability with neighbouring turn sizes
    by how close each size is to the target angle; every command keeps a small floor."""
    import math
    from .commands import COMMAND_IDS
    _last_turn_angle[0] = None
    choice = decide(state)
    p = {c: 0.0 for c in COMMAND_IDS}
    a = _last_turn_angle[0]
    if choice.startswith("turn_") and choice != "turn_around" and not choice.endswith("_fire") and a is not None:
        side = "left" if a > 0 else "right"
        w = {f"turn_{side}_{st}": math.exp(-abs(abs(a) - st) / temperature) for st in STEPS}
        z = sum(w.values())
        for k, v in w.items():
            p[k] = v / z
    else:
        p[choice] = 1.0
    n = len(COMMAND_IDS)
    return {k: (1 - smoothing) * v + smoothing / n for k, v in p.items()}
