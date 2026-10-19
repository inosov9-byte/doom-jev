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
SHELL_ITEMS = ("shells", "box of shells", "shotgun")   # low on shells, bullets are no detour (E1M5: walked off the switch for them)
HEALTH_ITEMS = ("stimpack", "medikit")      # a health bonus (+1) is not worth a detour
HEALTH_GAIN = {"stimpack": 10, "medikit": 25}
ARMOR_ITEMS = ("green armor", "blue armor")  # absorb a third / half of every hit
# Zombiemen and shotgun guys hit instantly: sidestepping does not dodge a bullet, only killing them or
# breaking line of sight helps. Left alone at 250-500 units they kept hitting the expert while it ran
# the route (E1M3 walkway: a zombieman 350 away took 71 -> 18).
HITSCAN = ("zombieman", "shotgun guy")
HITSCAN_ENGAGE = 500
# Imps left alone beyond 250 walked up and clawed at 50-60 units, where the second imp got the reloading
# expert (E1M4: most of the damage at 100 ms lag). Shot at range, a fireball takes ~1 s to arrive.
IMP_ENGAGE = 400
SHOOTERS = HITSCAN + ("imp",)
GROUP_COVER_NEAR = 200
EDGE = 120                     # a drop with no way back this close: no dodging, no long-range fights
GROUP_ENGAGE = 650
MELEE = ("demon", "spectre")   # bite only: keep them from reaching the player
MELEE_ENGAGE = 450             # demons are fought from this far: they come to the player anyway
BOSSES = ("baron",)            # fought at any range when in view
BOSS_NEAR, BOSS_FAR = 330, 520
AIM_LEAD = 0.35                # seconds before the shotgun is ready: stop dodging and aim
RELOAD_DODGE = 0.2             # seconds of shotgun reload worth a sidestep instead of a fire  # Baron kiting band
COVER_HEALTH = 70             # below this, a hit from a boss sends the expert behind cover
BARREL_BLAST = 160             # barrel explosion radius is 128; never shoot one closer than this
BARREL_KEEP = 135              # a barrel blast does 128 minus the box distance (axis distance less 16): keep this far under fire
DODGE_MIN = 0.25               # seconds: a fireball arriving sooner is not dodged from a stand
KILL_GOALS = ("baron of hell",)
DEAD_DROP = "drop with no way back"
# How far a move can carry the player, slide included (a run builds momentum that slides ~60-150 units
# after the next command replaces it), and the drop rays it must keep clear of: a strafe goes 90
# degrees sideways, which the 60 degree ray sees at a slant.
# A run also keeps sliding the old way after the next turn (Doom momentum): near a drop on either side
# the expert walks (E1M3 seed 1: ran along the walkway, turned at the bend, slid into the pool).
# Decisions 100 ms late add ~50 units of travel at a run (E1M3: the late expert ran off the walkway's
# bend on three seeds of four with 150 units of clearance ahead).
MOVE_REACH = {"forward": (("ahead", 240), ("30 left", 180), ("30 right", 180), ("60 left", 110), ("60 right", 110),
                          ("90 left", 80), ("90 right", 80)), "forward_short": (("ahead", 90),), "back": (("behind", 200),),
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
               "bullets": 0, "shells": 0, "weapon": "pistol", "moved": 0}
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
    m = re.search(r"Cover from \d+ shooters?: (\d+) units, ([^.]+)\.", s)
    f["group_cover"] = (float(m.group(1)), _angle(m.group(2))) if m else None
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
    f["on_nukage"] = "Standing on toxic nukage" in s
    f["on_lift"] = "Standing on a lift" in s
    m = re.search(r"Barrel (\d+) units away, ([^.]+)\.", s)
    f["barrel"] = (float(m.group(1)), _angle(m.group(2))) if m else None
    m = re.search(r"Being shot from out of view, ([^.]+)\.", s)
    if m:
        f["attacker"] = _angle(m.group(1))
    m = re.search(r"Recent commands, oldest first: ([^;]*);", s)
    f["recent"] = [c.strip().split(" ")[0] for c in m.group(1).split(",")] if m else []
    m = re.search(r"Last command: (\w+), moved (\d+) units( \(blocked\))?(, took (\d+) damage)?", s)
    if m:
        f["last"], f["blocked"] = m.group(1), bool(m.group(3))
        f["moved"] = int(m.group(2))
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


def _move_toward(dist: float, angle: float | None) -> str:
    """Step toward a spot without turning away from what is in front: back, strafe or forward."""
    if angle is None or abs(angle) >= 135:
        return "back"
    if abs(angle) >= 55:
        return "strafe_left" if angle > 0 else "strafe_right"
    if abs(angle) >= 20:
        return turn_toward(angle)
    return "forward_short" if dist < 120 else "forward"


def _go_to(angle: float, dist: float) -> str:
    if abs(angle) >= 7.5:
        return turn_toward(angle)
    return "forward_short" if dist < 160 else "forward"


def _decide(f: dict) -> str:
    ahead = f["rays"].get("ahead", (1500.0, "open"))
    has_ammo = f["bullets"] > 0 or f["shells"] > 0

    goal = f["goal"]
    # 1. the current goal up close: exit switch / switch (use), switch to shoot (fire), key or trigger line (walk on)
    # The centre ray, not a shoulder: use acts straight ahead (E1M1 seed 28: the switch at the right shoulder
    # 5 units off, the goal 118 degrees right, use pressed for 45 s).
    if ahead[1].startswith("exit switch") and "shoulder" not in ahead[1] and ahead[0] <= USE_REACH:
        return "use"
    if goal == "switch" and ahead[1] == "switch" and ahead[0] <= USE_REACH:
        return "use"
    # An item with a monster close by: fight first (E1M8 seed 3 at 100 ms: a demon 47 units
    # away in the crosshair, shotgun ready, and the expert pushed toward a box of shells behind it for 7 s,
    # 183 -> 21 health). The goal is still there afterwards.
    # Not a key or trigger line: those were walked onto through fire on E1M3 (stopping to fight short of the
    # stair trigger in the last corridor lost more runs than it saved).
    walk_on = goal not in ("exit", "switch", "switch to shoot", "trigger line") and "key" not in goal
    guarded = walk_on and has_ammo and any(e["angle"] is not None and e["dist"] <= 200 for e in f["enemies"])
    if f["exit_near"] is not None and goal not in KILL_GOALS and not guarded:     # a kill target is fought, not walked onto
        d, a = f["exit_near"]
        if a is not None and abs(a) >= (2.5 if goal == "switch to shoot" else 7.5):
            return turn_toward(a)
        if goal == "switch to shoot":
            return "fire"
        # A raised lift between the player and the goal: lower it first (E1M2's exit switch sits beyond a lift
        # that rises again; the expert pushed into its face for 45 s).
        if ahead[1].startswith("lift (raised)") and ahead[0] <= USE_REACH and f["blocked"]:
            return _lift_blocked(f)
        if goal in ("exit", "switch"):
            return "use" if d <= USE_REACH else ("forward_short" if d < 200 else "forward")
        return "forward_short" if d < 200 else "forward"     # key or trigger line: walk onto it

    # 1b. a fireball on course to hit: sidestep off its line first (E1M8: 585 damage taken from Barons
    #     at 300+ units, all fireballs). Blocked that way: the other side.
    #     Not at the edge of a drop with no way back: a fireball costs 3-24 health, the pit the whole run
    #     (E1M3's walkway: dodges at 100 ms lag carried the expert over the edge on most seeds).
    edge = min(f["drops"].values(), default=1e9) < EDGE
    # Wading through nukage: every second there costs ~4 health, and dodges, retreats and aiming turns kept
    # the late expert in E1M5's start pool for 15-20 s (37-49 health from the floor alone). Shoot only what
    # needs no more than one small turn while the shotgun is ready; otherwise keep walking the route out.
    wading = f["on_nukage"] and has_ammo and goal not in KILL_GOALS and not any(e["kind"] in BOSSES for e in f["enemies"])
    if f["incoming"] is not None and not edge and not wading and _dodge_fireball(f):
        side = f["incoming"][1]
        other = "right" if side == "left" else "left"
        last = f["last"] or ""
        if (f["rays"].get(f"60 {side}", (1500.0, ""))[0] < 40 and f["rays"].get(f"60 {other}", (0.0, ""))[0] >= 40
                or f["blocked"] and last.endswith("strafe_" + side) and not any(e["kind"] in BOSSES for e in f["enemies"])):
            side = other
        return "strafe_" + side

    # 1c. a barrel within blast range while something can shoot: use the shotgun's reload to step away from
    #     it. A stray bullet or fireball that hits it costs up to 128 health (E1M5, E1M6: gunmen set off
    #     barrels beside the expert, 68 health gone at once). A ready shotgun fights first: walking away
    #     from a barrel past a shotgun guy cost 30 health (E1M4).
    b = f["barrel"]
    # Something shooting from out of view counts as well (it is the one that sets the barrel off).
    if (b is not None and b[1] is not None
            and (any(e["dist"] <= 800 and e["kind"] in SHOOTERS for e in f["enemies"]) or f["attacker"] is not None)
            and (b[0] < BARREL_KEEP and f["reload"] > AIM_LEAD or b[0] < 80 and f["weapon"] != "shotgun")):
        move = _away_from(f, b[1])
        if move is not None:
            return move

    # 2. enemies within range (fist range when out of ammo). Near the exit or at low health,
    #    run instead: the switch ends the level, and a fight at 0.7 s per decision is a bad trade.
    #    E1M2: running past an imp and two zombiemen into the exit room killed the expert from behind
    #    on most failed seeds, so the near-exit sprint only applies with at most one enemy within 300.
    crowd = sum(1 for e in f["enemies"] if e["dist"] <= 300) >= 2
    # Not near a key: the enemies are still there once it is taken (E1M4's blue key room: the expert ran at
    # three imps with the shotgun ready and lost 57 health).
    sprint = ("key" not in goal and f["route_left"] is not None and f["route_left"] < 700 and not crowd) or f["health"] < 30
    reach = (100 if sprint else ENGAGE) if has_ammo else FIST
    # Demons close the gap fast and only bite: shoot them on the way in rather than walking into them
    # (E1M8's opening: the expert walked into the barrel field to meet the pack).
    melee_reach = MELEE_ENGAGE if has_ammo and not sprint else 0
    # Not from the edge of a drop with no way back either: stopping to aim on E1M3's narrow walkway left
    # the 100 ms-late expert sliding over the edge into the toxic pit (two seeds of four). Keep walking.
    # Two or more shooters in view with cover a few steps away: fight them from there, out to the range a
    # shotgun still hits, instead of walking into the open (E1M4's opening room: four gunmen, 71 -> 8
    # in five seconds once the expert stood in the middle of it).
    shooters = [e for e in f["enemies"] if e["kind"] in SHOOTERS]
    held = f["group_cover"] is not None and f["group_cover"][0] <= GROUP_COVER_NEAR and len(shooters) >= 2
    hitscan_reach = (GROUP_ENGAGE if held else HITSCAN_ENGAGE) if has_ammo and not sprint and not edge else 0
    if sprint and has_ammo and not edge:
        hitscan_reach = 300        # bullets outrun a sprint: a near gunman is fought even then
    threats = [e for e in f["enemies"] if e["angle"] is not None
               and (e["dist"] <= reach or (e["kind"] in BOSSES and has_ammo)
                    or (e["kind"] in HITSCAN and e["dist"] <= hitscan_reach)
                    or (e["kind"] == "imp" and e["dist"] <= min(hitscan_reach, IMP_ENGAGE))
                    or (e["kind"] in MELEE and e["dist"] <= melee_reach))]
    # A lift just used or stood on: ride it before fighting anything not close. Doom lifts stay down for 3 s, and a
    # switch-once lift that rises again strands the player (E1M2's exit lift: imps at 300 came into view as
    # it lowered; the expert fought them, the lift rose and use no longer worked).
    if f["on_lift"] or "use" in f["recent"][-3:] and (f["last"] == "use" or any(
            kind.startswith("lift") and d <= USE_REACH for k, (d, kind) in f["rays"].items())):
        threats = [e for e in threats if e["dist"] <= 200 or e["kind"] in BOSSES]
    # Hit hard from out of view while what is in view is farther: the unseen one is the gunman to face
    # (E1M7's start: 18, 18 and 21 from a shotgun guy out of view while the expert dodged a far imp).
    if (f["attacker"] is not None and f["damage"] >= 8 and has_ammo and not f["on_nukage"] and goal not in KILL_GOALS
            and all(e["dist"] > 200 for e in threats)):
        return turn_toward(f["attacker"])
    if wading:
        near = [e for e in threats if e["dist"] <= 450]
        threats = []
        if near:
            t = min(near, key=_priority)
            cone = any(kind.startswith("barrel") and d < BARREL_BLAST for k, (d, kind) in f["rays"].items()
                       if k in ("ahead", "30 left", "30 right"))
            if t["kind"] in MELEE and t["dist"] < 110:
                return _open_side(f, "fire_strafe_")
            if f["reload"] <= RELOAD_DODGE and not cone:
                if t["crosshair"]:
                    return "fire"
                turn = turn_toward(t["angle"])
                if turn.endswith(("_5", "_15", "_30")):
                    return turn + "_fire"
    if threats:
        t = min(threats, key=_priority)
        # A barrel in the firing cone inside its blast radius takes the shot, shotgun spread included
        # (E1M8: one blast, 100 -> 10 health; a pellet clipped a barrel 41 units off, 100 -> 35).
        # A barrel on a 30 degree ray lies half its distance off the aim line: only near ones meet the spread
        # (E1M4: a barrel 128 away on the 30 degree ray kept the expert from firing at an imp 79 away).
        cone = min((d for k, (d, kind) in f["rays"].items() if kind.startswith("barrel")
                    and (k == "ahead" or k in ("30 left", "30 right") and d < 64)), default=1500.0)
        if cone < BARREL_BLAST:
            move = "back" if f["rays"].get("behind", (0.0, ""))[0] > 120 else _open_side(f, "strafe_")
            if not _unsafe(f, move):
                return move
            threats = []            # boxed in by drops beside barrels (E1M3's walkway): hold fire, keep walking
    if threats:
        t = min(threats, key=_priority)
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
        # Two or more gunmen or imps with a line of fire: out of their sight while the shotgun reloads (with
        # bullets, whenever they hit), back out to fire when it is ready. They follow one at a time (E1M4's opening room:
        # 71 health gone in five seconds standing in the open). Moves keep facing them: no turning away.
        if (t["kind"] not in BOSSES and f["group_cover"] is not None and len(shooters) >= 2
                and (f["reload"] > AIM_LEAD
                     or (f["damage"] > 0 and (f["weapon"] != "shotgun" or f["health"] < 30)))):
            move = _move_toward(*f["group_cover"])
            if not _unsafe(f, move):
                return move
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
            # Shotgun reloading: move instead of a wasted fire. Against a gunman a sidestep dodges
            # nothing and moves the aim off him; "fire" holds the aim and shoots the moment it is ready.
            # At the edge of a drop with no way back a sidestep has nowhere to go: hold the aim instead
            # (E1M3's walkway at 100 ms: the swapped-in step back went into a barrel and over the edge).
            return "fire" if t["kind"] in HITSCAN or edge else _dodge(f)
        # Within claw reach and reloading: back off facing them instead of standing to wait for the shot
        # (E1M3's blue key trap: three imps clawed the expert 46 -> 16 while it waited out each reload).
        # Demons and spectres too: a player running backwards is three times as fast as they walk (E1M7's start at
        # 100 ms: sidestep-fires and turn-and-fires at a spectre 50-90 units away, 71 -> 17 in three seconds).
        if (f["reload"] > RELOAD_DODGE and t["dist"] <= (150 if t["kind"] in ("imp",) + MELEE else 110)
                and f["rays"].get("behind", (0.0, ""))[0] > 120 and not (f["blocked"] and f["last"] == "back")):
            return "back"
        # Back to a wall: sidestep out of reach instead, away from its side (a sidestep toward it stays in
        # reach: E1M5's start, 10-24 per strafe beside imps). "fire" stands still until the shot is out, up to
        # a second (E1M7's start: an imp and a spectre at 48-139 units took 57 health during one such wait).
        if f["reload"] > AIM_LEAD and t["dist"] <= 110 and t["kind"] not in HITSCAN + MELEE and abs(t["angle"]) >= 10:
            move = "strafe_right" if t["angle"] > 0 else "strafe_left"
            if f["rays"].get("60 " + move.split("_")[1], (0.0, ""))[0] > 96 and not _unsafe(f, move) and not (f["blocked"] and f["last"] == move):
                return move
        if t["crosshair"]:
            if f["damage"] > 0 and "fire" in (f["last"] or "") and t["kind"] not in HITSCAN:
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
    if f["attacker"] is not None and has_ammo and not sprint and not wading:
        return turn_toward(f["attacker"])

    # 4. supplies: ammo when out, health when low
    low_shells = f["weapon"] == "shotgun" and f["shells"] < 8      # Barons take ~15 shotgun blasts each
    # Health before ammo when low (E1M3: at 3 health the expert walked past a medikit 108 units away
    # to fetch shells). Below 60 a near medikit is worth the steps: carried health decides the next level.
    # A near item that would not be wasted is worth the steps too (E1M4: a medikit beside the start, taken
    # at 71 health it is 25 more for the zombieman room behind the first door).
    health = [p for p in f["pickups"] if p["name"] in HEALTH_ITEMS and p["angle"] is not None and not _cut_off(f, p)
              and ((p["dist"] < (600 if f["health"] < 40 else 300) and f["health"] < 60)
                   or (p["dist"] < 300 and f["health"] + HEALTH_GAIN[p["name"]] <= 100))]
    health += [p for p in f["pickups"] if p["name"] in ARMOR_ITEMS and p["angle"] is not None and p["dist"] < 900
               and f["armor"] < 50 and not _cut_off(f, p)]
    ammo = [p for p in f["pickups"] if p["name"] in AMMO_ITEMS and p["angle"] is not None and p["dist"] < 600
            and (not has_ammo or low_shells and p["name"] in SHELL_ITEMS) and not _cut_off(f, p)]
    # Health and armor first unless out of ammo altogether (E1M4: low on shells, the expert walked past
    # blue armor 460 units away to fetch a shotgun, then died at 23 health).
    items = health if health and (f["health"] < 40 or has_ammo or not ammo) else ammo
    if items:
        p = min(items, key=lambda p: p["dist"])
        return _go_to(p["angle"], p["dist"])

    route = f["route"]
    # 4b. a barrel on the way, outside its blast: shoot it before walking up to it. E1M3's walkway runs
    #     between barrels and the toxic pit; every expert that stood beside them died in one blast when a
    #     fireball or pellet set them off (57-58 health gone at once, three seeds).
    near_barrel = min((d for k, (d, kind) in f["rays"].items() if k != "ahead" and kind.startswith("barrel")),
                      default=1500.0)
    # Not with an enemy in view or the shotgun reloading: standing to wait for the shot is a free target
    # (E1M4: at 23 health the expert stopped to shoot a barrel while an imp threw fireballs).
    if (ahead[1].startswith("barrel") and BARREL_BLAST + 20 <= ahead[0] <= 400 and near_barrel >= 200 and has_ammo
            and route is not None and abs(route) < 20 and not f["enemies"] and f["reload"] <= RELOAD_DODGE):
        # Not with far monsters in view either: the shot wakes them (tried on E1M3: shotgun guys met the
        # expert at the next corner, rules 3/16 against 8/16).
        return "fire"
    # Just shot at a barrel still standing ahead: it bursts about half a second after the hit. Walking on
    # took the late expert into the blast (E1M7 at 100 ms: shot from 180 units, two steps on, 71 -> 41).
    if (ahead[1].startswith("barrel") and ahead[0] <= 400 and (f["last"] or "").endswith("fire")
            and not any(e["dist"] <= 200 for e in f["enemies"])):
        if f["rays"].get("behind", (0.0, ""))[0] > 120 and not _unsafe(f, "back"):
            return "back"
        return _open_side(f, "strafe_")
    # 5. doors on the way
    # A door still closed right after "use" does not open by hand (E1M9: a sector raised by a trigger
    # elsewhere; the expert pressed use 60 times while the route went 33 deg right).
    if (ahead[1].startswith("door (closed)") and ahead[0] <= USE_REACH and (route is None or abs(route) <= 60)
            and f["last"] != "use"):
        return "use"
    if ahead[1].startswith("door (opening"):
        return "forward_short"

    # 6. face the route. On nukage, coarser: every second there costs health, and fine 5 degree corrections
    #    left the expert wading for seconds (E1M5's pool before the switch: 5 health per second, 20 lost).
    if route is not None and abs(route) >= (20 if f["on_nukage"] else 7.5):
        return turn_toward(route)

    # 7. a lift: the route goes up a ledge too high to climb right in front: press use to lower it.
    #    A lift wall (manual lift line, E1M8 before the Barons) is used once, then walked onto as it lowers.
    if ahead[1].startswith("ledge too high") and ahead[0] <= USE_REACH and f["blocked"]:
        return "use"
    if ahead[1].startswith("lift (raised)") and ahead[0] <= USE_REACH:
        return _lift_blocked(f) if f["on_lift"] else "use"
    if ahead[1].startswith("lift (lowering)"):
        return "forward_short"                    # step on as it comes down

    # 8. unstick: sidestep toward the open side; if the sidestep was blocked too, back off
    if f["blocked"]:
        if f["last"] in ("strafe_left", "strafe_right"):
            return "back"
        return _open_side(f, "strafe_")
    if ahead[1] != "open" and ahead[0] < 40:
        return _open_side(f, "strafe_")

    return "forward" if ahead[0] > (120 if f["on_nukage"] else 200) else "forward_short"


def _lift_blocked(f: dict) -> str:
    """A raised lift face in the way. Standing on the lift itself, use does nothing (Doom lift switches work
    from their front side only): the lift rose again under the late player, and E1M2 runs stalled at 99.5%
    pressing use for 45 s (7 of 16 at 100 ms). Step off and back on, so its walk-over line lowers it again."""
    if not f["on_lift"]:
        return "use"
    # Keep backing until off it: a short back step left the instant player still on the 64-unit lift.
    if ((f["blocked"] or f["last"] == "back") and f["rays"].get("behind", (0.0, ""))[0] > 60
            and not _unsafe(f, "back")):
        return "back"
    return "forward_short"


def _priority(e: dict) -> float:
    """Target order: nearest first, shotgun guys first among similar distances (one blast takes 9-45 health,
    E1M4, E1M5 openings), and one already in the crosshair before an equally near one that needs a turn."""
    return e["dist"] * (0.6 if e["kind"] == "shotgun guy" else 1.0) - (40 if e["crosshair"] and e["kind"] not in BOSSES else 0)


def _away_from(f: dict, angle: float) -> str | None:
    """A move that takes the player away from a spot at this bearing without turning, if there is room."""
    if abs(angle) <= 45:
        cands = ["back", "strafe_right" if angle > 0 else "strafe_left"]
    elif abs(angle) >= 135:
        cands = ["forward", "strafe_right" if angle > 0 else "strafe_left"]
    else:
        cands = ["strafe_right" if angle > 0 else "strafe_left", "back" if abs(angle) < 90 else "forward"]
    # Not toward a monster in front (E1M7's start: away from a barrel behind meant walking into a spectre).
    # Gunmen ahead are no reason to stay by the barrel: walking toward them does not add to their fire.
    if any(e["angle"] is not None and abs(e["angle"]) <= 45 and e["kind"] not in HITSCAN
           and e["dist"] <= (300 if e["kind"] in MELEE else 160) for e in f["enemies"]):
        cands = [c for c in cands if c != "forward"] + (["strafe_left", "strafe_right"] if "forward" in cands else [])
    ray = {"back": "behind", "forward": "ahead", "strafe_left": "60 left", "strafe_right": "60 right"}
    last = f["last"] or ""
    for c in cands:
        if f["rays"].get(ray[c], (0.0, ""))[0] > 64 and not _unsafe(f, c) and not (f["blocked"] and last == c):
            return c
    return None


def _dodge_fireball(f: dict) -> bool:
    """Worth sidestepping the reported fireball. Against Barons always (E1M8). Against imps a sidestep
    needs ~0.25 s from a stand to clear the fireball's path (Doom strafe acceleration), more with a
    late decision: later ones were hit anyway, and a ready shotgun aimed at the imp kills it instead
    (E1M4: imps at 80-290 units, the expert dodged every fireball, never fired and lost 71 health)."""
    if any(e["kind"] in BOSSES for e in f["enemies"]):
        return True
    eta, side = f["incoming"]
    # An imp within claw reach: a sidestep stays in reach (E1M5's start at 100 ms: dodging fireballs beside
    # imps 42-65 units away, 26 and 16 from claws). Backing off facing it (the fight rules) gets out of reach.
    if any(e["kind"] == "imp" and e["dist"] <= 130 for e in f["enemies"]):
        return False
    if f["reload"] <= RELOAD_DODGE and any(e["crosshair"] and e["dist"] <= HITSCAN_ENGAGE for e in f["enemies"]):
        return False
    # A ready shotgun with an enemy near: turn and shoot (one tic to turn) rather than dodge. A shotgun guy
    # 63 units away blasted the dodging expert for 27 (E1M6's imp yard).
    if f["weapon"] == "shotgun" and f["reload"] <= RELOAD_DODGE and any(
            e["dist"] <= ENGAGE or e["kind"] in HITSCAN and e["dist"] <= 300 for e in f["enemies"]):
        return False
    moving = (f["last"] or "").endswith("strafe_" + side) and not f["blocked"]
    return eta >= DODGE_MIN or moving


MOVE_DIR = {"forward": 0, "forward_short": 0, "back": 180, "strafe_left": 90, "strafe_right": -90,
            "fire_strafe_left": 90, "fire_strafe_right": -90}


def _toward_barrel(f: dict, cmd: str) -> bool:
    """The move heads at the reported barrel while it is close and something that shoots is in view: a stray
    shot sets it off beside the player (E1M5 at 100 ms: a strafe toward a barrel 110 units off, 70 -> 0)."""
    b = f["barrel"]
    if cmd not in MOVE_DIR or b is None or b[1] is None or b[0] >= 150:
        return False
    if not (any(e["kind"] in SHOOTERS for e in f["enemies"]) or f["attacker"] is not None):
        return False
    diff = abs((MOVE_DIR[cmd] - b[1] + 180) % 360 - 180)
    return diff < 50


def _unsafe(f: dict, cmd: str) -> bool:
    """The move would carry the player over a drop with no way back."""
    return any(f["drops"].get(ray, 1e9) < reach for ray, reach in MOVE_REACH.get(cmd, ()))


def _still(f: dict) -> bool:
    """Standing nearly still with no drop within 60 ahead: a short step is safe there."""
    return f["moved"] < 10 and min(f["drops"].get(r, 1e9) for r in ("ahead", "30 left", "30 right")) > 60


def _flip_flop(f: dict) -> bool:
    """The last commands alternate between turning left and right: rules are undoing each other."""
    turns = [c.split("_")[1] for c in f["recent"][-3:] if c.startswith("turn_") and c != "turn_around"]
    return len(turns) == 3 == len(f["recent"][-3:]) and turns[0] == turns[2] != turns[1]


def decide(state: str) -> str:
    f = parse_state(state)
    cmd = _decide(f)
    # Still sliding from a run toward a drop: brake. Doom momentum carries a run ~150 units after the
    # next command, and a turn does not stop it (E1M3: turned at the walkway's bend mid-slide, fell in).
    if (f["last"] == "forward" and f["moved"] >= 70 and cmd != "back"
            and min(f["drops"].get(r, 1e9) for r in ("ahead", "30 left", "30 right")) < 200
            and f["rays"].get("behind", (0.0, ""))[0] > 60 and not _unsafe(f, "back")):
        return "back"
    # Turning back and forth beside a drop (E1M3 seed 1: 1.5 s of left-right turns at the pool's corner
    # while being shot): break it with a short step where there is room.
    if cmd.startswith("turn_") and not cmd.endswith("_fire") and _flip_flop(f):
        ahead = f["rays"].get("ahead", (1500.0, "open"))
        # From a standstill a short step carries 25-60 units, not the 90 kept clear after a run (E1M3 seed 3:
        # 1.8 s of left-right turns between barrels and the walkway's edge, then a strafe into the pit).
        if ahead[0] > 40 and (_still(f) or not _unsafe(f, "forward_short")):
            return "forward_short"
        # Nothing ahead but a wall or a drop: back off and let the route lead in at a new angle (E1M5 at
        # 100 ms: 60 s of left-right turns in a corner between a wall and a drop, ended as stuck).
        if f["moved"] < 5 and f["rays"].get("behind", (0.0, ""))[0] > 60 and not _unsafe(f, "back"):
            return "back"
    if not _unsafe(f, cmd) or cmd == "forward_short" and _still(f):
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
        if (not _unsafe(f, o) or o == "forward_short" and _still(f)) and not _toward_barrel(f, o):
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
