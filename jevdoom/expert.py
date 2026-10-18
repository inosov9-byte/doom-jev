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
HEALTH_ITEMS = ("stimpack", "medikit", "health bonus")


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
            if e:
                f["enemies"].append({"kind": e.group(1), "dist": int(e.group(2)), "angle": _angle(e.group(3)),
                                     "crosshair": e.group(3).startswith("in the crosshair")})
    m = re.search(r"Pickups in view: ([^.]*)\.", s)
    if m:
        for part in m.group(1).split(";"):
            p = re.match(r"\s*([a-z ]+?) (\d+) away, (.+)", part)
            if p:
                f["pickups"].append({"name": p.group(1), "dist": int(p.group(2)), "angle": _angle(p.group(3))})
    m = re.search(r"Obstacles: ([^.]*(?:\.\d[^.]*)*)\.", s)
    if m:
        for part in m.group(1).split(";"):
            r = re.match(r"\s*(ahead|behind|\d+ left|\d+ right) (open|(\d+) (.+))", part)
            if r:
                f["rays"][r.group(1)] = (1500.0, "open") if r.group(2) == "open" else (float(r.group(3)), r.group(4))
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


def _open_side(f: dict, prefix: str = "") -> str:
    """Side with more room, never the side the text reports nukage on."""
    if f.get("nukage") == "30 left":
        return prefix + "right"
    if f.get("nukage") == "30 right":
        return prefix + "left"
    left = min(f["rays"].get("30 left", (0.0, ""))[0], f["rays"].get("60 left", (0.0, ""))[0])
    right = min(f["rays"].get("30 right", (0.0, ""))[0], f["rays"].get("60 right", (0.0, ""))[0])
    return prefix + ("left" if left >= right else "right")


def _go_to(angle: float, dist: float) -> str:
    if abs(angle) >= 7.5:
        return turn_toward(angle)
    return "forward_short" if dist < 160 else "forward"


def decide(state: str) -> str:
    f = parse_state(state)
    ahead = f["rays"].get("ahead", (1500.0, "open"))
    has_ammo = f["bullets"] > 0 or f["shells"] > 0

    goal = f["goal"]
    # 1. the current goal up close: exit switch / switch (use), switch to shoot (fire), key or trigger line (walk on)
    if ahead[1].startswith("exit switch") and ahead[0] <= USE_REACH:
        return "use"
    if goal == "switch" and ahead[1] == "switch" and ahead[0] <= USE_REACH:
        return "use"
    if f["exit_near"] is not None:
        d, a = f["exit_near"]
        if a is not None and abs(a) >= (2.5 if goal == "switch to shoot" else 7.5):
            return turn_toward(a)
        if goal == "switch to shoot":
            return "fire"
        if goal in ("exit", "switch"):
            return "use" if d <= USE_REACH else ("forward_short" if d < 200 else "forward")
        return "forward_short" if d < 200 else "forward"     # key or trigger line: walk onto it

    # 2. enemies within range (fist range when out of ammo). Near the exit or at low health,
    #    run instead: the switch ends the level, and a fight at 0.7 s per decision is a bad trade.
    #    E1M2: running past an imp and two zombiemen into the exit room killed the expert from behind
    #    on most failed seeds, so the near-exit sprint only applies with at most one enemy within 300.
    crowd = sum(1 for e in f["enemies"] if e["dist"] <= 300) >= 2
    sprint = (f["route_left"] is not None and f["route_left"] < 700 and not crowd) or f["health"] < 30
    reach = (100 if sprint else ENGAGE) if has_ammo else FIST
    threats = [e for e in f["enemies"] if e["dist"] <= reach and e["angle"] is not None]
    if threats:
        t = min(threats, key=lambda e: e["dist"])
        if t["crosshair"]:
            if f["damage"] > 0 and "fire" in (f["last"] or ""):
                return _open_side(f, "fire_strafe_")       # getting hit while shooting: dodge and keep firing
            return "fire"
        turn = turn_toward(t["angle"])
        if turn.endswith(("_5", "_15", "_30")):
            return turn + "_fire"                          # small aim correction: aim and shoot in one decision
        return turn

    # 3. shot from out of view: face the attacker if we can fight back
    if f["attacker"] is not None and has_ammo and not sprint:
        return turn_toward(f["attacker"])

    # 4. supplies: ammo when out, health when low
    want = AMMO_ITEMS if not has_ammo else HEALTH_ITEMS if f["health"] < 40 else ()
    items = [p for p in f["pickups"] if p["name"] in want and p["dist"] < 600 and p["angle"] is not None]
    if items:
        p = min(items, key=lambda p: p["dist"])
        return _go_to(p["angle"], p["dist"])

    route = f["route"]
    # 5. doors on the way
    if ahead[1].startswith("door (closed)") and ahead[0] <= USE_REACH and (route is None or abs(route) <= 60):
        return "use"
    if ahead[1].startswith("door (opening"):
        return "forward_short"

    # 6. face the route
    if route is not None and abs(route) >= 7.5:
        return turn_toward(route)

    # 7. a lift: the route goes up a ledge too high to climb right in front: press use to lower it
    if ahead[1].startswith("ledge too high") and ahead[0] <= USE_REACH and f["blocked"]:
        return "use"

    # 8. unstick: sidestep toward the open side; if the sidestep was blocked too, back off
    if f["blocked"]:
        if f["last"] in ("strafe_left", "strafe_right"):
            return "back"
        return _open_side(f, "strafe_")
    if ahead[1] != "open" and ahead[0] < 40:
        return _open_side(f, "strafe_")

    return "forward" if ahead[0] > 200 else "forward_short"


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
