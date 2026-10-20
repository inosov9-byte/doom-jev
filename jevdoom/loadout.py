"""Campaign continuity: what the player carries from one level into the next.

Doom carries health, armor, weapons and ammo across a level exit; keys and power-ups are dropped.
A run that starts mid-episode gets the previous level's end state, applied with console give/take
commands right after spawn (needs cheats on). A pistol start is the loadout None.
"""
from __future__ import annotations

import json
from pathlib import Path

from .env import Obs, blank_action

PISTOL_START = {"health": 100, "armor": 0, "weapons": ["pistol"], "bullets": 50, "shells": 0, "rockets": 0}

# Stand-in for an E1M7 exit while the teacher cannot yet chain E1M3-E1M7: the weapons E1M1-E1M7 put
# on the main route (shotgun on E1M2/E1M3, chaingun on E1M2) and a modest ammo reserve. Full health,
# no armor. Replace with a real end state once the campaign chains to E1M8.
NOMINAL = {
    "E1M8": {"health": 100, "armor": 0, "weapons": ["pistol", "shotgun", "chaingun"],
             "bullets": 100, "shells": 30, "rockets": 0},
}

GIVE = {"shotgun": "shotgun", "chaingun": "chaingun", "rocket launcher": "rocketlauncher"}


def from_obs(obs: Obs) -> dict:
    weapons = ["pistol"] + [w for w, held in (("shotgun", obs.has_shotgun), ("chaingun", obs.has_chaingun),
                                               ("rocket launcher", obs.has_launcher)) if held]
    return {"health": round(obs.health), "armor": round(obs.armor), "weapons": weapons,
            "bullets": round(obs.bullets), "shells": round(obs.shells), "rockets": round(obs.rockets)}


def load(spec: str | None, map_name: str) -> dict | None:
    """None/'pistol' -> pistol start; 'nominal' -> NOMINAL[map]; otherwise a JSON file (a loadout,
    or a run's summary.json, whose end_loadout is used)."""
    if not spec or spec == "pistol":
        return None
    if spec == "nominal":
        return dict(NOMINAL[map_name]) if map_name in NOMINAL else None
    d = json.loads(Path(spec).read_text())
    return d.get("end_loadout", d)


def apply(g, lo: dict) -> None:
    """Turn the pistol-start player into the carried state, checked against the live values.
    Console quirks (ViZDoom 1.2): 'take armor N' removes all armor (every map used to start with 0 armor
    after a partial-armor exit), and 'give shell N' after the ammo was taken to 0 is capped at the normal
    maximum even with a backpack. So: armor as armor bonuses (exact amount; green absorption below 200),
    ammo as differences from what the player holds, repeated until it matches."""
    from .env import observe

    def send(c):
        g.send_game_command(c)
        g.make_action(blank_action(), 1)

    for w in lo["weapons"]:
        if w in GIVE:
            send(f"give {GIVE[w]}")
    # Doom keeps a backpack across exits; ammo above the normal maximum means one was carried
    if lo["bullets"] > 200 or lo["shells"] > 50 or lo.get("rockets", 0) > 50:
        send("give backpack")
    for _ in range(4):
        o = observe(g, 0)
        todo = [(item, lo.get(key, 0) - round(cur)) for item, key, cur in
                (("clip", "bullets", o.bullets), ("shell", "shells", o.shells), ("rocketammo", "rockets", o.rockets))]
        todo = [(item, d) for item, d in todo if d]
        if not todo:
            break
        for item, d in todo:
            send(f"give {item} {d}" if d > 0 else f"take {item} {-d}")
    if lo["health"] < 100:
        send(f"take health {100 - lo['health']}")
    elif lo["health"] > 100:
        send(f"give health {lo['health'] - 100}")
    if lo["armor"] >= 200:
        send("give bluearmor")
    elif lo["armor"] > 0:
        send(f"give armorbonus {lo['armor']}")
