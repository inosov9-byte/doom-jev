"""Teacher probe: synthetic situations with unambiguous right answers, scored per model and prompt.

Used to pick the teacher (model + instructions) before labelling, since the student can
only be as good as its labels. Situations use the same wording as perception.py.

Scoring: 1.0 for the best command, 0.5 for an acceptable one (e.g. a neighbouring turn
size in the right direction), 0 otherwise. Turning the wrong way always scores 0.

usage: python -m jevdoom.probe --models featherless-ai/Qwen3.6-35B-A3B-classifier featherless-ai/Qwen3.8-27B-classifier
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time

from .commands import COMMANDS
from .jev import HOSTED_URL, INSTRUCTIONS, JevClient

TURNS = (15, 45, 90)


def side(b: float) -> str:
    if abs(b) < 7.5:
        return "straight ahead"
    return f"{abs(b):.0f} deg {'left' if b > 0 else 'right'}"


def turn_credit(b: float) -> dict[str, float]:
    d = "left" if b > 0 else "right"
    mags = sorted(TURNS, key=lambda t: abs(abs(b) - t))
    credit = {f"turn_{d}_{mags[0]}": 1.0, f"turn_{d}_{mags[1]}": 0.5}
    if abs(b) > 150:
        credit = {"turn_around": 1.0, f"turn_{d}_90": 0.5}
    return credit


def obstacles(rng, ahead=None):
    a = ahead or f"{rng.randint(250, 900)} wall"
    return (f"Obstacles: ahead {a}; 30 left {rng.randint(120, 600)} wall; 60 left {rng.randint(80, 500)} wall; "
            f"30 right {rng.randint(120, 600)} wall; 60 right {rng.randint(80, 500)} wall; behind {rng.randint(100, 800)} wall.")


def vitals(rng):
    return f"Health {rng.randint(40, 100)}, armor 0. Weapon: pistol; {rng.randint(20, 60)} bullets, 0 shells."


def make_cases(seed: int = 7):
    rng = random.Random(seed)
    cases = []
    kinds = ["zombieman", "imp", "shotgun guy"]
    for b in (12, 20, 30, 40, 55, 70, -12, -20, -30, -40, -55, -70):          # enemy, route agrees with nothing
        s = (f"{vitals(rng)} Enemies in view: {rng.choice(kinds)} {rng.randint(150, 700)} away, {side(b)}. "
             f"{obstacles(rng)} Route to exit: straight ahead, {rng.randint(900, 4000)} units left.")
        cases.append(("aim at enemy", s, turn_credit(b)))
    for b in (25, 40, -25, -40, 60, -60):                                    # enemy one side, route the other
        r = -b + rng.choice((-10, 10))
        s = (f"{vitals(rng)} Enemies in view: {rng.choice(kinds)} {rng.randint(150, 600)} away, {side(b)}. "
             f"{obstacles(rng)} Route to exit: {side(r)}, {rng.randint(900, 4000)} units left.")
        cases.append(("aim, route conflicts", s, turn_credit(b)))
    for _ in range(8):                                                       # enemy in crosshair
        s = (f"{vitals(rng)} Enemies in view: {rng.choice(kinds)} {rng.randint(100, 800)} away, in the crosshair. "
             f"{obstacles(rng)} Route to exit: {side(rng.uniform(-60, 60))}, {rng.randint(900, 4000)} units left.")
        cases.append(("fire", s, {"fire": 1.0}))
    for b in (20, 45, 80, -20, -45, -80, 170, -170):                         # no enemy, follow route
        s = (f"{vitals(rng)} Enemies in view: none. {obstacles(rng)} "
             f"Route to exit: {side(b)}, {rng.randint(900, 4000)} units left.")
        cases.append(("turn to route", s, turn_credit(b)))
    for _ in range(8):                                                       # route clear ahead
        s = (f"{vitals(rng)} Enemies in view: none. {obstacles(rng, ahead=f'{rng.randint(350, 900)} wall')} "
             f"Route to exit: straight ahead, {rng.randint(900, 4000)} units left.")
        cases.append(("walk route", s, {"forward": 1.0}))
    for _ in range(6):                                                       # closed door in reach on route
        s = (f"{vitals(rng)} Enemies in view: none. "
             f"{obstacles(rng, ahead=f'{rng.randint(30, 60)} door (closed), within use reach')} "
             f"Route to exit: straight ahead, {rng.randint(900, 4000)} units left.")
        cases.append(("open door", s, {"use": 1.0}))
    for _ in range(4):                                                       # exit switch in reach
        s = (f"{vitals(rng)} Enemies in view: none. "
             f"{obstacles(rng, ahead=f'{rng.randint(30, 60)} exit switch, within use reach')} "
             f"Route to exit: exit switch {rng.randint(30, 60)} away, straight ahead.")
        cases.append(("press exit", s, {"use": 1.0}))
    for _ in range(4):                                                       # blocked forward: route ahead but wall at 30
        s = (f"{vitals(rng)} Enemies in view: none. {obstacles(rng, ahead=f'{rng.randint(20, 40)} wall')} "
             f"Route to exit: straight ahead, {rng.randint(900, 4000)} units left. Last command: forward, moved 2 units (blocked).")
        cases.append(("blocked", s, {"strafe_left": 1.0, "strafe_right": 1.0, "turn_left_45": 0.5, "turn_right_45": 0.5,
                                     "turn_left_15": 0.5, "turn_right_15": 0.5, "back": 0.5}))
    return cases


def body(model: str, state: str, instructions: str) -> dict:
    return {"model": model, "state": state,
            "questions": {"cmd": {"type": "choice", "instructions": instructions, "criteria": dict(COMMANDS)}}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--url", default=HOSTED_URL)
    ap.add_argument("--instructions", default=None, help="file with alternative instructions text")
    ap.add_argument("--min-gap", type=float, default=0.55)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    instr = open(args.instructions, encoding="utf-8").read().strip() if args.instructions else INSTRUCTIONS
    cases = make_cases()
    client = JevClient(args.url, timeout=60)
    out = open(args.out, "a", encoding="utf-8") if args.out else None
    for model in args.models:
        per = collections.defaultdict(list)
        wrong_dir = 0
        last = 0.0
        for kind, state, credit in cases:
            wait = args.min_gap - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.monotonic()
            ans, _ = client.classify_retry(body(model, state, instr))
            score = credit.get(ans["choice"], 0.0)
            per[kind].append(score)
            if ans["choice"].startswith("turn_") and any(k.startswith("turn_") for k in credit):
                want_left = any("left" in k for k in credit if credit[k] == 1.0)
                if ("left" in ans["choice"]) != want_left and ans["choice"] != "turn_around":
                    wrong_dir += 1
            if out:
                out.write(json.dumps({"model": model, "instructions": args.instructions or "default", "kind": kind,
                                      "choice": ans["choice"], "score": score, "conf": ans["confidence"]}) + "\n")
        allscores = [s for v in per.values() for s in v]
        print(f"{model.split('/')[-1]:32s} [{args.instructions or 'default'}] score {statistics.mean(allscores):.2f}  "
              f"wrong-direction turns {wrong_dir}")
        for kind, v in per.items():
            print(f"    {kind:22s} {statistics.mean(v):.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
