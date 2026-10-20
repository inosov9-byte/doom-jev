"""Checkpoint drills: replay a logged run to a takeover tic, then let an arm play on from that exact state.

A campaign ends at a few spots (E1M4's junction, E1M3's last fight). Playing whole maps to reach them costs
minutes per sample and mixes in everything before. A drill starts at the spot: the source run (any logged run,
teacher or student) is replayed tic by tic up to the takeover tic, exactly as replay.py does (the run's own
executor, command set and loadout code when they differ from the working copy; position and health checked
against the log at every decision). Perception (working copy) reads every logged view on the way, so its
route stage, history and memory match a live run. At the takeover tic the chosen arm gets control and plays
in real time as play.py does: 35 tics/s, the same worker thread, snapshot after the last command settles.

Outcome: "pass" when the spot's success test holds with health > 0 (reach a point, or exit the map),
"died", "stuck" (no route progress and no damage dealt for stuck_s), or "timeout". Each drill writes
decisions.jsonl and summary.json like play.py (decisions after the takeover only), so drill runs feed
dataset.py as DAgger data.

Variety: every source run of the spot's map that reaches the trigger point, at each takeover offset (seconds
from the moment it first came within the trigger radius).

Check: --arm logged keeps applying the logged commands after the takeover (no real time) and compares the
outcome, game time and kills with the source run's summary: the handover itself must not change the game.

Paired comparisons: --variant name=PATH[,ENV=VAL...] runs each job with the code under PATH (a source root
with jevdoom/, e.g. a `git archive` of the baseline commit) and extra environment variables. Jobs are
interleaved by source state, so every variant meets the same states under the same machine load.

usage:
  python -m jevdoom.drill --spot e1m4-junction --arm rules-lag-50 --jobs 6 --out C:\\tmp\\jev-doom\\tmp\\drill-x
  python -m jevdoom.drill --spot e1m3-late --arm logged --jobs 6 --out ...          # handover check
  python -m jevdoom.drill --spot e1m4-junction --arm rules-lag-50 --variant base=C:\\tmp\\snap --variant cand=. ...
  python -m jevdoom.drill --list
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TICRATE = 35
REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("JEVDOOM_RUNS_ROOT", r"C:\Users\inoso.INPC\repos\jev-doom"))
WAD = r"C:\tmp\jev-doom\wad\doom1.wad"

# name -> map, source globs (run folders, relative to RUNS), trigger point (x, y, radius: takeover clock starts the
# first time the player comes this close), takeover offsets in seconds from that moment, success test, time limit.
SPOTS = {
    # Every student campaign that reached E1M4 died here; most teacher deaths too. North along x=-1090 into the
    # junction around (-1100, 750): imps and demons from three sides. Pass: the switch beyond it.
    "e1m4-junction": dict(map="E1M4", sources=["runs/campaign-2b-*/s*/E1M4/*", "data/camp-teacher/*/E1M4/*"],
                          trigger=(-1085, 0, 200), offsets=(-2.0, 1.0, 4.0),
                          success={"reach": (-1088, 1504, 96)}, timeout=60),
    # After E1M3's last trigger line (-296, -1600): shotgun guys and a zombieman beside it, then an imp pack on the
    # way east to the exit. Pass: the exit.
    "e1m3-late": dict(map="E1M3", sources=["runs/campaign-2b-*/s*/E1M3/*", "data/camp-teacher/*/E1M3/*"],
                      trigger=(-560, -1690, 160), offsets=(-2.0, 0.0, 2.0),
                      success={"exit": True}, timeout=75),
    # E1M3's corner fight near (-480, -700) at ~30%: shotgun guys around the corner. Pass: past it to (-160, -860)
    # (the blue keycard).
    "e1m3-corner": dict(map="E1M3", sources=["runs/campaign-2b-*/s*/E1M3/*", "data/camp-teacher/*/E1M3/*"],
                        trigger=(-1000, -880, 160), offsets=(-2.0, 0.0, 2.0),
                        success={"reach": (-160, -864, 80)}, timeout=60),
}


# -- sources ---------------------------------------------------------------------------------------------------
def _tag(run_dir: Path) -> str:
    return f"{run_dir.parents[2].name}-{run_dir.parents[1].name}"


def find_states(spot: dict, max_sources: int = 0, offsets=None) -> list[dict]:
    """(source run, takeover tic) pairs: runs of the spot's map that reach the trigger, at each offset."""
    tx, ty, tr = spot["trigger"]
    out, n = [], 0
    for pat in spot["sources"]:
        for d in sorted(glob.glob(str(RUNS / pat))):
            d = Path(d)
            if not (d / "summary.json").exists() or not (d / "decisions.jsonl").exists():
                continue
            summ = json.loads((d / "summary.json").read_text())
            if summ.get("map") != spot["map"] or summ.get("stepped"):
                continue
            rows = [json.loads(l) for l in (d / "decisions.jsonl").open(encoding="utf-8") if l.strip()]
            hit = next((r for r in rows if math.dist(r["pos"], (tx, ty)) <= tr), None)
            if hit is None:
                continue
            n += 1
            if max_sources and n > max_sources:
                break
            last = rows[-1]["tic"]
            for off in (offsets if offsets is not None else spot["offsets"]):
                t = int(round(hit["tic"] + off * TICRATE))
                if 50 <= t < last:
                    out.append({"source": str(d), "tag": _tag(d), "tic": t, "offset": off,
                                "source_outcome": summ["outcome"], "seed": summ["seed"]})
    return out


# -- restoring a state --------------------------------------------------------------------------------------------
class _SyncPool:
    """Plans immediately: during the replay there is no clock, and the route must be ready on the next view."""

    def submit(self, fn, *a):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*a))
        return f


def build_perception(m, loadout, carry: bool, needs: bool = True):
    """Goals, grids and Perception exactly as play.run builds them (working copy)."""
    from .needs import Needs
    from .perception import Perception
    from .progression import Subgoal, load_or_solve, stage_grids, with_supply_detours, with_weapon_detours
    from .wadmap import exit_target, path_length
    start = next((float(t[0]), float(t[1])) for t in m.things if t[3] == 1)
    try:
        held = set(loadout["weapons"]) if loadout else set()
        goals = with_supply_detours(m, with_weapon_detours(m, load_or_solve(m, start), start, held=held), start,
                                    held=held, resources=carry and needs)
    except RuntimeError:
        st, mid, _ = exit_target(m)
        goals = [Subgoal("exit", "exit", st, mid)]
    grids = stage_grids(m, goals)
    stage_len, stage_routes, here = [], [], start
    for gl, gr in zip(goals, grids):
        r = gr.plan(here, gl.pos)
        stage_routes.append(r)
        stage_len.append(path_length(r or [here, gl.pos]))
        here = gl.pos
    nd = Needs(carry=carry) if needs else None
    perc = Perception(grids[-1], compass=True, goals=goals, stage_grids=grids, stage_routes=stage_routes, needs=nd)
    return perc, nd, goals, stage_len


def restore(source: str | Path, takeover: int, *, timeout_s: int = 400, record: bool = False):
    """Replay the source run up to the takeover tic. Returns the live game and everything the loop needs."""
    from .env import blank_action, make_game, observe
    from .replay import load_apply_loadout, load_executor
    from .wadmap import load_map
    source = Path(source)
    summ = json.loads((source / "summary.json").read_text())
    rows = [json.loads(l) for l in (source / "decisions.jsonl").open(encoding="utf-8") if l.strip()]
    by_tic, by_obs = {}, {}
    for r in rows:
        by_tic.setdefault(r["tic"], []).append(r)
        by_obs.setdefault(r["obs_tic"], []).append(r)
    Executor, exe_src = load_executor(summ.get("commit"))
    apply_loadout = load_apply_loadout(summ.get("commit"))
    loadout = summ.get("start_loadout")
    carry = bool(summ.get("carry", loadout is not None))
    m = load_map(WAD, summ["map"])
    perc, nd, goals, stage_len = build_perception(m, loadout, carry, needs=summ.get("needs") is not None)
    live_planner = perc._planner
    perc._planner = _SyncPool()
    if nd is not None:
        live_needs_pool, nd._pool = nd._pool, _SyncPool()

    g = make_game(WAD, summ["map"], summ.get("skill", 3), visible=False, timeout_s=timeout_s, seed=summ["seed"],
                  cheats=loadout is not None)
    g.new_episode()
    if loadout:
        apply_loadout(g, loadout)
    for _ in range(10):                       # same warm-up as play.py
        g.make_action(blank_action())
    exe = Executor()
    tic, obs, last_cmd, prev_ammo, shot_tic = 0, None, None, None, -10**6
    mismatches, n_dec = [], 0
    while tic < takeover and not g.is_episode_finished():
        obs = observe(g, tic)
        ammo = (obs.weapon, {3: obs.shells, 5: obs.rockets}.get(obs.weapon, obs.bullets))
        if prev_ammo is not None and ammo[0] == prev_ammo[0] and ammo[1] < prev_ammo[1]:
            shot_tic = tic
        prev_ammo, obs.shot_tic = ammo, shot_tic
        for r in by_obs.get(tic, []):         # the view the worker described: keeps perception's state in step
            try:
                perc.describe(obs, last_cmd=last_cmd)
            except Exception:
                pass
            if r.get("choice"):
                last_cmd = r["choice"]
        for r in by_tic.get(tic, []):
            n_dec += 1
            got = {"pos": [round(obs.x), round(obs.y)], "health": obs.health}
            if "dealt" in r:
                got.update(dealt=round(obs.damage_dealt), hits=obs.hits)
            diff = {k: (r[k], v) for k, v in got.items() if r[k] != v}
            if diff:
                mismatches.append({"decision": n_dec, "tic": tic, "diff": diff})
            if r.get("choice"):
                exe.start(r["choice"])
        g.make_action(exe.step(obs), 1)
        tic += 1
    perc._planner = live_planner              # live play plans in the background again, as in play.py
    if nd is not None:
        nd._pool = live_needs_pool
    return {"game": g, "exe": exe, "perc": perc, "needs": nd, "goals": goals, "stage_len": stage_len, "tic": tic,
            "last_cmd": last_cmd, "prev_ammo": prev_ammo, "shot_tic": shot_tic, "summary": summ, "rows": rows,
            "by_tic": by_tic, "mismatches": mismatches, "executor": exe_src, "loadout": loadout, "carry": carry,
            "decisions_before": n_dec}


# -- playing on from a state --------------------------------------------------------------------------------------
def drill(job: dict) -> dict:
    """Play one drill (one source state, one arm) and write its run folder."""
    from .env import observe
    from .loadout import from_obs
    from .play import PANEL_W, Brain, draw_panel, git_commit
    from .wadmap import path_length
    spot = SPOTS[job["spot"]]
    arm = job["arm"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    limit = spot["timeout"]
    st = restore(job["source"], job["tic"], timeout_s=int(job["tic"] / TICRATE + limit + 30))
    g, exe, perc, nd, stage_len = st["game"], st["exe"], st["perc"], st["needs"], st["stage_len"]
    tic0 = tic = st["tic"]
    prev_ammo, shot_tic = st["prev_ammo"], st["shot_tic"]
    logged = arm == "logged"
    succ = spot["success"]
    obs = observe(g, tic) if not g.is_episode_finished() else None
    start_state = from_obs(obs) if obs else None
    start_pos = [round(obs.x), round(obs.y)] if obs else None

    engine = brain = None
    if not logged:
        from .engines import make_engine
        engine = make_engine(arm, local_url=job.get("local_url") or "http://127.0.0.1:8012", seed=st["summary"]["seed"])
        brain = Brain(engine, perc)
        brain.last_cmd = st["last_cmd"]
        brain.start()
    ffmpeg = None
    if job.get("video") and obs is not None and obs.screen is not None:
        h, w = obs.screen.shape[:2]
        ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w + PANEL_W}x{h}", "-r", str(TICRATE), "-i", "-", "-c:v", "libx264",
                                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                   str(run_dir / "run.mp4")], stdin=subprocess.PIPE)
    info = {"arm": arm, "seed": st["summary"]["seed"], "cmd": st["last_cmd"], "probs": {}, "state": "", "lat": 0.0,
            "n": 0, "left": 0.0}
    log = (run_dir / "decisions.jsonl").open("w", encoding="utf-8")
    n_dec, n_err, lat = 0, 0, []
    exe_seq = 0
    outcome, health_pass = None, None
    progress_left, progress_tic, progress_dealt = float("inf"), tic, obs.damage_dealt if obs else 0.0
    stuck_s = job.get("stuck_s", 30.0)
    end_tic = round(st["summary"]["game_seconds"] * TICRATE)
    t_start = time.perf_counter()
    try:
        while True:
            if g.is_episode_finished():
                break
            obs = observe(g, tic)
            if obs is None:
                break
            ammo = (obs.weapon, {3: obs.shells, 5: obs.rockets}.get(obs.weapon, obs.bullets))
            if prev_ammo is not None and ammo[0] == prev_ammo[0] and ammo[1] < prev_ammo[1]:
                shot_tic = tic
            prev_ammo, obs.shot_tic = ammo, shot_tic
            if logged:
                for r in st["by_tic"].get(tic, []):
                    if r.get("choice"):
                        exe.start(r["choice"])
                if st["summary"]["outcome"] in ("stuck", "timeout", "stopped") and tic >= end_tic:
                    outcome = st["summary"]["outcome"]
                    break
                if tic > end_tic + 5 * TICRATE:
                    outcome = "unfinished"
                    break
            else:
                brain.offer(obs)
                if exe_seq == brain.issued and exe.settled:
                    brain.applied.set()
                while not brain.answers.empty():
                    d = brain.answers.get()
                    n_dec += 1
                    rec = {"id": f"{run_dir.name}-{n_dec:05d}", "group_id": f"{run_dir.name}-{n_dec // 20:04d}",
                           "tic": tic, "obs_tic": d["obs_tic"], "state": d["state"],
                           "perception_s": round(d["perception_s"], 3), "error": d["error"],
                           "pos": [round(obs.x), round(obs.y)], "health": obs.health,
                           "dealt": round(obs.damage_dealt), "hits": obs.hits}
                    dec = d["decision"]
                    if dec is not None:
                        rec.update(choice=dec.choice, probabilities=dec.probabilities, latency=round(dec.latency, 3))
                        if dec.probabilities:
                            rec["confidence"] = dec.probabilities.get(dec.choice)
                        exe.start(dec.choice)
                        exe_seq = d["seq"]
                        lat.append(dec.latency)
                        info.update(cmd=dec.choice, probs=dec.probabilities, state=d["state"], lat=dec.latency, n=n_dec)
                    else:
                        n_err += 1
                    log.write(json.dumps(rec) + "\n")
            g.make_action(exe.step(obs), 1)
            tic += 1
            if not logged:
                if "reach" in succ and obs.health > 0:
                    x, y, r = succ["reach"]
                    if math.dist((obs.x, obs.y), (x, y)) <= r:
                        outcome, health_pass = "pass", obs.health
                        break
                if tic - tic0 > limit * TICRATE:
                    outcome = "timeout"
                    break
                if perc.route and not (nd and nd.goal):
                    left = (math.dist((obs.x, obs.y), perc.route[min(perc.idx, len(perc.route) - 1)])
                            + path_length(perc.route[perc.idx:]) + sum(stage_len[perc.stage + 1:]))
                    info["left"] = left
                    if left < progress_left - 50 or obs.damage_dealt > progress_dealt:
                        progress_tic = tic
                        progress_left, progress_dealt = min(progress_left, left), obs.damage_dealt
                    elif tic - progress_tic > stuck_s * TICRATE:
                        outcome = "stuck"
                        break
            if ffmpeg is not None and obs.screen is not None:
                info.update(t=tic / TICRATE, health=obs.health, kills=obs.kills)
                ffmpeg.stdin.write(draw_panel(obs.screen, info).tobytes())
            if not logged:
                delay = t_start + (tic - tic0) / TICRATE - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        if outcome is None:
            if g.is_player_dead():
                outcome = "died"
            elif g.is_episode_finished() and g.get_episode_time() < g.get_episode_timeout() - 1:
                outcome = "exit"
            else:
                outcome = "timeout"
        if outcome == "exit" and not logged:
            outcome, health_pass = "pass", obs.health if obs else None
    except Exception:
        outcome = "crashed"
        traceback.print_exc()
    finally:
        if brain is not None:
            brain.stop.set()
        log.close()
        if ffmpeg is not None:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        wall = time.perf_counter() - t_start
        src = st["summary"]
        summary = {"arm": arm, "seed": src["seed"], "outcome": outcome, "passed": outcome == "pass",
                   "health_pass": None if health_pass is None else round(health_pass),
                   "spot": job["spot"], "variant": job.get("variant", "cur"), "source": str(job["source"]),
                   "source_tag": job.get("tag"), "source_outcome": src["outcome"], "takeover_tic": tic0,
                   "takeover_offset_s": job.get("offset"), "takeover_state": start_state, "takeover_pos": start_pos,
                   "replay_mismatches": len(st["mismatches"]),
                   "first_divergence": st["mismatches"][0] if st["mismatches"] else None,
                   "source_executor": st["executor"],
                   "game_seconds": round(tic / TICRATE, 1), "drill_seconds": round((tic - tic0) / TICRATE, 1),
                   "wall_seconds": round(wall, 1), "kills": obs.kills if obs else 0,
                   "damage_taken": round(obs.damage_taken) if obs else 0,
                   "health_end": round(obs.health) if obs else None, "end_loadout": from_obs(obs) if obs else None,
                   "start_loadout": st["loadout"], "carry": st["carry"], "map": src["map"],
                   "plan": [gl.kind + ":" + gl.label + ("?" if gl.need else "") for gl in st["goals"]],
                   "needs": [list(e) for e in nd.log] if nd else None,
                   "decisions": n_dec, "errors": n_err,
                   "latency_median": round(statistics.median(lat), 3) if lat else None,
                   "engine": engine.describe() if engine else {"name": "logged"},
                   "commit": git_commit(), "run_dir": str(run_dir)}
        if logged:
            want = {k: src[k] for k in ("outcome", "game_seconds", "kills")}
            got = {"outcome": outcome, "game_seconds": summary["game_seconds"], "kills": summary["kills"]}
            summary["check"] = {"expected": want, "got": got, "exact": want == got and not st["mismatches"]}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        g.close()
    return summary


# -- many drills --------------------------------------------------------------------------------------------------
def _parse_variant(spec: str) -> tuple[str, Path, dict]:
    name, _, rest = spec.partition("=")
    parts = rest.split(",")
    root = Path(parts[0] or ".").resolve()
    env = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    return name, root, env


def _run_child(job: dict, root: Path, env_extra: dict) -> dict:
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    jf = run_dir / "job.json"
    jf.write_text(json.dumps(job))
    env = {**os.environ, **env_extra, "JEVDOOM_GIT_REPO": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    with (run_dir / "drill.log").open("w", encoding="utf-8") as lf:
        subprocess.run([sys.executable, "-m", "jevdoom.drill", "--one", str(jf)], cwd=root, env=env,
                       stdout=lf, stderr=subprocess.STDOUT)
    try:
        return json.loads((run_dir / "summary.json").read_text())
    except (OSError, ValueError):
        return {**job, "outcome": "crashed", "passed": False, "health_pass": None}


def server_up(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            return b"ready" in r.read()
    except OSError:
        return False


def table(results: list[dict]) -> str:
    keys = sorted({(r.get("variant", "cur"), r["arm"]) for r in results})
    out = "| variant | arm | drills | pass | rate | died | stuck | timeout | other | median health at pass |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for v, a in keys:
        rs = [r for r in results if r.get("variant", "cur") == v and r["arm"] == a]
        n = len(rs)
        p = [r for r in rs if r.get("passed")]
        c = lambda o: sum(r["outcome"] == o for r in rs)
        hp = [r["health_pass"] for r in p if r.get("health_pass") is not None]
        other = n - len(p) - c("died") - c("stuck") - c("timeout")
        out += (f"| {v} | {a} | {n} | {len(p)} | {len(p) / max(n, 1):.0%} | {c('died')} | {c('stuck')} | {c('timeout')} "
                f"| {other} | {statistics.median(hp) if hp else '-'} |\n")
    return out


def paired(results: list[dict]) -> str:
    """Per source state: pass/fail of each variant x arm side by side, and discordant counts."""
    cols = sorted({(r.get("variant", "cur"), r["arm"]) for r in results})
    if len(cols) < 2:
        return ""
    states: dict = {}
    for r in results:
        states.setdefault((r.get("source_tag"), r.get("takeover_tic")), {})[(r.get("variant", "cur"), r["arm"])] = r
    a, b = cols[0], cols[1]
    both = [s for s in states.values() if a in s and b in s]
    only_a = sum(s[a].get("passed") and not s[b].get("passed") for s in both)
    only_b = sum(s[b].get("passed") and not s[a].get("passed") for s in both)
    return (f"\nPaired over {len(both)} states: only {a[0]}/{a[1]} passed {only_a}, only {b[0]}/{b[1]} passed {only_b}, "
            f"both {sum(s[a].get('passed') and s[b].get('passed') for s in both)}, "
            f"neither {sum(not s[a].get('passed') and not s[b].get('passed') for s in both)}.\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", choices=list(SPOTS))
    ap.add_argument("--arm", nargs="+", default=["rules-lag-50"], help="rules, rules-lag-50, jev-local, ... or logged")
    ap.add_argument("--variant", action="append", default=[], help="name=PATH[,ENV=VAL...] (default: cur=.)")
    ap.add_argument("--jobs", type=int, default=6, help="parallel games for rules/logged arms (jev-local runs one at a time)")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--offsets", nargs="+", type=float, help="takeover offsets in seconds (default: the spot's)")
    ap.add_argument("--max-sources", type=int, default=0)
    ap.add_argument("--sources", nargs="+", help="only source runs whose tag contains one of these")
    ap.add_argument("--repeat", type=int, default=1, help="play every state this many times per arm")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--local-url", default="http://127.0.0.1:8012")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true", help="list spots and their source states")
    args = ap.parse_args(argv)

    if args.one:
        s = drill(json.loads(Path(args.one).read_text()))
        print(json.dumps({k: s[k] for k in ("outcome", "health_pass", "drill_seconds", "replay_mismatches")
                          if k in s} | ({"check": s["check"]} if "check" in s else {})))
        return
    if args.list:
        for name, sp in SPOTS.items():
            if args.spot and name != args.spot:
                continue
            st = find_states(sp, args.max_sources, args.offsets)
            print(f"{name}: {sp['map']}, {len({s['tag'] for s in st})} source runs, {len(st)} states")
        return
    if not args.spot or not args.out:
        ap.error("--spot and --out are required")

    spot = SPOTS[args.spot]
    states = find_states(spot, args.max_sources, args.offsets)
    if args.sources:
        states = [s for s in states if any(k in s["tag"] for k in args.sources)]
    variants = [_parse_variant(v) for v in args.variant] or [("cur", REPO, {})]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "jev-local" in args.arm and not server_up(args.local_url):
        print(f"jev-local: no server at {args.local_url}; skipping the student arm", flush=True)
        args.arm = [a for a in args.arm if a != "jev-local"]
    jobs = []
    for rep in range(args.repeat):
        for s in states:
            for vname, root, env in variants:
                for arm in args.arm:
                    name = f"{s['tag']}-t{s['tic']}" + (f"-r{rep}" if args.repeat > 1 else "")
                    jobs.append(({**s, "spot": args.spot, "arm": arm, "variant": vname, "video": args.video,
                                  "local_url": args.local_url, "run_dir": str(out / vname / arm / name)}, root, env))
    print(f"{args.spot}: {len(states)} states from {len({s['tag'] for s in states})} runs, {len(jobs)} drills", flush=True)
    results, lock = [], threading.Lock()
    meta = {"spot": args.spot, "spec": {k: v for k, v in spot.items()}, "arms": args.arm,
            "variants": [(n, str(r), e) for n, r, e in variants], "states": len(states), "started": time.strftime("%Y%m%d-%H%M%S")}

    def done(s):
        with lock:
            results.append(s)
            print(f"  {s.get('variant')}/{s['arm']:13s} {s.get('source_tag')} t{s.get('takeover_tic')}: {s['outcome']:8s} "
                  f"hp {s.get('health_pass')}  ({len(results)}/{len(jobs)})", flush=True)
            (out / "results.json").write_text(json.dumps({"meta": meta, "runs": results}, indent=2))

    cpu = [j for j in jobs if j[0]["arm"] != "jev-local"]
    gpu = [j for j in jobs if j[0]["arm"] == "jev-local"]
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for f in as_completed([pool.submit(_run_child, *j) for j in cpu]):
            done(f.result())
    for j in gpu:                              # one student game at a time: the GPU server is shared
        done(_run_child(*j))
    md = f"# Drill {args.spot}\n\n`{json.dumps(meta)}`\n\n" + table(results) + paired(results)
    (out / "results.md").write_text(md)
    (out / "results.json").write_text(json.dumps({"meta": meta, "runs": results}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
