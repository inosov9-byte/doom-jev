"""Replay a recorded run tic by tic, with game audio.

A live run is deterministic given the map, seed, skill, start loadout and the command applied at
each tic: the game is synchronous and seeded, and the executor macros read only the observation
(no wall clock). decisions.jsonl logs the tic each command started, so the replay starts the same
episode, starts each logged command at its logged tic through the same executor, and steps one tic
at a time with ViZDoom's audio buffer on. No engine, no real time.

Checks: per decision, player position (rounded), health, damage dealt and hits at the logged tic
against the log (first divergence reported); final outcome, game time and kills against
summary.json.

Video: by default the replay audio is muxed onto the run's own run.mp4 (stream copy, so the picture
is exactly the recorded one); --render redraws game + decision panel from the replay instead (panel
text from the log; the "exit N units" figure comes from a synchronous Perception and can differ
slightly from the live one, whose route planner ran in the background).

The executor and command set are loaded from the run's recorded commit when they differ from the
working copy (runs before 2026-10-18 evening held `fire` for 20 tics, later ones for up to 40).

usage: python -m jevdoom.replay RUN_DIR [--out DIR] [--render] [--rate 44100]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import types
import wave
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .env import blank_action, make_game, observe

TICRATE = 35
REPO = Path(__file__).resolve().parent.parent
FFMPEG = r"C:\tmp\jev-doom\bin\ffmpeg.exe"


GIT_REPO = Path(os.environ.get("JEVDOOM_GIT_REPO", REPO))   # where recorded commits are looked up


def _git_show(commit: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=GIT_REPO, capture_output=True, text=True,
                       encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


def load_at_commit(commit: str | None, mods: tuple[str, ...] = ("commands", "executor")) -> tuple[dict, str]:
    """The given jevdoom modules as of the run's commit (working copy if unknown or identical). Each may
    import env and the other modules listed; env (the ViZDoom wrapper) always comes from the working copy."""
    import importlib
    cur = {m: importlib.import_module(f"{__package__}.{m}") for m in mods}
    base = (commit or "").replace("-dirty", "")
    srcs = {m: _git_show(base, f"jevdoom/{m}.py") for m in mods} if base else {}
    now = {m: (REPO / "jevdoom" / f"{m}.py").read_text(encoding="utf-8") for m in mods}
    if not srcs or any(v is None for v in srcs.values()) or all(
            srcs[m].replace("\r\n", "\n") == now[m].replace("\r\n", "\n") for m in srcs):
        return cur, "working copy"
    pkg = f"_jevdoom_{base}"
    if pkg not in sys.modules:
        sys.modules[pkg] = types.ModuleType(pkg)
        sys.modules[pkg].__path__ = []
        from . import env
        sys.modules[f"{pkg}.env"] = env
    out = {}
    for m in mods:                    # in order: a module may import the ones before it
        name = f"{pkg}.{m}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__package__ = pkg
            sys.modules[name] = mod
            exec(compile(srcs[m], f"{base}:jevdoom/{m}.py", "exec"), mod.__dict__)
        out[m] = sys.modules[name]
    return out, f"commit {base}"


def load_executor(commit: str | None):
    """Executor class as of the run's commit (working copy if unknown or identical)."""
    mods, src = load_at_commit(commit)
    return mods["executor"].Executor, src


def load_apply_loadout(commit: str | None):
    """loadout.apply as of the run's commit: the start state must be built the way the run built it (before
    d2dc68d, partial armor was dropped and ammo above the normal maximum clipped)."""
    mods, _ = load_at_commit(commit, ("loadout",))
    return mods["loadout"].apply


class _SyncPool:
    """Stand-in for Perception's background planner: plans immediately (replay has no clock)."""

    def submit(self, fn, *a):
        f = Future()
        f.set_result(fn(*a))
        return f


def _perception(m, wad_map_start, loadout):
    from .perception import Perception
    from .progression import Subgoal, load_or_solve, stage_grids, with_ambush_holds, with_supply_detours, with_weapon_detours
    from .wadmap import exit_target, path_length
    start = wad_map_start
    try:
        held = set(loadout["weapons"]) if loadout else set()
        goals = with_supply_detours(m, with_weapon_detours(m, load_or_solve(m, start), start, held=held), start,
                                    held=held)
    except RuntimeError:
        st, mid, _ = exit_target(m)
        goals = [Subgoal("exit", "exit", st, mid)]
    goals = with_ambush_holds(m, goals, start)
    grids = stage_grids(m, goals)
    stage_len, stage_routes, here = [], [], start
    for gl, gr in zip(goals, grids):
        r = gr.plan(here, gl.pos)
        stage_routes.append(r)
        stage_len.append(path_length(r or [here, gl.pos]))
        here = gl.pos
    perc = Perception(grids[-1], compass=True, goals=goals, stage_grids=grids, stage_routes=stage_routes)
    perc._planner = _SyncPool()
    return perc, stage_len


def replay(run_dir: str | Path, out_dir: str | Path, *, wad: str = r"C:\tmp\jev-doom\wad\doom1.wad",
           rate: int = 44100, render: bool = False, name: str | None = None) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or run_dir.name
    summ = json.loads((run_dir / "summary.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "decisions.jsonl").open(encoding="utf-8")]
    by_tic: dict[int, list] = {}
    for r in rows:
        by_tic.setdefault(r["tic"], []).append(r)
    Executor, exe_src = load_executor(summ.get("commit"))
    apply_loadout = load_apply_loadout(summ.get("commit"))
    loadout = summ.get("start_loadout")
    end_tic = round(summ["game_seconds"] * TICRATE)

    perc = stage_len = None
    by_obs: dict[int, list] = {}
    if render:
        from .wadmap import load_map, path_length
        m = load_map(wad, summ["map"])
        start = next((float(t[0]), float(t[1])) for t in m.things if t[3] == 1)
        perc, stage_len = _perception(m, start, loadout)
        for r in rows:
            by_obs.setdefault(r["obs_tic"], []).append(r)
        total_route = summ.get("route_total") or sum(stage_len)
        from .play import PANEL_W, draw_panel
        info = {"arm": summ["arm"], "seed": summ["seed"], "cmd": None, "probs": {}, "state": "", "lat": 0.0,
                "n": 0, "left": total_route}
        ff = None

    g = make_game(wad, summ["map"], summ.get("skill", 3), visible=False, timeout_s=summ.get("timeout_s", 300),
                  seed=summ["seed"], cheats=loadout is not None, audio_rate=rate)
    g.new_episode()
    if loadout:
        apply_loadout(g, loadout)
    for _ in range(10):                       # same warm-up as play.py
        g.make_action(blank_action())
    exe = Executor()

    chunks, mismatches, checked = [], [], 0
    tic, obs, n_dec, last_cmd = 0, None, 0, None
    prev_ammo, shot_tic = None, -10**6
    while not g.is_episode_finished() and tic < end_tic + 5 * TICRATE:
        if summ["outcome"] in ("stuck", "timeout", "stopped") and tic >= end_tic:
            break
        obs = observe(g, tic)
        if obs is None:
            break
        chunks.append(obs.audio)
        ammo = (obs.weapon, {3: obs.shells, 5: obs.rockets}.get(obs.weapon, obs.bullets))
        if prev_ammo is not None and ammo[0] == prev_ammo[0] and ammo[1] < prev_ammo[1]:
            shot_tic = tic
        prev_ammo, obs.shot_tic = ammo, shot_tic
        if perc is not None:
            for r in by_obs.get(tic, []):     # the worker described this view; keeps route tracking in step
                try:
                    perc.describe(obs, last_cmd=last_cmd)
                except Exception:
                    pass
                if r.get("choice"):
                    last_cmd = r["choice"]
        for r in by_tic.get(tic, []):
            n_dec += 1
            checked += 1
            got = {"pos": [round(obs.x), round(obs.y)], "health": obs.health}
            if "dealt" in r:
                got.update(dealt=round(obs.damage_dealt), hits=obs.hits)
            diff = {k: (r[k], v) for k, v in got.items() if r[k] != v}
            if diff:
                mismatches.append({"decision": n_dec, "tic": tic, "diff": diff})
            if r.get("choice"):
                exe.start(r["choice"])
                if perc is not None:
                    info.update(cmd=r["choice"], probs=r.get("probabilities"), state=r.get("state") or "",
                                lat=r.get("latency") or 0.0, n=n_dec)
        g.make_action(exe.step(obs), 1)
        tic += 1
        if perc is not None:
            if perc.route:
                info["left"] = (math.dist((obs.x, obs.y), perc.route[perc.idx]) + path_length(perc.route[perc.idx:])
                                + sum(stage_len[perc.stage + 1:]))
            if obs.screen is not None:
                if ff is None:
                    h, w = obs.screen.shape[:2]
                    ff = subprocess.Popen([FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                           "-s", f"{w + PANEL_W}x{h}", "-r", str(TICRATE), "-i", "-", "-c:v", "libx264",
                                           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                           str(out_dir / f"{name}-render.mp4")], stdin=subprocess.PIPE)
                info.update(t=tic / TICRATE, health=obs.health, kills=obs.kills)
                ff.stdin.write(draw_panel(obs.screen, info).tobytes())

    if g.is_episode_finished():
        outcome = "died" if g.is_player_dead() else (
            "exit" if g.get_episode_time() < g.get_episode_timeout() - 1 else "timeout")
    else:
        outcome = summ["outcome"] if tic >= end_tic else "unfinished"
    kills = obs.kills if obs else 0
    g.close()
    if perc is not None and ff is not None:
        ff.stdin.close()
        ff.wait()

    audio = np.concatenate([c if c is not None else np.zeros((rate // TICRATE, 2), np.int16) for c in chunks])
    wav = out_dir / f"{name}.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.ascontiguousarray(audio.astype(np.int16)).tobytes())

    report = {"run": str(run_dir), "executor": exe_src, "decisions_checked": checked,
              "mismatches": len(mismatches), "first_divergence": mismatches[0] if mismatches else None,
              "tics": tic, "game_seconds": round(tic / TICRATE, 1), "outcome": outcome, "kills": kills,
              "expected": {k: summ[k] for k in ("outcome", "game_seconds", "kills")},
              "audio_wav": str(wav), "audio_seconds": round(len(audio) / rate, 3),
              "audio_peak": int(np.abs(audio.astype(np.int32)).max()) if len(audio) else 0}
    report["exact"] = (not mismatches and outcome == summ["outcome"] and report["game_seconds"] == summ["game_seconds"]
                       and kills == summ["kills"])
    (out_dir / f"{name}-replay.json").write_text(json.dumps(report, indent=2))
    return report


def mux(video: str | Path, wav: str | Path, out: str | Path, vf: str | None = None, crf: int | None = None) -> None:
    """Video + replay audio. Without vf the picture is stream-copied; AAC 160 kb/s, faststart."""
    cmd = [FFMPEG, "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0"]
    if vf:
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", str(crf or 20), "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    cmd += ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=r"C:\tmp\jev-doom\audio")
    ap.add_argument("--wad", default=r"C:\tmp\jev-doom\wad\doom1.wad")
    ap.add_argument("--rate", type=int, default=44100, choices=(11025, 22050, 44100))
    ap.add_argument("--render", action="store_true", help="redraw game + panel from the replay")
    args = ap.parse_args(argv)
    rep = replay(args.run_dir, args.out, wad=args.wad, rate=args.rate, render=args.render)
    out = Path(args.out)
    name = Path(args.run_dir).name
    src = out / f"{name}-render.mp4" if args.render else Path(args.run_dir) / "run.mp4"
    mux(src, rep["audio_wav"], out / f"{name}-sound.mp4")
    rep["video"] = str(out / f"{name}-sound.mp4")
    print(json.dumps(rep, indent=2))
    return 0 if rep["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
