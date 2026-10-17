"""A decision engine plays DOOM E1M1 in real time.

The game advances at 35 tics/s no matter what. A worker thread takes the latest
observation, turns it into text, asks the engine (Jev or Haiku, see engines.py) for one
command, and hands the answer to the game loop, which plays it as a short macro. The next
request waits until that command has taken effect. No other policy acts.

usage: python -m jevdoom.play --arm jev-local [--seed 1] [--record-video] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import queue
import statistics
import subprocess
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from .commands import COMMAND_IDS
from .engines import ARMS, Engine, make_engine
from .env import blank_action, make_game, observe
from .executor import Executor
from .jev import LOCAL_URL
from .perception import Perception
from .wadmap import NavGrid, exit_target, load_map, path_length

TICRATE = 35
PANEL_W = 400
REPO = Path(__file__).resolve().parent.parent


class Brain(threading.Thread):
    """Serial decision loop: latest obs -> text -> engine -> command."""

    def __init__(self, engine: Engine, perc: Perception):
        super().__init__(daemon=True)
        self.engine, self.perc = engine, perc
        self.latest = None
        self.lock = threading.Lock()
        self.have_obs = threading.Event()
        self.answers: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.applied = threading.Event()      # set by the game loop once the last command took effect
        self.applied.set()
        self.issued = 0                       # sequence number of the latest answer with a command
        self.last_cmd: str | None = None

    def offer(self, obs):
        with self.lock:
            self.latest = obs
        self.have_obs.set()

    def run(self):
        last_start = 0.0
        while not self.stop.is_set():
            self.have_obs.wait(0.5)
            with self.lock:
                obs = self.latest
            if obs is None:
                continue
            self.applied.wait(1.5)                  # never decide on a view that predates the last command
            wait = self.engine.min_gap - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)
            with self.lock:
                obs = self.latest                   # freshest state right before the call
            last_start = time.monotonic()
            t0 = time.perf_counter()
            state = self.perc.describe(obs, last_cmd=self.last_cmd)
            t_perc = time.perf_counter() - t0
            try:
                dec, err = self.engine.decide(state), None
            except Exception as e:
                dec, err = None, f"{type(e).__name__}: {str(e)[:200]}"
            if dec is not None:
                self.issued += 1                    # bump first: the game loop compares against it before clear
                self.applied.clear()
                self.last_cmd = dec.choice
            self.answers.put({"obs_tic": obs.tic, "state": state, "decision": dec, "perception_s": t_perc,
                              "error": err, "seq": self.issued, "wall": time.perf_counter() - t0})
            if err:
                time.sleep(1.0)


def draw_panel(frame: np.ndarray, info: dict) -> np.ndarray:
    h = frame.shape[0]
    panel = np.full((h, PANEL_W, 3), 24, np.uint8)
    y = 28

    def put(text, size=0.5, color=(220, 220, 220), thick=1):
        nonlocal y
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y += int(26 * size / 0.5)

    put(f"{info['arm']}  seed {info['seed']}", 0.55, (120, 200, 255), 1)
    put(f"t={info.get('t', 0):.1f}s  decisions {info['n']}  latency {info['lat']:.2f}s", 0.45)
    put(f"health {info.get('health', 0):.0f}  kills {info.get('kills', 0)}  exit {info['left']:.0f} units", 0.45)
    y += 6
    put(f"> {info['cmd'] or '-'}", 0.8, (80, 255, 120), 2)
    y += 4
    probs = info.get("probs")
    if probs:
        for k in sorted(COMMAND_IDS, key=lambda c: -probs.get(c, 0))[:6]:
            p = probs.get(k, 0)
            cv2.rectangle(panel, (150, y - 12), (150 + int(230 * p), y + 2), (80, 160, 255), -1)
            put(f"{k:15s} {p:4.2f}", 0.42)
    elif info.get("cmd"):
        put("(no probabilities: generative engine)", 0.42, (150, 150, 150))
    else:
        put("(waiting for the first decision)", 0.42, (150, 150, 150))
    y += 8
    words, line = (info.get("state") or "").split(), ""
    for w in words:
        if len(line) + len(w) > 52:
            put(line, 0.36, (170, 170, 170))
            line = ""
        line += w + " "
        if y > h - 10:
            break
    if line and y <= h - 10:
        put(line, 0.36, (170, 170, 170))
    return np.hstack([frame, panel])


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def run(arm: str, seed: int | None = None, *, wad: str = r"C:\tmp\jev-doom\wad\doom1.wad", map_name: str = "E1M1",
        skill: int = 3, local_url: str = LOCAL_URL, timeout: int = 300, window: bool = False,
        record_video: bool = False, runs_dir: str = "runs", compass: bool = True, dry_run: bool = False,
        engine: Engine | None = None, stepped: bool = False, quiet: bool = False) -> dict:
    """stepped=True: the game waits for every decision (no real time, no worker thread). For
    developing the rules expert and recording expert trajectories; not used for the comparison."""
    engine = engine or make_engine(arm, local_url=local_url, dry_run=dry_run, seed=seed or 0)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(runs_dir) / (f"{stamp}-{arm}" + (f"-s{seed}" if seed is not None else ""))
    run_dir.mkdir(parents=True, exist_ok=True)
    m = load_map(wad, map_name)
    grid = NavGrid(m)
    perc = Perception(grid, compass=compass)
    stand, _, _ = exit_target(m)
    start = next((t[0], t[1]) for t in m.things if t[3] == 1)
    total_route = path_length(grid.plan(start, stand))

    g = make_game(wad, map_name, skill, visible=window, timeout_s=timeout, seed=seed)
    g.new_episode()
    for _ in range(10):                      # ViZDoom ignores input for the first 7 tics
        g.make_action(blank_action())
    exe = Executor()
    brain = Brain(engine, perc)
    if not stepped:
        brain.start()

    ffmpeg = None
    if record_video:
        h, w = g.get_screen_height(), g.get_screen_width()
        ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w + PANEL_W}x{h}", "-r", str(TICRATE), "-i", "-", "-c:v", "libx264",
                                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                   str(run_dir / "run.mp4")], stdin=subprocess.PIPE)

    log = (run_dir / "decisions.jsonl").open("w", encoding="utf-8")
    tic, n_dec, n_err, lat, best_left = 0, 0, 0, [], total_route
    tok_in = tok_out = 0
    cost = 0.0
    exe_seq = 0
    obs = None
    info = {"arm": arm, "seed": seed, "cmd": None, "probs": {}, "state": "", "lat": 0.0, "n": 0, "left": total_route}
    t_start = time.perf_counter()
    outcome = "timeout"
    try:
        while True:
            if g.is_episode_finished():
                break
            obs = observe(g, tic)
            if obs is None:
                break
            if stepped and exe.settled:
                if exe_seq == brain.issued:                  # previous command took effect: decide now
                    t0 = time.perf_counter()
                    st = perc.describe(obs, last_cmd=brain.last_cmd)
                    try:
                        dec, err = engine.decide(st), None
                    except Exception as e:
                        dec, err = None, f"{type(e).__name__}: {str(e)[:200]}"
                    if dec is not None:
                        brain.issued += 1
                        brain.last_cmd = dec.choice
                    brain.answers.put({"obs_tic": obs.tic, "state": st, "decision": dec, "perception_s": 0.0,
                                       "error": err, "seq": brain.issued, "wall": time.perf_counter() - t0})
            if not stepped:
                brain.offer(obs)
                # Release the worker only now: this obs was taken after the last command's tics ran.
                # (Setting it right after make_action let the worker read the pre-command obs.)
                if exe_seq == brain.issued and exe.settled:
                    brain.applied.set()
            while not brain.answers.empty():
                d = brain.answers.get()
                n_dec += 1
                rec = {"id": f"{run_dir.name}-{n_dec:05d}", "group_id": f"{run_dir.name}-{n_dec // 20:04d}",
                       "tic": tic, "obs_tic": d["obs_tic"], "state": d["state"], "perception_s": round(d["perception_s"], 3),
                       "error": d["error"], "pos": [round(obs.x), round(obs.y)], "health": obs.health}
                dec = d["decision"]
                if dec is not None:
                    rec.update(choice=dec.choice, probabilities=dec.probabilities, latency=round(dec.latency, 3),
                               input_tokens=dec.input_tokens, output_tokens=dec.output_tokens, cost_usd=dec.cost_usd)
                    if dec.probabilities:
                        rec["confidence"] = dec.probabilities.get(dec.choice)
                    exe.start(dec.choice)
                    exe_seq = d["seq"]
                    lat.append(dec.latency)
                    tok_in, tok_out, cost = tok_in + dec.input_tokens, tok_out + dec.output_tokens, cost + dec.cost_usd
                    info.update(cmd=dec.choice, probs=dec.probabilities, state=d["state"], lat=dec.latency, n=n_dec)
                else:
                    n_err += 1
                log.write(json.dumps(rec) + "\n")
            g.make_action(exe.step(obs), 1)
            tic += 1
            if perc.route:
                left = math.dist((obs.x, obs.y), perc.route[perc.idx]) + path_length(perc.route[perc.idx:])
                best_left = min(best_left, left)
                info["left"] = left
            if ffmpeg is not None and obs.screen is not None:
                info.update(t=tic / TICRATE, health=obs.health, kills=obs.kills)
                ffmpeg.stdin.write(draw_panel(obs.screen, info).tobytes())
            if not stepped:
                target = t_start + tic / TICRATE               # hold real time: 35 tics per second
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        if g.is_player_dead():
            outcome = "died"
        elif g.get_episode_time() < g.get_episode_timeout() - 1:   # ended early and alive: level exit
            outcome = "exit"
            best_left = 0.0
    except KeyboardInterrupt:
        outcome = "stopped"
    except Exception:
        outcome = "crashed"
        traceback.print_exc()
    finally:
        brain.stop.set()
        log.close()
        if ffmpeg is not None:
            ffmpeg.stdin.close()
            ffmpeg.wait()
        wall = time.perf_counter() - t_start
        lat_sorted = sorted(lat)
        summary = {"arm": arm, "seed": seed, "outcome": outcome,
                   "route_progress_pct": round(100 * (1 - best_left / total_route), 1),
                   "game_seconds": round(tic / TICRATE, 1), "wall_seconds": round(wall, 1),
                   "kills": obs.kills if obs else 0, "damage_taken": round(obs.damage_taken) if obs else 0,
                   "health_end": round(obs.health) if obs else None,
                   "decisions": n_dec, "errors": n_err, "decisions_per_s": round(n_dec / max(wall, 1e-9), 2),
                   "latency_median": round(statistics.median(lat), 3) if lat else None,
                   "latency_p90": round(lat_sorted[int(0.9 * (len(lat_sorted) - 1))], 3) if lat else None,
                   "input_tokens": tok_in, "output_tokens": tok_out, "cost_usd": round(cost, 4),
                   "route_total": round(total_route), "best_route_left": round(best_left),
                   "engine": engine.describe(), "skill": skill, "map": map_name, "timeout_s": timeout,
                   "compass": compass, "dry_run": dry_run, "stepped": stepped, "commit": git_commit(),
                   "run_dir": str(run_dir)}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        g.close()
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="jev-local", choices=ARMS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wad", default=r"C:\tmp\jev-doom\wad\doom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--local-url", default=LOCAL_URL)
    ap.add_argument("--timeout", type=int, default=300, help="episode limit in seconds of game time")
    ap.add_argument("--window", action="store_true", help="show the game window")
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--no-compass", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="mock engine with the arm's latency profile; free")
    ap.add_argument("--stepped", action="store_true", help="game waits for each decision (expert development)")
    args = ap.parse_args(argv)
    s = run(args.arm, args.seed, wad=args.wad, map_name=args.map, skill=args.skill, local_url=args.local_url,
            timeout=args.timeout, window=args.window, record_video=args.record_video, runs_dir=args.runs,
            compass=not args.no_compass, dry_run=args.dry_run, stepped=args.stepped)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
