"""Jev plays DOOM E1M1 in real time.

The game advances at 35 tics/s no matter what. A worker thread repeatedly takes the
latest observation, turns it into text, asks Jev for one command, and hands the answer
to the game loop, which plays it as a short macro. While Jev thinks, the player
finishes the previous macro and then stands still. No other policy acts.

usage: python -m jevdoom.play --url http://127.0.0.1:8012 [--teacher] [--record-video]
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
from .executor import Executor
from .env import blank_action, make_game, observe
from .jev import HOSTED_URL, TEACHER_MODEL, JevClient, student_request, teacher_request
from .perception import Perception
from .wadmap import NavGrid, exit_target, load_map, path_length

TICRATE = 35
PANEL_W = 400


class Brain(threading.Thread):
    """Serial decision loop: latest obs -> text -> Jev -> command."""

    def __init__(self, client: JevClient, perc: Perception, make_body, min_gap: float):
        super().__init__(daemon=True)
        self.client, self.perc, self.make_body, self.min_gap = client, perc, make_body, min_gap
        self.latest = None
        self.lock = threading.Lock()
        self.have_obs = threading.Event()
        self.answers: queue.Queue = queue.Queue()
        self.stop = threading.Event()
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
            wait = self.min_gap - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)
            with self.lock:
                obs = self.latest                       # freshest state right before the call
            last_start = time.monotonic()
            t0 = time.perf_counter()
            state = self.perc.describe(obs, last_cmd=self.last_cmd)
            t_perc = time.perf_counter() - t0
            try:
                ans, latency = self.client.classify(self.make_body(state))
                err = None
            except Exception as e:
                ans, latency, err = None, time.perf_counter() - t0, f"{type(e).__name__}: {e}"
            self.answers.put({"obs_tic": obs.tic, "state": state, "answer": ans, "latency": latency,
                              "perception_s": t_perc, "error": err})
            if ans is not None:
                self.last_cmd = ans["choice"]
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

    put(f"JEV  {info['model']}", 0.55, (120, 200, 255), 1)
    put(f"t={info['t']:.1f}s  decisions {info['n']}  latency {info['lat']:.2f}s", 0.45)
    put(f"health {info['health']:.0f}  kills {info['kills']}  exit {info['left']:.0f} units", 0.45)
    y += 6
    put(f"> {info['cmd'] or '-'}", 0.8, (80, 255, 120), 2)
    y += 4
    probs = info.get("probs") or {}
    for k in sorted(COMMAND_IDS, key=lambda c: -probs.get(c, 0))[:6]:
        p = probs.get(k, 0)
        cv2.rectangle(panel, (150, y - 12), (150 + int(230 * p), y + 2), (80, 160, 255), -1)
        put(f"{k:15s} {p:4.2f}", 0.42)
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=r"C:\tmp\jev-doom\wad\doom1.wad")
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--url", default="http://127.0.0.1:8012")
    ap.add_argument("--teacher", action="store_true", help="use the hosted 35B teacher instead of the local model")
    ap.add_argument("--full-prompt", action="store_true", help="send the teacher-style prompt to a local model (zero-shot)")
    ap.add_argument("--model-name", default="local")
    ap.add_argument("--min-gap", type=float, default=0.0, help="seconds between request starts (0.55 for the hosted demo)")
    ap.add_argument("--timeout", type=int, default=600, help="episode limit in seconds of game time")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--no-compass", action="store_true")
    args = ap.parse_args(argv)

    if args.teacher:
        client, make_body, name = JevClient(HOSTED_URL), teacher_request, TEACHER_MODEL.split("/")[-1]
        args.min_gap = max(args.min_gap, 0.55)
    else:
        client, name = JevClient(args.url), args.model_name
        make_body = (lambda s: student_request(s, full_prompt=args.full_prompt))

    run_dir = Path(args.runs) / (dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    m = load_map(args.wad, args.map)
    grid = NavGrid(m)
    perc = Perception(grid, compass=not args.no_compass)
    stand, _, _ = exit_target(m)
    start = next((t[0], t[1]) for t in m.things if t[3] == 1)
    total_route = path_length(grid.plan(start, stand))

    g = make_game(args.wad, args.map, args.skill, visible=not args.no_window, timeout_s=args.timeout)
    g.new_episode()
    for _ in range(10):                      # ViZDoom ignores input for the first 7 tics
        g.make_action(blank_action())
    exe = Executor()
    brain = Brain(client, perc, make_body, args.min_gap)
    brain.start()

    ffmpeg = None
    if args.record_video:
        h, w = g.get_screen_height(), g.get_screen_width()
        ffmpeg = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                   "-s", f"{w + PANEL_W}x{h}", "-r", str(TICRATE), "-i", "-", "-c:v", "libx264",
                                   "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                   str(run_dir / "run.mp4")], stdin=subprocess.PIPE)

    log = (run_dir / "decisions.jsonl").open("w", encoding="utf-8")
    tic, n_dec, lat, best_left = 0, 0, [], total_route
    info = {"model": name, "cmd": None, "probs": {}, "state": "", "lat": 0.0, "n": 0, "left": total_route}
    t_start = time.perf_counter()
    outcome = "timeout"
    try:
        while True:
            if g.is_episode_finished():
                break
            obs = observe(g, tic)
            if obs is None:
                break
            brain.offer(obs)
            while not brain.answers.empty():
                d = brain.answers.get()
                n_dec += 1
                rec = {"id": f"{run_dir.name}-{n_dec:05d}", "group_id": f"{run_dir.name}-{n_dec // 20:04d}",
                       "tic": tic, "obs_tic": d["obs_tic"], "state": d["state"], "latency": round(d["latency"], 3),
                       "perception_s": round(d["perception_s"], 3), "error": d["error"],
                       "pos": [round(obs.x), round(obs.y)], "health": obs.health}
                if d["answer"]:
                    a = d["answer"]
                    rec.update(choice=a["choice"], confidence=a["confidence"], probabilities=a["probabilities"])
                    exe.start(a["choice"])
                    lat.append(d["latency"])
                    info.update(cmd=a["choice"], probs=a["probabilities"], state=d["state"], lat=d["latency"], n=n_dec)
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
            target = t_start + tic / TICRATE                   # hold real time: 35 tics per second
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        dead = g.is_player_dead()
        if dead:
            outcome = "died"
        elif tic < args.timeout * TICRATE - 1:
            outcome = "exit"
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
        summary = {"outcome": outcome, "game_seconds": round(tic / TICRATE, 1), "wall_seconds": round(wall, 1),
                   "decisions": n_dec, "decisions_per_s": round(n_dec / max(wall, 1e-9), 2),
                   "latency_median": round(statistics.median(lat), 3) if lat else None,
                   "route_total": round(total_route), "best_route_left": round(best_left),
                   "route_progress_pct": round(100 * (1 - best_left / total_route), 1),
                   "model": name, "url": HOSTED_URL if args.teacher else args.url, "skill": args.skill}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(f"run dir: {run_dir}")
        g.close()


if __name__ == "__main__":
    main()
