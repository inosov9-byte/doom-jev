"""Render two runs into one synced comparison video (for posts), with live decision stats.

Each source run.mp4 is game + decision panel, one frame per tic (35 fps). This stacks the
two (portrait 1080x1350 for feeds, or wide 1920x1080), adds a title, per-engine labels and
running counters (decisions so far, latency of the last decision, health), freezes a run
that ended early under its outcome, and ends on a result card.

usage: python -m jevdoom.sbs_video RUN_A RUN_B --out runs/jev-vs-haiku.mp4 [--layout feed|wide] [--start 0 --duration 60]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .commands import COMMAND_IDS

N_CMD = len(COMMAND_IDS)

FPS = 35
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_B = r"C:\Windows\Fonts\segoeuib.ttf"
MONO = r"C:\Windows\Fonts\consola.ttf"
BG = (15, 17, 21)
ACCENT = {"a": (106, 168, 255), "b": (245, 196, 81)}
LABEL = {"jev-local": ("JEV", "Qwen3.5-4B + LoRA · local RTX 3090 · ~0.1 s per decision"),
         "rules": ("TEACHER", "hand-written rules reading the same text · instant"),
         "jev-hosted": ("JEV", "Qwen3.6-35B classifier · Featherless API"),
         "jev-hosted-gemma": ("JEV", "Gemma-4-26B classifier · Featherless API"),
         "haiku": ("CLAUDE HAIKU 5.5", "Anthropic API · structured output"),
         "haiku-think": ("CLAUDE HAIKU 5.5 + THINKING", "Anthropic API · structured output")}
CAPTION = {"jev-local": "Jev: a language model, one pass over the text, a probability per command.",
           "rules": "Teacher: hand-written rules over the same text; Jev learned by copying it.",
           "haiku": "Haiku: a generative model, writes its pick as JSON.",
           "haiku-think": "Haiku: a generative model, writes its pick as JSON."}


def font(path, size):
    return ImageFont.truetype(path, size)


class Run:
    def __init__(self, folder: Path):
        self.folder = folder
        self.summary = json.loads((folder / "summary.json").read_text())
        self.decisions = [json.loads(l) for l in (folder / "decisions.jsonl").open(encoding="utf-8")]
        self.cap = cv2.VideoCapture(str(folder / "run.mp4"))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last = None

    def frame(self, i: int):
        if i < self.frames:
            ok, f = self.cap.read()
            if ok:
                self.last = f
        return self.last, i >= self.frames

    def stats(self, tic: int) -> dict:
        done = [d for d in self.decisions if d["tic"] <= tic and d.get("choice")]
        last = done[-1] if done else None
        lat = [d["latency"] for d in done if d.get("latency")]
        return {"n": len(done), "cmd": last["choice"] if last else "-", "lat": last.get("latency") if last else None,
                "avg": sum(lat) / len(lat) if lat else None, "health": last["health"] if last else 100}


def outcome_text(s: dict) -> str:
    o = {"exit": "REACHED THE EXIT", "died": "DIED", "timeout": "TIME LIMIT"}.get(s["outcome"], s["outcome"].upper())
    return f"{o} · {s['game_seconds']:.0f} s · {s['route_progress_pct']:.0f}% of route · {s['kills']} kills"


def compose(a: Run, b: Run, t_frame: int, layout: str, title: str, subtitle: str) -> np.ndarray:
    W, H = (1080, 1350) if layout == "feed" else (1920, 1080)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 28), title, font=font(FONT_B, 44 if layout == "feed" else 50), fill=(235, 238, 243))
    d.text((40, 88), subtitle, font=font(FONT, 24), fill=(154, 163, 178))
    if layout == "feed":
        slots = [(0, 150, W, 548), (0, 150 + 548 + 8, W, 548)]
    else:
        slots = [(0, 150, W // 2 - 4, 860), (W // 2 + 4, 150, W // 2 - 4, 860)]
    for key, run, (x, y, w, h) in (("a", a, slots[0]), ("b", b, slots[1])):
        fr, ended = run.frame(t_frame)
        name, detail = LABEL.get(run.summary["arm"], (run.summary["arm"], ""))
        st = run.stats(min(t_frame, run.frames))
        d.rectangle([x, y, x + w, y + 50], fill=(23, 26, 33))
        d.rectangle([x, y, x + 6, y + 50], fill=ACCENT[key])
        d.text((x + 20, y + 6), name, font=font(FONT_B, 26), fill=ACCENT[key])
        nw = d.textlength(name, font=font(FONT_B, 26))
        d.text((x + 34 + nw, y + 13), detail, font=font(FONT, 19), fill=(154, 163, 178))
        lat = f"{st['lat']:.2f}s" if st["lat"] else "-"
        avg = f"{st['avg']:.2f}s" if st["avg"] else "-"
        stat = f"decisions {st['n']:3d}   last {lat}   avg {avg}   health {st['health']:.0f}"
        vid_h = h - 50 - 34
        if fr is not None:
            fh, fw = fr.shape[:2]
            scale = min(w / fw, vid_h / fh)
            fr2 = cv2.resize(fr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            if ended:
                fr2 = (fr2 * 0.35).astype(np.uint8)
            pil = Image.fromarray(cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB))
            img.paste(pil, (x + (w - pil.width) // 2, y + 50))
            if ended:
                txt = outcome_text(run.summary)
                f = font(FONT_B, 30 if layout == "feed" else 28)
                tw = d.textlength(txt, font=f)
                d.text((x + (w - tw) / 2, y + 50 + pil.height / 2 - 18), txt, font=f, fill=(255, 255, 255))
        d.text((x + 20, y + h - 30), stat, font=font(MONO, 20), fill=(201, 207, 218))
    if layout == "feed":
        for i, run in enumerate((a, b)):
            d.text((40, H - 92 + 30 * i), CAPTION.get(run.summary["arm"], ""), font=font(FONT, 21), fill=(154, 163, 178))
    t = t_frame / FPS
    clock = f"{int(t // 60)}:{t % 60:04.1f}  real time"
    d.text((W - 40 - d.textlength(clock, font=font(MONO, 24)), 92), clock, font=font(MONO, 24), fill=(154, 163, 178))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def end_card(a: Run, b: Run, layout: str) -> np.ndarray:
    W, H = (1080, 1350) if layout == "feed" else (1920, 1080)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H // 2 - 220
    d.text((60, y), "Same game, perception and commands", font=font(FONT_B, 40), fill=(235, 238, 243))
    d.text((60, y + 56), "The decision engine is the only difference.", font=font(FONT, 30), fill=(154, 163, 178))
    y += 150
    for key, run in (("a", a), ("b", b)):
        s = run.summary
        name, detail = LABEL.get(s["arm"], (s["arm"], ""))
        d.text((60, y), name, font=font(FONT_B, 32), fill=ACCENT[key])
        d.text((60, y + 44), detail, font=font(FONT, 22), fill=(154, 163, 178))
        d.text((60, y + 80), outcome_text(s), font=font(FONT, 26), fill=(235, 238, 243))
        d.text((60, y + 118), f"median decision {s['latency_median']:.2f} s · {s['decisions_per_s']:.2f} decisions/s · "
               f"cost ${s['cost_usd']:.3f}", font=font(MONO, 22), fill=(201, 207, 218))
        y += 190
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["feed", "wide"], default="feed")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the runs")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until both runs end")
    ap.add_argument("--title", default="Jev vs Claude Haiku play DOOM in real time")
    ap.add_argument("--subtitle", default="E1M1 · same seed · every move chosen by the model")
    ap.add_argument("--end-card-s", type=float, default=4.0)
    args = ap.parse_args(argv)

    a, b = Run(Path(args.run_a)), Run(Path(args.run_b))
    W, H = (1080, 1350) if args.layout == "feed" else (1920, 1080)
    first = int(args.start * FPS)
    last = max(a.frames, b.frames) + FPS                    # one second of the frozen outcome
    if args.duration:
        last = min(last, first + int(args.duration * FPS))
    for r in (a, b):                                        # skip to the start frame
        for _ in range(min(first, r.frames)):
            r.frame(_)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", "30", args.out], stdin=subprocess.PIPE)
    for i in range(first, last):
        ff.stdin.write(compose(a, b, i, args.layout, args.title, args.subtitle).tobytes())
    card = end_card(a, b, args.layout).tobytes()
    for _ in range(int(args.end_card_s * FPS)):
        ff.stdin.write(card)
    ff.stdin.close()
    ff.wait()
    print(f"wrote {args.out} ({(last - first) / FPS:.1f} s + {args.end_card_s:.0f} s card)")


if __name__ == "__main__":
    main()
