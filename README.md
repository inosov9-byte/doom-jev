# jev-doom

Jev plays DOOM E1M1 in real time, on a laptop, with every move chosen by the model.

Jev is [Simple Jev](https://github.com/featherless-ai/simple-jev): an open classifier API that reads an LLM's next-token scores over a fixed set of answer labels. Here the labels are ten DOOM commands. A small local model (Qwen3.5-0.8B on the Radeon 890M iGPU) is distilled from the hosted 35B Jev, then plays.

## How it works

```
ViZDoom (35 tics/s, never paused)
   │ observation every tic
   ▼
perception.py ── text: health, ammo, enemies in view (distance, bearing, in crosshair),
   │              raycast obstacles (wall / closed door / ledge / exit switch),
   │              nukage ahead, route compass to the exit, result of the last command
   ▼
Jev (local server) ── one choice question: forward, back, turn_left, turn_right,
   │                  turn_left_big, turn_right_big, strafe_left, strafe_right, fire, use
   ▼
commands.py ── plays the chosen command as a fixed 10-tic macro, then idles
```

Jev decides every command. The only non-Jev inputs are mechanical:

| Component | What it does | Decides anything? |
|---|---|---|
| Perception | Converts game state to text; no advice | No |
| Route compass | Bearing and distance to the exit along an A* path over the WAD map (like an automap). `--no-compass` turns it off | No, it is a sensor reading |
| Command macro | Replays a fixed button pattern (e.g. `turn_left` = 15 degrees) | No |
| Weapon select | Shotgun if it has shells, else pistol, else fist | Fixed rule, same as Doom's own auto-switch |

For comparison, Featherless's own [web DOOM demo](https://featherless.ai/simple-jev/cool-demo/doom/) uses a scripted route follower for navigation, doors and the exit, and asks Jev only for combat actions.

## Training: distill the hosted 35B into a local 0.8B

1. `collect.py` records situations with no bot in the loop: it teleports the player (cheat warp, god mode) along the exit route, near monsters, or anywhere walkable, records the text, then plays a few random commands and records again.
2. `label.py` sends each situation to the hosted `featherless-ai/Qwen3.6-35B-A3B-classifier` and stores its full probability distribution over the ten commands (free demo, ≤2 requests/s).
3. `dataset.py` writes RFDT records: short student prompt, teacher probabilities as soft targets.
4. RFDT (`src-simple-jev/RFDT/train.py`) LoRA-trains Qwen3.5-0.8B; `export.py` merges the adapter.
5. `evaluate.py` measures agreement with the teacher on held-out situations; `play.py` runs the real game.
6. DAgger: situations the student reaches in `play.py` runs (`decisions.jsonl`) go back through `label.py` and training.

## Measured on this machine (Ryzen AI 9 HX 375, Radeon 890M, 32 GB)

One classifier call, warm, median:

| Where | Model | Text | Screenshot |
|---|---|---|---|
| Local GPU (ROCm 7.2.1) | Qwen3.5-0.8B | 0.61 s | 0.78 s |
| Local GPU | Qwen3.5-2B | 1.00 s | 1.27 s |
| Local GPU | Qwen3.5-4B (short prompt) | 2.7 s | 3.0 s |
| Local CPU | Qwen3.5-0.8B | 4.3 s | 4.5 s |
| Hosted demo | Qwen3.6-35B-A3B | 0.87 s | 0.89–0.99 s |

## Run

```powershell
.\scripts\setup.ps1                      # venvs, WAD, Simple Jev, ROCm PyTorch under C:\tmp\jev-doom
.\scripts\serve.ps1 -Model <model dir>   # local Jev on the GPU, port 8012
C:\tmp\jev-doom\venv\Scripts\python.exe -m jevdoom.play --record-video
```

Each run writes `runs/<time>-<model>/`: `run.mp4` (game plus decision panel), `decisions.jsonl` (every state text, probabilities, latency), `summary.json` (outcome, route progress, decision rate).

## Layout

- `jevdoom/` source. `data/`, `runs/`, `models/` are git-ignored.
- Heavy files stay outside OneDrive: `C:\tmp\jev-doom\{venv, venv-rocm, wad, cache, src-simple-jev}`.
