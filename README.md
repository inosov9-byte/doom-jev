# doom-jev

Jev plays DOOM E1M1 in real time on a laptop, choosing every move itself.

Jev is [Simple Jev](https://github.com/featherless-ai/simple-jev), an open classifier API from Featherless. It wraps a language model so that, instead of writing text, the model returns a probability for each label in a fixed list. Here the list is 13 DOOM commands. A large model behind the hosted Jev API acts as teacher. A small model behind a local Jev server is fine-tuned to copy it, then plays.

Status (2026-10-17): the pipeline works end to end and a distilled local model plays in real time. No model, teacher included, has finished E1M1 yet.

## Two models, one Jev

| | Teacher | Student |
|---|---|---|
| Model inside Jev | `featherless-ai/gemma-4-26B-A4B-classifier` | `Qwen/Qwen3.5-0.8B`, LoRA fine-tuned |
| Runs on | Featherless hosted demo API | local GPU (Radeon 890M, ROCm) |
| Decision latency | ~1.4 s, max 2 calls/s | ~0.7 s, no limit |
| Role | labels training situations | plays the game |

Both sides use the same Jev server code, prompt format and labels. Gemma never runs during play.

### Training (offline)

```
ViZDoom E1M1 --teleport + random commands--> recorded situations (text)
        --> hosted Jev (Gemma-4-26B): probabilities over 13 commands
        --> RFDT (Simple Jev's trainer): LoRA on Qwen3.5-0.8B to match them
        --> merged local model
```

### Play (real time, fully local)

```
ViZDoom, 35 tics/s, never paused
   |  observation
   v
perception.py   text: health, ammo, enemies in view (distance, angle, in crosshair),
   |            raycast obstacles (wall / door state / ledge / exit switch), nukage,
   |            route compass to the exit, off-screen attacker direction, last command result
   v
local Jev       one choice over 13 commands
   v
executor.py     plays the command as a fixed macro; the next command cancels it
```

The next decision waits until the previous command has taken effect, so Jev never decides on a view that predates its own last move.

### What decides what

| Component | Does | Decides? |
|---|---|---|
| Jev | picks every command | yes |
| Perception | turns game state into text; facts only, no advice | no |
| Route compass | bearing and distance to the exit along an A* path over the WAD map, like an automap; `--no-compass` disables it | no, a sensor reading |
| Executor | replays a fixed button pattern per command | no |
| Weapon select | shotgun if shells, else pistol, else fist | fixed rule, like Doom's auto-switch |

Featherless's own [web DOOM demo](https://featherless.ai/simple-jev/cool-demo/doom/) works differently: a scripted route follower handles navigation, doors and the exit, and Jev picks only combat actions.

## Commands

`forward`, `back`, `strafe_left`, `strafe_right` (20 tics, about 0.57 s; ~300 units forward or back and ~250 strafing, measured including the slide), `turn_left_15/45/90`, `turn_right_15/45/90`, `turn_around` (exact angles), `fire`, `use`.

Turn amounts are in the label because a classifier picks labels and cannot output numbers. Moves have one length, close to the decision interval, because Doom momentum makes long timed moves overshoot.

## Choosing the teacher

`probe.py` scores teachers on 68 synthetic situations with unambiguous answers (aim, fire, follow route, open door, press exit, get unstuck). Wrong-direction turns score zero.

| Teacher (hosted) | Instructions | Score | Aim at enemy |
|---|---|---|---|
| Qwen3.6-35B-A3B | default | 0.50 | 0.08 |
| Qwen3.8-27B | default | 0.57 | 0.00 |
| gemma-4-12B | default | 0.62 | 0.71 |
| gemma-4-26B-A4B | default | 0.87 | 0.75 |
| **gemma-4-26B-A4B** | **facts (`prompts/facts.txt`)** | **0.90** | **0.83** |

The Qwen models follow the route compass even with an enemy in view. Known Gemma bias: it under-turns large right angles (90 to 180 degrees right usually gets `turn_right_45`).

## Results so far

| Run | Outcome |
|---|---|
| Zero-shot Qwen3.5-0.8B vs teacher | 8 to 10% agreement (chance is 7.7%) |
| Student after 1 epoch, 1,600 labels (older Qwen teacher) | 81 to 85% held-out agreement, 0.69 s latency |
| Hosted Qwen3.6-35B, live | died at 93 s, 47% of route |
| Hosted gemma-4-26B, live | died at 145 s, 26% of route (slow reactions, door toggling) |
| Local student (Qwen-taught), live | died at 43 s, 45% of route |

Fixes made after these runs: decisions wait for the previous command to apply; doors report closed / opening (with time to open) / closing, since pressing use on an opening door closes it in vanilla DOOM; off-screen attacks report a direction, like the status-bar face.

## Local speed, one classifier call

Ryzen AI 9 HX 375, Radeon 890M, 32 GB, warm median:

| Where | Model | Text | Screenshot |
|---|---|---|---|
| Local GPU (ROCm 7.2.1) | Qwen3.5-0.8B | 0.61 s | 0.78 s |
| Local GPU | Qwen3.5-2B | 1.00 s | 1.27 s |
| Local GPU | Qwen3.5-4B, short prompt | 2.7 s | 3.0 s |
| Local CPU | Qwen3.5-0.8B | 4.3 s | 4.5 s |
| Hosted demo | Qwen3.6-35B-A3B | 0.87 s | 0.9 to 1.0 s |

Training with RFDT on the 890M: about 0.36 examples/s, 2.5 GiB peak with gradient checkpointing.

## Run

```powershell
.\scripts\setup.ps1                     # venvs, WAD, Simple Jev, ROCm PyTorch under C:\tmp\jev-doom
.\scripts\serve.ps1 -Model <model dir>  # local Jev on the GPU, port 8012
C:\tmp\jev-doom\venv\Scripts\python.exe -m jevdoom.play --record-video
```

Pipeline modules (`python -m jevdoom.<module>`):

| Module | Purpose |
|---|---|
| `collect` | record situations (teleport + random commands, no bot) |
| `label` | label situations with the hosted teacher, resumable |
| `dataset` | teacher labels to RFDT records (short student prompt, soft targets) |
| `train_rfdt` | RFDT trainer with Windows-ROCm and last-position-logits adjustments |
| `export_model` | merge LoRA adapter, add processor configs |
| `evaluate` | student vs teacher agreement on held-out situations |
| `probe` | score teacher candidates |
| `play` | real-time run; writes `runs/<time>-<model>/` with `run.mp4`, `decisions.jsonl`, `summary.json` |

`data/`, `runs/`, `models/` are git-ignored. Heavy files live outside the repo under `C:\tmp\jev-doom`.

## Notes

- `train_rfdt.py` stubs `torch.distributed.tensor.DTensor` because the Windows ROCm PyTorch build ships without `torch.distributed`; on CUDA builds the stub is skipped.
- `DOOM1.WAD` is the freely distributable shareware v1.9 IWAD; `setup.ps1` checks its MD5.

## Branches

- `main`: the last state that ran end to end. Merge into it only after a live run confirms the change.
- `phase/<name>`: one branch per phase of work, e.g. `phase/gemma-student`, `phase/cuda-3090`.
- Datasets, models and runs are not in git; each run's `summary.json` records the model and settings used.
