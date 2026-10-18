# doom-jev

A language model plays DOOM in real time and chooses every move itself.

The model is Qwen3.5-4B on a home GPU (RTX 3090). It reads a short text description of the game ("zombieman 367 away, 33 deg left; ahead 409 wall; route to exit: straight ahead") and picks one of 26 commands in about 0.1 s, through Jev: [Simple Jev](https://github.com/featherless-ai/simple-jev)'s method of reading a model's next-token scores over a fixed list of options instead of letting it write text. It learned by copying a rule-based expert, then by having the expert correct the positions it reached on its own (DAgger).

Status (2026-10-18):

| Level | Student (Jev, Qwen3.5-4B) | Teacher (rules expert) |
|---|---|---|
| E1M1 | 20/20 on held-out seeds 26-45, median 29.9 s, ~0.1 s per decision | 10/10, median 19.5 s |
| E1M2 | 13/20, median 75.4 s | 10/10 on seeds 26-35, median 58.7 s (6/10 before the exit-room fix) |
| E1M3 | not trained; zero-shot dies at 9-32% of the route | dies at ~26% (dead-end pit) |
| E1M5, E1M6, E1M9 | not trained | planner solves them; the teacher dies at 15-35% |
| E1M4, E1M7 | — | no route found yet |

The student is at its teacher's level on E1M1 and E1M2; the next work is the teacher and planner on later levels.

Videos (OneDrive transfer folder, `pc-progress/share`): a 22.7 s E1M1 clear and a 71.7 s E1M2 clear with the command, latency and probabilities on screen, and student-vs-teacher side-by-sides for both levels. The full story, including what failed: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Current state and next steps: [docs/HANDOFF.md](docs/HANDOFF.md). Experiment design: [EXPERIMENT.md](EXPERIMENT.md).

## How a command happens

```
ViZDoom, 35 tics/s, never paused
   |  game state (engine variables + WAD map geometry)
   v
perception.py   text: health, ammo, weapon, keys; enemies and pickups in view (distance, angle,
   |            in crosshair); body-width rays (wall / door state / ledge / barrel / pillar);
   |            nukage; route compass to the next subgoal (key, switch, weapon, exit);
   |            off-screen attacker direction; recent commands and how far they moved
   v
local Jev       compact prompt (options A-T, state, "Answer:") -> one forward pass of Qwen3.5-4B + LoRA
   |            -> softmax over the 20 option letters -> command
   v
executor.py     plays the command as a fixed key macro; the next command cancels it
```

No screen pixels are used. Code describes the situation and presses keys; the model makes every decision.

| Component | Does | Decides? |
|---|---|---|
| Jev (model) | picks every command | yes |
| Perception | game state to text; facts only | no |
| Route compass | bearing and distance to the next subgoal along an A* path over the WAD map, like an automap | no, a sensor reading |
| Progression planner | order of subgoals: keys, switches, a short weapon detour, exit | no, it sets the compass target |
| Executor | fixed button pattern per command | no |
| Weapon select | best weapon with ammo (rocket launcher only for distant targets) | fixed rule, like Doom's auto-switch |

## How it learns

```
expert.py (rules, reads the same text) ──labels──► every recorded state
   states from: teleport situations, expert games, slowed-down expert games, the student's own games (DAgger)
        ──► train_fast.py: LoRA on Qwen3.5-4B, cross-entropy on the expert's option letter
        ──► fast_server.py serves base model + adapter on localhost:8012
```

| Round | Data | Held-out agreement with expert | Live E1M1, seeds 26-45 |
|---|---|---|---|
| Untrained Qwen3.5-4B | — | 22% | — |
| v4, 400 steps (33 min) | 10,155 expert-labelled states | 96% | 14/20 (stuck 3x on a barrel, 3 deaths) |
| DAgger round 1, +100 steps | + 2,248 states from the student's own games | 96% | 18/20 |

## Speed

One decision, RTX 3090, 40 recorded states, median:

| Model | Simple Jev server, stock prompt (~1,130 tokens) | fast_server, compact prompt (~355 tokens) |
|---|---|---|
| Qwen3.5-0.8B | 172 ms | — |
| Qwen3.5-2B | 183 ms | 70 ms |
| Qwen3.5-4B | 337 ms | 90 ms |
| Qwen3.5-9B | 452 ms | — |

The stock Simple Jev template lists the options twice after the state and its server runs two forward passes; `fastjev.py` lists them once before the state and scores in one pass. The rules expert, delayed artificially, shows why this matters: 5/5 clears at 0 s, 4/5 at 0.2 s, 2/5 at 0.35 s, 0/5 at 0.7 s.

## Run (Windows, PowerShell)

```powershell
.\scripts\setup.ps1 -Cuda                     # venvs, WAD, Simple Jev, CUDA PyTorch, ffmpeg under C:\tmp\jev-doom
.\scripts\build_data.ps1                      # E1M1 states: situations, expert and lagged-expert games
python -m jevdoom.dataset --expert <state files> --dedupe --out data\rfdt-v4.jsonl
python -m jevdoom.train_fast --data data\rfdt-v4.jsonl --model Qwen/Qwen3.5-4B --out C:\tmp\jev-doom\models\v4-4b --steps 400 --grad-ckpt
.\scripts\eval_student.ps1 -Adapter C:\tmp\jev-doom\models\v4-4b\best -Out runs\eval-v4   # serve + live seeds 26-45
python -m jevdoom.report runs\eval-v4         # HTML viewer: video synced to every decision
```

Use `C:\tmp\jev-doom\venv\Scripts\python.exe` for game-side modules and `C:\tmp\jev-doom\venv-rocm\Scripts\python.exe` (CUDA on the PC despite the name) for training and serving.

| Module | Purpose |
|---|---|
| `perception`, `executor`, `play` | text description, key macros, real-time loop |
| `expert` | rule-based teacher and reference arm |
| `progression` | subgoal planner (keys, switches, lifts, weapon detours) |
| `collect` | teleport situations (E1M1) |
| `compare` | live runs per arm and seed, `--map`; writes `results.md` |
| `dataset` | expert labels to training records, `--dedupe` |
| `fastjev`, `fast_server`, `train_fast` | compact prompt, one-pass server, LoRA trainer |
| `report`, `sbs_video` | HTML viewer, side-by-side videos |
| `jev`, `train_rfdt`, `label`, `probe`, `evaluate` | Simple Jev client, RFDT trainer and hosted-teacher tools from the laptop phase |

`data/`, `runs/`, `models/` are git-ignored. Heavy files live under `C:\tmp\jev-doom`, outside the repo and OneDrive.

## Branches

- `main`: the last state that ran end to end. Merge only after a live run confirms.
- `phase/<name>`: one branch per phase. Current: `phase/pc-student`.
