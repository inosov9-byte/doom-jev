# Handoff: laptop to RTX 3090 Ti PC (2026-10-18)

Read this first in a new Claude Code session on the PC. It says where the work stands, what moves through git, what moves through OneDrive, and what the PC session should do first.

## Where it stands

- **Pipeline:** perception text, 20 commands, real-time loop, engines (local Jev, hosted Jev, Haiku, rules), viewer and side-by-side videos. All of it works.
- **E1M1 comparison:** five seeds per arm in `runs/compare-final`; summary in [ARCHITECTURE.md](ARCHITECTURE.md).
  - No LLM arm finished the level.
  - Hosted Jev with Gemma did best: 52% median route progress, 75.6% best.
  - Haiku reached 12.7% on every seed.
- **Why Haiku stalled:** every Haiku run hit a barrel 27 units ahead at (1125, -2897). The old raycast checked walls only, so the text read "ahead 219 wall" while each forward move went nowhere. This is a perception bug, fixed in `phase/perception-v2`. The E1M1 arms have not been re-run with the fix yet.
- **Local student (Qwen3.5-0.8B, RFDT LoRA): not learned.**
  - On 5 easy commands it learns after a ~110-step plateau (75% held-out agreement at step 150).
  - On the full 20-command expert set it was still predicting `forward` at step 300 of 450 (6.7% agreement).
  - Training ran at ~32 s per step on the laptop. The run was stopped. No usable model exists.
- **Latency is the deciding factor.** The rule-based expert (the teacher) with an artificial decision delay, E1M1, perception v2, seeds 1-5 (`C:\tmp\jev-doom\smoke-v2b` on the laptop):

| Decision delay | Clears | Median route progress | Median time |
|---|---|---|---|
| 0 s | 5/5 | 100% | 24.0 s |
| 0.1 s | 4/5 | 100% | 32.2 s |
| 0.2 s | 4/5 | 100% | 35.9 s |
| 0.35 s | 2/5 | 89.3% | 45.4 s |
| 0.7 s (laptop Jev speed) | 0/5 | 64.9% | 50.0 s |

The target for the PC is local Jev under ~0.2 to 0.3 s per decision. On the laptop it takes 0.61 s, because Qwen3.5's linear-attention (DeltaNet) layers fall back to a slow PyTorch path on AMD under Windows. A third-party benchmark of 0.6 to 0.8B decision models measured 45 to 86 ms ([akitaonrails, 2026-10-10](https://akitaonrails.com/en/2026/10/10/which-jev-like-ai-decision-classifier-model-is-best/)). Model check, 2026-10-18: no Qwen3.6, 3.7 or 3.8 model smaller than 27B is published. Qwen3.5-0.8B is the newest small Qwen; Qwen3-0.6B is the standard-attention alternative.

- **Multi-level:**
  - The progression solver (keys, switches, lifts) works.
  - The expert clears E1M2 in real time (57.2 s).
  - E1M3 is unsolved: the planner misses a one-way climb.
  - No Jev arm has run E1M2.

## How the pieces move

| What | How | Where on the PC |
|---|---|---|
| Code, docs, prompts, scripts | git: `github.com/inosov9-byte/doom-jev`, branch **`phase/perception-v2`** (contains everything: E1M1 comparison + multi-level + perception v2) | `C:\Users\<you>\repos\jev-doom` |
| Datasets, run results, viewer, videos, logs, progression cache, Claude context | one zip through OneDrive: `OneDrive\Transfer\doom-jev\doom-jev-assets-2026-10-18.zip` | unzip as described below |
| Anthropic API key | **by hand**, never through OneDrive or git | `C:\tmp\jev-doom\.env` |
| Python envs, Simple Jev checkout, DOOM1.WAD, Hugging Face models | rebuilt by `scripts\setup.ps1 -Cuda` | `C:\tmp\jev-doom\...` |
| Trained LoRA adapters | not moved: none is usable; retraining on the PC is faster than copying 1.2 GB | — |

Branches on GitHub:

| Branch | Content |
|---|---|
| `main` | last state that ran end to end (older) |
| `phase/jev-vs-haiku` | E1M1 comparison, viewer, videos, ARCHITECTURE.md |
| `phase/multi-level` | progression solver, E1M2 |
| `phase/perception-v2` | both of the above merged, plus body-width rays, barrels and pillars, command history, chaingun and rocket launcher auto-select, latency sweep arms, `setup.ps1 -Cuda`, this file |

Rule kept so far: work on `phase/<name>` branches and merge to `main` only after a live run confirms.

## PC setup, step by step (PowerShell)

Prerequisites: Git, [uv](https://docs.astral.sh/uv/), an NVIDIA driver recent enough for CUDA 12.8, and the GitHub account `inosov9-byte` signed in (`gh auth login`).

```powershell
# 1. Code
mkdir $HOME\repos -Force; cd $HOME\repos
git clone https://github.com/inosov9-byte/doom-jev.git jev-doom
cd jev-doom; git switch phase/perception-v2

# 2. Environments, WAD, Simple Jev (outside OneDrive, under C:\tmp\jev-doom)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Cuda
#    the Jev venv keeps the name venv-rocm so every script still finds it, even though it is CUDA here

# 3. Assets from OneDrive (wait until OneDrive shows the zip as synced)
Expand-Archive "$env:OneDrive\Transfer\doom-jev\doom-jev-assets-2026-10-18.zip" C:\tmp\jev-doom\transfer
robocopy C:\tmp\jev-doom\transfer\data   .\data /E
robocopy C:\tmp\jev-doom\transfer\runs   .\runs /E
robocopy C:\tmp\jev-doom\transfer\cache  C:\tmp\jev-doom\cache /E
robocopy C:\tmp\jev-doom\transfer\logs   C:\tmp\jev-doom\logs-laptop /E

# 4. API key (type it, do not copy the file through OneDrive)
Set-Content C:\tmp\jev-doom\.env "ANTHROPIC_API_KEY=<key>"

# 5. Check
C:\tmp\jev-doom\venv-rocm\Scripts\python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
C:\tmp\jev-doom\venv\Scripts\python.exe -m jevdoom.compare --arms rules --seeds 1 --no-video --out C:\tmp\jev-doom\check
#    expected: seed 1 exit, about 24 s
```

`data/` and `runs/` are git-ignored, so copying them into the repo folder does not touch git. The repo is outside OneDrive, so runs do not cause sync churn.

The zip's `claude/` folder holds the laptop's global `CLAUDE.md` (writing and working preferences) and the project memory note. To give the PC session the same preferences, copy `CLAUDE.md` to `%USERPROFILE%\.claude\CLAUDE.md`. Skip this if the PC already has one.

## Starting the PC session

Open Claude Code in `C:\Users\<you>\repos\jev-doom` and say: "Read docs/HANDOFF.md and continue from 'First tasks'."

## First tasks for the PC session, in order

1. **Measure latency before anything else.**
   - Serve zero-shot `Qwen/Qwen3.5-0.8B` with `scripts\serve.ps1` and time 20 decisions on a recorded state.
   - Check whether `flash-linear-attention` is actually used (the warning about the fallback path disappears from the server log).
   - Time `Qwen/Qwen3-0.6B` the same way.
   - Pick the student model by measured latency. Target: under 0.2 s.
2. **Optional, if still above target:** cache the fixed prompt prefix. About 620 of the ~770 prompt tokens never change. Check whether the Simple Jev template puts them before the state.
3. **Recollect training data with perception v2.**
   - The text format changed: shoulder rays, barrels and pillars, a "Recent commands" line, rockets in the ammo line. `data/rfdt-v3` was built with the old text.
   - Rebuild with `jevdoom.collect` and the expert trajectories, then `jevdoom.dataset --expert ...`. The expert labels for free.
4. **Train the student.**
   - Use `jevdoom.train_rfdt` with `--eval-every`, to check progress during training (an explicit request from Ivan).
   - Expect the plateau seen on the laptop: flat until a sudden drop.
   - Go up to 1,000+ steps; the PC should take minutes per 100 steps. Measure that rather than assume it.
5. **Re-run E1M1, all arms, perception v2.**
   - Arms: `jev-local`, `jev-hosted`, `jev-hosted-gemma`, `haiku`, `haiku-think`, `rules`, `rules-lag`, and the sweep arms.
   - Output to `runs/compare-v2`.
   - Haiku budget: **under $0.40** for the two Haiku arms (the old run cost $0.26). Ivan approved under $1 in total for Haiku.
   - Rebuild the viewer with `jevdoom.report` and render a new `jevdoom.sbs_video` of Jev vs Haiku.
6. **E1M2 for Jev.**
   - `compare.py` has no `--map` option yet: add one and pass it to `play.run`. `play.py --map` already works and loads the progression stages.
   - Run hosted Gemma Jev, then the local student, on E1M2.
7. **Update ARCHITECTURE.md:** perception v2, the barrel finding, the latency sweep, PC numbers.

## Known issues

- Doom slides the player along walls. At an oblique angle a shoulder ray can report "wall 6 at the right shoulder" while forward still moves 85 to 130 units. The "Recent commands ... moved N units in total" line is the backstop.
- Solid things are treated as circles; Doom uses square boxes. Close enough for now.
- Weapon choice is a fixed rule, not a Jev decision:
  - Shotgun when it has shells, then chaingun, then pistol, then fist.
  - Rocket launcher only while every visible enemy is 300+ units away (kept down to 200), never with nothing in view.
  - Every shareware map has a chaingun and a rocket launcher, so this rule matters from E1M1 on.
- The probe (60 synthetic situations) predicts single decisions, not play. Haiku scores highest on it and does worst live.
- The free Featherless demo API is limited to 2 requests per second and needs a browser-like User-Agent (Cloudflare 1010 otherwise). Already handled in `jev.py`.

## Working agreements with Ivan

- Jev decides every command. Perception reports facts only; the executor plays fixed macros. No scripted bot behind Jev.
- Evaluate a proposal critically before implementing it. Include progress checks in long runs.
- Paid API calls cost Ivan money: estimate first, stay inside the approved budget.
- Disposable files go under `C:\tmp\jev-doom`, never in OneDrive. Stop only processes you started.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Writing: direct, factual, short; no hype phrases.

## Laptop state at handoff

- Training stopped; no GPU jobs running.
- The latency sweep finished; its results are in the table above.
- `C:\tmp\jev-doom\KEEP_AWAKE` keeps the laptop awake: delete the file to let it sleep.
- The viewer is served on port 4560 from `C:\Users\inoso\repos\jev-doom\runs`.
