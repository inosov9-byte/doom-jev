# Current state and next steps (2026-10-18, RTX 3090 PC)

Read this first in a new session. It says where the work stands, where things live, and what comes next. History: [ARCHITECTURE.md](ARCHITECTURE.md). Design: [../EXPERIMENT.md](../EXPERIMENT.md).

## Where it stands

- **Bar set by Ivan:** a student that clears a level about as often as its teacher is done; move to the next level. Haiku is no longer part of the experiment.
- **Student:** Qwen3.5-4B + LoRA behind `fast_server.py` (compact Jev prompt, one forward pass), 26 commands (incl. turn-and-fire), snapshot at tic 5, ~0.1 s per decision.
  - Joint model `C:\tmp\jev-doom\models\joint-a\best` (20 commands): E1M1 20/20 (29.9 s), E1M2 13/20 (75.4 s) on seeds 26-45.
  - v5 `C:\tmp\jev-doom\models\v5\best` (26 commands, matches current code): E1M1 20/20 (34.5 s), E1M2 12/20.
  - E1M3 zero-shot: dies at 9-32% of the route (teacher ~26%).
- **Teacher:** `expert.py` + planner. E1M1 10/10 (19.5 s), E1M2 10/10 on seeds 26-35 (58.7 s).
- **Later levels:** the planner solves E1M3, E1M5, E1M6, E1M9 (solids, stairs, live regrids). The teacher dies on all four: E1M3 dead-end damaging pit at ~26%, E1M5 shotgun-guy ambush at ~15%, E1M6 fights beside nukage at ~28%, E1M9 at ~35% (not checked). E1M4 and E1M7: no route found. E1M8: boss level, needs kill-target subgoals.

## Where things live

| What | Where |
|---|---|
| Code | `github.com/inosov9-byte/doom-jev`, branch `phase/pc-student` |
| Repo checkout | `C:\Users\inoso.INPC\repos\jev-doom` |
| Venvs, WAD, Simple Jev checkout, HF cache, ffmpeg, `.env` (API key) | `C:\tmp\jev-doom` (`venv` game side, `venv-rocm` model side, CUDA despite the name) |
| Adapters | `C:\tmp\jev-doom\models\<run>\best` and `\last`, with `metrics.jsonl` |
| Datasets, run folders | `data\`, `runs\` in the repo (git-ignored) |
| Progress for Ivan | `OneDrive\Transfer\doom-jev\pc-progress\`: training .md files, viewer + videos per comparison, `share\` for videos to pass on |

PC setup quirk: the Claude desktop app redirects AppData, which breaks uv-managed Python. Set `UV_PYTHON_INSTALL_DIR=C:\tmp\jev-doom\cache\python` before `scripts\setup.ps1 -Cuda`. causal-conv1d does not build without the CUDA toolkit (nvcc); everything runs without it.

## Next tasks, in order

1. **Dead-end warning in perception:** mark directions that drop into an area with no route back to the goal (reverse reachability on the stage grid) and keep the teacher out of them. Targets E1M3's pit, probably E1M6.
2. **Teacher fixes per level:** E1M5 (shotgun guys before any shotgun), E1M6, E1M9; each level is a short chain of fixes. Re-check E1M1/E1M2 (seeds 26-30) after every rule change.
3. **Planner on E1M4 and E1M7** (no route found; cause not investigated).
4. **Data and joint training** on every level the teacher clears, mixed with E1M1/E1M2 (`scripts\build_v5.ps1` relabels all states); keep one level out of training entirely for the transfer test.
5. **E1M8:** kill-target subgoals (compass to the Barons, done when dead) and an outcome that counts the Baron kills as success.
6. **Later:** speed (student ~4.5 decisions/s vs teacher ~7.5): CUDA graphs or `torch.compile` against the ~60 ms forward-pass floor.

Done on 2026-10-18: E1M1 reliability and DAgger, E1M2 transfer test and joint training, turn-and-fire commands, tic-5 snapshots, share videos and student-vs-teacher side-by-sides (`pc-progress\share\`).

## Commands

```powershell
.\scripts\eval_student.ps1 -Adapter C:\tmp\jev-doom\models\v5\best -Out runs\eval-x [-Map E1M2] [-Timeout 150] [-Video]
.\scripts\serve_fast.ps1 -Adapter <adapter>      # serve only, port 8012
C:\tmp\jev-doom\venv-rocm\Scripts\python.exe -m jevdoom.train_fast --data <jsonl...> --model Qwen/Qwen3.5-4B `
    --out C:\tmp\jev-doom\models\<run> --steps 300 --eval-every 100 --grad-ckpt [--init-adapter <adapter> --lr 1e-4] `
    --progress "$env:OneDrive\Transfer\doom-jev\pc-progress\training-<run>.md"
```

Stop the fast server before training (it holds ~9 GB; training peaks at 12.5 GB).

## Known issues

- Stuck positions remain the main failure: the E1M1 start barrel and wall corners on nukage. Each DAgger round adds the states the student actually reaches.
- Weapon choice is a fixed rule (shotgun when it has shells, then chaingun, pistol, fist; rocket launcher only at 300+ units).
- Solid things are circles in perception; Doom uses square boxes.
- The probe (60 synthetic situations) measures single decisions, not play.

## Working agreements with Ivan

- Jev decides every command. Perception reports facts; the executor plays fixed macros. No scripted bot behind Jev. Improving the teacher is allowed and expected.
- Evaluate proposals critically before implementing; add progress checks to long runs.
- Paid API calls need approval (Haiku is out of the experiment; no paid calls planned).
- Disposable files under `C:\tmp\jev-doom`, never in OneDrive; only small progress files go to `pc-progress`. Stop only processes you started.
- Work on `phase/<name>` branches; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Writing: direct, factual, short; no hype phrases.
