# Current state and next steps (2026-10-18, RTX 3090 PC)

Read this first in a new session. It says where the work stands, where things live, and what comes next. History: [ARCHITECTURE.md](ARCHITECTURE.md). Design: [../EXPERIMENT.md](../EXPERIMENT.md).

## Where it stands

- **Goal set by Ivan:** master E1M1 (fast, reliable, real-time play) before other levels; then E1M2 as a joint training run; final comparison is student vs teacher side by side. Haiku is no longer part of the experiment.
- **Student:** Qwen3.5-4B + LoRA behind `fast_server.py` (compact Jev prompt, one forward pass), ~0.1 s per decision.
  - E1M1, held-out seeds 26-45: 18/20 clears (DAgger round 1, step-100 adapter `C:\tmp\jev-doom\models\v4-4b-d1\best`), median 31.8 s.
  - E1M2: not trained yet.
- **Teacher:** `expert.py` + progression planner. E1M1 20/20 (median 23.5 s); E1M2 10/10 (median 60.1 s) since the shotgun detour. E1M3 unsolved (the planner misses a one-way climb).

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

1. **E1M2 transfer test:** the E1M1 student on E1M2 seeds 26-45, no E1M2 training.
2. **E1M2 data:** expert games seeds 6-25 and lagged-expert games seeds 6-15 (`data\expert-traj-e1m2`, `data\lag-traj-e1m2`), labelled with `dataset.py --expert ... --dedupe`.
3. **Joint training:** E1M1 data (`rfdt-v4`, `dagger-v4a`) + E1M2 data, from the round-1 adapter or fresh; checks every 100 steps.
4. **Live test on both maps,** seeds 26-45; DAgger round on E1M2 if below 90%.
5. **Deliverables:** E1M2 share video when Jev clears it with a good time (2x upscale, overlay, `pc-progress\share\`); final student vs teacher side-by-side video (`jevdoom.sbs_video`).
6. **Later:** speed (student ~4 decisions/s vs teacher ~6): CUDA graphs or `torch.compile` to cut the ~60 ms forward-pass floor, or the 2B student (70 ms); E1M3 planner; a level held out from training to measure transfer.

## Commands

```powershell
.\scripts\eval_student.ps1 -Adapter C:\tmp\jev-doom\models\v4-4b-d1\best -Out runs\eval-x [-Map E1M2] [-Video]
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
