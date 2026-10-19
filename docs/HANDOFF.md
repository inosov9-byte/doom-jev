# Current state and next steps (2026-10-19, RTX 3090 PC)

Read this first in a new session. It says where the work stands, where things live, and what comes next. History: [ARCHITECTURE.md](ARCHITECTURE.md). Design: [../EXPERIMENT.md](../EXPERIMENT.md).

## Where it stands

- **Bar set by Ivan:** a student that clears a level about as often as its teacher (at the student's decision speed) is done. Haiku is no longer part of the experiment.
- **Direction:** end-to-end campaign E1M1 -> E1M8 with continuity: each map starts with the health, armor, weapons and ammo carried out of the previous exit (`loadout.py`, `campaign.py`). Until the chain reaches E1M8, E1M8 starts from `NOMINAL["E1M8"]` (shotgun, chaingun, 100 bullets, 30 shells, 100 health).
- **E1M8 done (2026-10-19):** Jev model `C:\tmp\jev-doom\models\e1m8-d\best` (4 rounds: teacher data + 3 DAgger rounds) clears 8/20 unseen seeds (151-170), fastest 133.9 s. Teacher: ~70% instant, ~45% at 100 ms lag. The same model still clears E1M1 10/10 (seeds 26-35, median 31.8 s); E1M2 regression check pending. Videos and post facts in `pc-progress\share\` (`jev-plays-doom-e1m8-133.9s`, `jev-vs-teacher-e1m8`, `e1m8-facts.md`); watermarked cuts and the nosov.ai article brief in `share\nosov.ai\`.
- **Sound:** recordings so far are silent. `phase/audio-replay` (agent run) replays a logged run tic by tic with the audio buffer on, to re-render share videos with game sound.
- **Key finding:** a teacher tuned with instant reactions is unlearnable for a ~90 ms student: 50 ms of lag took the E1M8 teacher from ~70% to 0%. Teacher tactics are now measured with `rules-lag-100` too (shoot only when the shotgun is ready, dodge during the reload, stand and aim 0.35 s before the shot, keep distance from projectile throwers, openness-aware escape from dead-end corners).
- **Student speed:** `fastjev.Scorer` replays CUDA graphs per padded prompt length: 87 ms per decision (was 114), same choices. The 4B forward is compute-bound at ~400 tokens beyond that; a cached prompt prefix did not help.
- **Campaign teacher (rules / rules-lag-100):** E1M1, E1M2 reliable; E1M3 often without lag, rarely with it; E1M4 ~2/8; E1M5-E1M7 0 (routes exist on every map now; fights remain). Work continues on `phase/campaign-fights2` (agent run).

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

1. **Teacher on E1M4-E1M7** until a full `campaign.py --arm rules` run reaches E1M8; measure `rules` and `rules-lag-100` on each map; regression guard E1M2 seeds 26-30 and E1M8 seeds 1-8.
2. **Jev data and joint training** on every map the teacher clears at 100 ms lag (teacher + lagged-teacher games, then DAgger rounds), mixed with the E1M1/E1M2/E1M8 data so no level is forgotten.
3. **Jev campaign:** `campaign.py --arm jev-local` from E1M1 with real carry-over; replace `NOMINAL` with the real E1M7 exit state.
4. **Speed:** below ~85 ms needs fewer prompt tokens (shorter state text, retrain) or a smaller model.

Done on 2026-10-19: E1M8 teacher (boss fight robust to lag, supply sweep, lift state, fireball warnings, escape scoring), planner routes on every map, Jev E1M8 8/20, CUDA graphs, E1M8 videos.
Done on 2026-10-18: E1M1 reliability and DAgger, E1M2 transfer test and joint training, turn-and-fire commands, tic-5 snapshots, share videos and student-vs-teacher side-by-sides (`pc-progress\share\`).

## Commands

```powershell
.\scripts\eval_student.ps1 -Adapter C:\tmp\jev-doom\models\e1m8-d\best -Out runs\eval-x [-Map E1M8] [-Timeout 240] [-Loadout nominal] [-Video]
.\scripts\serve_fast.ps1 -Adapter <adapter>      # serve only, port 8012
C:\tmp\jev-doom\venv-rocm\Scripts\python.exe -m jevdoom.train_fast --data <jsonl...> --model Qwen/Qwen3.5-4B `
    --out C:\tmp\jev-doom\models\<run> --steps 300 --eval-every 100 --grad-ckpt [--init-adapter <adapter> --lr 1e-4] `
    --progress "$env:OneDrive\Transfer\doom-jev\pc-progress\training-<run>.md"
python -m jevdoom.compare --arms rules rules-lag-100 --map E1M8 --seeds 1 2 3 4 5 6 7 8 --jobs 8 --loadout nominal --timeout 240 --no-video --out <dir>
python -m jevdoom.campaign --arm rules --maps E1M1 E1M2 E1M3 E1M4 E1M5 E1M6 E1M7 E1M8 --seed 1 --timeout 300 --out <dir>
```

Teacher games run in parallel with `--jobs` (8-10 on 16 cores; more distorts real-time timing). Stop the fast server before training (it holds ~9 GB; training peaks at 12.5 GB).

## Known issues

- E1M8 student failures: deaths in the Baron fight (late dodges) and occasional stalls near the lift. Each DAgger round adds the states the student actually reaches.
- Weapon choice is a fixed rule (shotgun when it has shells, then chaingun, pistol, fist; rocket launcher only at 300+ units). Ivan wants it to become a Jev skill (switch commands + teacher labels) once campaign runs make shell economy matter.
- Solid things are circles in perception; Doom uses square boxes.
- The probe (60 synthetic situations) measures single decisions, not play.

## Working agreements with Ivan

- Jev decides every command. Perception reports facts; the executor plays fixed macros. No scripted bot behind Jev. Improving the teacher is allowed and expected.
- Evaluate proposals critically before implementing; add progress checks to long runs.
- Paid API calls need approval (Haiku is out of the experiment; no paid calls planned).
- Disposable files under `C:\tmp\jev-doom`, never in OneDrive; only small progress files go to `pc-progress`. Stop only processes you started.
- Work on `phase/<name>` branches; commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Writing: direct, factual, short; no hype phrases.
