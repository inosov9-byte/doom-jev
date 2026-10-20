# Current state and next steps (2026-10-19, RTX 3090 PC)

Read this first in a new session. It says where the work stands, where things live, and what comes next. History: [ARCHITECTURE.md](ARCHITECTURE.md). Design: [../EXPERIMENT.md](../EXPERIMENT.md).

## Where it stands

- **Bar set by Ivan:** a student that clears a level about as often as its teacher (at the student's decision speed) is done. Haiku is no longer part of the experiment.
- **Direction:** end-to-end campaign E1M1 -> E1M8 with continuity: each map starts with the health, armor, weapons and ammo carried out of the previous exit (`loadout.py`, `campaign.py`). Until the chain reaches E1M8, E1M8 starts from `NOMINAL["E1M8"]` (shotgun, chaingun, 100 bullets, 30 shells, 100 health).
- **E1M8 done (2026-10-19):** Jev model `C:\tmp\jev-doom\models\e1m8-d\best` (4 rounds: teacher data + 3 DAgger rounds) clears 8/20 unseen seeds (151-170), fastest 133.9 s. Teacher: ~70% instant, ~45% at 100 ms lag. The same model still clears E1M1 10/10 (seeds 26-35, median 31.8 s); E1M2 12/20 (median 86.3 s; joint model 13/20, 75.4 s): no forgetting, ~10 s slower; 2 runs stuck at 99.5% beside the exit. Videos and post facts in `pc-progress\share\` (`jev-plays-doom-e1m8-133.9s`, `jev-vs-teacher-e1m8`, `e1m8-facts.md`); watermarked cuts and the nosov.ai article brief in `share\nosov.ai\`.
- **Student moved to Qwen3.5-2B (2026-10-19):** with CUDA graphs a decision takes 0.8B 21 ms, 2B 37 ms, 4B 80 ms. The teacher at 50 ms delay clears E1M3 5/8 and E1M8 11/20, against 1-2/8 and 4/16 at 100 ms, so the 4B's 90 ms was the binding limit. `C:\tmp\jev-doom\models\2b-b\best` (2400 steps on all data, 90.6% agreement) beats the 4B camp-a on the same seeds: E1M2 carried 9/10 vs 6/10, pistol 8/10 vs 7/10, E1M8 5/10 vs 2/10, E1M3 carried 1/10 (first student clear, 121.2 s, video in `share\nosov.ai\`), E1M1 10/10. Serve with `serve_fast.ps1 -Model Qwen/Qwen3.5-2B -Adapter ...`. Teacher bar is now `rules-lag-50`.
- **Campaign state (2026-10-20):** planning, not only lag, limited the chain. (1) `loadout.apply` zeroed carried armor unless exactly 100/200 and clipped ammo above the normal max: fixed (all earlier campaign numbers have this bug). (2) Resource-aware planner (`with_resource_detours`, value-weighted limits: soulsphere 2000, blue armor 1800, chaingun 1500, green armor/RL/backpack 1200; armor early; monster-guarded items skipped; exit top-up from 2500 units). Teacher campaigns rules-lag-50, 20 paired seeds: 53 -> 59 maps cleared; E1M2 leaves with 90 armor (was 0); E1M4 7/10 with the chaingun; first E1M5 clears (2/7). Fights round 5: E1M4 16/32 vs 11/32, E1M8 20/32 vs 14/32 (paired). Student: `models\2b-e\best` (2B, 92.4%, trained on the fights-5 teacher, before the planner/loadout fix). Best student continuous run: E1M1-E1M3 cleared, E1M4 reached 91% (2b-d seed 70). Next: train on teacher campaigns with the new planner (`data\camp-teacher`), then student campaigns on the fixed code. Measure with paired runs (before and after in the same pass): load swings move sequential batches by up to 2x.
- **Decision timing:** HTTP 0.6 ms, perception 2 ms; the 5-tic settle wait before the next snapshot is optimal (teacher at 50 ms, E1M3/E1M8: settle 2: 0/8, 1/8; 3: 4/8, 3/8; 5: 4/8, 6/8; 7: 3/8, 3/8). Faster cadence with snapshots that do not yet show the last command loses.
- **Sound:** `replay.py` replays a logged run tic by tic (exact, checked per decision) with the audio buffer on; the E1M8 and E1M2 share videos now have game sound. Live recordings: `--sound`.
- **Key finding:** a teacher tuned with instant reactions is unlearnable for a ~90 ms student: 50 ms of lag took the E1M8 teacher from ~70% to 0%. Teacher tactics are now measured with `rules-lag-100` too (shoot only when the shotgun is ready, dodge during the reload, stand and aim 0.35 s before the shot, keep distance from projectile throwers, openness-aware escape from dead-end corners).
- **Student speed:** `fastjev.Scorer` replays CUDA graphs per padded prompt length: 87 ms per decision (was 114), same choices. The 4B forward is compute-bound at ~400 tokens beyond that; a cached prompt prefix did not help.
- **Campaign teacher (rules / rules-lag-100):** E1M1, E1M2 reliable; E1M3 often without lag, rarely with it; E1M4 ~2/8; E1M5-E1M7 0 (routes exist on every map now; fights remain). Work continues on `phase/campaign-fights2` (agent run).
- **Campaign supplies (`needs.py`, branch `phase/campaign-supplies`):** when health drops below 40 or the shotgun is nearly dry, perception swaps the compass to the nearest live item on a short walk (A* off the game loop, no monster near the item), then restores the stage route; in a campaign (`carry`), the last 1200 route units before each exit also top up health (<80), armor (<30) and shells (<15). Seeds 26-45, merged fights code: E1M3 exits 7/18 -> 10/18 (rules), 2/15 -> 4/16 (100 ms); E1M2 leaves with 92 health instead of 77-84; maps cleared 46 -> 48 (rules), 36 -> 40 (100 ms). E1M4 still dies at ~48% (imps at the switch stage, shells in hand). `--no-needs` (campaign.py, compare.py, play.py) turns it off. Guards, with vs without: E1M1 and E1M2 pistol start seeds 26-30 10/10 vs 10/10 and 10/10 vs 9/10; E1M8 nominal seeds 1-8 run twice, rules 11/16 vs 12/16, 100 ms 5/16 vs 4/16 (no detour started in any failed rules run). Tried and dropped: an armor detour (up to +800) on entering with <30 armor (E1M2 rules 6/9); bullets in the top-up and health <50 mid-map (E1M1 detours into the nukage room barrels and imps, 3 deaths in 20). Side effect: detours shift E1M2's view so the incidental green armor is missed more often (rules: 0 armor leaving E1M2 in 11 of 18 exits vs 6 of 18).

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
python -m jevdoom.replay runs\<eval>\<run> --out C:\tmp\jev-doom\audio [--render]   # exact tic replay + game audio -> <run>-sound.mp4
```

Recorded runs replay exactly (map, seed, skill, loadout, command per logged tic; checked per decision against the log). `replay.py` adds game sound to an old silent run.mp4; `play.py --record-video --sound` records sound live (no timing cost measured). Share videos with sound since 2026-10-19: `share\nosov.ai\` (E1M8 133.9 s, E1M2 71.7 s).

Teacher games run in parallel with `--jobs` (8-10 on 16 cores; more distorts real-time timing). Stop the fast server before training (it holds ~9 GB; training peaks at 12.5 GB).

## Known issues

- E1M2 exit at 100 ms: 2-4 of 20 campaigns stall beside the exit (stuck at 99%), standing on the lift in front of the switch and pressing use on the raised lift ahead (expert step 7) while the compass still reads 64 units left; appeared with or without supply detours.
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
