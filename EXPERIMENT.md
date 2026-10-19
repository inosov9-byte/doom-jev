# Experiment: a local language model learns real-time DOOM from a rule-based teacher

## Question

Can a general-purpose language model, used as a Jev classifier on a home GPU, learn to play DOOM levels in real time at close to its teacher's level: fast enough to play naturally, not turn by turn, and reliably on seeds it never saw?

Then: can one model play the whole episode, E1M1 to E1M8, in one run, carrying its health, armor, weapons and ammo from each exit into the next map?

## Design (since 2026-10-18)

The first phase (laptop, 2026-10-17) compared Jev with Claude Haiku. Every LLM arm, Haiku included, failed to finish E1M1; latency and missing training were the cause, not the model family. That comparison is archived in `runs/compare-final` and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Haiku is no longer an arm.

The experiment is **student vs teacher** on the same seeds:

| Arm | Engine | Where |
|---|---|---|
| `jev-local` (student) | Qwen3.5-4B + LoRA, compact Jev prompt, `fast_server.py` with CUDA graphs | RTX 3090, ~90 ms per decision |
| `rules` (teacher) | `expert.py`, instant | CPU |
| `rules-lag-50/100/200/350` | the teacher with a fixed decision delay | CPU; `rules-lag-100` is the teacher at the student's speed and the bar the student is measured against |

The teacher is tuned until it works at `rules-lag-100`: tactics that need instant reactions cannot be learned by a student that needs ~90 ms per decision (see the E1M8 finding below).

## Held constant

| Factor | Value | Where enforced |
|---|---|---|
| Difficulty | skill 3 ("Hurt me plenty") | `env.py` |
| Start state | E1M1, E1M2: pistol start; E1M8: `nominal` loadout (shotgun, chaingun, 100 bullets, 30 shells, 100 health), a stand-in for an E1M7 exit; campaign runs: the previous map's exit state | `--loadout`, `loadout.py`, `campaign.py` |
| Game RNG | ViZDoom seed per run; same seeds for every arm | `--seeds` |
| Clock | real time, 35 tics/s, never paused | `play.py` |
| Perception, commands | perception text, 26 commands, fixed macros | git commit in `summary.json` |
| Episode limit | 120-300 s per map (90 s for DAgger collection) | `--timeout` |

## Seeds

| Seeds | Use |
|---|---|
| 1-5 | first live check |
| 6-25 | teacher and lagged-teacher games in the training data |
| 26-45 | **held-out test** for E1M1 and E1M2, never used for training |
| 46-75 | DAgger collection (the student's own games, teacher-labelled) |
| E1M8: 1-16 teacher data and checks; 51-70 and later DAgger; 151-170 | **unseen test** for the final E1M8 model |

## Metrics

Per run (`summary.json`): outcome (`exit` / `died` / `timeout` / `stuck`), route progress, game seconds, kills, damage dealt, hits, decision latency, decisions per second, start and end loadout.

Primary: exit rate on held-out seeds, student vs teacher at the student's delay. Agreement with the teacher on held-out states is a training diagnostic only; a model at 94% agreement can still lose most runs. A level is done when the student clears about as often as `rules-lag-100`.

## Procedure

```powershell
.\scripts\eval_student.ps1 -Adapter <adapter dir> -Out runs\eval-<name> [-Map E1M8 -Loadout nominal -Timeout 240]   # student
python -m jevdoom.compare --arms rules rules-lag-100 --map <map> --seeds (26..45) --jobs 8 --no-video --out runs\eval-<name>  # teacher, same folder
python -m jevdoom.report runs\eval-<name>
```

## Results

E1M1, held-out seeds 26-45:

| Student | Clears | Median time | Failures |
|---|---|---|---|
| v4 (400 steps) | 14/20 | ~28.5 s | 3 stuck on a barrel at 13%, 3 died stuck at a wall near the exit |
| DAgger round 1 (step 100) | 18/20 | 31.8 s | 2 stuck early (12.5%, 15.4%) |
| Joint E1M1+E1M2 model | 20/20 | 29.9 s | — |
| E1M8 model e1m8-d (seeds 26-35, regression check) | 10/10 | 31.8 s | — |
| Teacher | 20/20 | 23.5 s (19.5 s after turn-and-fire) | — |

E1M2, held-out seeds 26-45:

| | Clears | Median time |
|---|---|---|
| E1M1-only student, zero-shot (6 seeds) | 0/6 | reached ~85% of the route; froze at the first mid-level switch |
| Joint model | 13/20 | 75.4 s |
| v5 (26 commands) | 12/20 | 79.2 s |
| Teacher, instant (seeds 26-35) | 6/10 before the exit-room fix, 10/10 after | 58.7 s |

E1M8 (boss finale), loadout `nominal`, timeout 240 s:

| | Clears | Fastest |
|---|---|---|
| Original teacher, instant | ~70% | ~120 s |
| Original teacher, 50 ms delay | 0% | — |
| Rebuilt teacher, 100 ms delay (seeds 1-16) | ~45% | — |
| Jev round 1 (teacher data only, seeds 1-10) | 0/10, stuck at the lift | — |
| Jev round 3 (2 DAgger rounds, seeds 51-70) | 7/20 | 160.5 s |
| Jev round 4 (3 DAgger rounds, unseen seeds 151-170) | **8/20** | 133.9 s |

Finding: tactics that need instant reactions do not transfer to a ~90 ms student. 50 ms of added delay took the original E1M8 teacher from ~70% to 0%. The teacher was rebuilt until it cleared at 100 ms delay (fire only when the shotgun is ready, dodge during the reload, aim 0.35 s before the shot, hold 330-520 units from the Barons, escape toward open space, dodge fireballs from their velocity), and the student then matched it.

Campaign teacher (E1M1 → E1M8 with carry-over): E1M1 and E1M2 reliable; E1M3 often without delay, rarely at 100 ms; E1M4 ~2/8; E1M5-E1M7 not yet. The planner has routes on every map; the remaining failures are fights.

## Next

1. Teacher fights on E1M3-E1M7 until a full `campaign.py --arm rules` run reaches E1M8 at 100 ms delay.
2. Joint training on every map the teacher clears, mixed with the E1M1/E1M2/E1M8 data so no level is forgotten.
3. `campaign.py --arm jev-local` from E1M1 with real carry-over, replacing the `nominal` E1M8 start.
4. Weapon choice as Jev commands instead of a fixed rule, once shell economy across maps matters.

## Known limits

- The student can at best match the teacher; the teacher's gaps are the student's gaps.
- 20 seeds per test: 40% vs 45% is one run.
- Perception and the route planner do the spatial work; the model chooses reactions to described facts. Removing the compass (`--no-compass`) is a separate, harder experiment.
- Weapon choice is a fixed executor rule (shotgun while it has shells, then chaingun, pistol, fist).
