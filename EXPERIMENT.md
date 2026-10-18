# Experiment: a local language model learns real-time DOOM from a rule-based teacher

## Question

Can a general-purpose language model, used as a Jev classifier on a home GPU, learn to play DOOM levels in real time at close to its teacher's level: fast enough to play naturally, not turn by turn, and reliably on seeds it never saw?

Then: how much of that skill carries to a level it was not trained on?

## Design (changed 2026-10-18)

The first phase (laptop, 2026-10-17) compared Jev with Claude Haiku. Every LLM arm, Haiku included, failed to finish E1M1; latency and missing training were the cause, not the model family. That comparison is archived in `runs/compare-final` and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Haiku is no longer an arm.

The experiment is now **student vs teacher** on the same seeds:

| Arm | Engine | Where |
|---|---|---|
| `jev-local` (student) | Qwen3.5-4B + LoRA, compact Jev prompt, `fast_server.py` | RTX 3090, ~0.1 s per decision |
| `rules` (teacher) | `expert.py`, instant | CPU |
| `rules-lag-100/200/350` | the teacher with a fixed decision delay | CPU, reference for the cost of latency |

## Held constant

| Factor | Value | Where enforced |
|---|---|---|
| Map, difficulty | E1M1 (then E1M2), skill 3, pistol start | `compare.py --map` |
| Game RNG | ViZDoom seed per run; same seeds for every arm | `--seeds` |
| Clock | real time, 35 tics/s, never paused | `play.py` |
| Perception, commands | perception v2, 20 commands, fixed macros | git commit in `summary.json` |
| Episode limit | 300 s (90 s for DAgger collection) | `--timeout` |

## Seeds

| Seeds | Use |
|---|---|
| 1-5 | first live check |
| 6-25 | expert and lagged-expert games in the training data |
| 26-45 | **held-out reliability test**: never used for training |
| 46-75 | DAgger round 1: the student's own games, expert-labelled, added to training |

## Metrics

Per run (`summary.json`): outcome (`exit` / `died` / `timeout`), route progress, game seconds, kills, damage, decision latency, decisions per second.

Primary: exit rate and median time on seeds 26-45, student vs teacher. Bar for moving on from a level: student clears at least 90% of held-out seeds.

## Procedure

```powershell
.\scripts\eval_student.ps1 -Adapter <adapter dir> -Out runs\eval-<name>           # student, seeds 26-45
python -m jevdoom.compare --arms rules --seeds (26..45) --no-video --out runs\eval-<name>   # teacher, same folder
python -m jevdoom.report runs\eval-<name>
```

## Results

E1M1, held-out seeds 26-45:

| Student | Clears | Median time | Failures |
|---|---|---|---|
| v4 (400 steps) | 14/20 | ~28.5 s | 3 stuck on a barrel at 13%, 3 died stuck at a wall near the exit |
| DAgger round 1 (step 100) | 18/20 | 31.8 s | 2 stuck early (12.5%, 15.4%) |
| Teacher | 20/20 | 23.5 s | — |

E1M2: teacher 10/10 (median 60.1 s) after adding a shotgun detour to the route planner (it was 4/10 with the pistol only). Student: next.

## Next

1. E1M2 transfer test: the E1M1 student on E1M2 with no E1M2 training.
2. Joint E1M1 + E1M2 training, then the same held-out test on both maps.
3. A level held out from training entirely, to measure transfer.

## Known limits

- The student can at best match the teacher; the teacher's gaps (E1M3's one-way climb) are the student's gaps.
- 20 held-out seeds per test: a 90% vs 100% difference is 2 runs.
- Perception and the route planner do the spatial work; the model chooses reactions to described facts. Removing the compass (`--no-compass`) is a separate, harder experiment.
