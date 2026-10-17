# Experiment: Jev vs Haiku at real-time DOOM

## Question

With the same game, the same perception text, the same 17 commands and the same real-time loop, how does a Jev classifier compare with Claude Haiku 5.5 at playing DOOM E1M1?

- **Jev** (Featherless Simple Jev): one forward pass of an open model; the answer is read from the next-token scores of the 17 labels. Returns a probability per command. Runs hosted or locally.
- **Haiku** (`claude-haiku-5-5`): a generative model behind Anthropic's API. Writes its answer as JSON constrained to the 17 command ids (structured outputs). No probabilities.

Only the decision engine changes between arms. Everything else is fixed by code, not by convention.

## Arms

| Arm | Engine | Model | Where | Prompt |
|---|---|---|---|---|
| `jev-local` | Jev | Qwen3.5-0.8B fine-tuned on the rule-based expert's labels (RFDT LoRA) | laptop GPU | short student prompt it was trained on |
| `jev-hosted` | Jev | `featherless-ai/Qwen3.6-35B-A3B-classifier` (the model in Featherless's own Doom demo) | Featherless demo API | shared instructions + command descriptions |
| `jev-hosted-gemma` | Jev | `featherless-ai/gemma-4-26B-A4B-classifier` (our teacher) | Featherless demo API | shared |
| `haiku` | Claude | `claude-haiku-5-5`, thinking off, effort `low` | Anthropic API | shared |
| `haiku-think` | Claude | `claude-haiku-5-5`, adaptive thinking, effort `low` | Anthropic API | shared |
| `rules` | reference | `expert.py`, the teacher, instant decisions | local | same state text, parsed |
| `rules-lag` | reference | `expert.py` delayed to 0.7 s per decision: what a perfect copy of the teacher could do at local Jev's speed | local | same |

"Shared" = `jev.INSTRUCTIONS` (identical to `prompts/facts.txt`) plus the descriptions in `commands.COMMANDS`, given to every engine word for word. The local student uses its training prompt because that prompt is part of that system.

The teacher changed during the project: hosted Gemma-4-26B labels first, then the rule-based expert, after Gemma died at 26% of the route live while the expert clears E1M1 on every seed. The two `rules` arms make the teacher's ceiling and the cost of latency visible in the same table.

## Held constant

| Factor | Value | Where enforced |
|---|---|---|
| Map, difficulty | E1M1, skill 3 (Hurt Me Plenty), pistol start | `play.py` defaults |
| Game RNG | ViZDoom seed per run; the same seed list for every arm | `--seed` |
| Clock | real time, 35 tics/s, never paused | `play.py` |
| Decision loop | serial; next request waits until the last command took effect | `play.Brain` |
| Perception | `perception.py` at the run's commit, compass on | git commit in `summary.json` |
| Commands | 17 labels, fixed macros | `commands.py`, `executor.py` |
| Episode limit | 300 s of game time | `--timeout` |
| Hosted Jev rate | ≥ 0.55 s between request starts (demo limit 2/s) | `--min-gap` |

Trajectories diverge once engines choose differently; the seed fixes the starting randomness, not the run.

## Metrics

Per run, from `summary.json` and `decisions.jsonl`:

| Metric | Meaning |
|---|---|
| `outcome` | `exit`, `died`, or `timeout` |
| `route_progress_pct` | furthest point reached along the A* route to the exit (primary) |
| `game_seconds` | time survived or time to exit |
| `kills`, `damage_taken` | combat result |
| `decisions_per_s`, `latency_median`, `latency_p90` | speed of the engine in the loop |
| `errors` | failed engine calls (player stands still) |
| `cost_usd` | Haiku: tokens x list price; Jev: 0 (free demo / local) |

Offline, no time pressure: the 60-situation probe (`probe.py`) per arm, to separate decision quality from speed.

Primary comparison: median `route_progress_pct` and exit rate per arm over the same seeds. Secondary: survival time, kills, latency, cost per run.

## Procedure

```powershell
# 1. local Jev server with the trained student
.\scripts\serve.ps1 -Model C:\tmp\jev-doom\models\jev-doom-v2
# 2. Anthropic credentials for the Haiku arms
$env:ANTHROPIC_API_KEY = "..."
# 3. offline probe per arm
C:\tmp\jev-doom\venv\Scripts\python.exe -m jevdoom.compare --probe --arms jev-local jev-hosted jev-hosted-gemma haiku haiku-think
# 4. live runs: 5 seeds per arm, interleaved by seed
C:\tmp\jev-doom\venv\Scripts\python.exe -m jevdoom.compare --arms jev-local jev-hosted jev-hosted-gemma haiku haiku-think --seeds 1 2 3 4 5
```

`compare.py` writes `runs/compare-<time>/results.md` and `results.json`, and keeps every run folder (video, decisions, summary). `--dry-run` swaps every engine for a mock with the same latency profile to test the harness for free.

## Cost

- Jev arms: free (local GPU, free Featherless demo).
- Haiku: about 600 input and 10 to 20 output tokens per decision without thinking, roughly 1.2 decisions/s, so about 200k input tokens per 5-minute run: roughly $0.02 to $0.05 per run at $0.10 / $0.50 per million tokens. The thinking arm costs more per decision. 5 seeds x 2 Haiku arms: under $1 (estimate; actual tokens are logged per run).

## Known limits

- Small N (5 seeds per arm); report medians and every run, not just averages.
- The hosted demo is a shared public endpoint; its latency varies with load. Latency is logged per decision.
- Network latency to Anthropic and Featherless differs; that is part of what is being compared.
- The local student imitates Gemma; its ceiling is the teacher's quality at its own speed.
