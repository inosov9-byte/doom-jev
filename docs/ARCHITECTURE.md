# What it took: Jev plays DOOM

How a text classifier ended up playing DOOM in real time, what the system looks like, and what broke on the way. Part 1 (2026-10-17) ran on a Ryzen AI 9 HX 375 laptop (Radeon 890M iGPU, 32 GB); no language model finished E1M1 there. Part 2 (2026-10-18, [below](#part-2-rtx-3090-the-student-clears-e1m1)) moved to an RTX 3090 PC, where a local 4B student reached its teacher's level on E1M1 and E1M2. Part 3 (2026-10-19, [below](#part-3-e1m8-the-teacher-at-the-students-speed)) took it to E1M8, the episode finale, and started the end-to-end campaign.

Each part describes the system as it was at the time. The current system is in the [README](../README.md).

## The starting point

A Featherless demo shows Jev, their classifier API, "playing" DOOM in a browser. Reading the demo's source changes the picture: a scripted route follower handles navigation, doors and the exit, and Jev only picks combat actions. Its README says level completion is not guaranteed.

The goal here was stricter: Jev picks every command, and runs locally.

## What Jev is

Jev is not a model. It is a way of using one: the model reads a prompt that lists labelled options, and instead of generating text, Jev reads the probabilities of the label tokens in one forward pass. Any open model can sit behind it. Featherless hosts some (Qwen3.6-35B, Gemma-4-26B, ...); the same server code runs locally.

Consequences that shaped everything:

- Jev cannot output a number. Every amount (how far to turn) has to be a label.
- Jev reads text. Something has to turn the game into text.
- Cost per decision is one prompt pass. Prompt length is latency.

## System (part 1, laptop)

```
                 ┌──────────── offline ─────────────┐
 ViZDoom E1M1 ──►│ collect.py   teleport + random   │
                 │ expert runs  rules play the level│──► states (text)
                 └──────────────────────────────────┘          │
                                                               ▼
                                    expert.py labels every state (free, instant)
                                                               │
                                                               ▼
                         RFDT (Simple Jev's trainer), LoRA on Qwen3.5-0.8B, AMD GPU
                                                               │
                                                               ▼
 ┌──────────────────────── real time, 35 tics/s ─────────────────────────────┐
 │ ViZDoom ─► perception.py ─► text ─► engine ─► command ─► executor.py ─► buttons │
 │            (facts only)            Jev local / Jev hosted / Haiku / rules        │
 └──────────────────────────────────────────────────────────────────────────┘
```

| Part | Role | Decides anything? |
|---|---|---|
| `perception.py` | game state to text: health, ammo, enemies (distance, exact angle, in crosshair), raycast obstacles with door state, nukage, route compass, off-screen attacker direction, result of the last command | no |
| `engines.py` | the decision engine for an arm: Jev endpoint, Claude Haiku, rules, mock | yes, this is the only decider |
| `executor.py` | plays a command as a fixed button macro; the next command cancels it | no |
| `play.py` | real-time loop; a worker thread asks the engine while the game keeps running | no |
| `expert.py` | rule-based player reading the same text; teacher and reference arm | yes, only in its own arm and for labels |

## The command set, and why it is 20 (26 since part 2)

| Version | Commands | What forced the change |
|---|---|---|
| 1 | 10: move, turn 15 / 60, strafe, fire, use | first design |
| 2 | 13: turns 15 / 45 / 90 / 180 | one decision per correction instead of four |
| 3 | 17: added 5 and 30 degree turns | measured on 743 teacher aiming decisions: with 15/45/90, 38% of turns could not get within 7 degrees of the enemy; far targets 8 to 11 degrees off could not be centred at all |
| 4 | 20: added `forward_short`, `fire_strafe_left/right` | the expert's runs: no way to dodge while shooting, no precise approach on the nukage bridge |

Moves last 20 tics (0.57 s), about one decision interval, so the player keeps moving between decisions. Turns are exact (no momentum in DOOM turning).

## Teachers

| Teacher | How it was tested | Result |
|---|---|---|
| Hosted Qwen3.6-35B (Featherless demo default) | probe + live | ignores enemies and follows the route arrow; died at 47% |
| Hosted Gemma-4-26B | probe + live | best LLM teacher (0.86 to 0.90 on the probe), died at 26% live: slow and toggled a door shut |
| Rule-based expert | live | clears E1M1 5 of 5 seeds without time pressure (27 to 34 s), 35 s in real time |

The probe is a fixed set of 56 (later 60) synthetic situations with unambiguous right answers. It measures single decisions, not play. Gemma scored well on it and still could not play, which is why the teacher became the rule-based expert: correct by construction for the basics, and free to label any state, including states the student reaches by itself.

## Bugs that looked like model failures

| Symptom | Cause | Fix |
|---|---|---|
| Agent oscillates left/right in place for a minute | the next decision used an observation taken before the previous command ran | wait until the command has taken effect |
| Same oscillation again, only at zero latency | the "taken effect" signal fired after a turn, but the worker read the observation from the start of that tic | release the worker only after a fresh post-command observation |
| Teacher spams `use` at a door that never opens | vanilla DOOM closes a door if you press use while it opens; the text said "closed" | door state in the text: closed / opening (time to open) / closing |
| Agent dies to enemies it cannot see | no cue for off-screen attackers | "being shot from out of view, N deg right", the status-bar face cue |
| Expert empties its pistol at an imp 1,060 units away | no engagement range | engage within 750 units |
| Every timeout scored as "exit" | episode clock counted warm-up tics | use ViZDoom's own episode clock |

## Latency is the constraint

The same rule-based expert, delayed to 0.7 s per decision (the local Jev speed), died at 64% of the route on seed 1, while the instant expert clears every seed. Decision quality alone does not finish the level; reaction time does.

| Engine | Decision latency (median) |
|---|---|
| Local Jev, Qwen3.5-0.8B on the 890M | 0.6 to 0.7 s |
| Hosted Jev, Qwen3.6-35B | ~0.9 s |
| Hosted Jev, Gemma-4-26B | ~1.4 s |
| Claude Haiku 5.5 | measured in the comparison |
| Rules expert | ~0 |

## Beyond E1M1 (branch `phase/multi-level`)

Later levels need things E1M1 does not: keys, locked doors, doors opened by a switch, a walk-over line or a shot, and lifts.

- `progression.py` models the map as a world state (keys held, trigger tags fired). A breadth-first search over key pickups and triggers finds the shortest sequence of subgoals after which the exit is reachable. Triggers are only expanded if they enlarge the reachable area, which keeps the search small.
- The compass follows the current subgoal: "Route to red keycard: ...", "Route to switch: ...", then "Route to exit: ...". The text adds keys held, switches, and "door (locked, needs red key)". On E1M1 the text is unchanged.
- The expert presses switches, fires at shoot-switches, walks onto keys and trigger lines, and presses use on a lift wall when the route is blocked by a ledge.

| Level | Progression found | Expert result (no time pressure) |
|---|---|---|
| E1M1 | exit | clears, 5 of 5 seeds |
| E1M2 | red keycard, switch, exit | clears 2 of 3 seeds (47 to 51 s); third died at 97.5% |
| E1M3 | not yet: a one-way climb the planner does not model | open |

## Running locally on an AMD laptop

- AMD's ROCm 7.2.1 PyTorch for Windows supports the Radeon 890M (gfx1150). 0.8B: 0.61 s per call vs 4.3 s on CPU.
- RFDT training needed two workarounds: a placeholder for `torch.distributed.tensor` (missing in the Windows ROCm build), and copying the image processor configs that RFDT's export leaves out.
- Training speed: about 0.35 examples/s, about 2.5 GiB of GPU memory. Most time goes to Qwen3.5's linear-attention layers running on the slow reference path.
- Simple Jev's prompt template is about 620 tokens of fixed instructions, examples and the option list printed twice; the game state is about 150. Each extra command label costs prompt length on every decision.

## Results

Live, real time, E1M1 skill 3, 5 seeds per arm, 300 s limit (`runs/compare-final/results.md`):

| Arm | Exit | Route progress, median (best) | Median decision | Decisions/s | Cost, 5 runs |
|---|---|---|---|---|---|
| Rules expert, instant | 5/5 | 100% in 21.7 s | ~0 s | 5.8 | 0 |
| Rules expert, 0.7 s per decision | 0/5 | 87.9% (89.4%) | 0.70 s | 1.17 | 0 |
| Jev hosted, Gemma-4-26B | 0/5 | 52.0% (75.6%) | 1.38 s | 0.66 | free demo |
| Jev hosted, Qwen3.6-35B | 0/5 | 23.6% (23.8%) | 0.68 s | 1.08 | free demo |
| Claude Haiku 5.5 | 0/5 | 12.7% (12.7%) | 0.96 s | 0.86 | $0.13 |
| Claude Haiku 5.5, thinking | 0/5 | 12.7% (12.7%) | 1.32 s | 0.58 | $0.13 |

What the table says:

- Rules that clear the level every time stop clearing it when each decision arrives 0.7 s late. In real time, latency costs as much as judgement.
- Both Haiku arms get stuck at the same corridor corner on every seed. The text says "moved 0 units (blocked)" after each step; Haiku keeps choosing forward and small turns. The expert reads the same line and sidesteps.
- Hosted Jev with Gemma is the strongest LLM arm despite being the slowest: it fights back and follows doors and the route, and it dies in the nukage room fights.

Single-decision quality does not predict play. On the 60-situation probe (20 commands), Haiku scores highest (0.875, Gemma 0.825, hosted Qwen 0.483), but its weakest category is exactly "blocked" (0.375; 0.125 with thinking), the one situation that ended all ten of its live runs. The rules expert scores only 0.617 on the probe, because it deliberately skips distant fights the probe expects it to take, and it is the only arm that finishes the level.

### The local student

Distilling the rule-based expert into Qwen3.5-0.8B takes far longer on this laptop than distilling an LLM teacher did. The first two runs (hard labels, then soft labels with a higher learning rate) were stopped at step 150 while still predicting "forward" for everything. The diagnosis:

| Test | Result |
|---|---|
| Logits-at-last-position shortcut on vs off | identical to three decimals; not the cause |
| Mid-training evaluation on vs off | identical; not the cause |
| Prompts and targets as compiled by RFDT | correct (20 answer letters, targets map to the right commands) |
| 5 clear-cut commands, 150 steps | flat at chance until about step 110, then a sharp drop: 75% held-out agreement at step 150 (chance 20%) |

So the pipeline learns, after a long plateau, and the earlier runs were stopped just before that point would likely have arrived. A 450-step run on the full expert dataset was started afterwards (about 3.8 hours at 0.35 examples/s).

The 450-step run ended the same way: held-out agreement 6.7% (always `forward`) at step 300, about 32 s per step. It was stopped.

## Part 2: RTX 3090, the student clears E1M1

Same perception, commands and teacher; a desktop with an RTX 3090 (24 GB). Perception v2 (body-width rays, barrels and pillars as obstacles, a "Recent commands ... moved N units" line) was already in place.

### Where the time went

Measured before training anything. A decision is one prefill forward pass; nothing is generated.

| Model | Simple Jev server, stock prompt (~1,130 tokens) | Bare forward pass, 1,168 tokens | Bare forward pass, 300 tokens |
|---|---|---|---|
| Qwen3.5-0.8B | 172 ms | 62 ms | 63 ms |
| Qwen3.5-2B | 183 ms | 88 ms | 62 ms |
| Qwen3.5-4B | 337 ms | 225 ms | 77 ms |

Two findings:

- The server added ~100 ms: it runs a prefix pass and a branch pass per request, plus a cache copy.
- The stock v1 template prints the 20 options twice, after the state: ~730 of ~1,130 tokens. Only ~185 tokens are the game state.

`fastjev.py` uses a compact prompt (options A-T once, then the state, then `Answer:`, ~355 tokens), and `fast_server.py` scores it in one forward pass behind the same `/v1/classifier` API. Qwen3.5-4B: 337 → 90 ms. The method is still Jev (scores over option letters); the template and server are ours, and training uses the same template.

Other candidates measured with the stock server (median, untrained agreement with the expert on 40 states): Qwen3-1.7B 228 ms (20/40), Gemma-4-E2B 262 ms (21/40), Qwen3-4B 429 ms (15/40), Qwen3.5-9B 452 ms (17/40). The standard-attention models are slower than Qwen3.5 at similar size. Meta's 2026 open models (Muse Spark Open 816B, Muse Glimmer 30B) are far too large for a 0.1 s decision.

### Why the laptop student never learned, and the fix

RFDT's laptop runs plateaued at "always forward". On the PC, a small trainer (`train_fast.py`: LoRA rank 32 on all linear layers, cross-entropy over the 20 option letters at the answer position, batch 16, lr 2e-4) learned at once: 2B reached 40% held-out agreement after 10 steps. Without gradient checkpointing, training spilled past 24 GB into system memory (52 s per step); with it, 2.6 s per step for 2B and ~5 s for 4B.

Learning-rate and class-imbalance explanations for the laptop plateau were checked and ruled out (those runs used 2e-4 to 5e-4; `forward` was 22% of labels). The likely cause is the stock template plus far too few steps; it was not investigated further once the new trainer worked.

### Data (v4, perception v2 text)

| Source | States |
|---|---|
| Teleport situations: general, heading-aligned, route tail (`collect.py`) | 5,000 |
| Instant expert games, seeds 6-25 | ~2,700 |
| Expert delayed 0.2 s and 0.35 s, seeds 6-15: the messy positions a slightly late player reaches | ~2,400 |
| Total, labelled by `expert.py` from the text | 10,155 |

### Results

Qwen3.5-4B, 400 steps (33 min): held-out agreement 22% → 77% (step 100) → 87% → 92% → 96%.

| Test | Student | Teacher |
|---|---|---|
| Seeds 1-5 | 5/5, median 27.4 s, 89-91 ms per decision | 5/5, median 21.4 s |
| Held-out seeds 26-45 | 14/20 | 20/20, median 23.5 s |

The first five seeds overstated reliability. The six failures: three runs pressed `forward_short` into the same barrel at (1125, -2897) for 300 s, the barrel that stopped every Haiku run in part 1, while the expert would sidestep. Two died standing against a wall on the nukage floor near the exit. Positions the student reached that the training data barely contained.

### DAgger

The student played seeds 46-75 (17/30 clears, 12 stuck on the barrel). Every state it reached was labelled by the expert, deduplicated (a stuck run repeats one state hundreds of times; 6,000 rows → 2,248 unique), weighted 3x and added to the data. Training continued from the v4 adapter at lr 1e-4.

| Student | Held-out seeds 26-45 |
|---|---|
| v4 | 14/20 |
| DAgger round 1, step 100 | 18/20, median 31.8 s, no deaths, 2 stuck early |

### E1M2: fixing the teacher first

The teacher cleared E1M2 on 4 of 10 seeds. Every death was the same: two imps at claw range near the end, fought with the pistol while carrying 20 useless shells. A shotgun lies 152 units off the route but never comes into view. A health top-up rule did not help (4/10).

The planner now inserts a weapon detour as a route stage when it costs at most 1,000 extra units: "Route to shotgun", done once the shotgun is held. E1M2's shotgun costs +961 units (about 2 s). E1M1 gets no detour (only a chaingun, +801, and E1M1 already works). Teacher on E1M2: 10/10, median 60.1 s.

### E1M2 transfer, then joint training

The E1M1 student on E1M2 with no E1M2 training (6 seeds): 0 clears, median ~85% of the route. It took the red key and the shotgun detour and won ~10 fights, then froze at the first mid-level switch ("Route to switch: switch 50 away"), sidestepping into a ledge 200 times. E1M1 has no mid-level switches. On single decisions it already agreed with the expert on 95% of E1M2 states; the misses were rare and decisive.

Joint training (E1M1 data + E1M1 DAgger + 10,368 E1M2 states, 400 steps from the DAgger adapter): held-out agreement 97.0%. Live, seeds 26-45: **E1M1 20/20** (median 29.9 s), **E1M2 13/20** (median 75.4 s). The instant teacher clears 6/10 of E1M2 seeds 26-35, so the student reached its teacher's level. Ivan's bar from here: a student at teacher level on a level is done.

### Latency: the teacher at the student's speed, and two ways to cut its cost

E1M2 punishes delay far more than E1M1: the teacher with 0.2 s per decision clears 1/10 (E1M1: 4/5).

| Change | Teacher at 0.1 s delay, E1M2 seeds 26-35 |
|---|---|
| Baseline (snapshot 8 tics into a move) | 5/10 |
| Snapshot at tic 5: inference overlaps the running command (Ivan's idea) | **7/10**, 4.84 vs 4.17 decisions/s |
| Snapshot at tic 3 | 4/10: the previous command has barely started |

54% of the teacher's E1M1 shots and 64% on E1M2 came right after a turn: two decisions per shot. Six commands `turn_{left,right}_{5,15,30}_fire` (turn on the first tic, then fire) do both in one, and release the next decision after the first shot. They are appended after the 20 existing commands, so trained students keep their option letters.

### Teacher fixes found on E1M2 and later levels

| Failure | Fix | Effect |
|---|---|---|
| Ran past an imp and two zombiemen into the E1M2 exit room (sprint mode near the exit) | sprint only with at most one enemy within 300 | E1M2 instant teacher 6/10 → 10/10 on seeds 26-35 |
| Pressed use 60 times on a door raised by a remote trigger (E1M9) | no second use on a door still closed | |
| Stood still while a demon bit it (E1M6) | sidestep-fire at demons/spectres within 110 | |

With turn-and-fire, tic-5 snapshots and these rules: E1M1 instant teacher median 19.5 s (was 23.5 s), E1M2 10/10.

### v5 student (26 commands)

All perception-v2 states collected so far (35,351 after deduplication; 18,862 of them from the student's own games) relabelled by the new teacher, 600 steps from the joint adapter: held-out agreement 97.4%. Live, seeds 26-45: E1M1 20/20 (median 34.5 s), E1M2 12/20. Same as the joint model: at teacher level.

### Planner fixes for E1M3-E1M9

A teacher survey of E1M3-E1M9 showed the planner, not the fighting, blocking most levels.

| Problem | Fix |
|---|---|
| Routes through pillars, torches and barrels: the grid modelled only walls (E1M9's plan went through a gap two torches close; also the E1M1 start barrel) | solid decorations present at skill 3 block their cells |
| E1M3 "one-way climb": a stair builder (special 8) raises a chain of untagged sectors; the planner raised only the tagged first step | stair building follows Doom's rule (same floor texture, +8 per step). E1M3 solves: blue key, stair trigger, exit |
| Waypoint line-of-sight checked on the final stage's grid, where later triggers have opened doors still shut now | checked on the current stage's grid |
| Within 120 units of a goal the compass pointed straight at it, through wall corners (E1M2 exit, 99.5% stalls) | only with a clear walk to the goal's standing spot |
| Map changed during play (trap doors, raised floors) | the stage grid is rebuilt from live sector heights when a non-door sector moves |
| Detour to a shotgun that exists only on easy skills (E1M3) | detour weapons filtered by skill and multiplayer flags |

| Level | Planner | Teacher (seeds 1-5) | Blocker |
|---|---|---|---|
| E1M3 | solved | dies at ~26% | drops into a dead-end damaging pit; no route back out |
| E1M4, E1M7 | no route (solved in part 3) | — | |
| E1M5 | solved | dies at ~15% | shotgun-guy ambush, pistol only |
| E1M6 | solved | dies at ~28% | fights beside nukage |
| E1M8 | no exit line (part 3 adds "kill target" subgoals) | — | boss level |
| E1M9 | solved | dies at ~35% | not investigated yet |

E1M3 zero-shot (v5 student, never trained on it, 10 seeds): dies at 9-32% of the route (median ~17%), against the teacher's 25-27%.

## Part 3: E1M8, the teacher at the student's speed

The goal changed from mastering single levels to one model playing E1M1 to E1M8 in one run. E1M8 ("Phobos Anomaly", two Barons of Hell, 1,000 HP each) went first because it is the hardest map of the episode.

### Continuity between maps

`loadout.py` reads health, armor, weapons and ammo at a map's exit and applies them at the next map's start through console commands (keys do not carry, as in Doom). `campaign.py` chains maps and stops at the first map without an exit. Until the teacher's chain reaches E1M8, E1M8 starts from `nominal`: shotgun, chaingun, 100 bullets, 30 shells, 100 health.

### The planner on every map

| Addition | Why |
|---|---|
| "Kill target" subgoals and a stand-spot route to them | E1M8 has no exit line until both Barons die; aiming the A* at an unreachable alcove froze perception for 2 s per decision |
| Supply detours (shells, armor, medikit, soulsphere) before the first boss | the boss fight needs more than the 30 starting shells |
| Lift state from the floor height: "lift (raised)" / "lift (lowering)" | the teacher's history-based lift rule was unlearnable from a single state |
| Floor movers at their end height, per-tag trigger lists, teleport lines | routes on E1M4 and E1M7 |

### Real-time budget

The teacher must decide in about 0 ms, otherwise lagged-teacher data is wrong. Perception had grown to 0.7 s on some decisions (A* every decision, NavGrid rebuilt on every live map change). Fixes: A* in a background thread with a 35-tic replan throttle, stage routes precomputed before the game, map geometry cached per map (3.2 s to 0.07 s per rebuild). Teacher games run 8-10 in parallel on 16 cores; more distorts the real-time clock.

The student: one CUDA graph per padded prompt length (256-576 tokens) took a decision from 114 to 87 ms with identical choices. Caching the fixed prompt prefix gave nothing; the 4B forward pass is compute-bound at ~400 tokens.

### The finding: tactics must survive the delay

| E1M8 teacher | Instant | 50 ms delay | 100 ms delay |
|---|---|---|---|
| Original | ~70% | 0% | — |
| Rebuilt | ~70% | — | ~45% |

The original teacher turned and fired the moment a Baron entered the crosshair and dodged a fireball when it came close. With 50 ms of delay both arrive late: the turn ends past the target, the dodge after the hit. The student, at ~90 ms, cannot learn from a teacher that only works at 0 ms. Changes that made the teacher work at 100 ms:

| Change | Mechanism |
|---|---|
| Shotgun readiness in the text: "Shotgun ready." / "reloading, ready in 0.6 s" | the shotgun fires once per 37 tics; the teacher fires only when ready and dodges during the reload |
| Aim lead 0.35 s | stop dodging and aim before the shotgun is ready, so the shot leaves on target |
| Baron distance band 330-520 units | close enough to hit with most pellets, far enough to sidestep fireballs |
| Escape scored by openness | the nearest gap was often a dead-end star tip |
| "Fireball incoming in 0.6 s, sidestep left" from projectile velocities | dodge before arrival, with time to spare at 100 ms |
| Line of fire | enemies behind other monsters or pillars read "out of the line of fire" |
| Chaingun-at-range rule removed | standing still to spray at range got the player killed |
| Damage dealt counts as progress | the stuck timer ended long boss fights |

### Student rounds on E1M8

| Round | Data | E1M8 clears |
|---|---|---|
| a | teacher and lagged-teacher games | 0/10, stuck at the lift |
| b | + DAgger round 1 | 0/10, reaches the Barons |
| c | + DAgger round 2, relabelled by the rebuilt teacher | 2/10 (seeds 1-10), 7/20 (seeds 51-70) |
| d | + DAgger round 3 | **8/20 on unseen seeds 151-170**, fastest 133.9 s, 94.4% agreement |

Each round trains from the previous adapter (`--init-adapter`), about 1 h on the 3090. Model d still clears E1M1 on 10/10 seeds.

At 94% agreement model d still loses 12 of 20 runs: agreement on teacher-visited states is a weak proxy, and clears are the measure.

### Open questions

- E1M3-E1M7 fights at 100 ms delay, then joint training on every map and a full campaign run by the student.
- Weapon choice is still a fixed rule; across eight maps shell economy matters, so switching should become a Jev command.
- Speed below ~85 ms needs a shorter prompt or a smaller model.
- Without the route compass the model would need memory; a separate experiment.
