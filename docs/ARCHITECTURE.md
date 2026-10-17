# What it took: Jev plays DOOM

How a text classifier ended up playing DOOM E1M1 in real time on a laptop, what the system looks like, and what broke on the way. Numbers are measured on a Ryzen AI 9 HX 375 laptop (Radeon 890M iGPU, 32 GB) unless stated.

## The starting point

A Featherless demo shows Jev, their classifier API, "playing" DOOM in a browser. Reading the demo's source changes the picture: a scripted route follower handles navigation, doors and the exit, and Jev only picks combat actions. Its README says level completion is not guaranteed.

The goal here was stricter: Jev picks every command, and runs locally.

## What Jev is

Jev is not a model. It is a way of using one: the model reads a prompt that lists labelled options, and instead of generating text, Jev reads the probabilities of the label tokens in one forward pass. Any open model can sit behind it. Featherless hosts some (Qwen3.6-35B, Gemma-4-26B, ...); the same server code runs locally.

Consequences that shaped everything:

- Jev cannot output a number. Every amount (how far to turn) has to be a label.
- Jev reads text. Something has to turn the game into text.
- Cost per decision is one prompt pass. Prompt length is latency.

## System

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

## The command set, and why it is 20

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

## Running locally on an AMD laptop

- AMD's ROCm 7.2.1 PyTorch for Windows supports the Radeon 890M (gfx1150). 0.8B: 0.61 s per call vs 4.3 s on CPU.
- RFDT training needed two workarounds: a placeholder for `torch.distributed.tensor` (missing in the Windows ROCm build), and copying the image processor configs that RFDT's export leaves out.
- Training speed: about 0.35 examples/s, about 2.5 GiB of GPU memory. Most time goes to Qwen3.5's linear-attention layers running on the slow reference path.
- Simple Jev's prompt template is about 620 tokens of fixed instructions, examples and the option list printed twice; the game state is about 150. Each extra command label costs prompt length on every decision.

## Results

Filled in from `runs/compare-*/results.md` after the live comparison.
