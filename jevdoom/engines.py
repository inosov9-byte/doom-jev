"""Decision engines for the Jev vs Haiku experiment. Every engine maps state text to one command.

All engines see the same state text and the same 13 commands. Jev engines read label
probabilities from a Simple Jev classifier endpoint. Haiku engines call Claude through the
Anthropic SDK with structured outputs, so the reply can only be one of the 13 command ids.
See EXPERIMENT.md for the arms.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .commands import COMMAND_IDS, COMMANDS
from .jev import HOSTED_URL, INSTRUCTIONS, LOCAL_URL, JevClient, student_request, teacher_request

# Anthropic list prices, USD per million tokens (claude-api skill table, cached 2026-10-06)
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}


ENV_FILE = Path(os.environ.get("JEVDOOM_ENV", r"C:	mp\jev-doom\.env"))   # outside the repo and OneDrive


def load_credentials() -> None:
    """Copy ANTHROPIC_* lines from ENV_FILE into the environment if not already set. Never prints values."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.startswith("ANTHROPIC_") and not os.environ.get(k):
            os.environ[k] = v.strip().strip('"')


@dataclass
class Decision:
    choice: str
    probabilities: dict[str, float] | None
    latency: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Engine:
    name = "engine"
    min_gap = 0.0          # seconds between request starts

    def decide(self, state: str) -> Decision:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name}


class JevEngine(Engine):
    def __init__(self, name: str, url: str, model: str, student_prompt: bool = False, min_gap: float = 0.0):
        self.name, self.url, self.model, self.student_prompt, self.min_gap = name, url, model, student_prompt, min_gap
        self.client = JevClient(url, timeout=20)

    def decide(self, state: str) -> Decision:
        body = student_request(state) if self.student_prompt else teacher_request(state, self.model)
        ans, dt = self.client.classify(body)
        return Decision(ans["choice"], ans["probabilities"], dt, ans.get("input_tokens") or 0, 0, 0.0)

    def describe(self) -> dict:
        return {"name": self.name, "engine": "jev", "url": self.url, "model": self.model,
                "prompt": "student" if self.student_prompt else "shared"}


class HaikuEngine(Engine):
    """Claude via the Anthropic SDK. Same instructions and command descriptions as the Jev teacher prompt."""

    def __init__(self, name: str, model: str = "claude-haiku-5-5", thinking: bool = False, effort: str = "low"):
        import anthropic                     # imported here so Jev-only runs need no SDK
        load_credentials()
        self.name, self.model, self.thinking, self.effort = name, model, thinking, effort
        self.client = anthropic.Anthropic(max_retries=2, timeout=20.0)
        options = "\n".join(f"- {k}" + (f": {v}" if v else "") for k, v in COMMANDS.items())
        self.system = f"{INSTRUCTIONS}\n\nCommands:\n{options}"
        self.schema = {"type": "object", "properties": {"command": {"type": "string", "enum": COMMAND_IDS}},
                       "required": ["command"], "additionalProperties": False}

    def decide(self, state: str) -> Decision:
        kwargs = dict(model=self.model, system=self.system,
                      messages=[{"role": "user", "content": f"State: {state}"}],
                      output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": self.schema}})
        if self.thinking:
            kwargs.update(max_tokens=2048, thinking={"type": "adaptive"})
        else:
            kwargs.update(max_tokens=64, thinking={"type": "disabled"})
        t = time.perf_counter()
        r = self.client.messages.create(**kwargs)
        dt = time.perf_counter() - t
        if r.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {getattr(r, 'stop_details', None)}")
        text = next(b.text for b in r.content if b.type == "text")
        choice = json.loads(text)["command"]
        pin, pout = PRICES[self.model]
        cost = (r.usage.input_tokens * pin + r.usage.output_tokens * pout) / 1e6
        return Decision(choice, None, dt, r.usage.input_tokens, r.usage.output_tokens, cost)

    def describe(self) -> dict:
        return {"name": self.name, "engine": "claude", "model": self.model, "thinking": self.thinking,
                "effort": self.effort, "prompt": "shared"}


class MockEngine(Engine):
    """Free stand-in with a latency profile, for testing the harness (--dry-run)."""

    def __init__(self, name: str, latency: float, min_gap: float = 0.0, seed: int = 0):
        self.name, self.latency, self.min_gap = name, latency, min_gap
        self.rng = random.Random(seed)

    def decide(self, state: str) -> Decision:
        dt = max(0.05, self.rng.gauss(self.latency, self.latency * 0.15))
        time.sleep(dt)
        weights = [5 if c == "forward" else 2 if c.startswith("turn_") else 1 for c in COMMAND_IDS]
        c = self.rng.choices(COMMAND_IDS, weights)[0]
        p = {k: (0.6 if k == c else 0.4 / (len(COMMAND_IDS) - 1)) for k in COMMAND_IDS}
        return Decision(c, p, dt)

    def describe(self) -> dict:
        return {"name": self.name, "engine": "mock", "latency": self.latency}


QWEN_HOSTED = "featherless-ai/Qwen3.6-35B-A3B-classifier"
GEMMA_HOSTED = "featherless-ai/gemma-4-26B-A4B-classifier"
# Typical latencies measured on this setup, used only by --dry-run mocks.
MOCK_LATENCY = {"jev-local": 0.7, "jev-hosted": 0.85, "jev-hosted-gemma": 1.35, "haiku": 0.8, "haiku-think": 2.0}


def make_engine(arm: str, local_url: str = LOCAL_URL, dry_run: bool = False, seed: int = 0) -> Engine:
    hosted_gap = 0.55                       # Featherless demo: at most 2 requests/s
    if dry_run:
        return MockEngine(arm, MOCK_LATENCY.get(arm, 0.8), hosted_gap if arm.startswith("jev-hosted") else 0.0, seed)
    if arm == "jev-local":
        return JevEngine(arm, local_url, "student", student_prompt=True)
    if arm == "jev-hosted":
        return JevEngine(arm, HOSTED_URL, QWEN_HOSTED, min_gap=hosted_gap)
    if arm == "jev-hosted-gemma":
        return JevEngine(arm, HOSTED_URL, GEMMA_HOSTED, min_gap=hosted_gap)
    if arm == "haiku":
        return HaikuEngine(arm, thinking=False, effort="low")
    if arm == "haiku-think":
        return HaikuEngine(arm, thinking=True, effort="low")
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


ARMS = ["jev-local", "jev-hosted", "jev-hosted-gemma", "haiku", "haiku-think"]
