"""Simple Jev classifier client and request builders.

Teacher format: full command descriptions (hosted 35B, used for labelling).
Student format: bare command ids (local fine-tuned 0.8B, shorter prompt = faster).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .commands import COMMANDS

HOSTED_URL = "https://simple-jev-demo-api.featherless.ai"
# Teacher chosen with probe.py (2026-10-17): gemma-4-26B-A4B scored 0.90 vs 0.50 for
# Qwen3.6-35B-A3B, which ignores enemies in view and follows the route compass instead.
TEACHER_MODEL = "featherless-ai/gemma-4-26B-A4B-classifier"
LOCAL_URL = "http://127.0.0.1:8012"

INSTRUCTIONS = ('You control the DOOM player in real time. Goal: survive and reach the level exit. Game facts: '
                'angles in the state are measured from where you face; "N deg left" means turn left by about N degrees '
                'to face it. Enemies in view shoot you. fire only hits what is in the crosshair. use only works on a '
                'door or switch within use reach straight ahead. Nukage floor drains health. '
                'Pick the single best next command.')
STUDENT_INSTRUCTIONS = "DOOM player. Next command?"


def teacher_request(state: str, model: str = TEACHER_MODEL) -> dict:
    return {"model": model, "state": state,
            "questions": {"cmd": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(COMMANDS)}}}


def student_request(state: str, model: str = "student", full_prompt: bool = False) -> dict:
    if full_prompt:
        return teacher_request(state, model)
    return {"model": model, "state": state,
            "questions": {"cmd": {"type": "choice", "instructions": STUDENT_INSTRUCTIONS,
                                  "criteria": {k: None for k in COMMANDS}}}}


class JevClient:
    def __init__(self, base_url: str, timeout: float = 20.0):
        self.url = base_url.rstrip("/") + "/v1/classifier"
        self.timeout = timeout

    def classify(self, body: dict) -> tuple[dict, float]:
        """Returns (answer for question 'cmd', latency seconds). Raises on HTTP errors."""
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "User-Agent": "jev-doom/0.1"})
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        dt = time.perf_counter() - t
        ans = out["answers"]["cmd"]
        ans["input_tokens"] = out.get("usage", {}).get("input_tokens")
        return ans, dt

    def classify_retry(self, body: dict, attempts: int = 4) -> tuple[dict, float]:
        for k in range(attempts):
            try:
                return self.classify(body)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as e:
                code = getattr(e, "code", None)
                if k == attempts - 1 or (code is not None and code < 500 and code != 429):
                    raise
                time.sleep(5 * (k + 1))
        raise RuntimeError("unreachable")
