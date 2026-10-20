"""Lean local Jev: compact classifier prompt and a one-forward-pass scorer.

Same idea as Simple Jev (a language model picks one answer letter; the answer distribution is the
softmax over the option letters' next-token logits), with a shorter prompt: options are listed
once, before the state, so the request is ~350 tokens instead of ~1,130. The Simple Jev v1
template lists the 20 options twice after the state, which costs ~730 tokens per decision.
Measured on the RTX 3090 (2026-10-18): Qwen3.5-4B forward 225 ms at 1,168 tokens, 89 ms at 450.

Training (train_fast.py) and serving (fast_server.py) both build prompts here, so they match.
No game imports.
"""
from __future__ import annotations

import string

from .commands import COMMAND_IDS

# A-Z, then a-z: 29 commands since the weapon commands (2026-10-20). " a".." z" are single tokens in Qwen3.5
# (digits are not: " 1" is two tokens). Upper-case letters keep their meaning, so old adapters still score.
LETTERS = string.ascii_uppercase + string.ascii_lowercase
GRAPH_BUCKETS = (256, 320, 384, 448, 512, 576)


def prompt_text(state: str, instructions: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}: {o}" for i, o in enumerate(options))
    return (f"<|im_start|>system\n{instructions} Answer with one option letter.\nOptions:\n{opts}<|im_end|>\n"
            f"<|im_start|>user\n{state}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:")


def request_parts(request: dict) -> tuple[str, str, list[str]]:
    """(state, instructions, option ids) from a /v1/classifier request with one choice question."""
    (q,) = request["questions"].values()
    return request["state"], q.get("instructions") or "Pick the best option.", list(q["criteria"])


def label_token_ids(tokenizer, n: int) -> list[int]:
    ids = []
    for ch in LETTERS[:n]:
        t = tokenizer.encode(" " + ch, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"option letter {ch!r} is not a single token: {t}")
        ids.append(t[0])
    return ids


class Scorer:
    """Loads a base model (optionally a LoRA adapter) and scores one request per forward pass."""

    def __init__(self, model: str, adapter: str | None = None, device: str = "cuda", dtype: str = "bfloat16",
                 graphs: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or model)
        m = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype))
        if adapter:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(device).eval()
        self.device = device
        self._labels: dict[int, list[int]] = {}
        self._graphs: dict[int, tuple] = {}
        if graphs and device == "cuda":
            self._capture(GRAPH_BUCKETS)

    def _capture(self, buckets) -> None:
        """One CUDA graph per padded prompt length: replaying a recorded forward skips ~2,000 kernel
        launches from Python (4B on the 3090: 100 -> 86 ms, same choices). Prompts are right-padded;
        a causal model's positions before the padding are unchanged, and the logits are read at the
        last real token."""
        torch, body = self.torch, self.model.model
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        with torch.inference_mode():
            for n in buckets:
                ids = torch.full((1, n), pad, dtype=torch.long, device=self.device)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        body(input_ids=ids, use_cache=False)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    hidden = body(input_ids=ids, use_cache=False).last_hidden_state
                self._graphs[n] = (ids, g, hidden, pad)

    def _last_logits(self, ids):
        n = ids.shape[1]
        fit = [b for b in self._graphs if b >= n]
        if not fit:
            return self.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
        static, g, hidden, pad = self._graphs[min(fit)]
        static.fill_(pad)
        static[:, :n].copy_(ids)
        g.replay()
        return self.model.lm_head(hidden[0, n - 1])

    def score(self, request: dict) -> dict:
        torch = self.torch
        state, instr, options = request_parts(request)
        if len(options) not in self._labels:
            self._labels[len(options)] = label_token_ids(self.tok, len(options))
        ids = self.tok(prompt_text(state, instr, options), return_tensors="pt",
                       add_special_tokens=False).input_ids.to(self.device)
        with torch.inference_mode():
            logits = self._last_logits(ids)
        p = logits[self._labels[len(options)]].float().softmax(-1).tolist()
        probs = dict(zip(options, p))
        choice = max(probs, key=probs.get)
        return {"choice": choice, "confidence": probs[choice], "probabilities": probs,
                "input_tokens": int(ids.shape[1])}


assert len(COMMAND_IDS) <= len(LETTERS)
