"""LLM policy on an OpenAI-compatible server (vLLM on ROCm in production).

* ``choose``: legal-move-constrained decoding (vLLM structured choice), then strict re-validation;
  the agent can never emit an illegal move.
* ``distribution``: a policy prior over legal moves from first-token log-probabilities, used as
  PUCT priors by the hybrid agent.
All calls are synchronous (search code is synchronous) and fail soft.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import httpx

from gamemaster.agents.base import Agent, Decision, GreedyAgent
from gamemaster.core import Action, Game
from gamemaster.prompts import messages_for_state, parse_move


class SyncLLM:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 timeout: float = 10.0, guided_style: str | None = None):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or None
        self.guided_style = guided_style or os.getenv("LLM_GUIDED_STYLE", "structured_outputs")
        self.client = httpx.Client(timeout=timeout,
                                   headers={"Authorization": f"Bearer {api_key or os.getenv('LLM_API_KEY', 'EMPTY')}"})
        self._healthy: bool | None = None
        self._failures = 0

    def available(self) -> bool:
        if os.getenv("LLM_ENABLED", "true").lower() in {"0", "false", "no"}:
            return False
        if self._healthy is None:
            try:
                data = self.client.get(f"{self.base_url}/models", timeout=2).json()
                self.model = self.model or data["data"][0]["id"]
                self._healthy = True
            except Exception:  # noqa: BLE001
                self._healthy = False
        return bool(self._healthy) and self._failures < 5  # circuit breaker

    def chat(self, messages: list[dict[str, str]], *, choices: list[str] | None = None, max_tokens: int = 16,
             logprobs: int | None = None, timeout: float | None = None, temperature: float = 0.0) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature}
        if choices and self.guided_style == "structured_outputs":
            body["structured_outputs"] = {"choice": choices}
        elif choices and self.guided_style == "guided":
            body["guided_choice"] = choices
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = logprobs
        if self.guided_style != "none":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = self.client.post(f"{self.base_url}/chat/completions", json=body, timeout=timeout)
            resp.raise_for_status()
            self._failures = 0
            return resp.json()
        except Exception:
            self._failures += 1
            raise


class LLMPolicyAgent(Agent):
    name = "llm"

    def __init__(self, seed: int | None = None, llm: SyncLLM | None = None, fallback: Agent | None = None,
                 history: int = 8):
        super().__init__(seed)
        self.llm = llm or SyncLLM()
        self.fallback = fallback or GreedyAgent(seed)
        self.history: list[str] = []
        self.history_len = history
        self.stats = {"llm_moves": 0, "fallback_moves": 0}

    def reset(self) -> None:
        self.history = []

    def choose(self, game: Game, state: Any, timeout: float | None = None) -> Action | None:
        if not self.llm.available():
            return None
        messages, legal = messages_for_state(game, state, self.history[-self.history_len:])
        try:
            data = self.llm.chat(messages, choices=legal, max_tokens=12, timeout=timeout)
        except Exception:  # noqa: BLE001
            return None
        move = parse_move(data["choices"][0]["message"].get("content") or "", legal)
        return None if move is None else game.str_to_action(state, move)

    def distribution(self, game: Game, state: Any, legal: list[Action], timeout: float | None = None) -> dict[Action, float]:
        """Prior over legal actions from first-token logprobs (requires distinct first tokens)."""
        if not self.llm.available():
            return {}
        messages, labels = messages_for_state(game, state, self.history[-self.history_len:])
        try:
            data = self.llm.chat(messages, max_tokens=1, logprobs=20, timeout=timeout)
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except Exception:  # noqa: BLE001
            return {}
        by_prefix: dict[str, float] = {}
        for item in top:
            tok = item["token"].strip().lower()
            if tok:
                by_prefix[tok] = max(by_prefix.get(tok, -math.inf), item["logprob"])
        out: dict[Action, float] = {}
        for action, label in zip(legal, [game.action_to_str(state, a) for a in legal], strict=True):
            label = label.lower()
            matches = [lp for tok, lp in by_prefix.items() if label.startswith(tok)]
            if matches:
                out[action] = math.exp(max(matches))
        return out if len(out) >= 1 else {}

    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        started = time.monotonic()
        action = self.choose(game, state, timeout=time_budget)
        if action is None:
            self.stats["fallback_moves"] += 1
            remaining = None if time_budget is None else max(0.05, time_budget - (time.monotonic() - started))
            decision = self.fallback.decide(game, state, remaining)
            decision.info["fallback"] = True
        else:
            self.stats["llm_moves"] += 1
            decision = Decision(action, {game.action_to_str(state, action): 1.0}, None, {"source": "llm"})
        self.history.append(game.action_to_str(state, decision.action))
        return decision


class HybridAgent(Agent):
    """PUCT search guided by LLM priors: the fine-tuned model proposes, search verifies."""

    name = "hybrid"

    def __init__(self, seed: int | None = None, llm: SyncLLM | None = None, iterations: int = 400,
                 time_limit: float | None = None, c_puct: float = 1.5, rollout: str = "random"):
        super().__init__(seed)
        from gamemaster.agents.mcts import MCTSAgent

        self.policy = LLMPolicyAgent(seed, llm)
        self._cache: dict[Any, dict[Action, float]] = {}

        def prior_fn(game: Game, state: Any, legal: list[Action]) -> dict[Action, float]:
            key = game.state_key(state)
            if key not in self._cache:
                self._cache[key] = self.policy.distribution(game, state, legal, timeout=3.0)
            return self._cache[key]

        self.search = MCTSAgent(seed, iterations=iterations, time_limit=time_limit, c_puct=c_puct,
                                rollout=rollout, prior_fn=prior_fn, prior_depth=1)

    def reset(self) -> None:
        self._cache.clear()
        self.policy.reset()

    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        decision = self.search.decide(game, state, time_budget)
        self.policy.history.append(game.action_to_str(state, decision.action))
        return decision
