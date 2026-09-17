"""Single-decision entrypoint shared by the harness CLI and the REST API.

Decision cascade, strongest available first, always inside the time budget, never illegal:
  known game + state   -> search agent (planner / MCTS, optionally LLM-guided hybrid)
  unknown game         -> LLM constrained choice over the provided legal actions
  anything failing     -> deterministic legal fallback
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

from academy_core import get_logger
from gamemaster.agents import default_agent_for, make_agent
from gamemaster.agents.llm_policy import SyncLLM
from gamemaster.games import get_game, list_games
from gamemaster.prompts import build_messages, parse_move

log = get_logger(__name__)


def _fallback(legal: list[str], seed_text: str) -> str:
    digest = int(hashlib.sha256(seed_text.encode()).hexdigest(), 16)
    return legal[digest % len(legal)]


def decide(payload: dict[str, Any], time_budget: float | None = None, agent_spec: str | None = None,
           llm: SyncLLM | None = None) -> dict[str, Any]:
    started = time.monotonic()
    budget = float(time_budget or payload.get("time_budget") or os.getenv("GAMEMASTER_MOVE_TIME", "5"))
    search_budget = max(0.05, budget * 0.85)
    game_name = str(payload.get("game") or payload.get("game_name") or "").lower()
    agent_spec = agent_spec or payload.get("agent") or os.getenv("GAMEMASTER_AGENT")

    if game_name in list_games() and payload.get("state") is not None:
        game = get_game(game_name)
        state = game.from_json(payload["state"])
        legal = game.legal_actions(state)
        if not legal:
            return {"action": None, "error": "terminal state", "source": "none"}
        spec = agent_spec or default_agent_for(game.num_players, game.deterministic)
        if not agent_spec and spec.startswith("mcts") and os.getenv("GAMEMASTER_HYBRID", "false").lower() == "true":
            spec = "hybrid:iterations=0"  # LLM priors + search; exact planners stay exact
        try:
            agent = make_agent(spec, seed=int(payload.get("seed", 0)))
            decision = agent.decide(game, state, search_budget)
            if decision.action in legal:
                return {"action": game.action_to_str(state, decision.action), "source": spec,
                        "policy": decision.policy, "value": decision.value,
                        "elapsed_s": round(time.monotonic() - started, 3)}
        except Exception as exc:  # noqa: BLE001
            log.warning("agent %s failed: %s", spec, exc)
        labels = [game.action_to_str(state, a) for a in legal]
        return {"action": _fallback(labels, game.render(state)), "source": "fallback"}

    # Unknown game: all we have is text, rules and legal actions.
    legal = [str(a) for a in payload.get("legal_actions") or payload.get("actions") or []]
    if not legal:
        return {"action": None, "error": "no legal actions provided", "source": "none"}
    observation = payload.get("observation") or payload.get("board") or payload.get("state") or ""
    obs_text = observation if isinstance(observation, str) else str(observation)
    llm = llm or SyncLLM(timeout=budget)
    if llm.available():
        messages = build_messages(game_name or "unknown game", str(payload.get("rules", "Rules are not given.")),
                                  obs_text, legal, payload.get("history"))
        try:
            data = llm.chat(messages, choices=legal, max_tokens=16, timeout=max(0.5, budget * 0.8))
            move = parse_move(data["choices"][0]["message"].get("content") or "", legal)
            if move is not None:
                return {"action": move, "source": "llm", "elapsed_s": round(time.monotonic() - started, 3)}
        except Exception as exc:  # noqa: BLE001
            log.warning("llm decision failed: %s", exc)
    return {"action": _fallback(legal, obs_text), "source": "fallback"}
