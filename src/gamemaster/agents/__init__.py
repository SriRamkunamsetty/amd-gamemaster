"""Agent registry and spec parsing: ``"mcts:iterations=400,c_puct=1.2"``."""

from __future__ import annotations

from typing import Any

from gamemaster.agents.base import Agent, Decision, GreedyAgent, RandomAgent
from gamemaster.agents.explorer import ExplorerAgent
from gamemaster.agents.llm_policy import HybridAgent, LLMPolicyAgent, SyncLLM
from gamemaster.agents.mcts import MCTSAgent
from gamemaster.agents.planner import PlannerAgent

AGENTS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "mcts": MCTSAgent,
    "planner": PlannerAgent,
    "llm": LLMPolicyAgent,
    "hybrid": HybridAgent,
}


def _coerce(value: str) -> Any:
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() == "none":
        return None
    return value


def make_agent(spec: str, seed: int | None = None) -> Agent:
    name, _, params = spec.partition(":")
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; available: {sorted(AGENTS)}")
    kwargs = {k.strip(): _coerce(v.strip()) for k, v in (p.split("=", 1) for p in params.split(",") if "=" in p)}
    return AGENTS[name](seed=seed, **kwargs)


def default_agent_for(game_num_players: int, deterministic: bool) -> str:
    """Sensible strongest-available agent when nothing is specified."""
    if game_num_players == 1 and deterministic:
        return "planner"
    if game_num_players == 1:
        return "mcts:iterations=0,rollout=none"  # stochastic puzzles: heuristic leaves beat random rollouts
    return "mcts:iterations=0"


__all__ = ["AGENTS", "Agent", "Decision", "ExplorerAgent", "GreedyAgent", "HybridAgent", "LLMPolicyAgent",
           "MCTSAgent", "PlannerAgent", "RandomAgent", "SyncLLM", "default_agent_for", "make_agent"]
