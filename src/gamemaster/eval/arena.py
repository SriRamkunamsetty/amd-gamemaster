"""Arena: play episodes, report win rate / score / latency / illegal moves.

    gamemaster arena --game connect4 --agents mcts:iterations=400 random --episodes 20
    gamemaster arena --game 2048 --agents greedy --episodes 5
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from gamemaster.agents import Agent, make_agent
from gamemaster.core import Game


@dataclass
class EpisodeResult:
    seed: int
    returns: list[float]
    steps: int
    seat_agents: list[str]
    move_times: list[float] = field(default_factory=list)
    illegal: int = 0


def play_episode(game: Game, agents: list[Agent], seed: int, move_time: float | None = None,
                 labels: list[str] | None = None) -> EpisodeResult:
    rng = random.Random(seed)
    state = game.initial_state(seed)
    for agent in agents:
        agent.reset()
    result = EpisodeResult(seed, [], 0, labels or [a.name for a in agents])
    while not game.is_terminal(state) and result.steps < game.max_steps:
        player = game.current_player(state)
        legal = game.legal_actions(state)
        started = time.monotonic()
        try:
            action = agents[player].act(game, state, move_time)
        except Exception:  # noqa: BLE001 - count as illegal, fall back to random
            action = None
        result.move_times.append(time.monotonic() - started)
        if action not in legal:
            result.illegal += 1
            action = rng.choice(legal)
        state = game.apply(state, action, rng)
        result.steps += 1
    result.returns = game.returns(state)
    return result


def evaluate(game: Game, specs: list[str], episodes: int = 10, move_time: float | None = None,
             seed: int = 0) -> dict[str, Any]:
    """For multi-player games seats rotate each episode, so every agent plays every seat."""
    n = game.num_players
    if len(specs) == 1:
        specs = specs * n
    if len(specs) != n:
        raise ValueError(f"{game.name} needs {n} agents, got {len(specs)}")
    per_agent: dict[str, dict[str, list[float]]] = {s: {"returns": [], "wins": [], "move_s": []} for s in set(specs)}
    illegal = 0
    for ep in range(episodes):
        shift = ep % n
        seats = specs[shift:] + specs[:shift]
        agents = [make_agent(spec, seed=seed + ep * 31 + i) for i, spec in enumerate(seats)]
        res = play_episode(game, agents, seed + ep, move_time, seats)
        illegal += res.illegal
        best = max(res.returns)
        for i, spec in enumerate(seats):
            per_agent[spec]["returns"].append(res.returns[i])
            per_agent[spec]["wins"].append(1.0 if n > 1 and res.returns[i] == best and res.returns.count(best) == 1 else 0.0)
        for t, i in zip(res.move_times, _movers(n, res.steps), strict=False):
            per_agent[seats[i]]["move_s"].append(t)
    report: dict[str, Any] = {"game": game.name, "episodes": episodes, "illegal_moves": illegal, "agents": {}}
    for spec, d in per_agent.items():
        report["agents"][spec] = {
            "mean_return": round(statistics.fmean(d["returns"]), 4) if d["returns"] else None,
            "win_rate": round(statistics.fmean(d["wins"]), 3) if n > 1 and d["wins"] else None,
            "mean_move_s": round(statistics.fmean(d["move_s"]), 4) if d["move_s"] else None,
            "max_move_s": round(max(d["move_s"]), 4) if d["move_s"] else None,
        }
    return report


def _movers(n: int, steps: int) -> list[int]:
    return [i % n for i in range(steps)]
