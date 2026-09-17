"""Exact planning for deterministic single-player games (BFS over canonical states)."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from gamemaster.agents.base import Agent, Decision, GreedyAgent
from gamemaster.core import Action, Game


class PlannerAgent(Agent):
    """Breadth-first search to the best reachable terminal return, with plan caching.

    Falls back to greedy one-ply lookahead when the node or time budget runs out.
    """

    name = "planner"

    def __init__(self, seed: int | None = None, max_nodes: int = 200_000, time_limit: float = 5.0):
        super().__init__(seed)
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self._plan: list[Action] = []
        self._expected_key: Any = None
        self._fallback = GreedyAgent(seed)

    def reset(self) -> None:
        self._plan, self._expected_key = [], None

    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        if game.num_players != 1 or not game.deterministic:
            return self._fallback.decide(game, state, time_budget)
        key = game.state_key(state)
        if self._plan and key == self._expected_key:
            action = self._plan.pop(0)
            self._expected_key = game.state_key(game.apply(state, action)) if self._plan else None
            return Decision(action, {game.action_to_str(state, action): 1.0}, None, {"cached": True})
        plan = self.plan(game, state, time_budget)
        if not plan:
            return self._fallback.decide(game, state, time_budget)
        action = plan[0]
        self._plan = plan[1:]
        self._expected_key = game.state_key(game.apply(state, action)) if self._plan else None
        return Decision(action, {game.action_to_str(state, action): 1.0}, None, {"plan_length": len(plan)})

    def plan(self, game: Game, state: Any, time_budget: float | None = None) -> list[Action] | None:
        end = time.monotonic() + (time_budget if time_budget is not None else self.time_limit)
        start_key = game.state_key(state)
        parents: dict[Any, tuple[Any, Action] | None] = {start_key: None}
        queue = deque([state])
        best_key, best_value = None, -float("inf")
        expanded = 0
        while queue and expanded < self.max_nodes:
            if expanded % 512 == 0 and time.monotonic() > end:
                break
            s = queue.popleft()
            expanded += 1
            for a in game.legal_actions(s):
                nxt = game.apply(s, a)
                k = game.state_key(nxt)
                if k in parents:
                    continue
                parents[k] = (game.state_key(s), a)
                if game.is_terminal(nxt):
                    value = game.returns(nxt)[0]
                    if value > best_value:
                        best_key, best_value = k, value
                else:
                    queue.append(nxt)
            # BFS reaches the shortest win first; a positive terminal return found at this depth is optimal
            # for step-penalised games, so stop once the frontier is deeper than the best solution.
            if best_key is not None and best_value > 0:
                break
        if best_key is None or best_value <= 0:
            return None
        actions: list[Action] = []
        k = best_key
        while parents[k] is not None:
            prev, a = parents[k]  # type: ignore[misc]
            actions.append(a)
            k = prev
        return list(reversed(actions))
