"""Generic Monte-Carlo Tree Search (UCT / PUCT).

* any number of players (value vectors, each node maximises its mover's value),
* stochastic games via open-loop search (the tree is indexed by action sequences and chance is
  re-sampled every simulation),
* optional ``prior_fn`` (policy priors, e.g. from a fine-tuned LLM -> AlphaZero-style PUCT) and
  ``value_fn`` (leaf evaluation instead of rollouts),
* adaptive min-max value normalisation for unbounded scores (e.g. 2048), as in MuZero,
* anytime: stops on iteration count or wall-clock budget, whichever comes first.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gamemaster.agents.base import Agent, Decision
from gamemaster.core import Action, Game

PriorFn = Callable[[Game, Any, list[Action]], dict[Action, float]]
ValueFn = Callable[[Game, Any], list[float]]


@dataclass
class Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: list[float] = field(default_factory=list)
    children: dict[Action, Node] = field(default_factory=dict)
    expanded: bool = False

    def q(self, player: int) -> float | None:
        return self.value_sum[player] / self.visits if self.visits and self.value_sum else None


class MinMax:
    def __init__(self, bounds: tuple[float, float] | None):
        self.fixed = bounds
        self.lo, self.hi = (bounds if bounds else (math.inf, -math.inf))

    def update(self, values: list[float]) -> None:
        if self.fixed:
            return
        for v in values:
            self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def norm(self, v: float) -> float:
        if self.hi > self.lo:
            return (v - self.lo) / (self.hi - self.lo)
        return 0.5


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(
        self,
        seed: int | None = None,
        iterations: int = 800,
        time_limit: float | None = None,
        c_puct: float = 1.4,
        rollout_depth: int = 40,
        rollout: str = "random",  # random | greedy | none (use evaluate at leaf)
        prior_fn: PriorFn | None = None,
        prior_depth: int = 1,
        value_fn: ValueFn | None = None,
        temperature: float = 0.0,
    ):
        super().__init__(seed)
        self.iterations = iterations  # <= 0 means "until the time budget runs out" (800 if there is none)
        self.time_limit = time_limit
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.rollout = rollout
        self.prior_fn = prior_fn
        self.prior_depth = prior_depth
        self.value_fn = value_fn
        self.temperature = temperature

    # ------------------------------------------------------------------ public
    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        legal = game.legal_actions(state)
        if len(legal) == 1:
            return Decision(legal[0], {game.action_to_str(state, legal[0]): 1.0}, None, {"simulations": 0})
        root, sims, minmax = self.search(game, state, time_budget)
        player = game.current_player(state)
        visits = {a: child.visits for a, child in root.children.items()}
        total = sum(visits.values()) or 1
        policy = {game.action_to_str(state, a): n / total for a, n in visits.items()}
        if self.temperature > 0:
            weights = [n ** (1 / self.temperature) for n in visits.values()]
            action = self.rng.choices(list(visits), weights=weights)[0]
        else:
            action = max(visits, key=lambda a: (visits[a], root.children[a].q(player) or 0.0, self.rng.random()))
        value = root.q(player)
        return Decision(action, policy, value, {"simulations": sims, "normalized_value": minmax.norm(value or 0.0)})

    def search(self, game: Game, state: Any, time_budget: float | None = None) -> tuple[Node, int, MinMax]:
        budget = time_budget if time_budget is not None else self.time_limit
        end = time.monotonic() + budget if budget is not None else None
        root = Node()
        minmax = MinMax(game.value_bounds())
        max_sims = self.iterations if self.iterations > 0 else (10**9 if end is not None else 800)
        sims = 0
        while sims < max_sims:
            if end is not None and sims > 0 and time.monotonic() >= end:
                break
            self._simulate(game, state, root, minmax)
            sims += 1
        return root, sims, minmax

    # ------------------------------------------------------------------ internals
    def _expand(self, game: Game, state: Any, node: Node, depth: int) -> None:
        legal = game.legal_actions(state)
        priors: dict[Action, float] = {}
        if self.prior_fn is not None and depth < self.prior_depth:
            try:
                priors = self.prior_fn(game, state, legal)
            except Exception:  # noqa: BLE001 - a failing policy model degrades to uniform priors
                priors = {}
        total = sum(priors.get(a, 0.0) for a in legal)
        for a in legal:
            p = priors.get(a, 0.0) / total if total > 0 else 1.0 / len(legal)
            node.children[a] = Node(prior=0.75 * p + 0.25 / len(legal) if priors else p)
        node.expanded = True

    def _select(self, node: Node, player: int, minmax: MinMax, legal: list[Action]) -> Action:
        sqrt_n = math.sqrt(max(1, node.visits))
        best, best_score = None, -math.inf
        for action in legal:
            child = node.children[action]
            q = child.q(player)
            q_norm = minmax.norm(q) if q is not None else 0.5  # first-play urgency: neutral
            u = self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            score = q_norm + u + 1e-9 * self.rng.random()
            if score > best_score:
                best, best_score = action, score
        return best  # type: ignore[return-value]

    def _leaf_value(self, game: Game, state: Any) -> list[float]:
        if game.is_terminal(state):
            return game.returns(state)
        if self.value_fn is not None:
            return self.value_fn(game, state)
        if self.rollout == "none":
            return game.evaluate(state)
        s = state
        for _ in range(self.rollout_depth):
            if game.is_terminal(s):
                return game.returns(s)
            legal = game.legal_actions(s)
            if self.rollout == "greedy" and len(legal) > 1:
                p = game.current_player(s)
                a = max(legal, key=lambda x: game.evaluate(game.apply(s, x, self.rng))[p] + 1e-6 * self.rng.random())
            else:
                a = self.rng.choice(legal)
            s = game.apply(s, a, self.rng)
        return game.evaluate(s)

    def _simulate(self, game: Game, root_state: Any, root: Node, minmax: MinMax) -> None:
        path = [root]
        movers: list[int] = []
        state = root_state
        node = root
        depth = 0
        while not game.is_terminal(state):
            if not node.expanded:
                self._expand(game, state, node, depth)
                break
            player = game.current_player(state)
            legal = game.legal_actions(state)
            if not game.deterministic:
                # open loop: a different chance outcome can change which moves are legal here
                for a in legal:
                    if a not in node.children:
                        node.children[a] = Node(prior=1.0 / len(legal))
            action = self._select(node, player, minmax, legal)
            state = game.apply(state, action, self.rng)
            node = node.children[action]
            movers.append(player)
            path.append(node)
            depth += 1
        values = self._leaf_value(game, state)
        minmax.update(values)
        for n in path:
            n.visits += 1
            if not n.value_sum:
                n.value_sum = [0.0] * len(values)
            for i, v in enumerate(values):
                n.value_sum[i] += v


def make_mcts(**kwargs: Any) -> MCTSAgent:
    return MCTSAgent(**kwargs)


__all__ = ["MCTSAgent", "Node", "PriorFn", "ValueFn", "make_mcts"]
