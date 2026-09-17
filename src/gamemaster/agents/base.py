from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from gamemaster.core import Action, Game


@dataclass
class Decision:
    action: Action
    policy: dict[str, float] = field(default_factory=dict)  # action label -> probability (for training data)
    value: float | None = None  # value estimate for the player to move
    info: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    name = "agent"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    @abstractmethod
    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision: ...

    def act(self, game: Game, state: Any, time_budget: float | None = None) -> Action:
        return self.decide(game, state, time_budget).action

    def reset(self) -> None:
        """Called at the start of every episode."""


class RandomAgent(Agent):
    name = "random"

    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        legal = game.legal_actions(state)
        return Decision(self.rng.choice(legal), {game.action_to_str(state, a): 1 / len(legal) for a in legal})


class GreedyAgent(Agent):
    """One-ply lookahead on ``game.evaluate`` (chance averaged over a few samples)."""

    name = "greedy"

    def __init__(self, seed: int | None = None, samples: int = 4):
        super().__init__(seed)
        self.samples = samples

    def decide(self, game: Game, state: Any, time_budget: float | None = None) -> Decision:
        player = game.current_player(state)
        legal = game.legal_actions(state)
        samples = 1 if game.deterministic else self.samples
        scored = []
        for a in legal:
            total = 0.0
            for _ in range(samples):
                nxt = game.apply(state, a, self.rng)
                total += game.evaluate(nxt)[player]
            scored.append((total / samples, self.rng.random(), a))
        best = max(scored)
        return Decision(best[2], {game.action_to_str(state, best[2]): 1.0}, best[0])


class Deadline:
    def __init__(self, seconds: float | None):
        self.end = None if seconds is None else time.monotonic() + seconds

    def expired(self) -> bool:
        return self.end is not None and time.monotonic() >= self.end
