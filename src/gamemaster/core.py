"""Game abstractions.

Two views of a game, because the challenge game may arrive either way:

* ``Game``: a *simulator* we can clone and search (the rules are known or were re-implemented).
  States are immutable and hashable, so tree search can branch freely. Chance events are drawn
  from the ``rng`` passed to ``apply``.
* ``Env``: a *black box* we can only reset and step (e.g. an HTTP game server with hidden rules).
  Explorer and LLM agents work here; ``GameEnv`` adapts any ``Game`` to this view.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

S = TypeVar("S", bound=Hashable)
Action = Hashable


class Game(ABC, Generic[S]):
    name: str = "game"
    num_players: int = 1
    deterministic: bool = True
    max_steps: int = 500
    rules: str = ""  # natural-language rules, used in LLM prompts

    @abstractmethod
    def initial_state(self, seed: int | None = None) -> S: ...

    @abstractmethod
    def current_player(self, state: S) -> int:
        """Index of the player to move; -1 when terminal."""

    @abstractmethod
    def legal_actions(self, state: S) -> list[Action]: ...

    @abstractmethod
    def apply(self, state: S, action: Action, rng: random.Random | None = None) -> S: ...

    @abstractmethod
    def is_terminal(self, state: S) -> bool: ...

    @abstractmethod
    def returns(self, state: S) -> list[float]:
        """Per-player return so far (final return at terminal states)."""

    @abstractmethod
    def render(self, state: S) -> str: ...

    @abstractmethod
    def to_json(self, state: S) -> dict[str, Any]: ...

    @abstractmethod
    def from_json(self, data: dict[str, Any]) -> S: ...

    # ---- optional hooks with sensible defaults -------------------------------------------
    def action_to_str(self, state: S, action: Action) -> str:
        return str(action)

    def str_to_action(self, state: S, text: str) -> Action:
        text = text.strip().lower()
        for action in self.legal_actions(state):
            if self.action_to_str(state, action).lower() == text:
                return action
        raise ValueError(f"illegal or unknown action {text!r}")

    def evaluate(self, state: S) -> list[float]:
        """Value estimate for non-terminal cut-offs; defaults to returns so far."""
        return self.returns(state)

    def value_bounds(self) -> tuple[float, float] | None:
        """Known return range (e.g. (-1, 1)); None lets search normalise adaptively."""
        return None

    def state_key(self, state: S) -> Hashable:
        return state


@dataclass
class Observation:
    text: str
    legal_actions: list[str]
    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    key: Hashable | None = None  # a hashable identity of the observation, used for exploration graphs


class Env(Protocol):
    rules: str

    def reset(self, seed: int | None = None) -> Observation: ...

    def step(self, action: str) -> Observation: ...


class GameEnv:
    """Adapts a single-agent (or self-play) ``Game`` to the black-box ``Env`` interface."""

    def __init__(self, game: Game, seed: int | None = None):
        self.game = game
        self.rules = game.rules
        self.rng = random.Random(seed)
        self.state = game.initial_state(seed)
        self._last_return = 0.0

    def _obs(self, reward: float) -> Observation:
        g, s = self.game, self.state
        legal = [] if g.is_terminal(s) else [g.action_to_str(s, a) for a in g.legal_actions(s)]
        return Observation(g.render(s), legal, reward, g.is_terminal(s), {"returns": g.returns(s)}, g.state_key(s))

    def reset(self, seed: int | None = None) -> Observation:
        self.rng = random.Random(seed)
        self.state = self.game.initial_state(seed)
        self._last_return = self.game.returns(self.state)[0]
        return self._obs(0.0)

    def step(self, action: str) -> Observation:
        act = self.game.str_to_action(self.state, action)
        player = max(0, self.game.current_player(self.state))
        self.state = self.game.apply(self.state, act, self.rng)
        ret = self.game.returns(self.state)[player]
        reward, self._last_return = ret - self._last_return, ret
        return self._obs(reward)
