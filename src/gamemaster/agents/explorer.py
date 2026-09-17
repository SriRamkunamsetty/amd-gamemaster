"""Black-box exploration for environments with unknown rules (ARC-AGI-3 style).

With no simulator we cannot search, so we *learn one*: every step records the transition
(obs_key, action) -> (next_obs_key, reward, done) into a graph. The agent

1. exploits: if the graph contains a known path to a rewarding transition, follow it;
2. explores: otherwise walk (via known transitions) to the nearest state with untried actions
   and try one, preferring actions that were novel elsewhere;
3. remembers across episodes/levels, so what it learns carries over.

It needs no LLM; an optional ``suggest`` callback (e.g. an LLM hypothesis about which action is
promising) is consulted to order untried actions.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

from gamemaster.core import Env, Observation

Suggest = Callable[[Observation, list[str]], list[str]]


@dataclass
class Transition:
    next_key: Hashable
    reward: float
    done: bool
    count: int = 1


@dataclass
class WorldModel:
    edges: dict[Hashable, dict[str, Transition]] = field(default_factory=lambda: defaultdict(dict))
    legal: dict[Hashable, list[str]] = field(default_factory=dict)
    action_novelty: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, key: Hashable, action: str, obs: Observation) -> None:
        nxt = obs.key if obs.key is not None else obs.text
        known = self.edges[key].get(action)
        if known and known.next_key == nxt:
            known.count += 1
        else:
            self.edges[key][action] = Transition(nxt, obs.reward, obs.done)
            if nxt != key:
                self.action_novelty[action] += 1
        self.legal.setdefault(nxt, obs.legal_actions)

    def untried(self, key: Hashable) -> list[str]:
        return [a for a in self.legal.get(key, []) if a not in self.edges.get(key, {})]

    def path_to(self, start: Hashable, goal: Callable[[Hashable, str, Transition], bool]) -> list[str] | None:
        """Shortest known action sequence from ``start`` whose last transition satisfies ``goal``."""
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            key, path = queue.popleft()
            for action, tr in self.edges.get(key, {}).items():
                if goal(key, action, tr):
                    return [*path, action]
                if not tr.done and tr.next_key not in seen:
                    seen.add(tr.next_key)
                    queue.append((tr.next_key, [*path, action]))
        return None


class ExplorerAgent:
    name = "explorer"

    def __init__(self, seed: int | None = None, suggest: Suggest | None = None):
        self.rng = random.Random(seed)
        self.model = WorldModel()
        self.suggest = suggest

    def choose(self, obs: Observation) -> str:
        key = obs.key if obs.key is not None else obs.text
        self.model.legal[key] = obs.legal_actions
        # 1. exploit a known rewarding transition
        path = self.model.path_to(key, lambda _k, _a, tr: tr.reward > 0)
        if path:
            return path[0]
        # 2. try something new here
        untried = self.model.untried(key)
        if untried:
            if self.suggest:
                ordered = [a for a in self.suggest(obs, untried) if a in untried]
                if ordered:
                    return ordered[0]
            return max(untried, key=lambda a: (self.model.action_novelty[a], self.rng.random()))
        # 3. travel to the nearest frontier state
        path = self.model.path_to(key, lambda _k, _a, tr: bool(self.model.untried(tr.next_key)) and not tr.done)
        if path:
            return path[0]
        return self.rng.choice(obs.legal_actions)

    def run_episode(self, env: Env, seed: int | None = None, max_steps: int = 500) -> dict[str, float]:
        obs = env.reset(seed)
        total, steps = 0.0, 0
        while not obs.done and steps < max_steps and obs.legal_actions:
            key = obs.key if obs.key is not None else obs.text
            action = self.choose(obs)
            obs = env.step(action)
            self.model.record(key, action, obs)
            total += obs.reward
            steps += 1
        return {"return": total, "steps": steps, "known_states": len(self.model.legal)}
