"""GRPO reward functions. Pure Python so they are unit-testable without a GPU.

TRL passes extra dataset columns as keyword arguments, so each function receives ``legal`` and
``policy`` lists aligned with ``completions``. Completions may be plain strings or chat messages.
"""

from __future__ import annotations

from typing import Any

from gamemaster.prompts import parse_move


def _text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def format_reward(completions: list[Any], legal: list[list[str]], **_: Any) -> list[float]:
    """+0.2 for a bare legal move, 0.0 for a legal move with extra text, -1.0 for illegal/unparseable."""
    out = []
    for comp, moves in zip(completions, legal, strict=True):
        text = _text(comp).strip()
        move = parse_move(text, moves)
        if move is None:
            out.append(-1.0)
        else:
            out.append(0.2 if text.lower() == move.lower() else 0.0)
    return out


def teacher_reward(completions: list[Any], legal: list[list[str]], policy: list[dict[str, float]], **_: Any) -> list[float]:
    """Soft agreement with the search teacher: prob(chosen) / max prob, in [0, 1]."""
    out = []
    for comp, moves, dist in zip(completions, legal, policy, strict=True):
        move = parse_move(_text(comp), moves)
        if move is None or not dist:
            out.append(0.0)
            continue
        top = max(dist.values()) or 1.0
        out.append(float(dist.get(move, 0.0)) / top)
    return out


def make_engine_reward(game_name: str, depth_agent: str = "greedy"):
    """Optional: score the move by simulating it in the real engine (value after the move).

    Slower than teacher_reward but has no label noise; use when the teacher is weak or for fine
    polishing. Requires ``state`` (game.to_json) in the dataset rows.
    """
    from gamemaster.games import get_game

    game = get_game(game_name)

    def engine_reward(completions: list[Any], legal: list[list[str]], state: list[dict], **_: Any) -> list[float]:
        out = []
        for comp, moves, st in zip(completions, legal, state, strict=True):
            move = parse_move(_text(comp), moves)
            if move is None:
                out.append(0.0)
                continue
            s = game.from_json(st)
            player = game.current_player(s)
            scores = {m: game.evaluate(game.apply(s, game.str_to_action(s, m)))[player] for m in moves}
            lo, hi = min(scores.values()), max(scores.values())
            out.append(1.0 if hi == lo else (scores[move] - lo) / (hi - lo))
        return out

    engine_reward.__name__ = f"engine_reward_{game_name}"
    return engine_reward
