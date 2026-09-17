"""One prompt format shared by data generation, SFT, GRPO and inference.

Train/serve skew is the most common way fine-tuned game models silently fail, so there is
exactly one function that turns a game state into chat messages.
"""

from __future__ import annotations

import re
from typing import Any

from gamemaster.core import Game

SYSTEM_TEMPLATE = (
    "You are an expert game-playing agent for the game '{name}'.\n"
    "Rules: {rules}\n"
    "Reply with exactly one legal move copied from the list, and nothing else."
)


def build_messages(game_name: str, rules: str, board_text: str, legal: list[str],
                   history: list[str] | None = None) -> list[dict[str, str]]:
    parts = [f"Current position:\n{board_text}"]
    if history:
        parts.append("Recent moves: " + ", ".join(history[-8:]))
    parts.append("Legal moves: " + ", ".join(legal))
    parts.append("Your move:")
    return [
        {"role": "system", "content": SYSTEM_TEMPLATE.format(name=game_name, rules=rules.strip())},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def messages_for_state(game: Game, state: Any, history: list[str] | None = None) -> tuple[list[dict[str, str]], list[str]]:
    legal = [game.action_to_str(state, a) for a in game.legal_actions(state)]
    return build_messages(game.name, game.rules, game.render(state), legal, history), legal


def parse_move(text: str, legal: list[str]) -> str | None:
    """Strict-then-lenient mapping of model output onto a legal move."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip().strip("`'\".").strip()
    lowered = {m.lower(): m for m in legal}
    if cleaned.lower() in lowered:
        return lowered[cleaned.lower()]
    first_line = cleaned.splitlines()[0].strip().lower() if cleaned else ""
    if first_line in lowered:
        return lowered[first_line]
    hits = [m for m in legal if re.search(rf"(?<![\w]){re.escape(m.lower())}(?![\w])", cleaned.lower())]
    return hits[0] if len(hits) == 1 else None
