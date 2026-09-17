"""Game registry. Plug the challenge's game in here once its rules are known."""

from __future__ import annotations

from collections.abc import Callable

from gamemaster.core import Game
from gamemaster.games.connect4 import ConnectFour
from gamemaster.games.game2048 import Game2048
from gamemaster.games.keydoor import KeyDoor

_REGISTRY: dict[str, Callable[[], Game]] = {
    "connect4": ConnectFour,
    "2048": Game2048,
    "keydoor": KeyDoor,
}


def register_game(name: str, factory: Callable[[], Game]) -> None:
    _REGISTRY[name] = factory


def get_game(name: str) -> Game:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        raise KeyError(f"unknown game {name!r}; available: {sorted(_REGISTRY)}") from exc


def list_games() -> list[str]:
    return sorted(_REGISTRY)
