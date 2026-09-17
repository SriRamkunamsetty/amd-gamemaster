"""2048: single-player, stochastic tile spawns, open-ended score."""

from __future__ import annotations

import random
from typing import Any, NamedTuple

from gamemaster.core import Game

N = 4
ACTIONS = ("up", "down", "left", "right")


class State2048(NamedTuple):
    board: tuple[int, ...]  # 16 cells, tile values (0 = empty)
    score: int
    steps: int
    over: bool


def _slide_row(row: list[int]) -> tuple[list[int], int]:
    tiles = [v for v in row if v]
    out: list[int] = []
    gained = 0
    i = 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            out.append(tiles[i] * 2)
            gained += tiles[i] * 2
            i += 2
        else:
            out.append(tiles[i])
            i += 1
    return out + [0] * (N - len(out)), gained


def _lines(action: str) -> list[list[int]]:
    """Cell indices of each line, ordered in the direction tiles move towards."""
    if action == "left":
        return [[r * N + c for c in range(N)] for r in range(N)]
    if action == "right":
        return [[r * N + c for c in reversed(range(N))] for r in range(N)]
    if action == "up":
        return [[r * N + c for r in range(N)] for c in range(N)]
    return [[r * N + c for r in reversed(range(N))] for c in range(N)]


def slide(board: tuple[int, ...], action: str) -> tuple[tuple[int, ...], int]:
    new = list(board)
    total = 0
    for line in _lines(action):
        values, gained = _slide_row([board[i] for i in line])
        total += gained
        for i, v in zip(line, values, strict=True):
            new[i] = v
    return tuple(new), total


class Game2048(Game[State2048]):
    name = "2048"
    num_players = 1
    deterministic = False
    max_steps = 3000
    rules = (
        "2048 on a 4x4 grid. Each move (up, down, left, right) slides all tiles in that direction; two tiles "
        "with the same value that collide merge into one tile with their sum, adding that sum to the score. "
        "After every move that changes the board a new tile (2 with 90% probability, otherwise 4) appears "
        "in a random empty cell. The game ends when no move changes the board. Maximise the score."
    )

    def _spawn(self, board: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
        empty = [i for i, v in enumerate(board) if v == 0]
        if not empty:
            return board
        new = list(board)
        new[rng.choice(empty)] = 2 if rng.random() < 0.9 else 4
        return tuple(new)

    def initial_state(self, seed: int | None = None) -> State2048:
        rng = random.Random(seed)
        board = self._spawn(self._spawn((0,) * (N * N), rng), rng)
        return State2048(board, 0, 0, False)

    def current_player(self, s: State2048) -> int:
        return -1 if s.over else 0

    def legal_actions(self, s: State2048) -> list[str]:
        if s.over:
            return []
        return [a for a in ACTIONS if slide(s.board, a)[0] != s.board]

    def apply(self, s: State2048, action: Any, rng: random.Random | None = None) -> State2048:
        rng = rng or random.Random()
        board, gained = slide(s.board, str(action))
        if board == s.board:
            raise ValueError(f"move {action} does not change the board")
        board = self._spawn(board, rng)
        steps = s.steps + 1
        nxt = State2048(board, s.score + gained, steps, False)
        over = steps >= self.max_steps or not any(slide(board, a)[0] != board for a in ACTIONS)
        return nxt._replace(over=over)

    def is_terminal(self, s: State2048) -> bool:
        return s.over

    def returns(self, s: State2048) -> list[float]:
        return [float(s.score)]

    def evaluate(self, s: State2048) -> list[float]:
        if s.over:
            return [float(s.score)]
        b = s.board
        empty = sum(1 for v in b if v == 0)
        rows = [b[r * N : r * N + N] for r in range(N)]
        cols = [tuple(b[r * N + c] for r in range(N)) for c in range(N)]
        mono = 0.0
        for line in rows + cols:
            inc = sum(max(0, line[i + 1] - line[i]) for i in range(N - 1))
            dec = sum(max(0, line[i] - line[i + 1]) for i in range(N - 1))
            mono += min(inc, dec)
        corner = max(b) if max(b) in (b[0], b[3], b[12], b[15]) else 0
        return [float(s.score) + 12.0 * empty * 4 + corner - 0.5 * mono]

    def render(self, s: State2048) -> str:
        rows = [" ".join(f"{v:>5}" if v else "    ." for v in s.board[r * N : r * N + N]) for r in range(N)]
        return "\n".join(rows + [f"score {s.score}{' (game over)' if s.over else ''}"])

    def to_json(self, s: State2048) -> dict[str, Any]:
        return {"board": list(s.board), "score": s.score, "steps": s.steps, "over": s.over}

    def from_json(self, d: dict[str, Any]) -> State2048:
        board = d["board"]
        flat = [v for row in board for v in row] if board and isinstance(board[0], list) else list(board)
        return State2048(tuple(int(v) for v in flat), int(d.get("score", 0)), int(d.get("steps", 0)),
                         bool(d.get("over", False)))
