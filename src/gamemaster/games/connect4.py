"""Connect Four: two-player, deterministic, adversarial."""

from __future__ import annotations

import random
from typing import Any, NamedTuple

from gamemaster.core import Game

ROWS, COLS = 6, 7
_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


class C4State(NamedTuple):
    board: tuple[int, ...]  # row 0 is the bottom; 0 empty, 1 player one, 2 player two
    to_move: int  # 0 or 1
    winner: int  # -1 none, 0/1 winner, 2 draw
    moves: int


def _idx(r: int, c: int) -> int:
    return r * COLS + c


class ConnectFour(Game[C4State]):
    name = "connect4"
    num_players = 2
    deterministic = True
    max_steps = ROWS * COLS
    rules = (
        "Connect Four on a 6-row, 7-column board. Players alternate dropping a disc into a column (0-6); "
        "it falls to the lowest empty cell. The first player to line up four of their discs horizontally, "
        "vertically or diagonally wins. A full board is a draw. Moves are column numbers."
    )

    def initial_state(self, seed: int | None = None) -> C4State:
        return C4State((0,) * (ROWS * COLS), 0, -1, 0)

    def current_player(self, s: C4State) -> int:
        return -1 if s.winner != -1 else s.to_move

    def legal_actions(self, s: C4State) -> list[int]:
        if s.winner != -1:
            return []
        return [c for c in range(COLS) if s.board[_idx(ROWS - 1, c)] == 0]

    def apply(self, s: C4State, action: Any, rng: random.Random | None = None) -> C4State:
        col = int(action)
        if s.winner != -1 or not 0 <= col < COLS or s.board[_idx(ROWS - 1, col)] != 0:
            raise ValueError(f"illegal move {action}")
        row = next(r for r in range(ROWS) if s.board[_idx(r, col)] == 0)
        piece = s.to_move + 1
        board = list(s.board)
        board[_idx(row, col)] = piece
        moves = s.moves + 1
        winner = s.to_move if self._wins(board, row, col, piece) else (2 if moves == ROWS * COLS else -1)
        return C4State(tuple(board), 1 - s.to_move, winner, moves)

    @staticmethod
    def _wins(board: list[int], row: int, col: int, piece: int) -> bool:
        for dr, dc in _DIRS:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and board[_idx(r, c)] == piece:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= 4:
                return True
        return False

    def is_terminal(self, s: C4State) -> bool:
        return s.winner != -1

    def returns(self, s: C4State) -> list[float]:
        if s.winner in (-1, 2):
            return [0.0, 0.0]
        return [1.0, -1.0] if s.winner == 0 else [-1.0, 1.0]

    def value_bounds(self) -> tuple[float, float]:
        return (-1.0, 1.0)

    def evaluate(self, s: C4State) -> list[float]:
        if self.is_terminal(s):
            return self.returns(s)
        score = self._window_score(s.board, 1) - self._window_score(s.board, 2)
        v = max(-0.9, min(0.9, score / 60.0))
        return [v, -v]

    @staticmethod
    def _window_score(board: tuple[int, ...], piece: int) -> float:
        other = 3 - piece
        total = 0.0
        for r in range(ROWS):
            for c in range(COLS):
                for dr, dc in _DIRS:
                    cells = [(r + i * dr, c + i * dc) for i in range(4)]
                    if not all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                        continue
                    vals = [board[_idx(rr, cc)] for rr, cc in cells]
                    if other in vals:
                        continue
                    n = vals.count(piece)
                    total += {2: 1.0, 3: 5.0, 4: 100.0}.get(n, 0.0)
        centre = sum(1 for r in range(ROWS) if board[_idx(r, 3)] == piece)
        return total + 2.0 * centre

    def render(self, s: C4State) -> str:
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = [" ".join(symbols[s.board[_idx(r, c)]] for c in range(COLS)) for r in reversed(range(ROWS))]
        footer = " ".join(str(c) for c in range(COLS))
        turn = "X" if s.to_move == 0 else "O"
        status = {-1: f"{turn} to move", 0: "X won", 1: "O won", 2: "draw"}[s.winner]
        return "\n".join(rows + [footer, status])

    def to_json(self, s: C4State) -> dict[str, Any]:
        return {"board": list(s.board), "to_move": s.to_move, "winner": s.winner, "moves": s.moves}

    def from_json(self, d: dict[str, Any]) -> C4State:
        board = tuple(int(x) for x in d["board"])
        moves = int(d.get("moves", sum(1 for x in board if x)))
        return C4State(board, int(d.get("to_move", moves % 2)), int(d.get("winner", -1)), moves)
