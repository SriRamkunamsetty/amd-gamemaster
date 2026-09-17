"""KeyDoor: a procedurally generated grid puzzle (single-player, deterministic, sparse reward).

Pick up the key (k), pass the locked door (D) and reach the goal (G). This is a stand-in for
ARC-AGI-3-style environments: long horizon, a hidden sub-goal, one success signal at the end.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, NamedTuple

from gamemaster.core import Game

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class KDState(NamedTuple):
    grid: tuple[str, ...]  # rows; '#' wall, '.' floor, 'k' key, 'D' door, 'G' goal
    pos: tuple[int, int]
    has_key: bool
    steps: int
    done: bool
    won: bool


class KeyDoor(Game[KDState]):
    name = "keydoor"
    num_players = 1
    deterministic = True
    max_steps = 120
    rules = (
        "Grid puzzle. You are '@'. Walls '#' block movement. Walking onto the key 'k' picks it up. The door 'D' "
        "is impassable until you hold the key. Reach the goal 'G' to win; fewer steps score higher. "
        "Moves: up, down, left, right. The episode ends after 120 steps."
    )

    def __init__(self, width: int = 9, height: int = 9):
        self.width, self.height = width, height

    def initial_state(self, seed: int | None = None) -> KDState:
        rng = random.Random(seed)
        for _ in range(1000):
            state = self._generate(rng)
            if state is not None:
                return state
        raise RuntimeError("could not generate a solvable level")

    def _generate(self, rng: random.Random) -> KDState | None:
        w, h = self.width, self.height
        cells = [["#" if r in (0, h - 1) or c in (0, w - 1) else "." for c in range(w)] for r in range(h)]
        wall_col = rng.randrange(3, w - 3)
        for r in range(1, h - 1):
            cells[r][wall_col] = "#"
        door_row = rng.randrange(1, h - 1)
        cells[door_row][wall_col] = "D"
        for _ in range((w * h) // 8):
            r, c = rng.randrange(1, h - 1), rng.randrange(1, w - 1)
            if c != wall_col:
                cells[r][c] = "#"
        left = [(r, c) for r in range(1, h - 1) for c in range(1, wall_col) if cells[r][c] == "."]
        right = [(r, c) for r in range(1, h - 1) for c in range(wall_col + 1, w - 1) if cells[r][c] == "."]
        if len(left) < 2 or not right:
            return None
        start, key = rng.sample(left, 2)
        goal = rng.choice(right)
        cells[key[0]][key[1]] = "k"
        cells[goal[0]][goal[1]] = "G"
        state = KDState(tuple("".join(row) for row in cells), start, False, 0, False, False)
        return state if self.shortest_solution(state) is not None else None

    def current_player(self, s: KDState) -> int:
        return -1 if s.done else 0

    def legal_actions(self, s: KDState) -> list[str]:
        if s.done:
            return []
        out = []
        for name, (dr, dc) in MOVES.items():
            r, c = s.pos[0] + dr, s.pos[1] + dc
            cell = s.grid[r][c]
            if cell != "#" and (cell != "D" or s.has_key):
                out.append(name)
        return out

    def apply(self, s: KDState, action: Any, rng: random.Random | None = None) -> KDState:
        if s.done or action not in self.legal_actions(s):
            raise ValueError(f"illegal move {action}")
        dr, dc = MOVES[str(action)]
        r, c = s.pos[0] + dr, s.pos[1] + dc
        grid = list(s.grid)
        has_key = s.has_key
        if grid[r][c] == "k":
            has_key = True
            grid[r] = grid[r][:c] + "." + grid[r][c + 1 :]
        won = grid[r][c] == "G"
        steps = s.steps + 1
        return KDState(tuple(grid), (r, c), has_key, steps, won or steps >= self.max_steps, won)

    def is_terminal(self, s: KDState) -> bool:
        return s.done

    def returns(self, s: KDState) -> list[float]:
        return [1.0 - 0.5 * s.steps / self.max_steps if s.won else 0.0]

    def value_bounds(self) -> tuple[float, float]:
        return (0.0, 1.0)

    def evaluate(self, s: KDState) -> list[float]:
        if s.done:
            return self.returns(s)
        dist = self._distance_to_win(s)
        if dist is None:
            return [0.0]
        return [0.5 * max(0.0, 1.0 - (s.steps + dist) / self.max_steps)]

    def _bfs(self, s: KDState, target: str, has_key: bool) -> int | None:
        seen = {s.pos}
        queue = deque([(s.pos, 0)])
        while queue:
            (r, c), d = queue.popleft()
            if s.grid[r][c] == target:
                return d
            for dr, dc in MOVES.values():
                nr, nc = r + dr, c + dc
                cell = s.grid[nr][nc]
                if (nr, nc) not in seen and cell != "#" and (cell != "D" or has_key):
                    seen.add((nr, nc))
                    queue.append(((nr, nc), d + 1))
        return None

    def _distance_to_win(self, s: KDState) -> int | None:
        if s.has_key:
            return self._bfs(s, "G", True)
        to_key = self._bfs(s, "k", False)
        if to_key is None:
            return None
        key_pos = next((r, row.index("k")) for r, row in enumerate(s.grid) if "k" in row)
        from_key = self._bfs(s._replace(pos=key_pos, has_key=True), "G", True)
        return None if from_key is None else to_key + from_key

    def shortest_solution(self, s: KDState) -> int | None:
        return self._distance_to_win(s)

    def render(self, s: KDState) -> str:
        rows = [list(row) for row in s.grid]
        rows[s.pos[0]][s.pos[1]] = "@"
        status = "won" if s.won else ("out of steps" if s.done else f"step {s.steps}")
        return "\n".join(["".join(r) for r in rows] + [f"key: {'yes' if s.has_key else 'no'}; {status}"])

    def to_json(self, s: KDState) -> dict[str, Any]:
        return {"grid": list(s.grid), "pos": list(s.pos), "has_key": s.has_key, "steps": s.steps,
                "done": s.done, "won": s.won}

    def from_json(self, d: dict[str, Any]) -> KDState:
        return KDState(tuple(d["grid"]), (int(d["pos"][0]), int(d["pos"][1])), bool(d.get("has_key")),
                       int(d.get("steps", 0)), bool(d.get("done")), bool(d.get("won")))

    def state_key(self, s: KDState) -> Any:
        return (s.pos, s.has_key, s.done)
