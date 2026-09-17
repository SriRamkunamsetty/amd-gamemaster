"""Teacher data generation.

A strong search agent (MCTS / planner) plays the game; every decision becomes a training record:

    {"messages": [...system, user...], "completion": "<teacher move>", "legal": [...],
     "policy": {"move": prob, ...}, "value": float|None, "outcome": float, "game": ..., "seed": ..., "step": ...}

* ``messages`` comes from gamemaster.prompts - the exact format used at inference.
* ``policy`` (visit distribution) powers the soft teacher-agreement reward in GRPO.
* ``outcome`` is the final return for the player who moved, enabling outcome filtering.

    gamemaster datagen --game connect4 --teacher mcts:iterations=800 --episodes 200 --out data/c4.jsonl --workers 8
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

from gamemaster.agents import make_agent
from gamemaster.games import get_game
from gamemaster.prompts import messages_for_state


def generate_episode(args: tuple[str, str, int, float, int]) -> list[dict[str, Any]]:
    game_name, teacher_spec, seed, epsilon, history_len = args
    game = get_game(game_name)
    rng = random.Random(seed)
    teachers = [make_agent(teacher_spec, seed=seed * 7 + p) for p in range(game.num_players)]
    state = game.initial_state(seed)
    records: list[dict[str, Any]] = []
    history: list[str] = []
    steps = 0
    while not game.is_terminal(state) and steps < game.max_steps:
        player = game.current_player(state)
        decision = teachers[player].decide(game, state)
        messages, legal = messages_for_state(game, state, history[-history_len:] if history_len else None)
        move = game.action_to_str(state, decision.action)
        records.append({
            "game": game_name, "seed": seed, "step": steps, "player": player,
            "messages": messages, "completion": move, "legal": legal,
            "policy": decision.policy or {move: 1.0}, "value": decision.value,
            "state": game.to_json(state),
        })
        # epsilon-exploration diversifies visited states; the *label* is still the teacher's move
        action = rng.choice(game.legal_actions(state)) if rng.random() < epsilon else decision.action
        history.append(game.action_to_str(state, action))
        state = game.apply(state, action, rng)
        steps += 1
    final = game.returns(state)
    for r in records:
        r["outcome"] = final[r["player"]]
    return records


def generate(game_name: str, teacher_spec: str, episodes: int, out: str | Path, seed: int = 0,
             workers: int = 1, epsilon: float = 0.1, history_len: int = 0, min_outcome: float | None = None) -> dict[str, Any]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(game_name, teacher_spec, seed + i, epsilon, history_len) for i in range(episodes)]
    seen: set[str] = set()
    written = kept_episodes = 0
    pool = mp.Pool(workers) if workers > 1 else None
    try:
        fh = out.open("w", encoding="utf-8")
        iterator = pool.imap_unordered(generate_episode, jobs) if pool else map(generate_episode, jobs)
        for records in iterator:
            if min_outcome is not None and records and records[-1]["outcome"] < min_outcome:
                continue
            kept_episodes += 1
            for r in records:
                key = r["messages"][-1]["content"]
                if key in seen:  # dedupe identical positions (common in openings)
                    continue
                seen.add(key)
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
        fh.close()
    finally:
        if pool:
            pool.close()
            pool.join()
    return {"records": written, "episodes_kept": kept_episodes, "episodes": episodes, "out": str(out)}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split(records: list[dict[str, Any]], eval_fraction: float = 0.05, seed: int = 0) -> tuple[list, list]:
    """Split by episode seed so evaluation positions never leak into training."""
    seeds = sorted({(r["game"], r["seed"]) for r in records})
    random.Random(seed).shuffle(seeds)
    held = set(seeds[: max(1, int(len(seeds) * eval_fraction))])
    train = [r for r in records if (r["game"], r["seed"]) not in held]
    evals = [r for r in records if (r["game"], r["seed"]) in held]
    return train, evals
