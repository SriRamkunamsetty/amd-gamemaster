"""GameMaster command line.

Harness mode (predicted contract, mirrors the published challenge format):
    python3 /app/app.py --input-state /app/input/state_01.json      -> /app/output/state_01_output.json
    input:  {"game": "connect4", "state": {...}}  or  {"rules": "...", "observation": "...", "legal_actions": [...]}
    output: {"action": "3", ...}

Tooling:
    gamemaster arena   --game connect4 --agents mcts:iterations=400 random --episodes 20
    gamemaster datagen --game keydoor --teacher planner --episodes 500 --out data/keydoor.jsonl
    gamemaster explore --game keydoor --episodes 20          (black-box explorer on a local game)
    gamemaster explore --env-url http://host:port --episodes 5 (black-box explorer on a remote game)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

from academy_core import output_path_for, setup_logging, write_json_atomic
from academy_core.harness import read_input_file


def _act(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="gamemaster act")
    p.add_argument("--input-state", "--input-file", "--input", "--input-json", dest="input", required=True)
    p.add_argument("--output-dir", default=os.getenv("HARNESS_OUTPUT_DIR", "/app/output"))
    p.add_argument("--output-file", default=None)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--agent", default=None)
    args = p.parse_args(argv)

    out = Path(args.output_file) if args.output_file else output_path_for(args.input, args.output_dir)
    payload = read_input_file(args.input)
    legal = payload.get("legal_actions") or []
    write_json_atomic(out, {"action": str(legal[0]) if legal else None, "source": "placeholder"})

    budget = args.time_budget or float(payload.get("time_budget") or os.getenv("GAMEMASTER_MOVE_TIME", "5"))
    watchdog = threading.Timer(budget + 10.0, lambda: os._exit(0))
    watchdog.daemon = True
    watchdog.start()

    from gamemaster.policy import decide

    try:
        result = decide(payload, time_budget=budget, agent_spec=args.agent)
    except Exception as exc:  # noqa: BLE001 - the placeholder answer stays in place
        print(f"decision failed: {exc}", file=sys.stderr)
        return 0
    write_json_atomic(out, result)
    print(json.dumps({"output": str(out), **result}, default=str))
    watchdog.cancel()
    return 0


def _arena(argv: list[str]) -> int:
    from gamemaster.eval import evaluate
    from gamemaster.games import get_game

    p = argparse.ArgumentParser(prog="gamemaster arena")
    p.add_argument("--game", required=True)
    p.add_argument("--agents", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--move-time", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default=None)
    args = p.parse_args(argv)
    report = evaluate(get_game(args.game), args.agents, args.episodes, args.move_time, args.seed)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).write_text(text)
    return 0


def _datagen(argv: list[str]) -> int:
    from gamemaster.training.datagen import generate

    p = argparse.ArgumentParser(prog="gamemaster datagen")
    p.add_argument("--game", required=True)
    p.add_argument("--teacher", default="mcts:iterations=800")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history", type=int, default=0)
    p.add_argument("--min-outcome", type=float, default=None)
    args = p.parse_args(argv)
    print(json.dumps(generate(args.game, args.teacher, args.episodes, args.out, args.seed, args.workers,
                              args.epsilon, args.history, args.min_outcome), indent=2))
    return 0


def _explore(argv: list[str]) -> int:
    from gamemaster.agents import ExplorerAgent
    from gamemaster.core import GameEnv
    from gamemaster.envs import HttpEnv
    from gamemaster.games import get_game

    p = argparse.ArgumentParser(prog="gamemaster explore")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--game")
    src.add_argument("--env-url")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--same-level", action="store_true", help="replay one seed to show learning across episodes")
    args = p.parse_args(argv)
    env = GameEnv(get_game(args.game)) if args.game else HttpEnv(args.env_url)
    agent = ExplorerAgent(seed=args.seed)
    for ep in range(args.episodes):
        seed = args.seed if args.same_level else args.seed + ep
        print(json.dumps({"episode": ep, "seed": seed, **agent.run_episode(env, seed)}))
    return 0


COMMANDS = {"act": _act, "arena": _arena, "datagen": _datagen, "explore": _explore}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return COMMANDS[argv[0]](argv[1:])
    if any(a.startswith("--input") for a in argv):  # harness invocation without a subcommand
        return _act(argv)
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
