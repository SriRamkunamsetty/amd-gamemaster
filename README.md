# GameMaster: search teaches, the LLM learns, search refines

**lablab.ai × AMD AI Academy Challenge, Mini-Challenge 6:** *teach a system to play a novel game, using what was
learned in challenges 1–5 plus model fine-tuning.*

A game-agnostic system for playing a **novel game**, built so that when the challenge game is released only the game
adapter needs writing. Agents, fine-tuning, evaluation, serving and packaging are ready and tested.

## Architecture

```
                 ┌──────────── Game (simulator: clone + search) ─────────────┐
 known rules ──► │ PlannerAgent (exact BFS, deterministic 1-player)           │
                 │ MCTSAgent (UCT/PUCT, n-player, stochastic open-loop,       │
                 │            min-max value normalization, anytime)           │──► action (always legal)
                 │ HybridAgent = PUCT + priors from the fine-tuned LLM        │
                 └─────────────────────────────────────────────────────────────┘
                 ┌──────────── Env (black box: reset/step only) ──────────────┐
 hidden rules ─► │ ExplorerAgent: learns a transition graph, exploits known    │
                 │ rewarding paths, travels to frontier states, keeps memory   │
                 │ across episodes/levels; optional LLM action suggestions     │
                 └─────────────────────────────────────────────────────────────┘

 Expert iteration (src/gamemaster/training):
   datagen: search teacher self-play → {prompt, teacher move, visit distribution, value, state, outcome}
   sft.py:  LoRA SFT on moves (prompt-masked loss)                   ─┐
   grpo.py: GRPO, rewards = legality/format + teacher agreement       ├─► export.py (merge) ─► vLLM on ROCm
            + optional engine-simulated move quality (verifiable)    ─┘
   prompts.py: ONE prompt builder shared by datagen, training and inference (no train/serve skew)
```

Why this beats "ask a big LLM for each move":
* **Never illegal:** vLLM choice-constrained decoding over the legal moves, re-validated client-side, with fallback.
* **Verifiable learning signal:** the game engine scores moves, so there is no human data and no reward from text.
* **Fits the hardware:** a small fine-tuned policy (1.7B–8B) plus CPU search stays within 48 GiB and per-move time limits.
* **Anytime:** every agent respects a per-move time budget.

## Practice games and results (CPU laptop)

| Game | Type | Results |
|---|---|---|
| `connect4` | 2-player, deterministic, adversarial | MCTS(300) beats random 6/6; blocks threats and takes immediate wins |
| `2048` | 1-player, stochastic, open-ended score | MCTS (open-loop, heuristic leaves, 50 ms/move) ≈ 29.7k points |
| `keydoor` | 1-player, sparse reward, hidden sub-goal (ARC-AGI-3 style) | planner optimal 10/10; black-box explorer solves an unseen level in 59 steps, then replays it in 9 (optimal) |

Multi-process data generation: 182 teacher records from 8 Connect Four games in 4 s.

## Quick start (CPU, no GPU needed)

```bash
uv venv --python 3.14 .venv
uv pip install -e ./academy-core -e ".[api]" pytest pytest-asyncio ruff
.venv/bin/python -m pytest -q          # Windows: .venv\Scripts\python -m pytest -q
```

```bash
python app.py --input samples/state_01.json --output-dir out        # harness mode -> out/state_01_output.json
gamemaster arena   --game connect4 --agents mcts:iterations=400 random --episodes 20
gamemaster datagen --game connect4 --teacher mcts:iterations=800 --episodes 400 --out data/c4.jsonl --workers 8
gamemaster explore --game keydoor --episodes 10 --same-level
gamemaster explore --env-url http://game-server:9000 --episodes 5
uvicorn gamemaster.service:app --port 8081
```

Harness input (predicted), either a known game plus state or a text-only observation:
```json
{"game": "connect4", "state": {"board": [...], "to_move": 1}, "time_budget": 2}
{"rules": "...", "observation": "...", "legal_actions": ["left", "right"]}
```
Output: `{"action": "3", "source": "mcts:iterations=0", "policy": {...}}`

## Fine-tuning on the AMD notebook (1 GPU, 48 GiB, 3 h/day)

```bash
bash scripts/notebook_setup.sh train
source /persistent/env-gamemaster.sh     # or /workspace/env-gamemaster.sh on rgapi-hackathon pods
GAME=connect4 TEACHER="mcts:iterations=800" BASE=Qwen/Qwen3-1.7B bash scripts/train_pipeline.sh
```

Every stage is idempotent and writes to persistent storage, so the pipeline resumes after a quota reset.

## Adding the challenge game (day-one checklist)

1. Implement `Game` (`initial_state, current_player, legal_actions, apply, is_terminal, returns, render, to_json, from_json`; optional `evaluate` heuristic and `value_bounds`). Register it in `src/gamemaster/games/__init__.py`.
2. Add it to the playout/JSON round-trip test in `tests/test_games_agents.py`.
3. Use `gamemaster arena` with `planner` / `mcts` for a strong baseline, and **submit that first** (Legend rank is first-come).
4. Run `scripts/train_pipeline.sh` → merged policy → `docker build --build-arg POLICY_DIR=checkpoints/<game>-merged` (with `GAMEMASTER_HYBRID=true`).
5. If the game is only available as a remote API with hidden rules, adapt `src/gamemaster/envs.py` and use `explore`.

## Grader-driven container design

* **ROCm base image checked by layer identity:** the final stage is `FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, and the image is never squashed.
* **New process per item:** the entrypoint starts the model server once during startup, and `app.py` is a thin client.
* **Peak VRAM between 1 and 48 GiB:** search runs on CPU, so the image serves the fine-tuned policy with vLLM (`LLM_VRAM_BUDGET_GIB=12`, converted into a fraction of the detected card), and the hybrid agent uses it.
* **Always a valid output:** a placeholder output is written first, and a watchdog guards against hangs.

```bash
docker build --build-arg POLICY_DIR=checkpoints/connect4-merged -t <registry>/amd-gamemaster:v1 .
bash scripts/check_submission.sh <registry>/amd-gamemaster:v1 samples 30
docker push <registry>/amd-gamemaster:v1   # public registry, but do not publish the image reference in a public repo
```

Deploy as a service: `cp .env.example .env && docker compose up -d --build` → `POST localhost:8081/v1/act`.

> The official spec for this mini-challenge was not published when this was built. The contract lives only in
> `src/gamemaster/cli.py`, `policy.py` and `envs.py`.

## Layout

```
src/gamemaster/  core.py (Game/Env) · games/ · agents/ (planner, mcts, explorer, llm_policy/hybrid) · prompts.py
                 training/ (datagen, rewards, sft, grpo, export) · eval/arena.py · policy.py · cli.py · service.py
academy-core/    shared runtime: LLM client, harness contract, deadlines, model-server entrypoint, VRAM monitor
scripts/         train_pipeline.sh · notebook_setup.sh · check_submission.sh
tests/           offline tests for rules, agents, rewards, datagen, LLM policy (mocked) and the harness CLI
```

## License

MIT
