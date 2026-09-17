#!/usr/bin/env bash
# Expert-iteration pipeline for one game, sized for the AMD notebook (1 GPU, 48 GiB, 3 h/day).
#
#   GAME=connect4 TEACHER="mcts:iterations=800" BASE=Qwen/Qwen3-1.7B bash scripts/train_pipeline.sh
#
# Stages are idempotent: each is skipped if its output exists, so the pipeline resumes after a
# quota reset. Everything is written to persistent storage.
set -euo pipefail

GAME="${GAME:-connect4}"
TEACHER="${TEACHER:-mcts:iterations=800}"
BASE="${BASE:-Qwen/Qwen3-1.7B}"
EPISODES="${EPISODES:-400}"
WORKERS="${WORKERS:-$(nproc)}"
if [ -d /persistent ] && [ -w /persistent ]; then PERSIST=/persistent; else PERSIST="${PERSIST:-/workspace}"; fi
DATA="$PERSIST/data/${GAME}.jsonl"
CK="$PERSIST/checkpoints/${GAME}"
mkdir -p "$(dirname "$DATA")" "$CK"
run() { echo -e "\n=== $* ==="; }

run "1/6 teacher data: $TEACHER x $EPISODES episodes"
[ -s "$DATA" ] || python3 -m gamemaster.cli datagen --game "$GAME" --teacher "$TEACHER" \
  --episodes "$EPISODES" --out "$DATA" --workers "$WORKERS" --epsilon 0.15

run "2/6 SFT (LoRA) on $BASE"
[ -f "$CK/sft/adapter_config.json" ] || python3 -m gamemaster.training.sft --data "$DATA" --model "$BASE" \
  --out "$CK/sft" $( [ -d "$CK/sft" ] && echo --resume )

run "3/6 merge SFT adapter"
[ -f "$CK/sft-merged/config.json" ] || python3 -m gamemaster.training.export --adapter "$CK/sft" --out "$CK/sft-merged"

run "4/6 GRPO with engine-verified rewards"
[ -f "$CK/grpo/adapter_config.json" ] || python3 -m gamemaster.training.grpo --data "$DATA" --model "$CK/sft-merged" \
  --game "$GAME" --out "$CK/grpo" --engine-reward

run "5/6 merge GRPO adapter -> final policy"
[ -f "$CK/policy-merged/config.json" ] || python3 -m gamemaster.training.export --adapter "$CK/grpo" \
  --base "$CK/sft-merged" --out "$CK/policy-merged"

run "6/6 evaluation: serve the policy with vLLM and compare agents"
cat <<EOF
Start the server in another terminal:
  python3 -m vllm.entrypoints.openai.api_server --model $CK/policy-merged --served-model-name policy \\
      --gpu-memory-utilization 0.25 --enable-prefix-caching
Then:
  python3 -m gamemaster.cli arena --game $GAME --agents llm mcts:iterations=200 --episodes 20
  python3 -m gamemaster.cli arena --game $GAME --agents hybrid:iterations=200 mcts:iterations=200 --episodes 20
Copy $CK/policy-merged into the repo as checkpoints/${GAME}-merged and build the image with
  --build-arg POLICY_DIR=checkpoints/${GAME}-merged
EOF
