"""GRPO with verifiable, engine-derived rewards (RL from the game itself).

    python -m gamemaster.training.grpo --data data/c4.jsonl --model checkpoints/c4-sft-merged \
        --game connect4 --out checkpoints/c4-grpo

Rewards (see rewards.py): format/legality + soft agreement with the search teacher's visit
distribution (+ optional engine-simulated move quality). The model cannot score by writing
text: anything that is not exactly a legal move is penalised.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True, help="base or SFT-merged model")
    p.add_argument("--game", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--max-completion-length", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.02)
    p.add_argument("--engine-reward", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--use-vllm", action="store_true", help="colocated vLLM generation (if installed for ROCm)")
    p.add_argument("--max-records", type=int, default=20000)
    args = p.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from gamemaster.training.datagen import load_records
    from gamemaster.training.rewards import format_reward, make_engine_reward, teacher_reward

    records = [r for r in load_records(args.data) if len(r["legal"]) > 1][: args.max_records]
    dataset = Dataset.from_list([
        {"prompt": r["messages"], "legal": r["legal"], "policy": r["policy"], "state": r.get("state")}
        for r in records
    ])
    reward_funcs = [format_reward, teacher_reward]
    weights = [1.0, 1.0]
    if args.engine_reward:
        reward_funcs.append(make_engine_reward(args.game))
        weights.append(0.5)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    config = GRPOConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        beta=args.beta,
        temperature=1.0,
        reward_weights=weights,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_steps=100,
        save_total_limit=3,
        use_vllm=args.use_vllm,
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, task_type="CAUSAL_LM",
                               target_modules="all-linear"),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "gamemaster_train.json").write_text(json.dumps(
        {"stage": "grpo", "base_model": args.model, "game": args.game, "records": len(records), "args": vars(args)},
        indent=2))


if __name__ == "__main__":
    main()
