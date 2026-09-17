"""Supervised fine-tuning (LoRA) on teacher trajectories. Runs on AMD GPUs via PyTorch ROCm.

    python -m gamemaster.training.sft --data data/c4.jsonl --model Qwen/Qwen3-1.7B --out checkpoints/c4-sft

Designed for the 3 h/day notebook quota: LoRA (small optimizer state), bf16, gradient
checkpointing, frequent checkpoints to persistent storage, and resume-from-checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dataset(records: list[dict]):
    from datasets import Dataset

    # Conversational prompt/completion format: TRL masks the prompt so loss is only on the move tokens.
    rows = [{"prompt": r["messages"], "completion": [{"role": "assistant", "content": r["completion"]}]}
            for r in records]
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from gamemaster.training.datagen import load_records, split

    records = load_records(args.data)[: args.max_records]
    train, evals = split(records, args.eval_fraction)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",  # SDPA works on ROCm without flash-attn
    )
    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_length=args.max_length,
        completion_only_loss=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=build_dataset(train),
        eval_dataset=build_dataset(evals) if evals else None,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
                               target_modules="all-linear"),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "gamemaster_train.json").write_text(json.dumps(
        {"stage": "sft", "base_model": args.model, "data": args.data, "train_records": len(train),
         "eval_records": len(evals), "args": vars(args)}, indent=2))


if __name__ == "__main__":
    main()
