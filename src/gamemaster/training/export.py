"""Merge a LoRA adapter into its base model and save a standalone checkpoint that vLLM can serve.

    python -m gamemaster.training.export --adapter checkpoints/c4-sft --out /models/c4-policy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default=None, help="defaults to the base recorded in the adapter config")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = args.base or PeftConfig.from_pretrained(args.adapter).base_model_name_or_path
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    Path(args.out, "gamemaster_export.json").write_text(json.dumps({"base": base, "adapter": args.adapter}, indent=2))
    print(f"merged model written to {args.out}")


if __name__ == "__main__":
    main()
