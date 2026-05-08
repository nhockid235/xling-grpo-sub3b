"""SFT trainer entrypoint.

Usage:
    python src/trainers/sft.py --config configs/sft_qwen15b.yaml --condition en --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.utils.io import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT trainer for xling-grpo-sub3b")
    parser.add_argument("--config", type=Path, required=True, help="YAML config")
    parser.add_argument("--condition", choices=["en", "vi", "enlang"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_short = cfg.get("model_short", "model")
    run_id = f"{model_short}_{args.condition}_{args.seed}"
    output_dir = args.output_dir or Path(cfg["paths"]["sft_ckpt_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_name = args.wandb_run_name or f"{run_id}_sft"

    seed_everything(args.seed)

    # Lazy heavy imports -- fail fast on bad CLI/config before importing torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from src.trainers.dataset_utils import (
        DEFAULT_SYSTEM_PROMPT_EN,
        DEFAULT_SYSTEM_PROMPT_VI,
        prepare_sft_dataset,
    )

    if os.environ.get("WANDB_DISABLED") != "1":
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            name=wandb_name,
            config=cfg,
            dir=str(output_dir),
        )

    # Tokenizer + base model
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFT: right pad

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    peft_config = None
    if cfg.get("lora", {}).get("enabled", False):
        from peft import LoraConfig

        lora_cfg = dict(cfg["lora"])
        lora_cfg.pop("enabled", None)
        peft_config = LoraConfig(**lora_cfg)

    train_dataset_path = cfg["conditions"][args.condition]["train_dataset"]
    system_prompt = (
        DEFAULT_SYSTEM_PROMPT_VI if args.condition == "vi" else DEFAULT_SYSTEM_PROMPT_EN
    )
    train_ds = prepare_sft_dataset(
        source=train_dataset_path,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )

    # SFTConfig (TRL 0.15+)
    sft_kwargs = dict(cfg["sft"])
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        seed=args.seed,
        dataset_text_field="text",
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "1" else "none",
        run_name=wandb_name,
        **sft_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "checkpoint-final"))
    tokenizer.save_pretrained(output_dir / "checkpoint-final")


if __name__ == "__main__":
    main()
