"""GRPO trainer entrypoint.

Usage:
    python src/trainers/grpo.py --config configs/grpo_en.yaml --model qwen15b --seed 42 \\
        --sft_checkpoint results/sft/qwen15b_en_42/checkpoint-final

Stage 2: GRPO post-training on top of SFT checkpoint, hoặc trực tiếp từ base model
cho `reproduce_open_rs.yaml`.

Critical setup (verified Open-RS Exp2 + TRL 0.15+ API):
    - reward_funcs = list of callables; signature (prompts, completions, **kwargs) -> list[float]
    - Dataset MUST có column "prompt"
    - processing_class=tokenizer (deprecated: tokenizer=)
    - num_generations=6, max_completion_length=3584, temperature=0.7
    - GRPOConfig.reward_weights yêu cầu TRL >= 0.15.0
"""

from __future__ import annotations

import argparse
import os
from functools import partial
from pathlib import Path

from src.utils.io import load_config
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO trainer cho xling-grpo-sub3b")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True, help="model_short")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sft_checkpoint",
        type=Path,
        default=None,
        help="SFT checkpoint path. Bỏ qua nếu reproduce Open-RS (start từ base).",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def _resolve_model_path(args: argparse.Namespace, cfg: dict) -> str:
    """Quyết định model start point — SFT ckpt (merge LoRA nếu có) hoặc base."""
    if args.sft_checkpoint is not None:
        if not (args.sft_checkpoint / "config.json").exists():
            raise FileNotFoundError(
                f"SFT checkpoint không hợp lệ (thiếu config.json): {args.sft_checkpoint}"
            )
        from src.trainers.checkpoint_utils import merge_lora_if_needed
        merged = merge_lora_if_needed(args.sft_checkpoint)
        return str(merged)
    return cfg["model_name_or_path"]


def _build_rewards(cfg: dict) -> tuple[list, list[float]]:
    """Lookup reward functions theo registry, partial-bind config-time args cho R5."""
    from src.rewards import get_reward

    reward_funcs: list = []
    reward_weights: list[float] = []
    for r in cfg["rewards"]:
        name = r["name"]
        fn = get_reward(name)
        # R5 (lang) cần config-time params bind trước khi pass cho TRL
        if name == "lang" and r.get("config"):
            fn = partial(fn, **r["config"])
            # TRL log reward theo __name__ → set lại để wandb có metric tên đúng
            try:
                fn.__name__ = "r5_lang_consistency"  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                pass
        reward_funcs.append(fn)
        reward_weights.append(float(r.get("weight", 1.0)))
    return reward_funcs, reward_weights


def _validate_fasttext_for_enlang(cfg: dict) -> None:
    """Pre-flight check: Cond C cần lid.176.bin tồn tại."""
    if cfg.get("condition") != "enlang":
        return
    lang_cfg = next((r for r in cfg.get("rewards", []) if r["name"] == "lang"), None)
    if not lang_cfg:
        return
    ft_path = Path(lang_cfg.get("config", {}).get("fasttext_model", "data/raw/lid.176.bin"))
    if not ft_path.exists():
        raise FileNotFoundError(
            f"Cond C (enlang) yêu cầu fastText model tại {ft_path}. "
            f"Tải qua: wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/"
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    condition = cfg.get("condition", "en")
    run_id = f"{args.model}_{condition}_{args.seed}"
    output_dir = args.output_dir or Path(cfg["paths"]["grpo_ckpt_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_name = args.wandb_run_name or f"{run_id}_grpo"

    seed_everything(args.seed)
    _validate_fasttext_for_enlang(cfg)

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from src.trainers.checkpoint_utils import KeepCheckpointStepsCallback
    from src.trainers.dataset_utils import (
        DEFAULT_SYSTEM_PROMPT_EN,
        DEFAULT_SYSTEM_PROMPT_VI,
        prepare_grpo_dataset,
    )

    if os.environ.get("WANDB_DISABLED") != "1":
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            name=wandb_name,
            config=cfg,
            dir=str(output_dir),
        )

    # Resolve model path (SFT-merged hoặc base)
    model_path = _resolve_model_path(args, cfg)

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO rollout decoding cần left pad

    # Dataset — nguồn từ JSONL (cfg.train_dataset) hoặc HF (cfg.train_dataset_hf)
    source = cfg.get("train_dataset_hf") or cfg["train_dataset"]
    split = cfg.get("train_split", "train")
    system_prompt = (
        DEFAULT_SYSTEM_PROMPT_VI if condition == "vi" else DEFAULT_SYSTEM_PROMPT_EN
    )
    train_ds = prepare_grpo_dataset(
        source=source,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        split=split,
    )
    if "prompt" not in train_ds.column_names or "answer" not in train_ds.column_names:
        raise ValueError(
            f"GRPO dataset phải có 'prompt' và 'answer'; got {train_ds.column_names}"
        )

    # Reward registry → partial-bind cho R5 nếu Cond C
    reward_funcs, reward_weights = _build_rewards(cfg)

    # GRPOConfig — `attn_implementation` không phải field GRPOConfig, pass qua model_init_kwargs
    grpo_kwargs = dict(cfg["grpo"])
    attn_impl = grpo_kwargs.pop("attn_implementation", "flash_attention_2")
    model_init_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_impl,
    }

    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        seed=args.seed,
        reward_weights=reward_weights,
        save_total_limit=3,
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "1" else "none",
        run_name=wandb_name,
        model_init_kwargs=model_init_kwargs,
        **grpo_kwargs,
    )

    # LoRA fallback (chỉ khi cfg.lora.enabled, mặc định Open-RS = full-param)
    peft_config = None
    if cfg.get("lora", {}).get("enabled", False):
        from peft import LoraConfig
        lora_cfg = dict(cfg["lora"])
        lora_cfg.pop("enabled", None)
        peft_config = LoraConfig(**lora_cfg)

    callbacks = []
    # Reproduce Open-RS: pin ckpt-50 và ckpt-100 để eval gating
    keep_steps = cfg.get("gating", {}).get("ckpt_step")
    if keep_steps is not None:
        callbacks.append(KeepCheckpointStepsCallback([int(keep_steps)]))

    trainer = GRPOTrainer(
        model=model_path,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks if callbacks else None,
    )

    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "GRPO OOM. Remediation options: "
            "(1) set lora.enabled=true cho fallback LoRA r=16; "
            "(2) giảm per_device_train_batch_size từ 6 → 4; "
            "(3) giảm num_generations từ 6 → 4 (sẽ lệch Open-RS). "
            f"Original error: {exc}"
        ) from exc

    trainer.save_model(str(output_dir / "checkpoint-final"))
    tokenizer.save_pretrained(output_dir / "checkpoint-final")


if __name__ == "__main__":
    main()
