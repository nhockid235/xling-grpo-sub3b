"""Checkpoint helpers -- LoRA merge for SFT->GRPO handoff and a checkpoint-keep callback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def merge_lora_if_needed(ckpt_path: str | Path) -> Path:
    """Merge LoRA adapter into base model if ``adapter_config.json`` is present.

    Args:
        ckpt_path: SFT checkpoint directory.

    Returns:
        Path to a merged-weights directory (``ckpt_path / "merged"``) ready for GRPO.
        Returns ``ckpt_path`` unchanged if no adapter is present.
    """
    ckpt_path = Path(ckpt_path)
    adapter_cfg = ckpt_path / "adapter_config.json"

    if not adapter_cfg.exists():
        return ckpt_path

    merged_dir = ckpt_path / "merged"
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        return merged_dir

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    import json
    with adapter_cfg.open() as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config["base_model_name_or_path"]

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )
    peft_model = PeftModel.from_pretrained(base, ckpt_path)
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)

    try:
        tok = AutoTokenizer.from_pretrained(ckpt_path)
        tok.save_pretrained(merged_dir)
    except Exception:
        tok = AutoTokenizer.from_pretrained(base_model_name)
        tok.save_pretrained(merged_dir)

    return merged_dir


class KeepCheckpointStepsCallback(TrainerCallback):
    """Persist a copy of specific checkpoint steps as ``keep_step{N}`` directories."""

    def __init__(self, steps_to_keep: list[int]) -> None:
        self.steps_to_keep = set(steps_to_keep)

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        step = state.global_step
        if step not in self.steps_to_keep:
            return control

        src = Path(args.output_dir) / f"checkpoint-{step}"
        if not src.exists():
            return control

        dst = Path(args.output_dir) / f"keep_step{step}"
        if dst.exists():
            return control

        try:
            shutil.copytree(src, dst, dirs_exist_ok=False)
        except Exception:
            pass
        return control
