"""Checkpoint helpers — LoRA merge cho SFT→GRPO handoff + ckpt rotation callback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def merge_lora_if_needed(ckpt_path: str | Path) -> Path:
    """Merge LoRA adapter vào base model nếu ckpt_path chứa adapter_config.json.

    Trả về path tới merged model (full-param) sẵn sàng cho GRPOTrainer.
    Nếu không phải LoRA ckpt → return ckpt_path nguyên.

    Args:
        ckpt_path: path tới checkpoint dir của SFT.

    Returns:
        Path tới full-param checkpoint (merged hoặc gốc).
    """
    ckpt_path = Path(ckpt_path)
    adapter_cfg = ckpt_path / "adapter_config.json"

    if not adapter_cfg.exists():
        # Không phải LoRA → trả về nguyên path
        return ckpt_path

    merged_dir = ckpt_path / "merged"
    if merged_dir.exists() and (merged_dir / "config.json").exists():
        # Đã merge trước đó → re-use
        return merged_dir

    # Lazy import để tránh phụ thuộc PEFT khi chưa cần
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Đọc adapter_config.json để biết base model
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

    # Save tokenizer (nếu có) để pipeline downstream load được
    try:
        tok = AutoTokenizer.from_pretrained(ckpt_path)
        tok.save_pretrained(merged_dir)
    except Exception:
        # Fallback từ base
        tok = AutoTokenizer.from_pretrained(base_model_name)
        tok.save_pretrained(merged_dir)

    return merged_dir


class KeepCheckpointStepsCallback(TrainerCallback):
    """Bảo toàn checkpoints ở các step chỉ định, kể cả khi save_total_limit rotate.

    Used cho Open-RS reproduction (cần ckpt-50, ckpt-100 để eval gating).
    Khi `save_total_limit=3` mà bạn muốn pin step 50 và 100 → callback này copy
    ra `output_dir/keep_step{N}/` ngay sau khi ckpt được save.
    """

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
            return control  # Đã copy

        try:
            shutil.copytree(src, dst, dirs_exist_ok=False)
        except Exception:
            # Không crash training nếu copy fail (disk full v.v.)
            pass
        return control
