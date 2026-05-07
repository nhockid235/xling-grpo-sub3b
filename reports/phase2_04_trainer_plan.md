# Phase 2 — Trainer Architecture Plan

**Plan date:** 2026-05-05
**Architect:** `Plan` subagent (read-only)
**Status:** ⛔ NEW BLOCKING discovery — must bump TRL pin to `>=0.15.0,<0.16.0`

---

## Critical TRL pin update

Plan agent verified via WebFetch of TRL source code:
- `GRPOConfig.reward_weights: Optional[list[float]]` field exists ONLY in TRL **>= 0.15.0**.
- `GRPOConfig.vllm_max_model_len`, `vllm_dtype` fields likewise 0.15+ only.
- TRL 0.15 spawns `vllm.LLM` internally on `use_vllm=True` — no external server needed.

**Action:** bump `pyproject.toml`:
```diff
-"trl>=0.14.0,<0.16.0",
+"trl>=0.15.0,<0.16.0",
```

And update CLAUDE.md pitfall #13 và Code standards.

---

## 1. Architecture overview

```
configs/*.yaml ──► load_config() ──► cfg dict
                                       │
                       ┌───────────────┼─────────────────────────┐
                       ▼                                          ▼
                 sft.py main                                grpo.py main
                       │                                          │
   tokenizer + AutoModelForCausalLM(bf16, FA2)        tokenizer + (sft_ckpt OR base)
   apply_chat_template → "text" col                   prepare_grpo_dataset → "prompt" col
   LoraConfig (if cfg.lora.enabled)                   build reward_funcs from registry
   SFTTrainer(model, SFTConfig, ds, peft_config)      GRPOTrainer(model, reward_funcs,
   trainer.train() ──► results/sft/{run}/             GRPOConfig(reward_weights=...),
   checkpoint-final/                                  ds, peft_config)
                                                       trainer.train() ──► results/grpo/{run}/
                                                                            checkpoint-{50,100,...}/
```

## 2. `sft.py` step-by-step plan

1. Parse CLI (skeleton OK). Resolve `output_dir` default = `results/sft/{model_short}_{condition}_{seed}`.
2. `cfg = load_config(args.config)`. Pull `train_dataset = cfg["conditions"][args.condition]["train_dataset"]`.
3. `seed_everything(args.seed)`.
4. `wandb.init(project=cfg["wandb"]["project"], name=args.wandb_run_name, config=cfg, dir=str(args.output_dir))`. Skip if `WANDB_DISABLED=1`.
5. Load tokenizer:
   ```python
   tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], use_fast=True)
   if tok.pad_token is None: tok.pad_token = tok.eos_token
   tok.padding_side = "right"   # SFT: right pad
   ```
6. Load base model:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       cfg["model_name_or_path"], torch_dtype=torch.bfloat16,
       attn_implementation="flash_attention_2")
   ```
7. LoRA wrap via `peft_config = LoraConfig(**cfg["lora"]) if cfg["lora"]["enabled"] else None`. Pass to `SFTTrainer(peft_config=...)` (do NOT call `get_peft_model` manually — TRL handles it).
8. Dataset prep via new helper `src/trainers/dataset_utils.py::prepare_sft_dataset(jsonl_path, tokenizer, system_prompt, condition)`:
   - `read_jsonl` → `datasets.Dataset.from_list`
   - `.map(lambda ex: {"text": tok.apply_chat_template([{role:system,...},{role:user,content:ex["problem"]},{role:assistant,content:ex["solution"]}], tokenize=False)})`
   - `.remove_columns([c for c in cols if c != "text"])`
9. `SFTConfig(**cfg["sft"], output_dir=str(args.output_dir), seed=args.seed, dataset_text_field="text", report_to="wandb", run_name=args.wandb_run_name)`.
10. `SFTTrainer(model=model, args=sft_config, train_dataset=ds, processing_class=tok, peft_config=peft_config)`. `trainer.train()`. `trainer.save_model(args.output_dir / "checkpoint-final")` + `tok.save_pretrained(...)`.

## 3. `grpo.py` step-by-step plan

1. Parse CLI. Resolve `output_dir` default `results/grpo/{model}_{cond}_{seed}`.
2. `cfg = load_config(args.config)`; resolve `model_path`:
   - if `args.sft_checkpoint` exists: use as model path; if SFT used LoRA, MERGE first via helper `merge_lora_if_needed(ckpt_path) -> Path` (see decision 4).
   - else: `cfg["model_name_or_path"]` (`reproduce_open_rs` path).
3. `seed_everything`, `wandb.init`.
4. Tokenizer load with `padding_side="left"` (rollout decoding).
5. Dataset via new helper `prepare_grpo_dataset(source, tokenizer, system_prompt)`:
   - source = JSONL path (`cfg["train_dataset"]`) or HF id (`cfg.get("train_dataset_hf")`).
   - Map each row → `{"prompt": tok.apply_chat_template([sys, user(problem)], tokenize=False, add_generation_prompt=True), "answer": ex["answer"]}`. Drop `solution`.
   - Returns `datasets.Dataset` with columns `["prompt", "answer"]`.
6. Build `reward_funcs` and `reward_weights`:
   ```python
   from src.rewards import get_reward
   from functools import partial
   reward_funcs, reward_weights = [], []
   for r in cfg["rewards"]:
       fn = get_reward(r["name"])
       if r["name"] == "lang" and r.get("config"):
           fn = partial(fn, **r["config"])
           fn.__name__ = "r5_lang_consistency"   # TRL logs by __name__
       reward_funcs.append(fn)
       reward_weights.append(r.get("weight", 1.0))
   ```
7. `GRPOConfig(**cfg["grpo"], output_dir=str(args.output_dir), seed=args.seed, reward_weights=reward_weights, save_total_limit=3, report_to="wandb", run_name=args.wandb_run_name)`. Pop `attn_implementation` from cfg["grpo"] (not a GRPOConfig field — applied at model load instead).
8. Pass model as path string. If LoRA fallback, `peft_config = LoraConfig(**cfg["lora"])` and model = base path (TRL wraps).
9. `GRPOTrainer(model=model_path, reward_funcs=reward_funcs, args=grpo_config, train_dataset=ds, processing_class=tok, peft_config=peft_config_or_None)`. `trainer.train()`. `trainer.save_model(output_dir/"checkpoint-final")`.

## 4. Decision points resolved

1. **R5 partial binding** — use `functools.partial(r5_lang_consistency, fasttext_model=..., no_penalty_for_en=..., min_response_tokens=...)`. Set `partial_obj.__name__` so wandb logs `rewards/r5_lang_consistency/mean` instead of generic `rewards/functools.partial`. Reason: TRL passes `**kwargs` from dataset cols only — config-time params must be bound.
2. **`reward_weights`** — confirmed in TRL 0.15+ (`Optional[list[float]] = None`). Pass via `GRPOConfig`. **Bump pin to `trl>=0.15.0,<0.16.0`**.
3. **vLLM** — TRL 0.15 spawns `vllm.LLM` internally on `use_vllm=True`; no external server. Single A100 must reserve memory: `vllm_gpu_memory_utilization=0.7` keeps ~24 GB for training.
4. **SFT-LoRA → GRPO** — recommend **option (a) merge then full-param GRPO**. Helper `merge_lora_if_needed(ckpt_path)` checks for `adapter_config.json`; if found, `PeftModel.from_pretrained(base, ckpt_path).merge_and_unload()`, save to `ckpt_path/merged/`, return that path. Reason: GRPO with LoRA-on-LoRA is brittle (target_modules collision); Open-RS uses full-param. For 3B OOM fallback set `cfg.lora.enabled=true` in GRPO config.
5. **Dataset prompt column** — single helper `prepare_grpo_dataset(source, tokenizer, system_prompt)` in `src/trainers/dataset_utils.py`. Detect HF id vs JSONL path via `Path(source).suffix == ".jsonl"`. Always renders chat template with `add_generation_prompt=True`, emits `["prompt", "answer"]`.
6. **Checkpoint lifecycle** — set `GRPOConfig.save_total_limit=3` (auto-rotate). `KeepCheckpointStepsCallback` (in `src/trainers/checkpoint_utils.py`) pins steps `{50, 100}` for Open-RS reproduction. For full plan: keep step 50, 100, final → ~9 GB/cell × 9 cells × 3 seeds = ~245 GB. Recommend **upload top-3 ckpts/cell to HF Hub private repo via `model.push_to_hub`**, then `rm -rf` local.
7. **Open-RS HF dataset path** — `prepare_grpo_dataset` branches: if `cfg.get("train_dataset_hf")` set → `load_dataset(hf_id, split=cfg.get("train_split","train"))`, no decontamination.

## 5. New files beyond `sft.py`/`grpo.py`

- `src/trainers/__init__.py` — empty package marker (already exists).
- `src/trainers/dataset_utils.py` — `prepare_sft_dataset`, `prepare_grpo_dataset`, `DEFAULT_SYSTEM_PROMPT_EN`, `DEFAULT_SYSTEM_PROMPT_VI` constants. ~80 lines.
- `src/trainers/checkpoint_utils.py` — `merge_lora_if_needed(ckpt_path) -> Path`, `KeepCheckpointStepsCallback(steps_to_keep)` subclassing `transformers.TrainerCallback`. ~60 lines.
- `tests/test_trainer_smoke.py` — dry-run with `max_steps=2`, `num_generations=2`, tiny model (`sshleifer/tiny-gpt2`) to exercise orchestration without GPU.

## 6. Edge cases & error handling

- **OOM in GRPO** — wrap `trainer.train()` in `try/except torch.cuda.OutOfMemoryError`; log clear remediation: "set lora.enabled=true OR reduce per_device_train_batch_size to 4 OR num_generations to 4". Do not auto-retry.
- **Missing fasttext** — already graceful in `lang.py` (returns 0.0). Trainer pre-flight: if `cfg.condition == "enlang"` and `not Path(cfg.lang.fasttext_model).exists()` → raise FileNotFoundError before training to fail fast.
- **Missing SFT ckpt for non-`reproduce_open_rs`** — script `train_grpo.sh` already checks; trainer additionally validates that `Path(args.sft_checkpoint, "config.json").exists()`.
- **Dataset schema mismatch** — assert `"problem" in ds.column_names and "answer" in ds.column_names` after loading; raise with explicit message.
- **`attn_implementation` in cfg["grpo"]** — pop before `**cfg["grpo"]` unpack into `GRPOConfig`. Pass instead via `model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16}` to GRPOConfig (TRL 0.15 supports this).
- **Reward column conflict** — `prepare_grpo_dataset` must NOT keep a column named `prompts` or `completions` (collision). Only `prompt` + `answer`.
- **Tokenizer pad-side** — SFT right-pad, GRPO left-pad. Set explicitly.

## 7. Verification plan (Phase 3 main agent)

1. **Static check** — `python -c "from src.trainers.sft import main; from src.trainers.grpo import main"` no GPU. Catches imports + registry wiring.
2. **Unit smoke test** — `pytest tests/test_trainer_smoke.py` running both trainers for 2 steps with tiny GPT-2 + 8 fake samples (mock dataset). Skips vLLM (`use_vllm=False`).
3. **Reward integration** — `tests/test_grpo_reward_path.py` builds 4-sample dataset, calls `reward_funcs[i](prompts, completions, answer=...)`, asserts each returns `list[float]` length 4.
4. **Dry-run on RunPod (1×A100)** — `bash scripts/train_grpo.sh qwen15b en 42` with `max_steps=5`, `save_steps=5`. Confirms (a) wandb logs reward/{name}/mean per reward, (b) GPU mem < 70 GB, (c) checkpoint dir exists post-train.
5. **Reproduce Open-RS gating run** (Week 2) — full 100 steps; eval ckpt-50 on AMC23 → must land in [77, 83].

## Files for Phase 3 implementation

- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/trainers/sft.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/trainers/grpo.py`
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/trainers/dataset_utils.py` (new)
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/src/trainers/checkpoint_utils.py` (new)
- `/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/pyproject.toml` (TRL pin bump)
- `/Users/vudang/PythonLab/Papper/CLAUDE.md` (TRL pin update)
