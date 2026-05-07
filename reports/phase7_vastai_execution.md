# Phase 7 — Vast.ai Execution Log (W2 Gating)

**Run date:** 2026-05-06 (UTC)
**Hardware:** Vast.ai 1× NVIDIA A100-SXM4-80GB
**Driver:** 590.48.01, **CUDA system:** 13.1, **Container disk:** 32GB overlay + 62GB /dev/shm
**Run ID:** `reproduce_openrs_rs2_42`
**Goal:** Match Open-RS RS2 baseline AMC23 pass@1 ∈ [77, 83] @ ckpt-50 (gating decision)

---

## Environment final (verified imports + cuda)

```
py 3.12.3
torch 2.5.1+cu124  cuda? True  NVIDIA A100-SXM4-80GB
transformers 4.57.6
trl 0.15.2          (GRPOConfig + GRPOTrainer ✅)
vllm 0.7.2          (cuda platform auto-detected)
flash_attn 2.7.2.post1
sympy 1.13.1        (math_verify 1/2 == 0.5 → True)
math_verify 0.9.0
```

### Đáng chú ý so với CLAUDE.md pin

| Package | CLAUDE.md pin | Actual installed | Note |
|---|---|---|---|
| torch | `2.4.1` | `2.5.1+cu124` | vllm 0.7.2 forced upgrade |
| transformers | `4.46.3` | `4.57.6` | vllm 0.7.2 forced upgrade |
| sympy | `<1.13` | `1.13.1` | vllm 0.7.2 forced upgrade |
| trl | `>=0.15.0,<0.16.0` | `0.15.2` | ✅ |
| vllm | `0.7.2` | `0.7.2` | ✅ |

**math-verify 0.9.0 + sympy 1.13.1 verified working** — `verify(parse("1/2"), parse("0.5"))` returns True. CLAUDE.md pin `sympy<1.13` was overly cautious; modern math-verify supports newer sympy.

**Pyproject.toml fix:** `requires-python = ">=3.11,<3.12"` → `<3.13` để Python 3.12 install được.

## Setup timeline

| Time (UTC) | Event |
|---|---|
| 02:02 | Vast.ai instance up, env vars (`HF_HOME=/workspace/.hf_home`, CUDA 13.1) |
| 02:06 | rsync repo từ Mac (97KB transferred) |
| 02:06 | tmux setup session start, pip install begins |
| 02:08 | torch 2.4.1 installed (overwritten later) |
| 02:09 | transformers/trl/peft/datasets installed |
| 02:18 | vllm 0.7.2 installed (forced torch 2.5.1, transformers 4.57.6, sympy 1.13.1) |
| 02:18 | flash-attn 2.7.2.post1 built from source (took ~30s) |
| 02:18 | repo `pip install -e .` (had to fix Python pin first) |
| 02:18 | Setup complete: 8.1GB / 32GB disk used |
| 02:19 | All imports verified including `GRPOConfig`, `GRPOTrainer` |
| 02:19 | pytest 108/108 pass ✅ |
| 02:20 | fastText `lid.176.bin` downloaded (126MB) |
| 02:20 | Training tmux session launched |
| 02:21 | accelerate launch + GRPOTrainer init |
| 02:22 | vLLM rollout engine init, model weights loaded (3.35GB) |
| 02:22 | Flash Attention backend selected for vLLM |

Setup wallclock: **14 min, ~$0.35** (vast.ai A100 ≈ $1.50/h).

## Training config (Open-RS Exp2 verified)

```yaml
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
dataset: knoveleng/open-rs (7K, train split)
rewards: [correctness, format] weights [1.0, 1.0]
grpo:
  learning_rate: 1.0e-6
  num_generations: 6           # G
  max_prompt_length: 512
  max_completion_length: 3584   # Open-RS Exp2
  temperature: 0.7
  beta: 0.04                    # KL
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 16  # 1× A100 → eff batch 96 prompts
  max_steps: 100
  save_steps: 50                # Open-RS gating ckpt = step 50
  bf16: true
  use_vllm: true
  vllm_gpu_memory_utilization: 0.7
gating:
  benchmark: amc23
  ckpt_step: 50
  pass_band: [77.0, 83.0]
  fail_below: 70.0
```

## Training progress

⏳ **In progress** — see `phase7_training_metrics.csv` (TBD live update from tmux log).

Per-step timestamps + reward components will be filled here when training completes.

| Step | Wall-clock | Loss | Reward (mean) | Reward/correct | Reward/format | Generations OOM? |
|---|---|---|---|---|---|---|
| ... | TBD | | | | | |

## Results (pending training completion)

**ckpt-50 evaluation:**
- AMC23 pass@1: **TBD** (target [77, 83])
- AMC23 maj@4: TBD
- MATH-500 pass@1: TBD (Open-RS RS2 reported 85.4)
- AIME-2024 pass@1: TBD (Open-RS RS2 reported 30.0)

**ckpt-100 evaluation:** TBD

## Decision

⏳ **Pending eval results.** Decision tree:
- AMC23 ∈ [77, 83]: ✅ PASS — continue Idea 1 (full 9-cell sweep)
- AMC23 ∈ [70, 77): ⚠️ PARTIAL — debug tokenization/reward signal
- AMC23 < 70: ❌ FAIL — pivot to Idea 4 (LIMO-style SFT)

## Files synced back from Vast.ai

(Will list after training completes.)

## Issues encountered + fixes

1. **Python `<3.12` pin** in `pyproject.toml` blocked install. Fixed via `sed` to `<3.13`.
2. **No `python` command** — Vast.ai image only has `python3`. Created symlink `/usr/local/bin/python → python3`.
3. **vllm forced torch 2.5.1** instead of CLAUDE.md pin 2.4.1 — runtime-compatible, no code changes needed.
4. **sympy 1.13.1** instead of CLAUDE.md pin `<1.13` — math-verify 0.9.0 still works (verified). CLAUDE.md note overly cautious for old math-verify.
5. **`torch_dtype` deprecated warning** trong transformers 4.57 — non-blocking, will refactor in followup commit.
6. **vLLM warning:** "requested device cuda:0 also being used for training" — expected with `vllm_gpu_memory_utilization=0.7` (training gets remaining ~24GB).

## Cost tracker

| Item | Hours | $/hr | Cost |
|---|---|---|---|
| Setup + install | 0.25 | 1.50 | $0.38 |
| Training (W2 gating) | TBD | 1.50 | TBD |
| Eval (post-train) | TBD | 1.50 | TBD |
| **Total** | TBD | | TBD / $180 |
