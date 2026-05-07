# Phase 0 — Open-RS Reproduction Guide

**Verify date:** 2026-05-05
**Researcher:** `general-purpose` subagent
**Source authority:** arXiv:2503.16219 + GitHub `knoveleng/open-rs` repo + HF model/dataset cards
**Scope:** Hyperparams + setup chính xác để reproduce Open-RS baseline (W2 gating)

---

## Verbatim agent output

> Note: Table 5 content didn't render in HTML. The repo `grpo.yaml` is the canonical source — paper appendix references back to it. KL beta is not in the YAML, so it uses TRL `GRPOConfig` default (`beta=0.04`).

### Open-RS Stage-1 reproduction guide

**Critical correction up front:** Open-RS does **NOT** use Qwen2.5-Math-1.5B. The base model is **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`**. Also, "Stage-1" terminology is misleading — the paper runs **3 parallel Experiments (Exp1/2/3)**, producing checkpoints **Open-RS1/2/3**. The 80% AMC23 number you must match comes from **Open-RS2 @ 50 steps** (Exp 2), not "Stage-1".

### Open-RS hyperparams (verified from `recipes/grpo.yaml` + paper §3.5)

| Param | Value | Source |
|---|---|---|
| `model_name_or_path` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | [grpo.yaml](https://raw.githubusercontent.com/knoveleng/open-rs/main/recipes/grpo.yaml) |
| `dataset_name` | `knoveleng/open-rs` (7,000 rows, single `train` split) | grpo.yaml + HF |
| `learning_rate` | `1.0e-06` | grpo.yaml |
| `lr_scheduler_type` | `cosine_with_min_lr` (`min_lr_rate: 0.1`) | grpo.yaml |
| `warmup_ratio` | `0.1` | grpo.yaml |
| `max_steps` | `500` (Exp1); paper picks ckpts @ **100 / 50 / 50** for RS1/RS2/RS3 | grpo.yaml + paper §3.5.4 |
| `num_generations` (G) | **6** (not 4 or 8) | grpo.yaml |
| `temperature` | `0.7` (rollouts) | grpo.yaml |
| `max_prompt_length` | `512` | grpo.yaml |
| `max_completion_length` | **`3584`** (Exp2/3, used in repo); Exp1 used 4096 | grpo.yaml + paper §3.5.1 |
| `vllm_max_model_len` | `4608` | grpo.yaml |
| `per_device_train_batch_size` | `6` | grpo.yaml |
| `gradient_accumulation_steps` | `4` | grpo.yaml |
| Effective batch (4 GPU × 6 × 4) | **96 prompts/step** (× G=6 = 576 generations) | derived |
| `beta` (KL) | **not set in YAML → TRL default `0.04`** | TRL `GRPOConfig` |
| `num_train_epochs` | `1` | grpo.yaml |
| `seed` | `42` | grpo.yaml |
| `bf16` | `true`, `attn_implementation: flash_attention_2` | grpo.yaml |
| `use_vllm` | `true` (`gpu_memory_utilization: 0.7`, `enforce_eager: true`) | grpo.yaml |
| `gradient_checkpointing` | `true` (`use_reentrant: false`) | grpo.yaml |
| LoRA | **None — full-parameter** | repo (no PEFT config) |
| `reward_funcs` | **`[format, cosine]`** with weights **`[1.0, 2.0]`** (this is Exp3 / RS3 setup) | grpo.yaml |

**Reward variations across the 3 experiments** (paper §3.5):
- Exp 1 (RS1): `accuracy + format`, `max_completion=4096`, dataset = open-s1 (18,615) — uses 100 ckpt steps
- Exp 2 (RS2): `accuracy + format`, `max_completion=3584`, dataset = `knoveleng/open-rs` 7K mix — 50 ckpt steps
- Exp 3 (RS3): `cosine + format` (cosine replaces accuracy), same 7K mix — 50 ckpt steps **← what `grpo.yaml` ships**

### Reported numbers (paper Table 1, zero-shot pass@1)

| Benchmark | Base (R1-Distill-Qwen-1.5B) | Open-RS1 (Exp1, 100s) | Open-RS2 (Exp2, 50s) | Open-RS3 (Exp3, 50s) |
|---|---|---|---|---|
| AIME24 | 28.8 | 30.0 | 30.0 | **46.7** |
| MATH-500 | 82.8 | 83.8 | 85.4 | 84.4 |
| **AMC23** | **62.9** | 70.0 | **80.0** | 72.5 |
| Minerva | 26.5 | 29.0 | 30.5 | 26.8 |
| OlympiadBench | 43.3 | 52.4 | 52.4 | 51.3 |
| Avg | 48.9 | 53.0 | 55.7 | 56.3 |

GSM8K is **not** evaluated in the Open-RS paper. Hardware: **4× A40 48GB**, 24h, ~$42, full-param. They use vLLM for rollouts + lighteval for eval.

### Dataset schema (`knoveleng/open-rs`)

Single `train` split, 7,000 rows. Composition: 3K hard from `open-s1` + 3K hard from `open-deepscaler` + 1K easy from raw DeepScaleR. Fields: `problem` (str), `solution` (str), `answer` (str), `level` (str: `"Hard"` ~6K / `"Easy"` ~1K). No Math-difficulty levels 1–5.

### Gotchas khi reproduce (W2 gating)

- **Base model là `DeepSeek-R1-Distill-Qwen-1.5B` chứ không phải `Qwen2.5-1.5B-Instruct` hay `Qwen2.5-Math-1.5B`.** AMC23 baseline 62.9% chỉ đúng với checkpoint distill này. Nếu bạn train từ `Qwen2.5-1.5B-Instruct` thì baseline AMC23 ~30% và không kỳ vọng vọt 80%.
- **Để match 80% AMC23, bạn phải reproduce Exp2 (RS2)**: reward = `accuracy + format` (NOT `cosine + format` mặc định trong YAML), `max_completion=3584`, lấy ckpt **50 global steps**. YAML mặc định trong repo là Exp3 setup, sẽ ra ~72.5% AMC23 (RS3).
- **Effective batch = 4 GPU × 6 batch × 4 grad_accum = 96 prompts**. Trên 1×A100 80GB phải tăng grad_accum lên 16 để match.
- **G=6, không phải G=4** (Open-RS chọn 6 cho A40 48GB). G=6 với 1.5B + max_len=3584 cần ~40GB; OK trên A100.
- **KL beta không khai báo trong YAML → mặc định TRL `0.04`**. CLAUDE.md của bạn ghi `kl_beta=0.04` nên trùng. Confirm khi load `GRPOConfig` (TRL 0.13 default đã đổi qua nhiều version, pin lại).
- **Peak performance trong 50-100 steps rồi degrade** — paper báo cáo length drift + multilingual drift sau 150-200 steps. Nhớ save ckpt mỗi 50 steps (`save_steps: 50`) và đánh giá nhiều ckpt, đừng chỉ lấy step cuối.
- **Reward weights `[1.0, 2.0]` ưu tiên cosine 2× format** — không phải sum đều. Khi swap về `accuracy+format` cho RS2, bạn nên dùng `[1.0, 1.0]` (paper không chỉ trọng số khác).
- **System prompt cố định** trong YAML có chỉ thị "respond by English, NOT use other languages" — đây là Exp3 hack thay cho lang-consistency reward. Nếu bạn muốn lang-consistency thực sự (Cond C của bạn), giữ system prompt EN nhưng thêm R5 fastText reward.
- **`vllm_enforce_eager: true`** — chậm hơn nhưng tránh CUDA graph OOM trên A40. Trên A100 80GB có thể tắt để tăng tốc rollout.
- **Checkpoint paper @ 50 steps là ~30 phút training** trên 4×A40. Trên 1×A100 ratio steps/wallclock sẽ khác — đừng so wallclock, so global step count.
- **lighteval với custom tasks** trong `src/open_r1/evaluate.py` — dùng đúng template này để con số AMC23 so sánh được; lighteval defaults khác sẽ ra điểm lệch 2-3pp.

### Sources

- [arXiv 2503.16219 (HTML)](https://arxiv.org/html/2503.16219)
- [knoveleng/open-rs GitHub](https://github.com/knoveleng/open-rs)
- [recipes/grpo.yaml raw](https://raw.githubusercontent.com/knoveleng/open-rs/main/recipes/grpo.yaml)
- [knoveleng/open-rs HF dataset](https://huggingface.co/datasets/knoveleng/open-rs)
- [knoveleng/OpenRS-GRPO HF model](https://huggingface.co/knoveleng/OpenRS-GRPO)

---

## Phân tích & action items (main agent)

### Decision: target Exp2 (RS2) để gating, không Exp3

| Tiêu chí | Exp2 (RS2) | Exp3 (RS3) |
|---|---|---|
| AMC23 target | **80.0%** ← chosen | 72.5% |
| Reward setup | `accuracy + format`, weights `[1.0, 1.0]` | `cosine + format`, weights `[1.0, 2.0]` |
| max_completion | 3584 | 3584 |
| Ckpt | step 50 | step 50 |
| Lý do chọn | Số AMC23 cao hơn, dễ argue "pipeline works"; reward đơn giản hơn để debug | — |

### Mapping Open-RS hyperparams → CLAUDE.md GRPO defaults

Đã capture trong `phase0_00_summary` Edit 2. Toàn bộ hyperparams set theo Open-RS Exp2 (chứ không phải Exp3 default trong YAML), với 2 điểm khác nhỏ:
1. `gradient_accumulation_steps`: 4 → 16 (vì 1× A100 thay vì 4× A40 — phải scale eff batch)
2. `vllm_enforce_eager`: true → false (A100 80GB đủ VRAM)

### Reproduction script template

```bash
# scripts/reproduce_open_rs.sh — W2 gating run
#!/bin/bash
set -e

RUN_ID="reproduce_openrs_rs2_42"
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATASET="knoveleng/open-rs"
OUTPUT="results/grpo/${RUN_ID}"

accelerate launch \
  --config_file configs/accelerate_single_a100.yaml \
  src/trainers/grpo.py \
    --model_name_or_path "${MODEL}" \
    --dataset_name "${DATASET}" \
    --reward_funcs accuracy format \
    --reward_weights 1.0 1.0 \
    --num_generations 6 \
    --max_prompt_length 512 \
    --max_completion_length 3584 \
    --temperature 0.7 \
    --beta 0.04 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_ratio 0.1 \
    --max_steps 100 \
    --save_steps 50 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
    --bf16 \
    --gradient_checkpointing \
    --use_vllm \
    --vllm_gpu_memory_utilization 0.7 \
    --output_dir "${OUTPUT}" \
    --seed 42

# Eval ckpt-50
bash scripts/eval_amc23.sh "${OUTPUT}/checkpoint-50"
# expected: AMC23 pass@1 ∈ [77, 83]  (80.0 ± 3pp)
```

### Cost ước tính cho gating run

- 1× A100 80GB on RunPod, $1.19/hr
- Open-RS Exp2 = 50 steps; trên 1× A100 với grad_accum=16 → ~3-4× chậm hơn 4× A40 setup
- Wallclock estimate: 50 steps × 6 prompts × 6 generations × 3.5K tokens ≈ 6-8 hours
- **Cost: ~$10-12 cho gating run** (thuộc Task 1.2 budget $12)

### Decision tree khi run

```
Run reproduce_openrs_rs2_42 → eval AMC23 ckpt-50

├── AMC23 ∈ [77, 83]: ✅ PASS gating
│   └── Continue Idea 1 (full plan: 9 cells × 3 seeds)
│
├── AMC23 ∈ [70, 77): ⚠️ PARTIAL
│   ├── Try ckpt-100 (Exp1 setting): if better, switch to RS1 reproduction
│   └── Else debug: tokenization, reward signal, lr, vLLM rollout sanity
│
└── AMC23 < 70: ❌ FAIL
    └── Pivot to Idea 4 (LIMO-style SFT)
```

### Risk còn open

1. **Reward function `accuracy` / `cosine` của Open-RS:** chưa inspect source `src/open_r1/rewards/`. Phải verify Math-Verify implementation Open-RS dùng có khác Math-Verify upstream không. Sẽ check tại Phase 2 (rewards agent).
2. **System prompt:** Open-RS có system prompt cố định trong `grpo.yaml` (chứa hint "answer in English"). Nếu reproduce bỏ system prompt này → có thể lệch điểm. Phải copy verbatim từ YAML.
3. **lighteval custom tasks:** chưa inspect template eval của Open-RS. Khả năng cao Open-RS dùng prompt eval khác lighteval default → số AMC23 lệch. Mitigation: dùng đúng `src/open_r1/evaluate.py` của Open-RS thay vì viết lại từ đầu cho gating run.
4. **Grad accum scaling:** scaling từ 4 GPU → 1 GPU bằng grad_accum không hoàn toàn equivalent (effective LR/momentum khác). Có thể cần điều chỉnh LR nhỏ về 0.5e-6.
