# Phase 8 — Pivot Decision: "Open-RS Extended to Cross-Lingual"

**Decision date:** 2026-05-06 (UTC), after W2 gating retrain attempt failed.
**Status:** Restructuring paper plan để align 1-1 với Open-RS baseline.

---

## What we learned from Phase 7 (failed gating)

### Failure summary

W2 gating run (DeepSeek-R1-Distill-Qwen-1.5B + LoRA r=16, 50 steps GRPO):

| Benchmark | Open-RS RS2 reported | Our base eval | Our ckpt-50 |
|---|---|---|---|
| AMC23 pass@1 | 80% | **45%** | **32.5%** |
| MATH-500 pass@1 | 85.4% | 54% | 46.2% |
| AIME-2024 pass@1 | 30% | 16.7% | 3.3% |

**Two layers of bias detected:**

### Layer 1 — Eval-time bias (system prompt mismatch)

Our eval system prompt:
```
Solve the math problem step by step.
Put your reasoning in <think>...</think>
and final answer in <answer>...</answer>.
```

DeepSeek-R1-Distill native distribution: `<think>...</think>\n\nFinal: $\boxed{N}$` — **NO `<answer>` tag.**

Our prompt forces a format the model has never seen → distribution shift → 17pp drop on base alone (AMC23 62.9% reported → 45% ours).

### Layer 2 — Training-time bias (R2 format reward)

Our R2 regex: `<think>.+?</think>.*<answer>.+?</answer>`

DeepSeek-R1-Distill never matches → R2 = 0 throughout 50 steps → only R1 (correctness) drives gradient → high variance + push toward spurious patterns → model degrades from 45% → 32.5% AMC23.

### Root cause

We mixed two incompatible designs:
1. **Open-RS Stage-1 protocol** (DeepSeek base, no-SFT, native `<think>` format)
2. **Our paper format design** (`<think>` + `<answer>` tags, requires SFT to teach)

For W2 gating reproduction, must match Open-RS protocol exactly.

---

## Restructured paper plan: "Open-RS Extension"

### One concrete baseline

**Primary baseline:** Open-RS (arXiv:2503.16219, Knoveleng et al. 2025).

Paper claim:
> "Building on Open-RS RS2 (Knoveleng et al., 2025), we extend along three axes — training language, reward function, and base model — to study cross-lingual transfer of GRPO at sub-3B scale."

This gives reviewers a clean diff: every cell is "Open-RS + 1 axis change".

### Experimental matrix (6 cells)

| Cell | Base | Train data | Rewards | Note |
|---|---|---|---|---|
| **A1** | DeepSeek-R1-Distill-Qwen-1.5B | open-rs (EN, 7K) | acc + fmt | ✅ exact RS2 reproduction |
| **A2** | DeepSeek-R1-Distill-Qwen-1.5B | open-rs translated VI | acc + fmt | language axis |
| **A3** | DeepSeek-R1-Distill-Qwen-1.5B | open-rs (EN) | acc + fmt + lang | reward axis |
| **B1** | Qwen2.5-1.5B-Instruct | open-rs (EN) | acc + fmt | base-model axis |
| **B2** | Qwen2.5-1.5B-Instruct | open-rs translated VI | acc + fmt | language × base |
| **B3** | Qwen2.5-1.5B-Instruct | open-rs (EN) | acc + fmt + lang | reward × base |

All hyperparameters held identical to Open-RS Exp2.

### Key changes from original CLAUDE.md plan

| Aspect | Original | Restructured |
|---|---|---|
| Base models | Qwen2.5-1.5B/3B + Llama-3.2-3B (3) | DeepSeek-Distill + Qwen2.5-1.5B (2) |
| 3B/Llama support | Required | **Future work** |
| Pipeline | SFT → GRPO | **GRPO only** (matches Open-RS) |
| R2 format reward | `<think>.+</think>.+<answer>.+</answer>` | **Open-RS native: `<think>.+?</think>`** |
| R4 tag count | Required | **DROP** (not in Open-RS) |
| R5 lang-consistency | Cond C only | **Same** — paper contribution |
| Cells × seeds | 9 × 3 = 27 | **6 × 1 = 6** (Tier 1 arXiv); 6 × 3 if budget |
| Eval benchmarks | GSM8K + MATH-500 + AIME + MGSM + MSVAMP | Same + **AMC23** (matches Open-RS) |

### Budget revision

| Item | Original | Restructured |
|---|---|---|
| Setup + reproduce | $25 | $7.5 |
| Main training | $115 | $37.5 (6 cells × $6.25) |
| 3 seeds expansion | — | +$75 (Tier 2) |
| Eval | $36 | $6 (6 ckpts × 1h) |
| Buffer | $14 | $7 |
| **Tier 1 (arXiv)** | **$190** | **$58** |
| Tier 2 (Findings push) | — | $133 |

→ **Save $130** vs original. Tier 1 fits in $180 budget với buffer.

### Acceptance rate targets

| Venue | Tier 1 (6 cells × 1 seed) | Tier 2 (6 cells × 3 seeds) |
|---|---|---|
| arXiv preprint cs.CL | **~100%** | ~100% |
| Workshop (MRL@EMNLP, ML4LR@NeurIPS) | ~70% | ~85% |
| EMNLP/ACL Findings | ~30% | ~50% |
| Main ACL/NeurIPS | ~10% | ~25% |

**Strategy:** Ship Tier 1 to arXiv week 8. If results strong, Tier 2 expansion → Findings submission.

---

## Code fixes required (Phase 8 immediate)

### Fix 1: R2 format reward → Open-RS native

```python
# src/rewards/format.py
@register("format")
def r2_format(prompts, completions, **kwargs):
    # Open-RS native: model puts reasoning trong <think>...</think>
    pattern = re.compile(r"<think>.+?</think>", re.DOTALL)
    return [1.0 if pattern.search(c) else 0.0 for c in completions]
```

### Fix 2: System prompt → DeepSeek minimal

```python
# src/eval/prompts.py + src/trainers/dataset_utils.py
DEFAULT_SYSTEM_PROMPT_EN = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
```

(Match DeepSeek-R1 paper Appendix A.1 prompt.)

### Fix 3: Configs

`configs/reproduce_open_rs.yaml`: rewards = [correctness, format] only (drop length, tag).
`configs/grpo_*.yaml`: rewards = [correctness, format] (+ lang for enlang).

### Fix 4: Update tests

`tests/test_rewards_format.py`: update fixtures — remove `<answer>` tag expectation.

### Fix 5: Drop SFT step

`scripts/train_sft.sh`: keep but unused for now.
`src/trainers/grpo.py`: handle missing SFT checkpoint gracefully (already does — uses `model_name_or_path` if no SFT).

---

## Execution roadmap

| Phase | Action | Cost | Time |
|---|---|---|---|
| **8.1** (now) | Code fixes 1-4 | $0 | 30 min |
| **8.2** | Re-train A1 (reproduce Open-RS RS2) | $7.5 | 5h |
| **8.3** | Eval A1 — confirm AMC23 ∈ [77, 83] | $0.5 | 15 min |
| **8.4** | Translate open-rs dataset → VI cho A2/B2 | $1 | 30 min |
| **8.5** | Train A2, A3, B1, B2, B3 (5 cells parallel/serial) | $30 | 25h |
| **8.6** | Eval all 6 cells × 6 benchmarks (MGSM × 10 lang + AMC23 + MATH-500 + AIME + GSM8K + MSVAMP × 10 lang) | $5 | 4h |
| **8.7** | Aggregate, tables, figures, paper draft | $0 | 2 days |
| **8.8** | arXiv submit | $0 | 30 min |
| **TOTAL** | | **~$44** | ~4 days wallclock |

---

## Risks (open)

1. **A1 retrain still fails** — i.e., Open-RS protocol broken on our infra. Then either (a) deeper debug (eval pipeline still wrong somewhere), or (b) accept partial reproduction and document gap.
2. **VI translation quality** — auto-translate open-rs to VI may produce broken math symbols. Fallback: use existing 5CD-AI MetaMathQA-VI for A2/B2 (different dataset, but verified VI math).
3. **R5 lang-consistency drowns R1** — too strong → model outputs short VI text without reasoning. Mitigation: tune weight 0.5 vs 1.0.
4. **B-series Qwen2.5 doesn't have native `<think>` format** — may need different format reward, or shorter base eval to verify GRPO works at all.
5. **Vast.ai instance interrupted** — saved everything to Mac via auto-rsync. Can resume.

---

## Decision

**Proceed with Phase 8.1** — code fixes immediate, then A1 retrain.

If A1 hits AMC23 ∈ [70, 83] → ship full plan.
If A1 still fails → escalate (check raw responses, evaluate Open-RS released checkpoint as sanity).
