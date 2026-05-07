# Twitter/X thread template (post day 1 after arXiv accept)

Replace `<arxiv-id>` with the assigned arXiv ID after publication.

---

## Tweet 1/8 (hook + main finding)

🧵 New paper: training a 1.5B math reasoning model with GRPO on a single
A100 + LoRA ($20 budget).

Surprising finding: a "lang-consistency" reward that fires constant 1.0
on EN data (zero content signal!) recovers **+13.3pp on AIME-2024**.

[Image: Fig 3 — effect sizes bar chart]

[arXiv:<arxiv-id>]

## Tweet 2/8 (problem)

Open-RS reports 80% AMC23 with full-param GRPO on 4× A40s (~$200 budget).

Most independent researchers can't afford that. Q: what works on 1× GPU
+ LoRA + $20?

We tested 3 arms varying exactly one axis: training language or reward.

## Tweet 3/8 (setup)

Base: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Hyperparams: Open-RS RS2 verbatim (G=6, max_completion=3584, lr=1e-6,
50 steps)
LoRA: r=16, single A100 80GB

A1: EN training (open-rs 7K)
A2: VI training (5CD-AI MetaMathQA-VI 5K)
A3: EN training + R5 (fastText lang-consistency reward)

## Tweet 4/8 (Finding 1)

A1 (vanilla EN GRPO) hits the *expected* benchmark-specific overfit:

✅ AMC23: +7.5pp over base
❌ AIME-2024: -16.7pp ← collapses

Model concentrates on AMC-style intermediate reasoning, loses depth
needed for AIME.

[Image: Fig 1 — bar chart]

## Tweet 5/8 (Finding 2)

A2 (VI training) → AIME-2024 maj@8 stays AT base (33.3%). Different
training distribution naturally regularizes.

But VI translation infrastructure isn't free. Could we get the same
effect with EN data?

## Tweet 6/8 (Finding 3 — main contribution)

A3 (EN + R5) recovers AIME-2024 by +13.3pp over A1, achieves highest
mean accuracy (43.3%, vs 42.4% base).

Critically: R5 = 1.0 uniformly on EN training (model rarely
code-switches). It contributes ZERO content signal.

So why does it work?

## Tweet 7/8 (mechanism hypothesis)

We think it's PPO clipping geometry, not reward content:

- Mean reward shifts E[R₁+R₂] ≈ 0.2 → E[R₁+R₂+R₅] ≈ 1.2
- Advantage normalization (r-r̄)/std cancels mean shift
- BUT clip ratios + KL penalties are magnitude-sensitive

Auxiliary reward = implicit regularizer. Testable claim.

## Tweet 8/8 (limitations + close)

Honest limits:
- 1 seed (multi-seed extension coming)
- LoRA r=16 only (not full-param)
- 22.5pp gap from Open-RS reported (LoRA + 1 GPU constraint)
- AMC23 maj@4 has a sampling bug

Code, LoRA adapters, eval JSONs, training logs:
[github.com/anonymous/xling-grpo-sub3b]

[arXiv:<arxiv-id>]
