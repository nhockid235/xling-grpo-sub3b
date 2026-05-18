# Beyond English-Only GRPO

### Training Language and Auxiliary Reward as Implicit Regularizers in Sub-3B Math Reasoning

**Vu Dang** · Independent Researcher · [`vu.dh4494@gmail.com`](mailto:vu.dh4494@gmail.com) · ORCID [0009-0005-2344-1030](https://orcid.org/0009-0005-2344-1030)

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/main.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20061328.svg)](https://doi.org/10.5281/zenodo.20061328)
[![tests](https://github.com/vudang4494/xling-grpo-sub3b/actions/workflows/test.yml/badge.svg)](https://github.com/vudang4494/xling-grpo-sub3b/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Code License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Paper License](https://img.shields.io/badge/Paper-CC--BY--4.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![AI Disclosure](https://img.shields.io/badge/AI--Assisted-Disclosed-yellow.svg)](#ai-assistance-disclosure)

---

## TL;DR

A multi-seed empirical study of Group Relative Policy Optimization (GRPO)
post-training on a **1.5B distilled reasoning model** under a **single A100 +
LoRA `r=16`** budget. Across **three seeds × four arms**, we find:

- ✅ **A3 (English + language-consistency reward `R5`)** achieves the
  highest mean AIME-2024 `maj@8` (**37.8 ± 1.9 %**, +4.4 pp over the
  untrained base), the only arm with a positive Δ on the hardest benchmark.
- ⚠️ **Vanilla English GRPO (A1)** exhibits **σ = 11.3 pp** seed variance
  on AMC-23 — single-seed claims at this scale are unreliable.
- ❌ **No cross-lingual transfer** on 10-language MGSM: all four arms
  converge to within ±0.5 pp of the untrained base mean.
- 🔬 The mechanism behind `R5` is **content-specific, not pure
  reward-magnitude**: a constant-bias control (A4) fails to reproduce A3's
  effect (CI [+1.13, +11.10] excludes 0).

> 📄 Full paper: [`paper/main.pdf`](paper/main.pdf) (11 pages, ACL/arXiv style) ·
> IEEE Access rendering: [`paper/ieee/main.pdf`](paper/ieee/main.pdf)

## Table of contents

- [Headline results](#headline-results)
- [Findings](#findings)
- [Quick reproduction](#quick-reproduction)
- [Hardware and software](#hardware-and-software)
- [Repository layout](#repository-layout)
- [Reward functions](#reward-functions)
- [Evaluation protocol](#evaluation-protocol)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [AI assistance disclosure](#ai-assistance-disclosure)

## Headline results

Mean ± σ across three seeds (`42, 123, 7`). Cells without σ are
single-seed (`Base` and `A4` AMC-23 `pass@1` only).

| Arm | Training data | Rewards | AMC-23 `pass@1` | AMC-23 `maj@4` | MATH-500 | AIME `pass@1` | AIME `maj@8` |
|---|---|---|---|---|---|---|---|
| Base (untrained) | — | — | 50.0 | 70.0 | 59.4 | 26.7 | 33.3 |
| **A1** — English-only | `knoveleng/open-rs` (7 K) | R1 + R2 | 56.7 ± 11.3 | 70.0 ± 4.3 | 59.7 ± 1.7 | 14.4 ± 7.7 | 32.2 ± 6.9 |
| **A2** — Vietnamese | `5CD-AI/Vietnamese-MetaMathQA` (5.2 K) | R1 + R2 | 56.7 ± 5.2 | 70.0 ± 2.5 | 60.8 ± 1.0 | 20.0 ± 5.8 | 34.4 ± 1.9 |
| **A3** — English + `R5` | `knoveleng/open-rs` (7 K) | R1 + R2 + R5 | 57.5 ± 6.6 | 68.3 ± 3.8 | 60.7 ± 0.6 | 21.1 ± 1.9 | **37.8 ± 1.9** |
| **A4** — English + constant bias | `knoveleng/open-rs` (7 K) | R1 + R2 + 1.0 | 52.5 | 70.0 ± 2.5 | 61.2 ± 0.9 | 24.4 ± 1.9 | 32.2 ± 5.1 |

**Base model:** [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
· **GRPO recipe:** Open-RS RS2 verbatim
· **LoRA:** `r=16, α=32`, target `{q,k,v,o}_proj`
· **Steps:** 50 · **Eval:** Open-RS `lighteval` `MATH_QUERY_TEMPLATE`,
greedy decoding (`T=0`, `max_tokens=8192`).

## Findings

1. **A3 is best on the hardest benchmark (positive).** A3 achieves the
   highest mean AIME-2024 `maj@8` (37.8 ± 1.9 %) and is the only arm with
   a positive Δ over the untrained base (+4.4 pp), robust across three
   seeds.
2. **Vanilla English GRPO is unstable (methodological).** A1 exhibits
   σ = 11.3 pp seed variance on AMC-23 and consistently degrades
   AIME-2024 `pass@1` (mean −12.2 pp across 3/3 seeds), confirming that
   single-seed claims at sub-3B + LoRA scale are unreliable.
3. **No cross-lingual transfer (negative).** On the MGSM 10-language
   benchmark, all four arms converge to within ±0.5 pp of the untrained
   base mean. Training-language choice does not produce cross-lingual
   transfer at this scale.

A constant-bias ablation (`A4`) **fails to reproduce** A3's effect on
AIME-2024 `maj@8` (A3 − A4 = +5.58 pp, 95 % bootstrap CI
[+1.13, +11.10], excluding 0), demonstrating that `R5`'s contribution is
**content-specific** rather than a pure reward-magnitude perturbation.

See [`paper/main.pdf`](paper/main.pdf) for the full discussion, including
the Preliminaries on the GRPO advantage-invariance identity and the
PPO-clipping-geometry hypothesis that A4 refutes.

## Quick reproduction

```bash
# 1. Clone and install (Python 3.11 or 3.12)
git clone https://github.com/vudang4494/xling-grpo-sub3b.git
cd xling-grpo-sub3b
pip install -e ".[dev,analysis]"

# 2. Download the fastText language identifier (used by R5)
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
     -P data/raw/

# 3. Train each arm (≈ 3–4 hours per seed on 1× A100 80 GB)
bash scripts/reproduce_open_rs.sh                # A1: Open-RS RS2 setup
bash scripts/train_a2_a3.sh a2                   # A2: Vietnamese-translated
bash scripts/train_a2_a3.sh a3                   # A3: English + R5
bash scripts/w19_train_a4_multiseed.sh           # A4: constant-bias, 3 seeds

# 4. Evaluate on AMC-23 / MATH-500 / AIME-2024 / MGSM
bash scripts/eval_all.sh

# 5. Build the paper PDF
brew install tectonic                            # macOS, one-time
cd paper && tectonic -X compile main.tex
```

Three seeds (`42, 123, 7`) per arm total ≈ 10–12 A100-hours of training
plus ≈ 4 A100-hours of evaluation across the full benchmark suite.

## Hardware and software

| Component | Used | Minimum |
|---|---|---|
| **GPU** | 1× NVIDIA A100-SXM4-80 GB | 1× ≥ 40 GB VRAM with FlashAttention 2 |
| **CPU RAM** | 64 GB | 32 GB |
| **Disk** | ~ 80 GB | 40 GB (model + adapters + eval JSONs) |
| **Python** | 3.12.3 | 3.11 |

Tested with the pinned dependencies in [`pyproject.toml`](pyproject.toml):

```text
torch==2.5.1+cu124   transformers==4.49.0   trl==0.15.2
vllm==0.7.2          accelerate==1.2.1      peft==0.14.0
datasets==4.8.5      flash_attn==2.7.2.post1
sympy==1.13.1        math_verify==0.9.0     fasttext-wheel==0.9.2
```

## Repository layout

```text
xling-grpo-sub3b/
├── README.md              — this file
├── LICENSE                — Apache-2.0
├── CITATION.cff           — citation metadata
├── CONTRIBUTING.md        — contribution guidelines
├── VERIFICATION.md        — per-component data audit trail
├── pyproject.toml         — Python dependencies (pinned)
├── Makefile               — reproduce shortcuts
│
├── configs/               — YAML hyperparameters (one per arm + reproduce + eval)
├── data/                  — dataset prep and decontamination scripts
├── docs/                  — project documentation (paper checklist, pipeline, submission guide)
├── paper/                 — LaTeX source, figures, tables, compiled PDFs
│   ├── main.{tex,pdf}     — primary manuscript (ACL/arXiv style)
│   ├── ieee/              — IEEE Access rendering of the same paper
│   ├── tables/            — LaTeX tables (table_main.tex, table_delta.tex, table_mgsm.tex)
│   └── figures/           — figures (PDF) + make_figures.py
├── results/               — eval JSONs (in git) and LoRA adapters (local-only)
├── scripts/               — training, evaluation, and integrity pipelines
├── src/
│   ├── rewards/           — R1 (correctness), R2 (format), R5 (lang), R_const
│   ├── trainers/          — TRL 0.15 GRPO + LoRA wrappers
│   ├── eval/              — vLLM eval adapters (AMC-23, MATH-500, AIME-2024, MGSM)
│   └── analysis/          — aggregate.py, bootstrap.py, plot_curves.py
└── tests/                 — pytest (CPU-safe subset runs in CI)
```

See [`docs/README.md`](docs/README.md) for the full documentation index.

## Reward functions

| Reward | Definition | Used by |
|---|---|---|
| **R1** — correctness | Indicator of [Math-Verify](https://github.com/huggingface/Math-Verify) equivalence between the extracted answer and the gold. Extraction priority: `<answer>` tag → `\boxed{}` → last-number fallback. | All arms |
| **R2** — format | Indicator that the completion matches a regex requiring both `<think>...</think>` and `<answer>...</answer>` blocks (in order). | All arms |
| **R5** — language consistency | Indicator that fastText `lid.176.bin` predicts the same language for the prompt and the completion (guard: completions shorter than 10 tokens score 0). Fires on every prompt (`no_penalty_for_en=False`). | A3 only |
| **R<sub>const</sub>** | Deterministic constant reward of 1.0 on every input. Used to test the reward-magnitude hypothesis. | A4 only |

## Evaluation protocol

- **Template:** Open-RS [`lighteval`](https://github.com/huggingface/lighteval)
  `MATH_QUERY_TEMPLATE` verbatim. No system prompt; the evaluation
  template is embedded in the user message.
- **Decoding:** Greedy (`T=0`) for `pass@1`. `T=0.7`, `top_p=0.95` for
  `maj@k` with seeds `42, 43, …` for the `k` stochastic samples.
- **Stop:** `max_tokens=8192`, no early stop.
- **Bootstrap CIs:** 10 000 subject-bootstrap resamples over seed-level
  means, fixed RNG seed. Implementation in
  [`scripts/w110_bootstrap_ci.py`](scripts/w110_bootstrap_ci.py).

## Citation

```bibtex
@misc{dang2026beyond,
  title     = {Beyond English-Only {GRPO}: Training Language and Auxiliary
               Reward as Implicit Regularizers in Sub-3{B} Math Reasoning},
  author    = {Dang, Vu},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20061328},
  url       = {https://doi.org/10.5281/zenodo.20061328}
}
```

Machine-readable citation metadata is also provided in
[`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button using it).

## License

- **Code** (this repository): [Apache-2.0](LICENSE).
- **LoRA adapters:** Apache-2.0 (same as code).
- **Manuscript and figures** (`paper/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Evaluation JSONs** (with `responses[]` arrays): CC BY 4.0.
- **Upstream artifacts** retain their original licenses
  (see [Acknowledgments](#acknowledgments)).

## Acknowledgments

This work builds directly on:

- **Open-RS** ([Dang & Ngo, 2025](https://arxiv.org/abs/2503.16219)) —
  GRPO protocol, hyperparameters, RS2 reward setup, evaluation template,
  and the AMC-23 dataset release.
- **DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)) —
  base distilled checkpoint and the original GRPO recipe at scale.
- **TRL** ([Hugging Face](https://github.com/huggingface/trl)) v0.15.2 —
  `GRPOTrainer` with in-process vLLM rollouts.
- **vLLM** ([Kwon et al., 2023](https://arxiv.org/abs/2309.06180)) v0.7.2 —
  PagedAttention rollout engine.
- **Math-Verify** ([Hugging Face](https://github.com/huggingface/Math-Verify)) —
  symbolic equivalence checker that backs `R1`.
- **fastText** ([Joulin et al., 2017](https://arxiv.org/abs/1607.01759)) —
  `lid.176.bin` language identifier underlying `R5`.
- **5CD-AI** Vietnamese AI team — Vietnamese MetaMathQA translation.

## AI assistance disclosure

This project was developed with assistance from Anthropic Claude (via the
[Claude Code](https://claude.com/claude-code) command-line interface).
AI assistance covered drafting and refactoring of source code, manuscript
composition, literature exploration, and LaTeX formatting.

**Human accountability.** All experimental results were produced by
training and evaluation runs that the author launched, monitored, and
verified independently. All cited references were checked against
original sources. The author bears sole responsibility for the technical
claims, methodology, and any errors in this work.

This disclosure complies with the
[ICLR 2026 LLM Policy](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/),
NeurIPS Ethics Guidelines, and IEEE Access submission requirements
regarding the use of large language models in academic work. See
[`VERIFICATION.md`](VERIFICATION.md) for the per-component verification
record.
