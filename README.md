# xling-grpo-sub3b

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20061328.svg)](https://doi.org/10.5281/zenodo.20061328)
[![tests](https://github.com/vudang4494/xling-grpo-sub3b/actions/workflows/test.yml/badge.svg)](https://github.com/vudang4494/xling-grpo-sub3b/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Paper License](https://img.shields.io/badge/Paper-CC--BY--4.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![AI-Assisted](https://img.shields.io/badge/AI--Assisted-Disclosed-yellow.svg)](#ai-assistance-disclosure)

Code, configs, and checkpoints for the paper:

> **Beyond English-Only GRPO: Training Language and Auxiliary Reward as
> Implicit Regularizers in Sub-3B Math Reasoning**
> *Vu Dang, 2026.*
> Zenodo: [doi:10.5281/zenodo.20061328](https://doi.org/10.5281/zenodo.20061328)
> · PDF: [paper/main.pdf](paper/main.pdf)

**v2 (May 2026):** Multi-seed extension with A4 constant-bias ablation and 10-language MGSM evaluation. The v1 single-seed findings have been weakened or refuted; this document reflects v2.

We compare four GRPO post-training arms at sub-3B scale on a single A100 + LoRA constraint, across three random seeds per primary arm.

Mean ± σ across 3 seeds (single seed for A4):

| Arm | Training data | Rewards | AMC23 | MATH-500 | AIME p@1 | AIME m@8 |
|---|---|---|---|---|---|---|
| Base (untrained) | — | — | 50.0 | 59.4 | 26.7 | 33.3 |
| **A1** (EN) | `knoveleng/open-rs` 7K | R1 + R2 | 56.7 ± 11.3 | 59.7 ± 1.7 | 14.4 ± 7.7 | 32.2 ± 6.9 |
| **A2** (VI) | `5CD-AI/Vietnamese-MetaMathQA` 5.2K | R1 + R2 | 56.7 ± 5.2 | 60.8 ± 1.0 | 20.0 ± 5.8 | 34.4 ± 1.9 |
| **A3** (EN+R5) | `knoveleng/open-rs` 7K | R1 + R2 + R5 | 57.5 ± 6.6 | 60.7 ± 0.6 | 21.1 ± 1.9 | **37.8 ± 1.9** |
| **A4** (EN+const) | `knoveleng/open-rs` 7K | R1 + R2 + 1.0 | 52.5 | 62.0 | 26.7 | 33.3 |

Three orthogonal findings (see `paper/main.pdf` for full discussion):

1. **Positive — A3 best on hardest benchmark.** A3 achieves the highest mean AIME-2024 maj@8 (37.8 ± 1.9%) and is the only arm with a positive Δ over the untrained base (+4.4pp), robust across three seeds.
2. **Methodological — Vanilla EN GRPO unstable.** A1 exhibits σ = 11.3pp seed variance on AMC-23 and consistently degrades AIME-2024 pass@1 (mean −12.2pp across 3/3 seeds), confirming that single-seed claims at sub-3B + LoRA scale are unreliable.
3. **Negative — No cross-lingual transfer.** On MGSM 10-language benchmark, all four arms converge to within ±0.5pp of the untrained base mean. Training-language choice does not produce cross-lingual transfer at this scale.

The A4 constant-bias ablation partially but not fully reproduces A3's effect on AIME-2024 maj@8, suggesting the mechanism is not purely reward-magnitude based; the precise decomposition remains an open question.

## Quick reproduction

Hardware required: 1× NVIDIA A100 80GB GPU.

```bash
git clone <this repo>
cd xling-grpo-sub3b
pip install -e ".[dev,analysis]"
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/

# Reproduce A1 (Open-RS RS2 setup)
bash scripts/reproduce_open_rs.sh

# Reproduce A2 (VI training)
bash scripts/train_a2_a3.sh a2

# Reproduce A3 (EN + R5)
bash scripts/train_a2_a3.sh a3
```

Each arm requires approximately 3–4 hours of training plus 15 minutes of evaluation.

## Repository layout

```
xling-grpo-sub3b/
├── README.md           — this file
├── LICENSE             — Apache-2.0
├── CITATION.cff        — citation metadata
├── CHANGELOG.md        — version history
├── CONTRIBUTING.md     — contribution guidelines
├── VERIFICATION.md     — per-component data audit trail
├── pyproject.toml      — Python deps (pinned versions)
├── Makefile            — reproduce commands
│
├── configs/            — YAML hyperparameters (one per arm + reproduce + eval)
├── data/               — dataset prep, decontamination scripts
├── docs/               — project docs (paper checklist, pipeline, submission guide)
├── paper/              — LaTeX source, figures, tables, compiled PDFs
│   ├── main.tex/.pdf   — primary manuscript (v2)
│   ├── ieee/           — IEEE Access submission version (v2)
│   ├── archive/        — v1 Phase 8 PDF (historical)
│   ├── tables/         — auto-generated LaTeX tables
│   └── figures/        — multi-seed figures (PDF)
├── reports/            — internal development notes (gitignored)
├── results/            — eval JSONs (in git) + LoRA adapters (local only)
├── scripts/            — training + eval + integrity pipeline scripts
├── src/
│   ├── rewards/        — R1 (correctness), R2 (format), R5 (lang), R_const
│   ├── trainers/       — TRL 0.15 GRPO + LoRA wrappers
│   ├── eval/           — vLLM eval adapters (AMC23, MATH-500, AIME, MGSM, ...)
│   └── analysis/       — aggregate.py, bootstrap.py, plot_curves.py
└── tests/              — pytest (104 tests pass on CPU-only env)
```

See [`docs/README.md`](docs/README.md) for project documentation index.

## Reward definitions

- **R1 (correctness):** Math-Verify equivalence on extracted answer.
- **R2 (format):** Regex match `<think>...</think>...<answer>...</answer>`.
- **R5 (lang-consistency):** fastText langID match between prompt and completion. Fires on every prompt (`no_penalty_for_en=False`).

## Eval template

Verbatim Open-RS `lighteval` `MATH_QUERY_TEMPLATE` (no system prompt; template embedded in user message). Greedy decoding, `max_tokens=8192`, no early stop.

## Citation

```bibtex
@misc{dang2026beyond,
  title  = {Beyond English-Only GRPO: Training Language and Auxiliary Reward
            as Implicit Regularizers in Sub-3B Math Reasoning},
  author = {Dang, Vu},
  year   = {2026},
  doi    = {10.5281/zenodo.20061328},
  url    = {https://doi.org/10.5281/zenodo.20061328},
  publisher = {Zenodo}
}
```

## License

Code: Apache-2.0 (see `LICENSE`).
LoRA adapters: same.
Eval JSONs (with `responses[]` arrays): CC BY 4.0.

## Acknowledgments

This work builds directly on:
- Open-RS ([Knoveleng et al., 2025](https://arxiv.org/abs/2503.16219)) — protocol, hyperparameters, RS2 reward setup, eval template, and the AMC-23 dataset release.
- DeepSeek-R1 ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)) — base distilled checkpoint and GRPO recipe.
- TRL ([Hugging Face](https://github.com/huggingface/trl)) v0.15.2 — `GRPOTrainer` with in-process vLLM rollouts.
- 5CD-AI Vietnamese AI team — Vietnamese MetaMathQA translation.

## AI Assistance Disclosure

This project was developed with assistance from Anthropic Claude (via the
[Claude Code](https://claude.com/claude-code) command-line interface).
AI assistance included drafting and refactoring of source code, manuscript
composition, literature exploration, and LaTeX formatting.

**Human accountability.** All experimental results were produced by training
and evaluation runs that the author launched, monitored, and verified
independently. All cited references were checked against original sources.
The author bears sole responsibility for the technical claims, methodology,
and any errors in this work.

This disclosure complies with the
[ICLR 2026 LLM Policy](https://blog.iclr.cc/2025/08/26/policies-on-large-language-model-usage-at-iclr-2026/),
NeurIPS Ethics Guidelines, and IEEE Access submission requirements regarding
the use of large language models in academic work. See `VERIFICATION.md`
for the per-component verification record.
