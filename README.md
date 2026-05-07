# xling-grpo-sub3b

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20061328.svg)](https://doi.org/10.5281/zenodo.20061328)
[![tests](https://github.com/nhockid235/xling-grpo-sub3b/actions/workflows/test.yml/badge.svg)](https://github.com/nhockid235/xling-grpo-sub3b/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![Paper License](https://img.shields.io/badge/Paper-CC--BY--4.0-orange.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Code, configs, and checkpoints for the paper:

> **Beyond English-Only GRPO: Training Language and Auxiliary Reward as
> Implicit Regularizers in Sub-3B Math Reasoning**
> *Vu Dang, 2026.*
> Zenodo: [doi:10.5281/zenodo.20061328](https://doi.org/10.5281/zenodo.20061328)
> · arXiv: TBD (pending endorsement)
> · PDF: [paper/main.pdf](paper/main.pdf)

We compare three GRPO post-training arms at sub-3B scale on a single A100 + LoRA budget:

| Arm | Training data | Rewards | AMC23 pass@1 | AIME-2024 pass@1 |
|---|---|---|---|---|
| Base (untrained) | — | — | 50.0% | 26.7% |
| **A1** (EN) | `knoveleng/open-rs` 7K | R1 + R2 | **57.5%** | 10.0% |
| **A2** (VI) | `5CD-AI/Vietnamese-MetaMathQA` 5.2K | R1 + R2 | 52.5% | 16.7% |
| **A3** (EN+R5) | `knoveleng/open-rs` 7K | R1 + R2 + R5 | 52.5% | **23.3%** |

Three findings (see `paper/main.tex` for full discussion):

1. **English-only GRPO causes benchmark-specific overfit** — A1 gains +7.5pp on AMC23 but loses 16.7pp on AIME-2024.
2. **Vietnamese training acts as a regularizer** — A2 preserves AIME-2024 maj@8 at base level (33.3%).
3. **Lang-consistency reward is an implicit regularizer** — A3's R5 fires uniformly at 1.0 on English data (no content signal) yet recovers 13.3pp on AIME-2024.

## Quick reproduction

Hardware: 1× NVIDIA A100 80GB. Total budget: ~$20 GPU.

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

Each arm: ~3-4h training + 15min eval = ~$5-7 on Vast.ai A100 spot.

## Repository layout

```
xling-grpo-sub3b/
├── configs/        — YAML hyperparameters (one per arm + reproduce + eval)
├── data/           — dataset prep, decontamination
├── src/
│   ├── rewards/    — R1 (correctness), R2 (format), R5 (lang) + tests
│   ├── trainers/   — TRL 0.15 GRPO + LoRA wrappers
│   ├── eval/       — vLLM-based eval (AMC23, MATH-500, AIME-2024, ...)
│   └── analysis/   — aggregate.py, bootstrap.py, plot_curves.py
├── scripts/        — train + eval shell launchers
├── tests/          — pytest (108 tests pass on CPU-only env)
├── paper/          — LaTeX source, figures, tables, build script
└── results/        — checkpoints, eval JSONs, master.csv (released)
```

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

(arXiv preprint pending endorsement; will add `eprint` field once published.)

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
