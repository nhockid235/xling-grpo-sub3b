# Contributing

Thanks for your interest in this project. The repository accompanies a research
paper, so contributions fall into a few well-defined categories.

## What's welcome

- **Bug reports** for the training, evaluation, or analysis code (`src/`,
  `data/`, `scripts/`).
- **Reproduction reports**: ran the pipeline at a different scale or with
  different hyperparameters? Open an issue with your results, and link to a
  branch / fork with the diff.
- **Eval extensions**: new benchmark adapters in `src/eval/` (follow the
  existing pattern from `gsm8k.py` / `math500.py`).
- **Reward function variants**: new functions registered via
  `src/rewards/__init__.py::register("name")`.
- **Documentation fixes**: typos, clearer wording, additional examples.

## What's out of scope

- Renaming hyperparameters or reward names that are referenced verbatim in the
  paper (would break reproduction).
- Removing or rewriting the `reports/phase*` files (they are the durable
  research log for the paper appendix).

## Workflow

1. **Fork** the repo and create a branch:
   `git checkout -b fix/short-description`.
2. **Run tests locally**:
   ```bash
   pip install -e ".[dev]"
   pytest tests/ -v
   ```
   The CPU-only subset (matching the GitHub Actions CI) skips four tests
   that require `data/raw/lid.176.bin`. Download that file (~131 MB) if you
   want to run R5 lang-consistency tests:
   ```bash
   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -P data/raw/
   ```
3. **Commit** with a message that describes *why*, not just *what*.
4. **Open a PR** referencing the issue (if any). Keep the diff small.

## Code style

- Python 3.11 / 3.12 (3.11 in CI).
- Type hints required for public functions in `src/`.
- Comments: Vietnamese for inline code explanation is fine; English for
  docstrings (the rest of the world reads them).
- Run `ruff check src/ scripts/ data/ tests/` before submitting.

## Reward function contract

Reward functions in `src/rewards/` must follow the TRL 0.15 `GRPOTrainer`
signature exactly:

```python
def reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]
```

Length of returned list must equal `len(prompts)`. Returning a scalar, tensor,
or numpy array silently breaks reward aggregation in TRL.

## Releasing checkpoints

LoRA adapters (~17 MB each) live in `results/training/grpo/`. Larger
checkpoints (full-param models, optimizer states) should be released on
Hugging Face Hub rather than committed here.

## Questions

Open a GitHub Discussion for high-level questions, or email the maintainer
(see `CITATION.cff`).
