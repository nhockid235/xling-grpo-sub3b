# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### In progress (W1, 2026-05-15)
- W1.7 re-eval all 10 LoRAs × seeds on AMC-23 (post-fix maj@4) — running on A100-SXM4-80GB at vast.ai.
- W1.9 A4 multi-seed expansion (seeds 123, 7) — queued after W1.7.
- W1.10 bootstrap 95% CI for AIME-2024 m@8 headline — partial run on existing data, A3 vs Base CI [+3.3, +6.7], significant.
- W20 update `paper/tables/table_main_v2.tex` + `table_delta_v2.tex` with post-fix maj@4 numbers.

### Fixed (2026-05-15)
- AMC23 `maj@4` sampling-bug: `configs/eval.yaml` now sets `temperature_maj=0.7`; `src/eval/_common.py::extract_prediction()` strips whitespace at all return paths.
- 4 Anonymous citations resolved to verified authors (Park 2025, Zhang 2025 ×2, Yong 2025).
- Eval-pipeline-gap claim refuted: re-evaluating Open-RS's own public RS2 checkpoint with our pipeline gives `maj@4=75.0%` vs paper's `80%` — within 5pp variance. The original 12.9pp gap was a metric mismatch (`pass@1` vs `maj@k`), not a pipeline bug.

### Planned
- MGSM multi-seed eval (optional, acknowledged in Limitations as single-seed).
- arXiv preprint listing once endorsement is confirmed.

## [v1.0.0] — 2026-05-07

Initial public release. Single-seed preliminary results across three GRPO arms
(English-only, Vietnamese-translated, English+R5) on
DeepSeek-R1-Distill-Qwen-1.5B base, single A100 + LoRA budget.

### Added
- Three reproducible training arms (`configs/grpo_a2_vi.yaml`, `grpo_a3_enlang.yaml`, `reproduce_open_rs.yaml`).
- Five reward functions (`src/rewards/{correctness,format,length,tag,lang}.py`)
  matching Open-RS RS2 verbatim plus an additional fastText
  language-consistency reward (`R5`).
- Six evaluation benchmark adapters (`src/eval/{gsm8k,math500,aime,amc23,mgsm,msvamp}.py`)
  using the Open-RS `lighteval` `MATH_QUERY_TEMPLATE` verbatim.
- 115 pytest tests (CI-safe subset of 110 passes on GitHub Actions Ubuntu 3.11).
- Full LoRA adapters for the three arms (`results/training/grpo/`, ~17 MB each).
- Eval JSONs with full `responses[]` arrays for post-hoc analysis
  (`results/training/eval/`).
- Phase reports `reports/phase{0..8}_*.md` documenting design decisions,
  failures, and recovery.
- Paper LaTeX source with ACL style files, three figures, and two tables.
- Twitter thread template (`paper/TWITTER_THREAD.md`).
- Pre-flight environment check (`scripts/preflight.sh`).
- arXiv submission bundle build script (`paper/build_arxiv_bundle.sh`).
- GitHub Actions CI: pytest + ruff lint.

### Known limitations
- Single seed (seed=42) only.
- LoRA r=16 instead of full-param: 22.5pp gap to Open-RS reported AMC23 80%.
- Vietnamese training uses 5CD-AI MetaMathQA (different from open-rs translated)
  → confounds language with distribution.
- AMC23 maj@4 sampling implementation has a bug; pass@1 numbers verified.
- Eval pipeline reports −12.9 pp (AMC23) and −23.4 pp (MATH-500) gaps to
  Open-RS reported base, even with verbatim `lighteval` prompt.

### Acknowledgements
- Open-RS protocol and AMC-23 dataset release ([Knoveleng & Ngo, 2025](https://arxiv.org/abs/2503.16219)).
- DeepSeek-R1 base distilled checkpoint ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)).
- TRL 0.15.2 in-process vLLM rollout integration.
- 5CD-AI Vietnamese MetaMathQA translation.
