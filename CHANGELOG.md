# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Tier 2 multi-seed expansion (seeds 42, 123, 7) — push Findings competitiveness
- MGSM 10-language eval for cross-lingual claim
- AMC23 `maj@4` sampling-bug fix
- arXiv preprint listing once endorsement is confirmed

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
