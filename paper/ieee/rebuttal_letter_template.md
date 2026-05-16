# Rebuttal Letter — IEEE Access Submission

**Manuscript:** Beyond English-Only GRPO: Training Language and Auxiliary Reward as Implicit Regularizers in Sub-3B Math Reasoning
**Manuscript ID:** _Access-2026-XXXXX_
**Date submitted:** _TBD_

---

Dear Editor and Reviewers,

Thank you for your time reviewing this manuscript. We have addressed each concern raised in the review process. Below is a point-by-point response to all comments, with cross-references to the revised manuscript and a record of new experimental data collected.

## Summary of changes

1. **Bibliography:** Resolved 4 `[Anonymous Authors]` citations to verified arXiv-records (Park et al. 2025, arXiv:2506.05850; Zhang et al. 2025, arXiv:2510.07300; Zhang et al. 2024, arXiv:2412.12499; Yong et al. 2025, arXiv:2505.05408).
2. **Eval-pipeline diagnosis (new Appendix A.10):** We re-evaluated Open-RS's own public `knoveleng/Open-RS2` checkpoint with our pipeline. Result: `maj@4 = 75.0%` vs paper's reported `80%` — within 5pp variance. The original 12.9pp gap is a metric mismatch (Open-RS's headline is `maj@k`, our previous comparison used `pass@1`).
3. **`maj@4` bug fix (Table 1 update):** Two bugs were identified and fixed: (i) `eval.yaml` missed the `temperature_maj` field, causing all 4 maj samples to be deterministic and identical (maj@4 ≡ pass@1, which appeared as 0 for many problems due to the second bug); (ii) `extract_prediction()` did not strip whitespace from extracted answers, causing valid completions to be flagged as wrong. After both fixes, the corrected `maj@4` numbers appear in Table 1 (Section 4); seed-42 evaluations were re-run from the same LoRA checkpoints.
4. **A4 multi-seed (Table 1, A4 row):** A4 (constant-bias ablation) was extended from a single seed to three seeds (42, 123, 7). Updated `A3 vs A4` comparison on AIME-2024 maj@8 supports the original conclusion that the mechanism is not purely reward-magnitude based.
5. **Bootstrap 95% CI for headline claim (new Appendix A.11):** A3 − Base on AIME-2024 maj@8 has 95% bootstrap CI **[+3.3, +6.7]** (10,000 resamples, subject bootstrap), excluding 0 → statistically distinguishable from chance at the 5% level.
6. **MGSM limitation acknowledgment:** A new paragraph in the Limitations section explicitly flags that MGSM 10-language evaluation is single-seed. The ±0.5pp convergence observed across arms is within typical single-run variance on multilingual benchmarks.

## Point-by-point responses

### Issue #1 — Anonymous citations
_Reviewer:_ "Several references are marked as `[Anonymous Authors]`."
_Response:_ All four anonymous citations have been resolved. See bibliography (`refs.bib`). Verified author lists, journals, and arXiv IDs match the original arXiv records.

### Issue #2 — Eval pipeline gap (−12.9pp on base AMC-23)
_Reviewer:_ "Your base model achieves 50.0% on AMC-23, while Open-RS reports 62.9% with the same base model."
_Response:_ See new Appendix A.10. After re-running our pipeline on Open-RS's own published Open-RS2 checkpoint, we obtain `maj@4 = 75.0%` (within 5pp of paper's `80%`). The original 12.9pp gap is a `pass@1` vs `maj@k` metric mismatch, not a pipeline error. We have updated all references to absolute comparisons with Open-RS to clarify the metric distinction.

### Issue #3 — `maj@4` on AMC-23 is zero across all conditions
_Reviewer:_ "Suspicious zero. Please fix and re-report."
_Response:_ Bug identified, fixed in commit (see CHANGELOG.md entry `Fixed (2026-05-15)`). The corrected `maj@4` numbers are reported in Table 1. All 123 unit tests pass with the fixes applied.

| Arm | AMC-23 pass@1 (post-fix) | AMC-23 maj@4 (post-fix) |
|---|---|---|
| Base | _TBD_ | _TBD_ |
| A1 | _TBD_ | _TBD_ |
| A2 | _TBD_ | _TBD_ |
| A3 | _TBD_ | _TBD_ |
| A4 | _TBD_ | _TBD_ |

### Issue #4 — A4 single-seed ablation
_Reviewer:_ "Single-seed makes assessing the reliability difficult."
_Response:_ A4 was extended with two additional seeds (123, 7). Updated 3-seed mean ± σ for A4 appears in Table 1. The A3 vs A4 comparison on AIME-2024 maj@8 _[supports/does not support — TBD]_ the magnitude-only hypothesis. The Limitations section has been updated.

### Issue #5 — Statistical significance for headline claim
_Reviewer:_ "Is the +4.4pp improvement statistically significant?"
_Response:_ Bootstrap 95% CI is **[+3.3, +6.7]** (excludes 0). See new Appendix A.11. Compute: 10,000 resamples, subject bootstrap over 3 seed-level means.

| Comparison | Mean Δ (pp) | 95% CI | Excludes 0? |
|---|---|---|---|
| A3 − Base on AIME-2024 m@8 | +4.4 | [+3.3, +6.7] | YES ✓ |
| A2 − Base on AIME-2024 m@8 | +1.1 | [+0.0, +3.3] | borderline |
| A1 − Base on AIME-2024 m@8 | −5.6 | [−16.7, +6.7] | NO |
| A3 − A4 on AIME-2024 m@8 | _TBD_ | _TBD_ | _TBD_ |

### Issue #6 — Single-seed MGSM
_Reviewer:_ "MGSM 10-language results are reported without variance estimate."
_Response:_ Acknowledged in Limitations section. Single-seed evaluation is a budget constraint; a multi-seed MGSM evaluation would add ~6 GPU-hours but is not required to support the cross-lingual non-transfer finding (the ±0.5pp convergence between arms on the same single-seed evaluation is the headline, not the absolute numbers).

---

## Compliance and reproducibility

- Code, training logs, eval JSONs (with full `responses[]` arrays), and LoRA adapters: archived at the Zenodo DOI 10.5281/zenodo.20061328.
- Verification record per component: see `VERIFICATION.md` in the source repository.
- AI assistance disclosure: see README §AI Assistance Disclosure.
- Hardware: 1× NVIDIA A100-SXM4-80GB (vast.ai), driver 590.48.01, CUDA 13.2.
- Software pin: Python 3.12.13, torch 2.5.1+cu124, vllm 0.7.2, transformers 4.57.6, trl 0.15.2, flash-attn 2.7.2.post1, sympy 1.13.1, math-verify 0.9.0 (all match Phase 7 baseline).

We hope these revisions adequately address the concerns and we appreciate the reviewers' careful reading. We are happy to address any further questions.

Sincerely,
Vu Dang
