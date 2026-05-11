# Paper Readiness Checklist

Comprehensive list of every artifact, claim, and verification step
needed for submission. This file is the audit trail: every item must
be either ✅ done or ⏳ explicitly tracked.

Last updated: 2026-05-10

---

## 1. Manuscript content

| Section | Status | File | Verification |
|---|---|---|---|
| Title | ✅ | `paper/main.tex`, `paper/ieee/main.tex` | Same in both versions |
| Abstract | ✅ | both | Will need rewrite for multi-seed |
| Keywords (IEEE) | ✅ | `paper/ieee/main.tex` | 10 keywords listed |
| 1. Introduction | ✅ | both | Hook + research questions |
| 2. Related Work | ✅ | both | Sober Look, Open-RS cited |
| 3. Preliminaries (theory) | ✅ | IEEE only | GRPO objective, advantage-invariance |
| 4. Method | ✅ | both | 3 arms, reward functions |
| 5. Experimental Setup | ✅ | both | Multi-seed, evaluation |
| 6. Results | 🟡 | both | **Need rewrite with 9-cell numbers** |
| 7. Discussion / Mechanism | ✅ | both | Hypothesis section |
| 8. Limitations | ✅ | both | Honest enumeration |
| 9. Conclusion | ✅ | both | Summary |
| References | ✅ | `paper/refs.bib` | 21 entries verified |
| Appendix | ✅ | `paper/appendix.tex` | Reproducibility |
| AI Disclosure | ✅ | both | ICLR/NeurIPS/IEEE compliant |

⚠️ **TODO before submission:**
- Update tables 1-2 with multi-seed numbers (currently single-seed)
- Update headline claims in abstract from "+7.5pp" to weakened multi-seed
- Add A4 ablation results to Discussion (when A4 finishes)
- Add MGSM multilingual results (when Phase 9.4 done)

---

## 2. Tables (LaTeX, in `paper/tables/`)

| Table | Status | File | Source |
|---|---|---|---|
| Main results (3 arms × benchmarks) | 🟡 single-seed | `table_results.tex` | Need multi-seed rewrite |
| Effect vs base | 🟡 single-seed | `table_delta_vs_base.tex` | Need rewrite |
| Variance comparison (NEW) | ❌ pending | `table_variance.tex` | Phase 9.5 will create |
| Bootstrap 95% CIs | ❌ pending | `table_bootstrap.tex` | Phase 9.5 |
| Hyperparameters | ✅ | inline appendix | Verified |
| Compute cost | ✅ | inline appendix | Verified |
| Software versions | ✅ | inline appendix | Verified |
| MGSM multilingual | ❌ pending | `table_mgsm.tex` | Phase 9.4 result |

---

## 3. Figures (PDF, in `paper/figures/`)

| Figure | Status | File | Story |
|---|---|---|---|
| Arm means ± σ | ✅ | `fig1_arm_means_with_ci.pdf` | Bar chart 3 arms × 5 metrics |
| Per-seed scatter | ✅ | `fig2_seed_scatter.pdf` | Show variance directly |
| Variance ratios | ✅ | `fig3_variance_ratios.pdf` | **Headline** finding |
| Effect Δ from base | ✅ | `fig4_effect_vs_base.pdf` | A3 only positive on AIME m@8 |
| Training curves multi-seed | ✅ | `fig5_training_curves_multiseed.pdf` | Reward/KL/length |
| AIME focus | ✅ | `fig6_aime_focus.pdf` | A3 wins hardest benchmark |
| MGSM language sweep | ❌ pending | `fig7_mgsm.pdf` | Phase 9.4 result |
| A4 mechanism comparison | ❌ pending | `fig8_a4_ablation.pdf` | After A4 done |

Auto-regenerate: `python3 paper/figures/make_figures_v2.py`

---

## 4. Data evidence (in `results/eval/`)

| Cell | JSONs | Status | SHA256 verified |
|---|---|---|---|
| Base (deepseek_r1_distill_15b) | 3 | ✅ in git | ✅ |
| Base (v3 openrs_eval) | 3 | ✅ in git | ✅ |
| A1 seed=42 (Phase 8 v3 ckpt50) | 3 | ✅ in git | ✅ |
| A1 seed=42 (Phase 8 v2) | 3 | ✅ in git | ✅ |
| A1 seed=123 | 3 | ✅ in git | ✅ |
| A1 seed=7 | 3 | ✅ in git | ✅ |
| A2 seed=42 | 3 | ✅ in git | ✅ |
| A2 seed=123 | 3 | ✅ in git | ✅ |
| A2 seed=7 | 3 | ✅ in git | ✅ |
| A3 seed=42 | 3 | ✅ in git | ✅ |
| A3 seed=123 | 3 | ✅ in git | ✅ |
| A3 seed=7 | 3 | ✅ in git | ✅ |
| A4 seed=42 | 0 | 🔄 in progress | pending |
| MGSM 10 langs × 9 cells | 0 | ❌ pending | Phase 9.4 |

**Total committed JSONs:** 30/24 expected core + Phase 9.4 pending

Each JSON contains:
- `pass_at_1`, `maj_at_4` (where applicable), `maj_at_8` (where applicable)
- Full `responses[]` array (model outputs)
- Metadata: model_path, eval_date, vllm_version, seed, etc.

---

## 5. Code evidence (in repository root)

| Component | Status | Notes |
|---|---|---|
| Reward functions | ✅ | `src/rewards/{correctness,format,length,tag,lang,bias_const}.py` |
| Eval adapters | ✅ | `src/eval/{gsm8k,math500,aime,amc23,mgsm,msvamp,math500_vi_hand,_common,prompts,runner}.py` |
| Trainer wrapper | ✅ | `src/trainers/{grpo,sft,checkpoint_utils,dataset_utils}.py` |
| Tests | ✅ 104 pass | `tests/` |
| CI workflow | ✅ | `.github/workflows/test.yml` |
| Configs | ✅ | `configs/{base,reproduce_open_rs,grpo_a2_vi,grpo_a3_enlang,grpo_a4_const_bias,eval}.yaml` |
| License | ✅ | Apache-2.0 (`LICENSE`) |
| README | ✅ | with AI disclosure section |
| pyproject.toml | ✅ | exact version pins |
| Makefile | ✅ | reproducible build commands |

---

## 6. Reproducibility artifacts

| Artifact | Status | Location |
|---|---|---|
| LoRA adapters (17MB each × 9 cells) | ✅ on local | `results/grpo/{cell}/checkpoint-50/adapter_model.safetensors` |
| Trainer state JSONs | ✅ in git | `results/grpo/{cell}/checkpoint-{N}/trainer_state.json` |
| Per-step training logs | ✅ on local | `reports/phase9_runs/raw_logs/` |
| SHA256 manifest | ✅ in git | `reports/phase9_runs/manifest.csv` |
| VERIFICATION.md audit trail | ✅ in git | `VERIFICATION.md` |
| fastText langID model location | ✅ documented | `data/raw/lid.176.bin` (130MB, downloaded externally) |

---

## 7. Compliance / metadata

| Requirement | Status | Notes |
|---|---|---|
| Apache-2.0 code license | ✅ | `LICENSE` |
| CC-BY-4.0 paper license | ✅ | README badge |
| Zenodo DOI | ✅ | `10.5281/zenodo.20061328` |
| GitHub repo public | ✅ | `vudang4494/xling-grpo-sub3b` |
| AI disclosure (ICLR 2026) | ✅ | both manuscript + README |
| Author affiliation | ✅ | "Independent Researcher" |
| Author email | ✅ | vu.dh4494@gmail.com |
| **ORCID iD** | ❌ | **User task** — register at orcid.org/register |
| HuggingFace upload | ❌ pending | Phase 9.7 — paper PDF + adapters |
| arXiv listing | ❌ blocked | Endorsement needed |

---

## 8. Submission materials (write at submit time)

| Item | Status | Notes |
|---|---|---|
| Cover letter (1 page) | ❌ pending | Phase 9.6 — explain fit + novelty |
| Suggested reviewers (3-5) | ❌ pending | Phase 9.6 — research GRPO+RLHF authors |
| Conflict of interest | ✅ "None declared" | Standard |
| Funding statement | ✅ "Self-funded" | Standard |
| Data availability | ✅ | GitHub + Zenodo links |
| Ethics statement | ✅ "No human subjects" | Standard |
| Author contributions | ✅ "Single author" | Standard |
| Software/data availability | ✅ | Apache-2.0, public |
| Compute disclosure | ✅ | $87 single A100 documented |

---

## 9. Verification (cross-check)

| Check | Status | Tool |
|---|---|---|
| SHA256 manifest matches local + remote | ✅ | `data_integrity_pipeline.py` |
| Tests pass on clean checkout | ✅ 104 pass | `pytest tests/` |
| Paper claims trace to eval JSONs | 🟡 | `data_integrity_pipeline.py` flags 4 unmatched (means + external) |
| Paper compiles (LaTeX) | ✅ | `tectonic main.tex` (both versions) |
| 0 overfull boxes | ✅ | manuscript polish |
| Citations resolve | ✅ | bibtex passes |
| All figures load | ✅ | PDF check |
| README links work | ✅ | manual verify |
| GitHub Actions CI green | ✅ | `.github/workflows/test.yml` |

---

## 10. Pending work (in order)

```
Phase 9.2 (multi-seed):    9/9 cells eval done ✅
Phase 9.3 (A4 ablation):   IN PROGRESS — A4 training 78%, eval pending (~3h)
Phase 9.4 (MGSM):          NOT STARTED — ~3h on Vast.ai when A4 done
Phase 9.5 (aggregate):     NOT STARTED — bootstrap CIs, no GPU (~2h)
Phase 9.6 (paper rewrite): NOT STARTED — incorporate all data (~3 days)
Phase 9.7 (submit):        NOT STARTED — TMLR/IEEE Access/HF Papers
```

**ETA full submission ready: ~7-10 days** from now (May 10).

---

## 11. Audit commands

Run these to verify entire repo state at any time:

```bash
# 1. Pull all data, verify checksums, validate JSONs
python3 scripts/data_integrity_pipeline.py

# 2. Run all tests
/Users/vudang/miniconda3/bin/python3 -m pytest tests/ -q

# 3. Regenerate figures from latest data
/Users/vudang/miniconda3/bin/python3 paper/figures/make_figures_v2.py

# 4. Compile both manuscript versions
cd paper && tectonic main.tex
cd paper/ieee && bash build.sh

# 5. Check git state matches expectations
git status --short
git log --oneline -5
git ls-files results/eval/ | wc -l   # should equal eval JSON count
```

---

## 12. What CANNOT go missing

These items, if lost, would make the paper unsubmittable:

| Critical artifact | Backup |
|---|---|
| 30 eval JSONs | ✅ in git, multiple branches |
| LoRA adapters (9 × 17MB) | ✅ on local, SHA256 verified |
| `manifest.csv` | ✅ in git |
| `VERIFICATION.md` | ✅ in git |
| paper/main.tex + paper/ieee/main.tex | ✅ in git |
| 6 figures | ✅ in git |
| Apache-2.0 LICENSE | ✅ in git |
| Zenodo DOI metadata | ✅ external (Zenodo permanent) |

→ **All redundant**: GitHub + local + Zenodo + (when uploaded) HuggingFace.

---

## 13. Risks tracked

| Risk | Mitigation |
|---|---|
| Vast.ai instance crashes | Local mirror SHA256-verified |
| Local disk dies | Git remote (GitHub) + Zenodo backup |
| GitHub repo goes down | Zenodo permanent DOI archive |
| arXiv endorsement never comes | TMLR/IEEE Access/HF Papers don't need it |
| Reviewer demands specific claim verification | All eval JSONs have full responses[] arrays |
| Reproducibility challenge from reviewer | Public Apache-2.0 code + LoRA adapters |
