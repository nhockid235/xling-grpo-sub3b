# Paper v2 Pipeline — Logical Structure + Step-by-Step TODO

This document specifies the complete pipeline to upgrade `paper/main.tex`
and `paper/ieee/main.tex` from Phase 8 single-seed to Phase 9 multi-seed.
Every step must be ✅ before the paper is submission-ready.

**Constraints (hard rules):**
- ❌ NO compute cost / dollar amounts anywhere in manuscript or README
- ❌ NO "$X per hour", "$Y total", "$Z budget" mentions
- ✅ Hardware described neutrally: "one NVIDIA A100 80GB GPU"
- ✅ Every numerical claim traceable to an eval JSON
- ✅ Zero overfull boxes in compiled PDF
- ✅ All citations resolve

---

## 1. Paper structure (target — 14 sections)

```
┌────────────────────────────────────────────────────────────────┐
│                     PAPER v2 STRUCTURE                          │
├────────────────────────────────────────────────────────────────┤
│  §0  Title + Author + Affiliation                              │
│  §0  Abstract (structured: Background/Methods/Results/Concl.)  │
│  §0  Keywords (5-10 for IEEE)                                  │
│                                                                 │
│  §I  Introduction                                              │
│       — Motivation                                             │
│       — Research questions (Q1, Q2)                            │
│       — Contributions (4 items)                                │
│       — Paper structure paragraph                              │
│                                                                 │
│  §II  Related Work                                             │
│       — GRPO and small reasoning models                        │
│       — Reproducibility in LLM reasoning (cite Sober Look)     │
│       — Cross-lingual transfer                                 │
│       — Reward shaping (cite Zheng 2018)                       │
│                                                                 │
│  §III  Preliminaries                                           │
│        — GRPO objective formal                                 │
│        — Advantage-invariance identity                         │
│        — Multi-seed methodology                                │
│                                                                 │
│  §IV  Method                                                   │
│        — Base model + setup                                    │
│        — Four training arms (A1, A2, A3, A4)                   │
│        — Reward function formal definitions                    │
│        — Hyperparameters table                                 │
│                                                                 │
│  §V  Experimental Setup                                        │
│       — Benchmarks (AMC, MATH, AIME, MGSM)                     │
│       — Evaluation protocol                                    │
│       — Random seeds (3 seeds per arm; A4 single seed)         │
│                                                                 │
│  §VI  Results                                                  │
│        §VI.A  English benchmarks — multi-seed comparison       │
│        §VI.B  A4 mechanism ablation                            │
│        §VI.C  MGSM multilingual sweep                          │
│                                                                 │
│  §VII  Discussion                                              │
│         — Variance pattern across benchmark sizes              │
│         — A4 implication for mechanism hypothesis              │
│         — Negative cross-lingual finding                       │
│                                                                 │
│  §VIII  Limitations                                            │
│         — Single base model                                    │
│         — A4 single seed                                       │
│         — A2 dataset confound                                  │
│         — Evaluation pipeline gap to Open-RS reported          │
│                                                                 │
│  §IX  Conclusion                                               │
│                                                                 │
│  §*  Acknowledgment (no $$$, only thanks to community)         │
│  §*  AI Assistance Disclosure                                  │
│                                                                 │
│  References (BibTeX, IEEEtran or natbib)                       │
│                                                                 │
│  Appendix                                                      │
│        A. Reproducibility (hyperparameters, software)          │
│        B. Per-cell results full table                          │
│        C. MGSM per-language full table                         │
│                                                                 │
│  AUTHOR BIO (IEEE version only)                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. TODO Pipeline (sequential, 25 steps)

### Phase A: Strip + Clean (steps 1-5)

- [ ] **1.** Remove all `$X.YZ`, `$/h`, "budget" mentions from `paper/main.tex`
- [ ] **2.** Remove all `$X.YZ`, `$/h`, "budget" mentions from `paper/ieee/main.tex`
- [ ] **3.** Remove cost mentions from `paper/appendix.tex`
- [ ] **4.** Remove cost mentions from `README.md` (or rephrase neutrally)
- [ ] **5.** Audit `git grep -nE "\\\$[0-9]"` returns nothing in tex/md

### Phase B: Update Abstract + Introduction (steps 6-9)

- [ ] **6.** Rewrite abstract for v2:
  - Background: GRPO at sub-3B has reproducibility issues (Sober Look)
  - Methods: 4 arms × 3 seeds × 5 benchmarks (AMC, MATH, AIME × 2 + MGSM)
  - Results: A3 best AIME m@8 (+4.4pp), A1 catastrophic AIME (-12pp), MGSM null
  - Conclusion: variance > mean is the main story
- [ ] **7.** Update IEEE structured abstract similarly
- [ ] **8.** Rewrite §I Introduction:
  - Motivation pivot from "cross-lingual breakthrough" to "reproducibility study"
  - Q1: How stable are GRPO results across seeds at sub-3B?
  - Q2: Does training language affect cross-lingual transfer?
  - 4 contributions
- [ ] **9.** Add boxed main claim with new framing

### Phase C: Update Methodology + Setup (steps 10-13)

- [ ] **10.** §IV: Add A4 (constant-bias) as fourth arm
- [ ] **11.** §IV: Formal reward function definitions (R1, R2, R5, R_const)
- [ ] **12.** §V: Update setup section — 3 seeds × 3 arms + 1 ablation
- [ ] **13.** §V: Add MGSM evaluation protocol

### Phase D: Rewrite Results (steps 14-17)

- [ ] **14.** §VI.A: Multi-seed EN benchmarks
  - Use `table_main_v2.tex` (mean ± σ)
  - Use `table_delta_v2.tex` (Δ vs base)
  - Reference `fig1_arm_means_with_ci.pdf`, `fig2_seed_scatter.pdf`,
    `fig3_variance_ratios.pdf`
  - Narrative: A1 unstable, A2/A3 tighter, A3 best AIME m@8
- [ ] **15.** §VI.B: A4 mechanism ablation
  - Reference A4 single-seed numbers
  - Compare with A3 multi-seed
  - Honest negative: A4 does not reproduce A3 on AIME m@8
- [ ] **16.** §VI.C: MGSM multilingual
  - Use `table_mgsm.tex` (5 arms × 10 langs)
  - Reference `fig6_aime_focus.pdf` and new MGSM figure
  - Honest negative: all arms ≈ base on MGSM
- [ ] **17.** Add new figure for MGSM null result if needed

### Phase E: Update Discussion + Limitations + Conclusion (steps 18-21)

- [ ] **18.** §VII Discussion:
  - Variance scales inversely with benchmark size
  - A4 ablation: mechanism is NOT purely magnitude
  - MGSM null: training language doesn't transfer cross-lingually
- [ ] **19.** §VIII Limitations:
  - Single base model
  - A4 only 1 seed
  - A2 dataset confound (not translated Open-RS)
  - Open-RS reproduction gap unexplained
- [ ] **20.** §IX Conclusion: 3-paragraph summary
- [ ] **21.** Verify §AI Disclosure section is intact + accurate

### Phase F: Tables, Figures, References (steps 22-25)

- [ ] **22.** Verify all tables in `paper/tables/` are committed and referenced
- [ ] **23.** Verify all figures in `paper/figures/` are committed and referenced
- [ ] **24.** `paper/refs.bib`: add Sober Look (2504.07086) if missing,
        verify all cites resolve
- [ ] **25.** Appendix: per-cell full table + per-language MGSM table

### Phase G: Compile + QA (steps 26-30)

- [ ] **26.** Compile `paper/main.tex` → `main.pdf`, check 0 overfull boxes
- [ ] **27.** Compile `paper/ieee/main.tex` → `ieee/main.pdf`, check 0 overfull
- [ ] **28.** Cross-check every percent/number in paper against `master.csv`
- [ ] **29.** Verify all citations resolve (no `[?]` in PDF)
- [ ] **30.** Run `data_integrity_pipeline.py` final time

### Phase H: Submission prep (steps 31-35)

- [ ] **31.** Write cover letter (1 page) for TMLR
- [ ] **32.** Write cover letter for IEEE Access (alt)
- [ ] **33.** List 3-5 suggested reviewer emails
- [ ] **34.** Verify AI disclosure compliant (ICLR 2026 / NeurIPS / IEEE)
- [ ] **35.** Update `CITATION.cff` with v2 metadata

### Phase I: Release v2 (steps 36-40)

- [ ] **36.** Archive v1 PDF → `paper/archive/main_v1_phase8.pdf`
- [ ] **37.** Replace `paper/main.pdf` with v2
- [ ] **38.** Update `README.md` v2 findings (no $$$)
- [ ] **39.** Tag git: `git tag -a v2.0 -m "Multi-seed extension"`
- [ ] **40.** Upload v2 to Zenodo (new version of existing DOI)

### Phase J: Submit (steps 41-45)

- [ ] **41.** Submit to HuggingFace Papers (instant)
- [ ] **42.** Submit to TMLR
- [ ] **43.** Submit to NeurIPS MATH-AI workshop (when call opens)
- [x] **44.** (skipped — arXiv submission not pursued, no endorser available)
- [ ] **45.** Track all submissions in spreadsheet

---

## 3. Logical flow of the paper (narrative arc)

```
┌────────────────────────────────────────────────────────────┐
│ HOOK (Intro): Sober Look showed GRPO reasoning             │
│              papers may not reproduce. We investigate       │
│              the cross-lingual angle.                       │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ SETUP (Method): 4 arms — varying training language        │
│                 (EN/VI) and reward (with/without R5,       │
│                 and constant-bias control A4).             │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RESULT 1 (Positive): A3 (R5) achieves best AIME m@8       │
│                       +4.4pp over base, robust across      │
│                       3 seeds (σ=1.9pp).                   │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RESULT 2 (Methodological): A1 (vanilla EN) exhibits        │
│                             σ=11.3pp variance on AMC-23    │
│                             — single-seed claims unreliable │
│                             at sub-3B + LoRA scale.        │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RESULT 3 (Mechanism test): A4 (constant-bias) partially    │
│                              but not fully reproduces A3's  │
│                              effect. Mechanism is not       │
│                              purely reward-magnitude.       │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ RESULT 4 (Negative): All 4 arms ≈ base on MGSM 10 langs    │
│                       (Δ ≤ 0.5pp). Cross-lingual transfer  │
│                       does NOT occur with this recipe.     │
└────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ DISCUSSION: Three orthogonal findings — positive, methodo- │
│              logical, negative — each useful for the field.│
│              Mechanism remains open question for follow-up.│
└────────────────────────────────────────────────────────────┘
```

This is a balanced narrative: positive + negative + methodological, none
overstated.

---

## 4. Verification checklist (run after every step)

```bash
# 1. No money/cost mentions
grep -rE "\\\$[0-9]" paper/main.tex paper/ieee/main.tex paper/appendix.tex README.md
# Expected: no matches

# 2. No old single-seed claims
grep -E "single seed|seed=42 only|+7.5" paper/main.tex paper/ieee/main.tex
# Expected: no matches OR only as historical reference

# 3. All tables present
ls paper/tables/table_*_v2.tex paper/tables/table_mgsm.tex
# Expected: 3 files

# 4. All figures present  
ls paper/figures/fig*.pdf | wc -l
# Expected: 6+ files

# 5. Compile clean
tectonic paper/main.tex 2>&1 | grep -E "warning|overfull"
# Expected: <=1 underfull (badness ok), 0 overfull

# 6. Numbers match eval JSONs
python3 scripts/data_integrity_pipeline.py
# Expected: cross-check flags only acceptable mismatches (means, Open-RS reported)
```

---

## 5. Risks + mitigation

| Risk | Mitigation |
|---|---|
| Forget to strip $$$ somewhere | grep audit before push |
| Number mismatch between paper and JSON | run pipeline cross-check |
| Compile fails | tectonic with --keep-intermediates for debug |
| Reviewer rejects negative finding | Frame as "we save the field $$$" (don't say $) |
| AI disclosure outdated | Verify ICLR 2026 latest policy |

---

## 6. Acceptance criteria

Paper is "done" when ALL of:
1. ✅ All 40 TODO items checked
2. ✅ 0 overfull boxes in both main.pdf and ieee/main.pdf
3. ✅ No $X mentions anywhere
4. ✅ Cross-check pipeline shows ≤4 unmatched claims (acceptable: means, external citations)
5. ✅ All figures + tables referenced in text
6. ✅ Final QA review by human (user)
7. ✅ Compile reproducible via `bash paper/build.sh` (or equivalent)

---

## 7. Timeline

```
Day 1: Phase A (strip) + Phase B (abstract/intro)
Day 2: Phase C (method) + Phase D (results)
Day 3: Phase E (discussion) + Phase F (refs/tables) + Phase G (compile)
Day 4: User review + adjustments
Day 5: Phase H (submission prep) + Phase I (release)
Day 6+: Phase J (submit to venues)
```

Total: ~6 days for full v2 + submission.
