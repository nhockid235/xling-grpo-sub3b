# IEEE Access Submission Guide — Step-by-step

Complete guide for submitting `paper/ieee/main.pdf` to IEEE Access.

---

## 1. Pre-submission checklist

| Item | Status | Notes |
|---|---|---|
| Paper PDF (IEEEtran format) | ✅ | `paper/ieee/main.pdf` (10 pages, 241 KB) |
| Source files (.tex, .bib) | ✅ | `paper/ieee/main.tex`, `paper/ieee/refs.bib` |
| Figures (.pdf) | ✅ | `paper/ieee/figures/` |
| Tables (.tex) | ✅ | `paper/ieee/tables/` |
| AI Assistance Disclosure | ✅ | In manuscript + VERIFICATION.md |
| ORCID iD | ❌ | **User must register** at https://orcid.org/register |
| Suggested reviewers list | ❌ | See Section 6 below |
| Cover letter | ❌ | Template in Section 5 |

---

## 2. IEEE Access overview (verified facts)

- **Type:** Open-access multidisciplinary journal
- **Scope:** Engineering, computing, technology (broad — accepts ML/NLP)
- **Review:** Single-blind peer review, ~4–6 weeks first decision
- **APC (2026):** USD **\$2,160** per accepted article (only paid on acceptance)
- **Member discounts:** IEEE member 5% off, IEEE+Society 20% off
- **Page limit:** No strict limit (typically 10–20 pages recommended)
- **License:** CC BY 4.0 (open access)
- **Submission URL:** https://ieee.atyponrex.com/journal/ieee-access

Source: [IEEE Access APC page](https://ieeeaccess.ieee.org/about/article-processing-charges/)

---

## 3. Account setup

1. Go to https://ieee.atyponrex.com/journal/ieee-access
2. Click **"Create Account"** if you don't have IEEE web account
3. Fill profile:
   - Full name (matching paper authorship)
   - Email (matches `vu.dh4494@gmail.com`)
   - Country: Vietnam
   - ORCID iD (register first at https://orcid.org/register if needed)
   - Areas of expertise: Machine Learning, NLP, Reinforcement Learning

---

## 4. Submission step-by-step

### Step 1 — Start submission
- Click **"New submission"**
- Choose article type: **Research article**

### Step 2 — Manuscript details
- **Title:** Beyond English-Only GRPO: A Multi-Seed Empirical Study at Sub-3B Scale
- **Running title:** (short version, ≤50 chars)
- **Abstract:** Copy the structured abstract from `paper/ieee/main.tex` (Background/Methods/Results/Conclusion)
- **Keywords:** Reinforcement learning from rewards, GRPO, large language models, mathematical reasoning, LoRA, cross-lingual transfer, auxiliary rewards, reward shaping, regularization, multilingual evaluation

### Step 3 — Author info
- **Corresponding author:** Yes (single author = both)
- **Affiliation:** Independent Researcher
- **Country:** Vietnam
- **ORCID iD:** [your ID after registration]

### Step 4 — File upload

Required files:
1. **Manuscript PDF** — `paper/ieee/main.pdf`
2. **Source files (.zip)** — bundle:
   - `main.tex`
   - `refs.bib`
   - `tables/*.tex`
   - `figures/*.pdf`
3. **Cover letter** (Section 5 below)
4. **Reviewer suggestions** (Section 6 below)

### Step 5 — Declarations
- **Conflict of interest:** None declared
- **Funding:** Self-funded
- **Data availability:** "All data, code, and trained model weights publicly available at https://github.com/vudang4494/xling-grpo-sub3b under Apache-2.0 license; permanent archive at Zenodo DOI 10.5281/zenodo.20061328"
- **Ethics statement:** "This work uses publicly available datasets and pre-trained models. No human subjects research."
- **AI assistance disclosure:** Refer to manuscript Section X (AI Assistance Disclosure)

### Step 6 — Submission summary review
- Verify all details correct
- Confirm originality (not under review elsewhere)
- Submit

---

## 5. Cover letter template

Save as `cover_letter.pdf`:

```
[Date: 2026-05-11]

Dear IEEE Access Editorial Office,

I am submitting the manuscript titled "Beyond English-Only GRPO: A Multi-Seed
Empirical Study at Sub-3B Scale" for consideration in IEEE Access.

This work provides a multi-seed empirical study of Group Relative Policy
Optimization (GRPO) post-training of a 1.5B distilled reasoning model under
a single-GPU LoRA constraint. Building on the recent observation by
Hochlehnert et al. (COLM 2025) that small-model RL reasoning gains may not
survive multi-seed evaluation, we test three training-condition arms across
three random seeds plus a single-seed mechanism ablation. The work makes
four contributions:

1. Multi-seed empirical evidence that vanilla English GRPO at sub-3B + LoRA
   scale exhibits sigma=11.3pp seed variance on AMC-23 and consistently
   degrades AIME-2024 pass@1, confirming single-seed claims unreliable.

2. Documentation of the auxiliary-reward arm (A3) achieving the only positive
   AIME-2024 maj@8 effect (+4.4 +- 1.9 pp over base), robust across three
   seeds.

3. A constant-bias ablation that partially refutes a pure reward-magnitude
   interpretation of the auxiliary-reward effect.

4. A 10-language MGSM evaluation showing null cross-lingual transfer for
   all training conditions.

The work fits IEEE Access's broad scope on engineering applications of
machine learning, with particular relevance to deployment-constrained
post-training of language models. Our reproducibility commitment (full code,
LoRA adapters, evaluation JSONs, manifest-checksummed artifacts) addresses
the IEEE emphasis on rigorous and verifiable research.

In compliance with IEEE Access policies on AI-assisted academic work, the
manuscript and source code were prepared with assistance from Anthropic
Claude (via the Claude Code command-line interface). All empirical results
were independently produced and verified by the author, who bears sole
responsibility for the technical claims, methodology, and any errors. The
complete AI-assistance scope is detailed in the manuscript and in the
VERIFICATION.md audit document of the released codebase.

The corresponding author and sole author is Vu Dang, an independent
researcher based in Vietnam. The work has not been published elsewhere and
is not under review at any other venue.

Thank you for your consideration.

Sincerely,
Vu Dang
vu.dh4494@gmail.com
ORCID: [register and insert]
https://github.com/vudang4494/xling-grpo-sub3b
```

---

## 6. Suggested reviewers (3–5)

Search for active researchers in GRPO/RLHF for small LMs. Candidates:

1. **Andreas Hochlehnert** (Sober Look authors, COLM 2025)
   - Affiliation: University of Cambridge / Bethge Lab
   - Why fit: Multi-seed methodology — directly extends their work
   - Conflict check: None (we cite, don't co-author)

2. **Quy-Anh Dang** (Open-RS, AAAI 2026)
   - Why fit: Original Open-RS recipe author — knows the protocol
   - Conflict check: We extend their recipe (no co-authorship)

3. **Authors of Cross-lingual Collapse** (arXiv:2506.05850)
   - Why fit: Cross-lingual + GRPO at 3B+
   - Caveat: anonymous paper, may not be findable

4. **Zichen Liu** (Dr.GRPO, COLM 2025)
   - Why fit: GRPO variance analysis
   - Conflict check: None

5. **A researcher from VinAI / FPT AI / Zalo AI** (Vietnamese AI community)
   - Why fit: Vietnamese language angle, connection to local researchers

→ Provide 3–5 with email addresses (search via Google Scholar profiles).

---

## 7. After submission

### Timeline
- **First decision:** 4–6 weeks
- **Likely outcome:** Major revision (most common for IEEE Access first-decision)
- **Revision deadline:** Usually 2 months
- **Total to acceptance:** 3–6 months (estimate, no guarantee)

### If accepted
1. Receive acceptance email
2. **Pay APC \$2,160** (or apply for waiver if eligible — Vietnam may qualify)
3. Sign IEEE copyright form (CC BY 4.0)
4. Galley proof review (1 week)
5. Publication in IEEE Xplore
6. Receive permanent DOI

### If revision requested
1. Read reviewer comments carefully
2. Address each comment in revision letter (point-by-point response)
3. Update manuscript
4. Resubmit within deadline

### If rejected
1. Read reviewer comments
2. Improve based on feedback
3. Submit to alternative venue (TMLR, ACL Findings, workshop)
4. Don't take it personally — most papers get rejected at first venue

---

## 8. APC waiver eligibility (Vietnam)

IEEE Access offers waivers for authors from low-income countries (per World Bank classification). Vietnam may qualify for:

- **Full waiver** if classified as "low income"
- **Partial discount** if "lower-middle income"

**Action:** Contact IEEE Access support after acceptance to request waiver assessment.
URL: https://open.ieee.org/for-authors/article-processing-charges/

---

## 9. Alternative venues (if IEEE Access not preferred)

Same paper can be submitted to alternative venues — pick ONE primary, can have secondary as workshop:

| Venue | APC | Review time | Notes |
|---|---|---|---|
| **TMLR** | \$0 | 3-6 months | Open review, accepts rigorous empirical work, no novelty threshold |
| **NeurIPS MATH-AI workshop** | \$0 | ~2 months | Workshop, lighter review, fits content |
| **ACL Findings** | \$0 | ~6 months | Need to submit through ARR rolling review |
| **HuggingFace Papers** | \$0 | Instant | No review, just indexing |

**Recommended strategy:** Submit IEEE Access + TMLR + HF Papers in parallel.

---

## 10. Risks and mitigation

| Risk | Mitigation |
|---|---|
| Desk-reject for format | Verify IEEEtran compiles, page limits |
| Reviewer "not novel enough" | Cite Sober Look + emphasize 3-arm × multi-seed novelty |
| APC unaffordable | Apply for low-income country waiver |
| Single-author concerns | Emphasize reproducibility + independent verification |

---

## 11. Final pre-submission verification

Run this command before submitting:

```bash
cd /Users/vudang/PythonLab/Papper/xling-grpo-sub3b
python3 scripts/data_integrity_pipeline.py
```

Expected output:
- `27+ eval JSONs valid`
- `0 unmatched paper claims (excl. means and external)`
- Manifest CSV updated

Then check:
```bash
ls paper/ieee/main.pdf                    # Manuscript exists
ls paper/ieee/figures/*.pdf | wc -l        # Should be 3+
ls paper/ieee/tables/*.tex | wc -l         # Should be 2+
grep -E "OR" paper/ieee/main.tex            # Should not match weird patterns
```

---

## 12. Submission flow summary

```
1. Register ORCID (5 min) — https://orcid.org/register
2. Bundle source files (zip)
3. Write cover letter (use template Section 5)
4. List 3-5 suggested reviewers (Section 6)
5. Submit at https://ieee.atyponrex.com/journal/ieee-access
6. Wait 4-6 weeks
7. Address revision comments
8. Pay APC OR request waiver
9. Galley proof
10. Published!
```

Total realistic timeline: **3-6 months** from submission to publication.

---

## Sources

- [IEEE Access APC official](https://ieeeaccess.ieee.org/about/article-processing-charges/)
- [IEEE Author Center](https://journals.ieeeauthorcenter.ieee.org/)
- [IEEE Open Access Author Portal](https://open.ieee.org/for-authors/article-processing-charges/)
- [IEEE Access Submission Portal](https://ieee.atyponrex.com/journal/ieee-access)
