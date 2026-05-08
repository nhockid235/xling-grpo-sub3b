# IEEE Access submission package

This directory contains the IEEE Access submission version of the paper.
The ACL/arXiv version lives in `paper/` (one level up); this version is
restructured for IEEE conventions (Roman-numeral sections, structured
abstract, IEEE keywords, IEEEtran bibliography style, formal mathematical
notation).

## Files

| File | Purpose |
|---|---|
| `main.tex` | IEEEtran journal-class manuscript (9 pages typeset) |
| `refs.bib` | Bibliography (shared with the ACL/arXiv version) |
| `tables/` | LaTeX tables (shared content with ACL version) |
| `figures/` | PDF figures (shared content with ACL version) |
| `build.sh` | One-shot build script (calls `tectonic`) |
| `main.pdf` | Compiled output |

## Build

```bash
brew install tectonic           # macOS, one-time
bash build.sh                   # produces main.pdf (~240 KB, 9 pages)
```

The build is hermetic via Tectonic — all packages are downloaded from
CTAN on first compile and cached locally. No `pdflatex`/`bibtex` toolchain
required.

## Differences from the ACL/arXiv version

- **Class:** `IEEEtran` (journal mode) vs `article` + `acl.sty`
- **Abstract:** Structured (Background / Methods / Results / Conclusion)
  per IEEE Access guidelines, vs single-paragraph ACL abstract
- **Sections:** Roman numerals (I, II, ...) per IEEE convention
- **Citations:** Numbered IEEEtran style `[1]` vs author-year `\citet`
- **Preliminaries section:** Formal GRPO objective with PPO-clipping
  geometry derivation, used to support the auxiliary-reward mechanism
  hypothesis in Section VII
- **Mechanism section:** Standalone Section VII proposing the
  PPO-clipping-geometry hypothesis, with falsifiable prediction operationalized
  by a constant-bias ablation arm (\armD{})
- **Reproducibility statement:** Inline subsection with full version pins
  (Python, torch, transformers, trl, vllm, etc.)
- **Author biography:** IEEE-style biography block at end

## Submission target

**IEEE Access** — open access, broad ML/NLP scope, ~6-week first-decision
review. Article processing charge \$1,950 (2026 rate). Submission portal:
<https://ieee.atyponrex.com/journal/ieee-access>.

## Status

- [x] IEEE format draft (single-seed results, 9 pages)
- [ ] Multi-seed bootstrap CIs (Phase 9.2 in progress)
- [ ] \armD{} (constant-bias ablation) results (Phase 9.3 pending)
- [ ] Multilingual evaluation (MGSM 10 langs, Phase 9.4 pending)
- [ ] Final author biography photo + ORCID
- [ ] IEEE Access copyright + author agreements
