# IEEE Access rendering

IEEEtran-class rendering of the manuscript in `paper/main.tex`. Same
content, different style for IEEE Access journal submission (Roman-numeral
sections, structured abstract, IEEE keywords, IEEEtran bibliography
style, formal mathematical notation).

## Files

| File | Purpose |
|---|---|
| `main.tex` | IEEEtran journal-class manuscript |
| `refs.bib` | Bibliography (synced with the ACL `paper/refs.bib`) |
| `tables/` | LaTeX tables (synced with `paper/tables/`) |
| `figures/` | PDF figures (synced with `paper/figures/`) |
| `build.sh` | One-shot build script (calls `tectonic`) |
| `main.pdf` | Compiled output (11 pages) |

## Build

```bash
brew install tectonic           # macOS, one-time
bash build.sh                   # produces main.pdf
```

The build is hermetic via Tectonic — all packages are downloaded from
CTAN on first compile and cached locally. No `pdflatex`/`bibtex`
toolchain required.

## Differences from the ACL rendering

- **Class:** `IEEEtran` (journal mode) vs `article` + `acl.sty`
- **Abstract:** Same structured abstract (Background / Methods / Results /
  Conclusion) in both renderings
- **Sections:** Roman numerals (I, II, …) per IEEE convention
- **Citations:** Numbered IEEEtran style `[1]` vs author-year `\citet`
- **Mechanism section:** Standalone Section VII proposing the
  PPO-clipping-geometry hypothesis, refuted by the constant-bias
  ablation A4
- **Reproducibility statement:** Inline subsection with full version
  pins (Python, torch, transformers, trl, vllm, etc.)
- **Author biography:** IEEE-style biography block at end

## Submission target

**IEEE Access** — open access, broad ML/NLP scope, ~6-week first-decision
review. Article processing charge \$1,950 (2026 rate). Submission portal:
<https://ieee.atyponrex.com/journal/ieee-access>.
