# Phase 0 — Dataset Availability & Schema Verification

**Verify date:** 2026-05-05
**Researcher:** `huggingface-skills:AGENTS` subagent
**Source authority:** HuggingFace Hub (live query)
**Scope:** 9 datasets dùng trong training, eval, và baseline reproduction

---

## Verbatim agent output

> Note from agent: `juletxara/mgsm` actually lists 10 non-en languages (es, fr, de, ru, zh, ja, th, sw, bn) plus `en`, totaling 11 — `te` (Telugu) is in the description but not in the tag listing. Let me verify quickly.

### Dataset Verification Report

| Dataset | Avail | License | Rows / Splits | Key Fields | Notes |
|---|---|---|---|---|---|
| `AI-MO/NuminaMath-CoT` | OK | apache-2.0 | ~860K, `train`/`test` | `problem`, `solution`, `source` | Updated Nov 2024, 213K downloads. Stable. |
| `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated` | OK | not declared in card metadata (parent MetaMathQA = MIT) | 40K, JSON format | `query_vi`, `response_vi`, plus EN parallel fields | MT via Google Translate — quality is mid; expect math-symbol/equation breakage. License field empty on Hub — verify before publish. |
| `knoveleng/open-rs` | OK | mit | 7K `train` only | `problem`, `solution`, `answer`, `level` | NO separate Stage-1 split — single train file. Filter by `level`/source to reproduce Stage-1. arXiv:2503.16219. |
| `openai/gsm8k` (canonical) | OK | mit | 7.47K train / 1.32K test | `question`, `answer` | Live, 11M downloads. Use `main` config. |
| `HuggingFaceH4/MATH-500` | OK | not declared (source = MIT PRM800K) | 500 in `test` | `problem`, `solution`, `answer`, `subject`, `level`, `unique_id` | Single split. |
| `Maxwell-Jia/AIME_2024` | OK | mit | 30 records | `ID`, `Problem`, `Solution`, `Answer` | Field names capitalized — careful in loader. AIME I + II. |
| `juletxara/mgsm` | OK | cc-by-sa-4.0 | 250/lang test + small train shots | `question`, `answer`, `answer_number`, `equation_solution` | Tags list 10 langs (en, es, fr, de, ru, zh, ja, th, sw, bn). Card text mentions Telugu (`te`) too — confirm by inspecting splits at load time. Realistically plan for 10 langs guaranteed. |
| `Mathoctopus/MSVAMP` | OK | apache-2.0 | ~1K/lang × 10 langs | `id`, `question`, `answer`, `answer_number` | Languages: bn, zh, en, fr, de, ja, ru, es, sw, th. Last updated Oct 2023 — stable but stale. |
| **MGSM-Pro** | PARTIAL | cc-by-4.0 (`McGill-NLP/mgsm-pro`) | 10K-100K, 9 langs (NOT 47) | unspecified — card says "complete dataset will be updated soon" | Found `McGill-NLP/mgsm-pro` (official-looking, 9 langs, symbolic+IC variants) and a near-empty `Tiany1/Mgsm_pro`. **No 47-language MGSM-Pro exists publicly as of 2026-05-05.** Activate backup plan: hand-translate 100 MATH-500 VI samples + use juletxara/mgsm + MSVAMP for breadth. |

### Top 5 dataset risks

- **MGSM-Pro 47-lang version does not exist**: only McGill-NLP partial 9-lang release (still marked "to be updated"). Drop from primary eval or pin McGill version with caveat in paper.
- **5CD-AI Vietnamese MetaMathQA license unset** on Hub metadata; must email authors or fall back to manual MT before public release — blocker for arXiv reproducibility appendix.
- **MGSM language coverage mismatch**: card claims 11 langs incl. Telugu, tags expose 10. Validate at load time; do not hardcode 11 in `eval/mgsm.py`.
- **GSM8K leakage into NuminaMath / MetaMathQA-VI**: both training pools known to overlap GSM8K test. Decontamination (8-gram match) is mandatory before SFT — paper-rejection risk.
- **Open-RS has no Stage-1 split**: 7K is one flat train file. Reproduction script must filter by source field (`open-s1` vs `open-deepscaler`) — cannot rely on dataset splits.

---

## Phân tích & action items (main agent)

### Mapping vào CLAUDE.md hiện tại

| Dataset | CLAUDE.md status | Cần update? |
|---|---|---|
| NuminaMath-CoT | ✅ correct | Không |
| MetaMathQA-VI | ✅ ID đúng | Note license blocker vào Pitfalls |
| Open-RS | ⚠️ "check repo" | Đổi thành "mit" + note no Stage-1 split |
| GSM8K | ⚠️ thiếu canonical name | Đổi `gsm8k` → `openai/gsm8k` config `main` |
| MATH-500 | ✅ correct | Không |
| AIME-2024 | ⚠️ thiếu cảnh báo field caps | Note vào Pitfalls |
| MGSM | ⚠️ ghi 11 langs | Đổi thành 10 langs guaranteed |
| MSVAMP | ✅ correct | Không |
| MGSM-Pro | ⚠️ "TBD" | Đổi sang `McGill-NLP/mgsm-pro` partial + backup plan |

### Action items đã lưu vào `phase0_00_summary`

- Edit 3 trong summary đã capture các thay đổi ở § Datasets.
- Pitfall #15, #18 trong summary đã capture MGSM-Pro fallback và AIME field caps.

### Risk còn open (chưa giải quyết tại Phase 0)

1. **License `5CD-AI/...VI`:** chưa biết tác giả response email khi nào. Mitigation paper-time: nếu chưa có response trước W7, fallback dùng `meta-math/MetaMathQA` + tự MT 7K samples bằng GPT-4o (có cost ~$10).
2. **Decontamination quy mô:** 860K NuminaMath-CoT × 1.32K GSM8K test × 500 MATH-500 × 30 AIME = O(billion) 8-gram comparisons. Cần dùng Bloom filter hoặc datasketch MinHashLSH. Sẽ research tại Phase 2 (data agent).
3. **Open-RS source filter:** cần inspect dataset thực tế xem có column `source` không hay phải dùng `level`. Nếu không có → cần re-derive từ `knoveleng/open-s1` + `knoveleng/open-deepscaler` riêng. Sẽ check tại Phase 2.

### Quy chuẩn loader cho `data/prepare_*.py`

```python
# AIME — chú ý field capitalization
DATASETS = {
    "aime2024": {
        "id": "Maxwell-Jia/AIME_2024",
        "fields": {"problem": "Problem", "answer": "Answer", "id": "ID"},  # MAP capital
    },
    "gsm8k": {
        "id": "openai/gsm8k",   # CANONICAL not just "gsm8k"
        "config": "main",
        "split": "test",
        "fields": {"problem": "question", "answer": "answer"},
    },
    "math500": {
        "id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "fields": {"problem": "problem", "answer": "answer", "level": "level"},
    },
    "mgsm": {
        "id": "juletxara/mgsm",
        "langs": ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn"],  # 10 GUARANTEED
        "split": "test",
        "fields": {"problem": "question", "answer": "answer_number"},
    },
    "msvamp": {
        "id": "Mathoctopus/MSVAMP",
        "langs": ["bn", "zh", "en", "fr", "de", "ja", "ru", "es", "sw", "th"],
        "split": "test",
        "fields": {"problem": "question", "answer": "answer_number"},
    },
}
```

### Sources cited

- `https://huggingface.co/datasets/AI-MO/NuminaMath-CoT`
- `https://huggingface.co/datasets/5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated`
- `https://huggingface.co/datasets/knoveleng/open-rs`
- `https://huggingface.co/datasets/openai/gsm8k`
- `https://huggingface.co/datasets/HuggingFaceH4/MATH-500`
- `https://huggingface.co/datasets/Maxwell-Jia/AIME_2024`
- `https://huggingface.co/datasets/juletxara/mgsm`
- `https://huggingface.co/datasets/Mathoctopus/MSVAMP`
- `https://huggingface.co/datasets/McGill-NLP/mgsm-pro`
