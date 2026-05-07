# Reports — xling-grpo-sub3b

Thư mục này chứa toàn bộ research reports, decision logs, và verification artifacts cho dự án **"Does English GRPO Transfer? A Controlled Study of Cross-Lingual Math Reasoning in Sub-3B Models"**.

> ⚠️ **Mọi file trong thư mục này là tư liệu cần để upgrade paper.** Không xoá. Reference trực tiếp từ `paper/appendix.tex` khi viết Reproducibility section.

---

## Index

### Phase 0 — Pre-build research (2026-05-05)

| File | Mục đích | Verdict |
|---|---|---|
| `phase0_00_summary_critical_fixes.md` | Tổng hợp 4 critical fixes phải áp vào CLAUDE.md trước khi build skeleton | 🔴 4 critical, 🟡 5 medium, 🟢 7 confirmed |
| `phase0_01_datasets_verification.md` | Verify availability + schema 9 HuggingFace datasets | 8/9 OK, MGSM-Pro 47-langs không tồn tại |
| `phase0_02_trl_api_research.md` | TRL API surface (GRPOTrainer, GRPOConfig, SFTTrainer) | TRL 0.13.0 KHÔNG có GRPOTrainer — phải đổi sang 0.14+ |
| `phase0_03_open_rs_reproduction.md` | Open-RS hyperparams + Stage-1/Exp2 setup chính xác | Base = DeepSeek-R1-Distill-Qwen-1.5B (không phải Qwen2.5) |

### Phases sau (sẽ thêm khi tiến triển)

- `phase1_bootstrap_log.md` — TBD
- `phase2_*` — TBD (rewards, data, eval, trainer-plan)
- `phase3_trainer_integration.md` — TBD
- `phase4_*` — TBD
- `phase5_validation_log.md` — TBD
- `weekN_decision_log.md` — gating decisions W2/W4/W4-7
- `experiment_runs/` — per-cell run logs (9 cells × 3 seeds)

---

## Quy ước

- **Tên file:** `phaseN_NN_topic.md` hoặc `weekN_topic.md`. NN là số thứ tự trong phase.
- **Nội dung:** Phần "Verbatim agent output" giữ NGUYÊN văn từ agent — không paraphrase. Phần "Phân tích & action items" của tôi (main agent) thêm phía dưới.
- **Sources:** Mọi finding phải có URL/source. Không quote từ trí nhớ.
- **Date stamp:** Ghi ngày verify ở đầu mỗi file.
