# Phase 0 Summary — Critical fixes phải áp dụng

**Verify date:** 2026-05-05
**Source:** 3 parallel research agents (xem 3 reports chi tiết: `phase0_01`, `phase0_02`, `phase0_03`)
**Status:** ⛔ BLOCKING — phải sửa CLAUDE.md trước khi launch Phase 1

---

## Tóm tắt 30 giây

CLAUDE.md gốc có **4 lỗi configuration sẽ phá vỡ pipeline ngay tại bước import** hoặc khi reproduce baseline. Mỗi lỗi đã được trace tới source code/dataset trên GitHub/HF. Toàn bộ tham số GRPO hyperparams cần điều chỉnh để khớp Open-RS thật.

---

## 🔴 CRITICAL fixes (4)

### 1. TRL version: `0.13.0` → `0.14.0+`

**Vấn đề:** `GRPOTrainer` và `GRPOConfig` **KHÔNG TỒN TẠI** trong TRL 0.13.0. GRPOTrainer chỉ xuất hiện từ TRL 0.14.0 (release 2025-01-29).

**Pin `trl==0.13.0` → `from trl import GRPOTrainer` → ImportError ngay lập tức.**

**Hành động:**
- `pyproject.toml`: `trl>=0.14.0,<0.16.0` (recommended `trl==0.15.x` cho vLLM-stable GRPO)
- `CLAUDE.md` § Code standards / Core deps: sửa version.

**Source:** GitHub TRL tag v0.13.0 — không có file `trl/trainer/grpo_*.py`. Xác nhận tại `phase0_02_trl_api_research.md`.

---

### 2. Open-RS base model: `Qwen2.5-1.5B-Instruct` → `DeepSeek-R1-Distill-Qwen-1.5B`

**Vấn đề:** Open-RS paper (arXiv:2503.16219) baseline AMC23 62.9% → 80.0% (RS2) chỉ đúng với base model `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Nếu reproduce từ `Qwen2.5-1.5B-Instruct` thì baseline AMC23 ~30% và **không thể vọt lên 80%** trong 50 GRPO steps.

**Hành động:**
- W2 gating reproduction phải dùng đúng `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- Sau khi pass gate → primary experiments mới dùng Qwen2.5-1.5B-Instruct + Qwen2.5-3B-Instruct + Llama-3.2-3B-Instruct theo plan gốc.
- `CLAUDE.md` § Decision points W2: thêm "Reproduce với DeepSeek-R1-Distill, target = match RS2 @ 50 steps".

**Source:** `recipes/grpo.yaml` của repo `knoveleng/open-rs`. Xác nhận tại `phase0_03_open_rs_reproduction.md`.

---

### 3. GRPO hyperparams: 6 chỗ sai so với Open-RS thật

| Param | CLAUDE.md hiện tại | Open-RS thật | Note |
|---|---|---|---|
| `num_generations` | 4 | **6** | TRL field, không phải `group_size` |
| `max_new_tokens` | 2048 | **3584** (Exp2/3); Exp1 dùng 4096 | tránh truncate CoT |
| `temperature` | 1.0 | **0.7** | rollout temperature |
| `lr_scheduler` | `cosine` | **`cosine_with_min_lr`** (`min_lr_rate=0.1`) | min lr floor |
| LoRA r=16 default | có | **full-param** (no PEFT) | Open-RS không LoRA cho 1.5B |
| `group_size=8` | có | **không phải param TRL** | xóa, dùng `num_generations` |

**Hành động:** Sửa CLAUDE.md § Hyperparameters / GRPO. Set defaults theo Open-RS Exp2.

**Note quan trọng cho reproduction (RS2):** reward = `accuracy + format` (KHÔNG phải `cosine + format` như default YAML), checkpoint @ 50 steps. YAML mặc định trong repo là Exp3 setup → ra ~72.5% AMC23, không match 80%.

**Source:** `recipes/grpo.yaml` raw + paper Table 1, §3.5. Xác nhận tại `phase0_03_open_rs_reproduction.md`.

---

### 4. MGSM-Pro 47 langs: KHÔNG TỒN TẠI

**Vấn đề:** Search HuggingFace cho "MGSM-Pro" / "MGSM Pro" / "X-MGSM" / "MGSM-X" — không có dataset 47 ngôn ngữ. Có duy nhất `McGill-NLP/mgsm-pro` (9 langs, marked "complete dataset will be updated soon"). Một upload near-empty `Tiany1/Mgsm_pro` không dùng được.

**Hành động:**
- Activate backup plan ngay lập tức (đã ghi trong `luuy`):
  - **Hand-translate** 100 samples MATH-500 → VI (tôi native VI).
  - Eval breadth: `juletxara/mgsm` (10 langs guaranteed) + `Mathoctopus/MSVAMP` (10 langs).
  - Optional: thêm `McGill-NLP/mgsm-pro` 9 langs với caveat "partial release".
- `CLAUDE.md` § Datasets: ghi rõ MGSM-Pro = `McGill-NLP/mgsm-pro` partial, không dùng làm primary.
- `CLAUDE.md` § Common pitfalls: thêm "MGSM-Pro 47-lang không tồn tại — backup là MGSM 10 + MSVAMP 10 + 100 hand-translated VI MATH-500".

**Source:** HF Hub search 2026-05-05. Xác nhận tại `phase0_01_datasets_verification.md`.

---

## 🟡 MEDIUM notes (5)

### 5. MGSM language coverage: 10 langs guaranteed, không phải 11

`juletxara/mgsm` tags expose **10 langs**: `en, es, fr, de, ru, zh, ja, th, sw, bn`. Card text mention Telugu (`te`) nhưng không có trong tags. → **Đừng hardcode 11 trong `eval/mgsm.py`**, validate splits at load time.

### 6. Dataset license blocker: `5CD-AI/Vietnamese-MetaMathQA-40K-gg-translated`

License field **unset** trên HF metadata. Parent MetaMathQA = MIT, nhưng MT subset chưa declare license rõ. → Liên hệ tác giả (5CD-AI / Vietnamese AI team) hoặc fallback hand-MT trước khi public arXiv repo. Blocker cho § Reproducibility appendix.

### 7. Dataset field naming gotchas

- `Maxwell-Jia/AIME_2024`: fields **CAPITALIZED** (`Problem`, `Solution`, `Answer`, `ID`) — careful trong loader, không dùng `question`/`answer`.
- `gsm8k` → dùng tên canonical `openai/gsm8k` config `main`.
- `knoveleng/open-rs`: KHÔNG có Stage-1 split — single 7K train file, filter theo `level` (`Hard` ~6K / `Easy` ~1K) hoặc theo source field.

### 8. TRL 0.14+ API gotchas (xem chi tiết `phase0_02`)

- Reward function MUST return `list[float]`, length == `len(prompts)`. Return scalar/tensor → silent break.
- Reward function signature keyword-only: `def reward(prompts, completions, **kwargs)`.
- Dataset GRPO MUST có column tên `"prompt"` (không phải `"text"`/`"question"`).
- `num_generations` default = 8 (đã set 6 theo Open-RS).
- `max_completion_length` default = 256 (phải override 3584).
- Use `processing_class=tokenizer`, không `tokenizer=` (deprecated).
- SFTTrainer trong 0.13/0.14 không auto-handle `messages` column — phải `apply_chat_template` trước hoặc dùng `formatting_func`.

### 9. Open-RS effective batch size

Open-RS report: **4× A40 48GB**, eff batch = `4 GPU × 6 batch × 4 grad_accum = 96 prompts/step` (× G=6 = 576 generations). Trên 1× A100 80GB (RunPod budget) phải tăng `gradient_accumulation_steps` lên **16** để match effective batch. Nếu giảm batch sẽ mismatch reward variance ước lượng.

---

## 🟢 CONFIRMED (không đổi)

1. VRAM 1.5B + G=6 + max_len=3584 + full-param ≈ 40GB → OK trên A100 80GB.
2. `transformers==4.46.3`, `accelerate==1.2.1`, `peft==0.14.0`, `vllm==0.7.2` — compatible với TRL 0.14+.
3. `bf16=true`, `attn_implementation: flash_attention_2`, `gradient_checkpointing=true` (use_reentrant=false) — giữ nguyên.
4. `vllm_enforce_eager: true` cho Open-RS A40, có thể tắt trên A100 để tăng tốc.
5. KL beta `0.04` — match TRL default.
6. Save checkpoint mỗi 50 steps (Open-RS recommend) — peak performance @ 50-100, sau đó degrade do length drift + multilingual drift.
7. lighteval với custom tasks Open-RS — dùng đúng template để số AMC23 so sánh được.

---

## Action checklist trước Phase 1

- [ ] User approve sửa CLAUDE.md (xem proposed diff bên dưới)
- [ ] Edit CLAUDE.md theo 4 critical fixes
- [ ] Edit `pyproject.toml` template trong CLAUDE.md (TRL version)
- [ ] Đảm bảo `phase0_01`, `phase0_02`, `phase0_03` được lưu đầy đủ trong `reports/`
- [ ] Launch Phase 1 (skeleton bootstrap)

## Proposed CLAUDE.md edits

### Edit 1: § Code standards / Core deps

```diff
-  - `trl==0.13.0` (verify GRPOTrainer signature in this version)
+  - `trl>=0.14.0,<0.16.0` (GRPOTrainer first appeared in 0.14.0, NOT 0.13.0)
```

### Edit 2: § Hyperparameters / GRPO

```diff
 **GRPO:**
 - `lr=1e-6` <- critical, do NOT use SFT lr
-- `num_generations=4` (G doubling doubles VRAM)
-- `max_new_tokens=2048`
-- `temperature=1.0` for rollouts
+- `num_generations=6` (Open-RS Exp2 setup; G doubling doubles VRAM)
+- `max_completion_length=3584` (Open-RS Exp2; Exp1 used 4096)
+- `max_prompt_length=512`
+- `temperature=0.7` for rollouts (Open-RS verified)
 - `kl_beta=0.04`
-- `group_size=8`
-- `total_steps=500` (matches Open-RS Stage-1)
+- `total_steps=500` (save_steps=50, peak ckpt @ 50-100)
+- `lr_scheduler=cosine_with_min_lr` (min_lr_rate=0.1, warmup_ratio=0.1)
+- `bf16=true`, `flash_attention_2`, `gradient_checkpointing=true`
+- LoRA: full-param recommended for 1.5B (Open-RS does this); LoRA r=16 fallback for 3B if VRAM OOM
+- Effective batch: 96 prompts/step (6 batch × 4 grad_accum × 4 GPU). On 1×A100, set grad_accum=16 to match.
```

### Edit 3: § Datasets

```diff
-| MGSM-Pro | TBD — verify HF availability W1 | Multilingual eval (47 langs) | TBD | TBD |
+| MGSM-Pro | `McGill-NLP/mgsm-pro` (9 langs only, partial release) | Optional eval | cc-by-4.0 | ~10K |
```

Thêm note dưới bảng:

```diff
+**MGSM 10 langs guaranteed:** en, es, fr, de, ru, zh, ja, th, sw, bn (Telugu mention in card but NOT in tags — verify at load time).
+**MSVAMP 10 langs:** bn, zh, en, fr, de, ja, ru, es, sw, th.
+**AIME-2024 schema:** fields are CAPITALIZED (`Problem`, `Solution`, `Answer`, `ID`).
+**GSM8K canonical name:** `openai/gsm8k` config `main`.
+**Open-RS dataset note:** Single `train` split (7K), no Stage-1 separation. Filter by `level` (Hard ~6K, Easy ~1K) or source field.
```

### Edit 4: § Common pitfalls — thêm 4 mục mới

```diff
+13. **TRL version pin.** `trl==0.13.0` does NOT contain GRPOTrainer. Use `trl>=0.14.0,<0.16.0`. Adding 0.13 = ImportError.
+14. **Open-RS reproduction base model.** Must be `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, not `Qwen2.5-1.5B-Instruct`. Distill checkpoint AMC23 baseline 62.9%; from-Qwen2.5 baseline ~30% — cannot match 80% in 50 steps.
+15. **MGSM-Pro 47-language version does not exist publicly.** Only `McGill-NLP/mgsm-pro` 9 langs (partial). Backup plan = MGSM 10 + MSVAMP 10 + 100 hand-translated VI MATH-500.
+16. **GRPO reward function signature.** Must `def f(prompts, completions, **kwargs) -> list[float]`. Length must match `len(prompts)`. Return scalar = silent break in TRL.
+17. **Dataset GRPO column name.** TRL GRPOTrainer expects `"prompt"` column. If dataset has `"question"`, rename before passing.
+18. **AIME 2024 field caps.** `Maxwell-Jia/AIME_2024` uses `Problem`/`Solution`/`Answer`/`ID` (capitalized). Lowercase access = KeyError.
```

### Edit 5: § Decision points W2

```diff
-| W2 | Reproduce Open-RS within 3pp on AMC23 | continue Idea 1 | pivot to Idea 4 (LIMO-style SFT) |
+| W2 | Reproduce Open-RS RS2 (base=DeepSeek-R1-Distill-Qwen-1.5B, reward=accuracy+format, ckpt @ 50 steps) within 3pp of AMC23 80.0% | continue Idea 1 | pivot to Idea 4 (LIMO-style SFT) |
```
