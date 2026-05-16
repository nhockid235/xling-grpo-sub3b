#!/bin/bash
# W1.7 — Re-eval all 10 LoRA ckpts × seeds on AMC23 với fixed maj@4 code.
# Mục đích: đóng Issue #3 (maj@4=0 bug).
# Ckpts: base + a1(42,123,7) + a2(42,123,7) + a3(42,123,7) + a4(42 only for now).
# After A4 multi-seed (W1.9) hoàn tất, sẽ chạy thêm a4 seeds 123 + 7.
set -euo pipefail

source ~/.env_papper 2>/dev/null || true
source /venv/main/bin/activate
cd /workspace/xling-grpo-sub3b

LOG=results/eval/w17_$(date -u +%Y%m%dT%H%M%SZ).log
echo "[W1.7] start $(date -u +%FT%TZ)" | tee -a "$LOG"

# Eval list: (run_id, ckpt_path_or_hf_id, seed)
ARMS=(
  "base_distill_v2|deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|42"
  "reproduce_openrs_rs2_42_v2|results/grpo/reproduce_openrs_rs2_42/keep_step50|42"
  "reproduce_openrs_rs2_123_v2|results/grpo/reproduce_openrs_rs2_123/keep_step50|123"
  "reproduce_openrs_rs2_7_v2|results/grpo/reproduce_openrs_rs2_7/keep_step50|7"
  "a2_vi_42_v2|results/grpo/a2_vi_42/keep_step50|42"
  "a2_vi_123_v2|results/grpo/a2_vi_123/keep_step50|123"
  "a2_vi_7_v2|results/grpo/a2_vi_7/keep_step50|7"
  "a3_enlang_42_v2|results/grpo/a3_enlang_42/keep_step50|42"
  "a3_enlang_123_v2|results/grpo/a3_enlang_123/keep_step50|123"
  "a3_enlang_7_v2|results/grpo/a3_enlang_7/keep_step50|7"
  "a4_const_bias_42_v2|results/grpo/a4_const_bias_42/keep_step50|42"
)

for ARM in "${ARMS[@]}"; do
  IFS='|' read -r RID CKPT SEED <<< "$ARM"
  OUT="results/eval/w17_${RID}/"
  mkdir -p "$OUT"
  echo "[W1.7] === $RID  ckpt=$CKPT  seed=$SEED ===" | tee -a "$LOG"
  python -u src/eval/runner.py \
      --checkpoint "$CKPT" \
      --benchmarks amc23 \
      --config configs/eval.yaml \
      --output_dir "$OUT" \
      --run_id "$RID" \
      --seed "$SEED" 2>&1 | tee -a "$LOG"
done

echo "[W1.7] done $(date -u +%FT%TZ)" | tee -a "$LOG"

# Aggregate report
python - <<'PY' | tee -a "$LOG"
import json, glob, statistics, re
arms = {}
for p in sorted(glob.glob("results/eval/w17_*_v2/*amc23*.json")):
    d = json.load(open(p))
    rid = d["run_id"]
    m = re.match(r"(.+?)_(\d+)_v2$", rid)
    if not m:
        arms.setdefault(rid, []).append((rid, d.get("pass_at_1"), d.get("maj_at_4")))
        continue
    arm, seed = m.group(1), int(m.group(2))
    arms.setdefault(arm, []).append((seed, d.get("pass_at_1"), d.get("maj_at_4")))

print("\n========== W1.7 AMC23 maj@4 RE-EVAL (post-fix) ==========")
print(f"{'arm':<30} {'seeds':<14} {'pass@1 (mean±σ)':<22} {'maj@4 (mean±σ)':<22}")
print("-" * 90)
for arm in sorted(arms):
    rows = arms[arm]
    seeds = [r[0] for r in rows]
    p1 = [r[1] for r in rows if r[1] is not None]
    m4 = [r[2] for r in rows if r[2] is not None]
    p1_s = f"{statistics.mean(p1)*100:.2f}±{(statistics.stdev(p1)*100 if len(p1)>1 else 0):.2f}" if p1 else "—"
    m4_s = f"{statistics.mean(m4)*100:.2f}±{(statistics.stdev(m4)*100 if len(m4)>1 else 0):.2f}" if m4 else "—"
    print(f"{arm:<30} {str(seeds):<14} {p1_s:<22} {m4_s:<22}")
PY
