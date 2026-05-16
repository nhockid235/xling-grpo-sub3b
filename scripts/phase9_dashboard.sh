#!/bin/bash
# Phase 9 Mac-side dashboard. Pulls Vast.ai state + parses local trainer_state
# to print a paper-grade status report. Run as needed (no daemon).
#
# Usage: bash scripts/phase9_dashboard.sh

set -uo pipefail

VAST_HOST="root@202.122.49.242"
VAST_PORT="22569"
LOCAL_ROOT="/Users/vudang/PythonLab/Papper/xling-grpo-sub3b"
REMOTE_ROOT="/workspace/xling-grpo-sub3b"
START_EPOCH=1746686856  # 2026-05-08 06:47:36 UTC

NOW=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Use SSH process etime as source of truth (avoids Mac date epoch issues)
ETIME_RAW=$(ssh -p "$VAST_PORT" "$VAST_HOST" 'PID=$(pgrep -f "src/trainers/grpo.py" | head -1); ps -o etime= -p "$PID" 2>/dev/null | tr -d " "' 2>/dev/null | tail -1)
# Parse HH:MM:SS or DD-HH:MM:SS
if [[ "$ETIME_RAW" == *-* ]]; then
    DAYS=${ETIME_RAW%%-*}
    REST=${ETIME_RAW#*-}
    IFS=: read -r H M S <<< "$REST"
    ELAPSED=$((DAYS*86400 + H*3600 + M*60 + S))
else
    IFS=: read -r H M S <<< "$ETIME_RAW"
    ELAPSED=$((H*3600 + M*60 + S))
fi
H=$((ELAPSED / 3600)); M=$(((ELAPSED % 3600) / 60))
COST=$(awk -v e=$ELAPSED 'BEGIN{printf "%.2f", (e/3600)*1.50}')

echo "================================================================"
echo "  Phase 9 dashboard — $NOW"
echo "  Elapsed since launch: ${H}h${M}m   Cost: \$${COST}"
echo "================================================================"

# 1. Pull live state from Vast.ai
echo
echo "── Vast.ai live state ──"
ssh -p "$VAST_PORT" "$VAST_HOST" '
echo "GPU:  $(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader)"
PROC=$(pgrep -af "src/trainers/grpo.py" | grep -v grep | head -1)
if [[ -n "$PROC" ]]; then
    PID=$(echo "$PROC" | awk "{print \$1}")
    ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d " ")
    echo "Train: ALIVE pid=$PID elapsed=$ETIME"
else
    echo "Train: NO PROC (cell finished or crashed)"
fi
echo
echo "── Cells completed (PASS markers) ──"
grep -c "PASS" /workspace/phase9.log 2>/dev/null
echo "── Checkpoints saved ──"
find /workspace/xling-grpo-sub3b/results/grpo -maxdepth 2 -name "checkpoint-*" -type d 2>/dev/null | sort
echo "── Eval JSONs ──"
find /workspace/xling-grpo-sub3b/results/eval -name "*.json" 2>/dev/null | wc -l
find /workspace/xling-grpo-sub3b/results/eval -name "*.json" 2>/dev/null | head -20
' 2>&1 | grep -v "^Welcome\|^Have fun"

# 2. Pull all trainer_state.json files to local
echo
echo "── Rsync trainer_state files ──"
rsync -az -e "ssh -p $VAST_PORT" --include="*/" --include="trainer_state.json" --exclude="*" \
    "$VAST_HOST:$REMOTE_ROOT/results/grpo/" "$LOCAL_ROOT/results/grpo/" 2>&1 | tail -3

# 3. Pull eval JSONs
rsync -az -e "ssh -p $VAST_PORT" --include="*/" --include="*.json" --exclude="*" \
    "$VAST_HOST:$REMOTE_ROOT/results/eval/" "$LOCAL_ROOT/results/eval/" 2>&1 | tail -3

# 4. Parse trainer_state files for paper-grade metrics
echo
echo "── Per-cell metrics (parsed from local trainer_state.json) ──"
python3 - <<'PY'
import json
from pathlib import Path

root = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b/results/grpo")
for ts_file in sorted(root.glob("*/checkpoint-*/trainer_state.json")):
    cell = ts_file.parent.parent.name
    step = ts_file.parent.name.replace("checkpoint-", "")
    try:
        with ts_file.open() as f:
            d = json.load(f)
    except Exception as e:
        print(f"  {cell}/ckpt-{step}: parse error {e}")
        continue
    history = d.get("log_history", [])
    if not history:
        continue
    last = history[-1]
    first = history[0]
    print(f"  {cell:42s} step={d['global_step']:3d}/{d['max_steps']:<3d} "
          f"r1={last.get('rewards/r1_correctness', 0):.3f} "
          f"r2={last.get('rewards/r2_format', 0):.3f} "
          f"reward={last.get('reward', 0):.3f} "
          f"kl={last.get('kl', 0):.5f} "
          f"len={last.get('completion_length', 0):.0f}")
    # Trend r1 first→last
    r1_first = first.get("rewards/r1_correctness", 0)
    r1_last  = last.get("rewards/r1_correctness", 0)
    r2_first = first.get("rewards/r2_format", 0)
    r2_last  = last.get("rewards/r2_format", 0)
    delta_r1 = r1_last - r1_first
    delta_r2 = r2_last - r2_first
    print(f"  {'':42s} Δr1={delta_r1:+.3f} Δr2={delta_r2:+.3f} "
          f"({len(history)} log points step {first['step']}→{last['step']})")
PY

# 5. Check for eval results
echo
echo "── Eval JSONs (local) ──"
find "$LOCAL_ROOT/results/eval" -name "*.json" 2>/dev/null | sort | while read -r f; do
    base=$(basename "$f" .json)
    pass=$(python3 -c "import json; d=json.load(open('$f')); print(f\"pass@1={d.get('pass_at_1','?'):.4f}\" if isinstance(d.get('pass_at_1'),(int,float)) else 'pass@1=?')" 2>/dev/null)
    echo "  $base   $pass"
done

# 6. Phase 9 cell schedule + ETA
echo
echo "── Phase 9 cell schedule ──"
echo "  Cell 1/7: reproduce_open_rs seed=123  (A1 multi-seed)"
echo "  Cell 2/7: reproduce_open_rs seed=7    (A1 multi-seed)"
echo "  Cell 3/7: a2 seed=123                  (A2 multi-seed)"
echo "  Cell 4/7: a2 seed=7                    (A2 multi-seed)"
echo "  Cell 5/7: a3 seed=123                  (A3 multi-seed)"
echo "  Cell 6/7: a3 seed=7                    (A3 multi-seed)"
echo "  Cell 7/7: a4 seed=42                   (A4 const-bias ablation)"
echo
echo "  Per-cell budget: 100 steps × ~5min = 8h33m + eval 30-60min ≈ 9.5h"
echo "  Total 7 cells:   ~66h × \$1.50 = \$99 (revised from \$42 plan)"
echo "  Sequence ETA:    launch + 66h ≈ May 11 00:00 UTC"
echo "  Cell 1 ETA:      ~16:00 UTC May 8 (training) + eval"

echo
echo "================================================================"
