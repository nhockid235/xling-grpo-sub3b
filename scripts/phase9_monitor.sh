#!/bin/bash
# Phase 9 cost + progress monitor — run trên Mac, polls Vast.ai.
# Auto-rsync ckpts + log mỗi 60s. Reports cost dựa trên wallclock.
#
# Usage:
#   bash scripts/phase9_monitor.sh <ssh_host_port> <hourly_rate>
#   Example: bash scripts/phase9_monitor.sh "ssh -p 22569 root@202.122.49.242" 1.20

set -uo pipefail

SSH_CMD="${1:?Usage: $0 \"ssh -p PORT root@HOST\" RATE_PER_HOUR}"
HOURLY="${2:-1.50}"

LOCAL_DIR="/Users/vudang/PythonLab/Papper/results/training_phase9"
mkdir -p "$LOCAL_DIR"

START=$(date +%s)
echo "[$(date)] Phase 9 monitor start. SSH=$SSH_CMD rate=\$$HOURLY/h"

while true; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START))
    HOURS=$(echo "scale=2; $ELAPSED/3600" | bc)
    COST=$(echo "scale=2; $HOURS * $HOURLY" | bc)

    # Sync log + metrics
    eval "$SSH_CMD" 'tail -100 /workspace/phase9.log 2>/dev/null > /tmp/phase9_tail.log; tail -50 /workspace/training.log 2>/dev/null > /tmp/training_tail.log; tail -50 /workspace/training_a3.log 2>/dev/null > /tmp/training_a3_tail.log'

    eval "$SSH_CMD" -o ServerAliveInterval=15 'cat /workspace/phase9.log 2>/dev/null' > "$LOCAL_DIR/phase9.log" 2>/dev/null || true

    # Status
    echo "[$(date)] Elapsed: ${HOURS}h, Cost: \$$COST"

    # Check if done
    if eval "$SSH_CMD" 'grep -q "PHASE9_EXIT=0" /workspace/phase9.log 2>/dev/null'; then
        echo "[$(date)] PHASE 9 DONE ✓"
        # Final sync ckpts + eval JSONs
        rsync -az -e "$SSH_CMD" --exclude='*/optimizer.pt' --exclude='*/keep_step50' \
            root@:/workspace/xling-grpo-sub3b/results/grpo/ "$LOCAL_DIR/grpo/" 2>/dev/null || true
        rsync -az -e "$SSH_CMD" \
            root@:/workspace/xling-grpo-sub3b/results/eval/ "$LOCAL_DIR/eval/" 2>/dev/null || true
        break
    fi

    sleep 60
done

echo "[$(date)] Total elapsed: ${HOURS}h, Final cost: \$$COST"
