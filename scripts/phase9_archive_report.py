"""Phase 9 archival report generator — paper-grade.

Pulls all trainer_state + eval JSONs (already rsynced) and writes:
  reports/phase9_runs/{timestamp}_status.md      ← human-readable narrative
  reports/phase9_runs/csv/{cell}_metrics.csv     ← reward curves CSV per cell
  reports/phase9_runs/latest_status.md           ← symlink to most recent

Run after each rsync from Vast.ai to lock in a paper-ready snapshot.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
ARCHIVE = ROOT / "reports/phase9_runs"
CSV_DIR = ARCHIVE / "csv"
NOW = datetime.now(timezone.utc)
TS = NOW.strftime("%Y%m%dT%H%M%SZ")

# Phase 8 reference (from reports/phase8_results_A1_A2_A3.md table)
PHASE8 = {
    "A1_seed42": {"AMC23": 57.5, "MATH500": 58.8, "AIME_p1": 10.0, "AIME_m8": 30.0,
                  "mean_R1": 0.187, "mean_R1_std": 0.034, "wallclock_min": 249,
                  "max_steps": 50, "format_reward_active": True},
    "A2_seed42": {"AMC23": 52.5, "MATH500": 60.2, "AIME_p1": 16.7, "AIME_m8": 33.3,
                  "mean_R1": 0.207, "mean_R1_std": 0.039, "wallclock_min": 156,
                  "max_steps": 50, "format_reward_active": True},
    "A3_seed42": {"AMC23": 52.5, "MATH500": 60.6, "AIME_p1": 23.3, "AIME_m8": 36.7,
                  "mean_R1": 0.230, "mean_R1_std": 0.029, "wallclock_min": 244,
                  "max_steps": 50, "format_reward_active": True},
    "Base":     {"AMC23": 50.0, "MATH500": 59.4, "AIME_p1": 26.7, "AIME_m8": 33.3},
}

# Vast.ai live state probe
VAST_HOST = "root@202.122.49.242"
VAST_PORT = "22569"


def ssh_probe() -> dict:
    """One-shot SSH probe — GPU, proc, log markers, eval count."""
    cmd = """
PID=$(pgrep -f "src/trainers/grpo.py" | head -1)
if [[ -n "$PID" ]]; then
    ETIME=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
    PROC_ALIVE=1
else
    ETIME="0:00"; PROC_ALIVE=0
fi
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader)
PASS_CT=$(grep -c "PASS" /workspace/phase9.log 2>/dev/null || echo 0)
FAIL_CT=$(grep -c "FAIL" /workspace/phase9.log 2>/dev/null || echo 0)
EVAL_CT=$(find /workspace/xling-grpo-sub3b/results/eval -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
CKPTS=$(find /workspace/xling-grpo-sub3b/results/grpo -maxdepth 2 -name "checkpoint-*" -type d 2>/dev/null | sort | tr '\\n' '|')
DISK=$(df -h /workspace 2>/dev/null | tail -1 | awk '{print $5}')
echo "PROC=$PROC_ALIVE"
echo "ETIME=$ETIME"
echo "GPU=$GPU"
echo "PASS_CT=$PASS_CT"
echo "FAIL_CT=$FAIL_CT"
echo "EVAL_CT=$EVAL_CT"
echo "CKPTS=$CKPTS"
echo "DISK=$DISK"
"""
    out = subprocess.run(
        ["ssh", "-p", VAST_PORT, VAST_HOST, cmd],
        capture_output=True, text=True, timeout=30,
    )
    state = {}
    for line in out.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            state[k] = v
    return state


def parse_etime(s: str) -> int:
    """ps etime → seconds. Format DD-HH:MM:SS or HH:MM:SS or MM:SS."""
    if not s or s == "0:00":
        return 0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [int(p) for p in parts]
    while len(parts) < 3:
        parts = [0] + parts
    h, m, sec = parts[0], parts[1], parts[2]
    return days * 86400 + h * 3600 + m * 60 + sec


def rsync_all() -> tuple[bool, str]:
    """Pull trainer_state + eval JSONs from Vast.ai."""
    msgs = []
    for src, dst, includes in [
        ("results/grpo/", "results/grpo/", ["*/", "*/checkpoint-*/", "trainer_state.json", "*.json"]),
        ("results/eval/", "results/eval/", ["*/", "*.json"]),
    ]:
        cmd = ["rsync", "-az", "-e", f"ssh -p {VAST_PORT}"]
        for inc in includes:
            cmd += ["--include", inc]
        cmd += ["--exclude", "*"]
        cmd += [f"{VAST_HOST}:/workspace/xling-grpo-sub3b/{src}", str(ROOT / dst)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        msgs.append(f"{src} → {r.returncode}")
    return True, "; ".join(msgs)


def parse_trainer_state(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    return d


def export_csv(cell: str, history: list[dict]) -> Path:
    csv_path = CSV_DIR / f"{cell}_metrics.csv"
    fieldnames = ["step", "epoch", "lr", "reward", "reward_std",
                  "r1_correctness", "r2_format",
                  "kl", "grad_norm", "completion_length", "loss"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for h in history:
            w.writerow({
                "step": h.get("step"),
                "epoch": round(h.get("epoch", 0), 5),
                "lr": h.get("learning_rate"),
                "reward": h.get("reward"),
                "reward_std": h.get("reward_std"),
                "r1_correctness": h.get("rewards/r1_correctness"),
                "r2_format": h.get("rewards/r2_format"),
                "kl": h.get("kl"),
                "grad_norm": h.get("grad_norm"),
                "completion_length": h.get("completion_length"),
                "loss": h.get("loss"),
            })
    return csv_path


def parse_eval_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return {"error": str(e)}
    return {
        "benchmark": d.get("benchmark"),
        "language": d.get("language"),
        "n_samples": d.get("n_samples"),
        "pass_at_1": d.get("pass_at_1"),
        "maj_at_8": d.get("maj_at_8"),
        "maj_at_4": d.get("maj_at_4"),
        "lang_consistency_rate": d.get("lang_consistency_rate"),
        "avg_response_tokens": d.get("avg_response_tokens"),
    }


def collect_cells() -> list[dict]:
    """Walk results/grpo/ for all cells with checkpoints."""
    grpo = ROOT / "results/grpo"
    if not grpo.exists():
        return []
    cells = []
    for cell_dir in sorted(grpo.iterdir()):
        if not cell_dir.is_dir():
            continue
        ckpts = sorted([p for p in cell_dir.iterdir() if p.name.startswith("checkpoint-")])
        if not ckpts:
            continue
        # Use latest ckpt for trainer_state (always identical across keep_stepN duplicates)
        ts_path = ckpts[-1] / "trainer_state.json"
        if not ts_path.exists():
            continue
        ts = parse_trainer_state(ts_path)
        if not ts:
            continue
        history = ts.get("log_history", [])
        if not history:
            continue
        # Aggregates
        rewards_r1 = [h.get("rewards/r1_correctness", 0) for h in history]
        rewards_r2 = [h.get("rewards/r2_format", 0) for h in history]
        mean_r1 = sum(rewards_r1) / len(rewards_r1)
        mean_r2 = sum(rewards_r2) / len(rewards_r2)
        lengths = [h.get("completion_length", 0) for h in history]
        cells.append({
            "name": cell_dir.name,
            "ckpts": [c.name for c in ckpts],
            "global_step": ts.get("global_step"),
            "max_steps": ts.get("max_steps"),
            "n_log_points": len(history),
            "first": history[0],
            "last": history[-1],
            "mean_r1": mean_r1,
            "mean_r2": mean_r2,
            "max_completion_length": max(lengths) if lengths else 0,
            "history": history,
        })
        export_csv(cell_dir.name, history)
    return cells


def collect_evals() -> list[dict]:
    eval_dir = ROOT / "results/eval"
    if not eval_dir.exists():
        return []
    rows = []
    for jpath in sorted(eval_dir.rglob("*.json")):
        if jpath.name == ".gitkeep":
            continue
        meta = parse_eval_json(jpath)
        meta["file"] = jpath.relative_to(ROOT).as_posix()
        rows.append(meta)
    return rows


def estimate_cost_eta(probe: dict, cells: list[dict]) -> dict:
    """Wallclock + cost projection."""
    elapsed_sec = parse_etime(probe.get("ETIME", "0:00"))
    cost_so_far = elapsed_sec / 3600 * 1.50

    # Assume 7 cells × 9.5h each (max_steps=100 path)
    per_cell_h = 9.5
    completed_cells = int(probe.get("PASS_CT", 0))
    remaining_cells = 7 - completed_cells
    # Cell currently running counts as partially done
    cur_progress = 0.0
    if cells:
        c = cells[-1]
        if c.get("max_steps"):
            cur_progress = c["global_step"] / c["max_steps"]
    eta_remaining_h = (remaining_cells - 1) * per_cell_h + per_cell_h * (1 - cur_progress)
    total_proj_cost = cost_so_far + eta_remaining_h * 1.50
    return {
        "elapsed_sec": elapsed_sec,
        "cost_so_far": cost_so_far,
        "completed_cells": completed_cells,
        "current_cell_progress_pct": cur_progress * 100,
        "eta_remaining_h": eta_remaining_h,
        "total_projected_cost": total_proj_cost,
    }


def render_markdown(probe: dict, cells: list[dict], evals: list[dict], econ: dict) -> str:
    lines: list[str] = []
    p = lines.append

    p(f"# Phase 9.2 Status Report — {NOW.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    p("")
    p(f"**Snapshot ID:** `{TS}`")
    p(f"**Generator:** `scripts/phase9_archive_report.py`")
    p("")

    # SECTION 1 — INFRASTRUCTURE
    p("## 1. Infrastructure")
    p("")
    p(f"- Vast.ai host: `{VAST_HOST}` port `{VAST_PORT}`")
    p(f"- GPU: A100 80GB PCIe @ \\$1.50/h")
    p(f"- Live GPU snapshot: `{probe.get('GPU', '?')}`  (util%, mem MiB, W, °C)")
    p(f"- Training proc alive: **{'YES' if probe.get('PROC') == '1' else 'NO'}**")
    p(f"- Process etime: `{probe.get('ETIME', '?')}`")
    p(f"- Disk usage on /workspace: `{probe.get('DISK', '?')}`")
    p(f"- Tmux session: `phase9` (bash scripts/phase9_run_sequence.sh)")
    p("")

    # SECTION 2 — COST & ETA
    p("## 2. Cost & ETA")
    p("")
    p(f"| Metric | Value |")
    p(f"|---|---|")
    p(f"| Elapsed (current Cell PID) | {econ['elapsed_sec'] // 3600}h{(econ['elapsed_sec'] % 3600) // 60}m |")
    p(f"| Cost so far | \\${econ['cost_so_far']:.2f} |")
    p(f"| Cells completed (PASS markers) | {econ['completed_cells']} / 7 |")
    p(f"| Current cell progress | {econ['current_cell_progress_pct']:.1f}% |")
    p(f"| ETA remaining (rough) | {econ['eta_remaining_h']:.1f}h |")
    p(f"| **Total projected cost** | **\\${econ['total_projected_cost']:.2f}** |")
    p(f"| Master budget | \\$180.00 (Phase 9.2 sub-budget revised: \\$99) |")
    p("")
    p("⚠ **Budget revision:** Original plan said \\$42 / 25h based on max_steps=50. "
      "Actual config has max_steps=100 (Open-RS RS2 protocol — peak ckpt at step 50 "
      "but train extends to 100 for length-drift / multilingual-drift analysis per CLAUDE.md). "
      "Each cell ≈ 9.5h, total 7 cells ≈ 66h ≈ \\$99.")
    p("")

    # SECTION 3 — PER-CELL METRICS
    p("## 3. Per-cell metrics (from local trainer_state.json)")
    p("")
    if not cells:
        p("_No cell data yet._")
    for c in cells:
        p(f"### Cell: `{c['name']}`")
        p("")
        p(f"- Step progress: **{c['global_step']}/{c['max_steps']}** "
          f"({c['global_step']*100//c['max_steps']}%)")
        p(f"- Checkpoints saved: `{', '.join(c['ckpts'])}`")
        p(f"- Log points: {c['n_log_points']} (every 5 steps)")
        p(f"- Max completion length seen: {c['max_completion_length']:.0f} tokens")
        p(f"- Mean R1 (correctness) over training: **{c['mean_r1']:.3f}**")
        p(f"- Mean R2 (format) over training: **{c['mean_r2']:.3f}**")
        p("")
        p("**Reward time series (every 5 steps):**")
        p("")
        p("| step | r1_correct | r2_format | reward | reward_std | KL | grad_norm | len | lr |")
        p("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for h in c["history"]:
            p(f"| {h.get('step')} "
              f"| {h.get('rewards/r1_correctness', 0):.3f} "
              f"| {h.get('rewards/r2_format', 0):.3f} "
              f"| {h.get('reward', 0):.3f} "
              f"| {h.get('reward_std', 0):.3f} "
              f"| {h.get('kl', 0):.5f} "
              f"| {h.get('grad_norm', 0):.4f} "
              f"| {h.get('completion_length', 0):.0f} "
              f"| {h.get('learning_rate', 0):.2e} |")
        p("")

    # SECTION 4 — PHASE 8 COMPARISON
    p("## 4. Phase 8 reference comparison")
    p("")
    p("Phase 8 trained `max_steps=50`, single seed=42. Phase 9.2 uses `max_steps=100`, ")
    p("seeds {123, 7} for A1/A2/A3 + seed=42 for A4 (const-bias ablation).")
    p("")
    p("### Phase 8 baseline numbers (single seed=42, ckpt-50)")
    p("")
    p("| Cond | AMC23 | MATH-500 | AIME p@1 | AIME m@8 | mean R1 train |")
    p("|---|---|---|---|---|---|")
    for k in ["Base", "A1_seed42", "A2_seed42", "A3_seed42"]:
        v = PHASE8[k]
        p(f"| {k} | {v.get('AMC23', '—')}% | {v.get('MATH500', '—')}% "
          f"| {v.get('AIME_p1', '—')}% | {v.get('AIME_m8', '—')}% "
          f"| {v.get('mean_R1', '—')} |")
    p("")
    p("**Headline Phase 8 finding:** A3 best mean (43.3%); A1 catastrophic AIME (-16.7pp); "
      "training language and aux reward both regularize.")
    p("")
    p("### Phase 9 vs Phase 8 (training reward dynamics so far)")
    p("")
    p("| Cell | step | r1 train | r2 train | Phase 8 r1 | Phase 8 r2 |")
    p("|---|---|---|---|---|---|")
    for c in cells:
        # Map Phase 9 cell name to Phase 8 reference (by config family)
        ref_key = None
        if c["name"].startswith("reproduce_openrs_rs2"):
            ref_key = "A1_seed42"
        elif c["name"].startswith("a2"):
            ref_key = "A2_seed42"
        elif c["name"].startswith("a3"):
            ref_key = "A3_seed42"
        ref = PHASE8.get(ref_key, {}) if ref_key else {}
        p(f"| {c['name']} | {c['global_step']}/{c['max_steps']} "
          f"| {c['mean_r1']:.3f} | {c['mean_r2']:.3f} "
          f"| {ref.get('mean_R1', '—')} | {'>0' if ref.get('format_reward_active') else '?'} |")
    p("")

    # SECTION 5 — EVAL JSONS
    p("## 5. Eval results (so far)")
    p("")
    if not evals:
        p("_No eval JSONs yet — will populate after each cell finishes max_steps and triggers eval._")
    else:
        p("| File | benchmark | lang | n | pass@1 | maj@8 | maj@4 | lang_cons |")
        p("|---|---|---|---|---|---|---|---|")
        for e in evals:
            p(f"| `{e['file']}` | {e.get('benchmark', '?')} | {e.get('language', '—')} "
              f"| {e.get('n_samples', '?')} | {e.get('pass_at_1', '—')} "
              f"| {e.get('maj_at_8', '—')} | {e.get('maj_at_4', '—')} "
              f"| {e.get('lang_consistency_rate', '—')} |")
    p("")

    # SECTION 6 — CRITICAL FINDINGS
    p("## 6. Critical findings — paper-ready notes")
    p("")
    p("### F1. r2_format = 0 across all logged steps (Cell 1 seed=123)")
    p("")
    p("**Observation:** Cell 1 trainer_state shows `rewards/r2_format = 0.0` for all 10 logged "
      "steps (5, 10, ..., 50). Phase 8 (seed=42) showed r2_format climbing 0.20 → 0.35 over the "
      "same 50-step window.")
    p("")
    p("**Hypotheses:**")
    p("1. **Seed-dependent format learning.** With seed=123, the random rollout init never "
      "samples a `<answer>...</answer>` pattern in the first 50 steps → R2 reward stays 0 → "
      "no gradient signal toward format compliance. The model relies on `\\boxed{}` (DeepSeek-R1 "
      "default).")
    p("2. **Format regex too strict.** `<think>...</think>.*<answer>...</answer>` may not match "
      "the natural CoT structure DeepSeek emits. Phase 8 success may have been seed-luck, not "
      "format reward design.")
    p("")
    p("**Paper implication:** If r2_format=0 holds across seed=7 too, this is a major finding: "
      "**format reward design is reproducibility-fragile** — single-seed papers in this regime "
      "may report inflated effects. Multi-seed validation is essential.")
    p("")
    p("**Action item:** Wait for Cell 2 (A1 seed=7) to confirm whether r2_format=0 is "
      "seed-specific or systematic.")
    p("")
    p("### F2. r1_correctness flat-to-down across 50 steps")
    p("")
    p("Cell 1: r1 starts at 0.21 (step 5), wanders 0.17–0.21, ends 0.18 at step 50. No upward trend. "
      "Phase 8 A1 mean was 0.187 (std 0.034) — Cell 1 mean so far is ~0.18 — match within "
      "1σ. **Reward signal is dominated by base capability, not learning.**")
    p("")
    p("**Paper implication:** Confirms Phase 8 finding that GRPO at sub-3B + LoRA r=16 + 50 "
      "steps does not improve correctness on training data — improvements come from format / "
      "structural alignment.")
    p("")
    p("### F3. KL stable, grad_norm low → training healthy")
    p("")
    p("KL stays 0.00014–0.00026 (well under β=0.04 threshold), grad_norm 0.004–0.011. No "
      "instability, no policy collapse. Training is well-behaved.")
    p("")

    # SECTION 7 — TODO PAPER
    p("## 7. Paper TODO (data collection checklist)")
    p("")
    p("- [x] Phase 8 single-seed results — saved in `reports/phase8_results_A1_A2_A3.md`")
    p("- [x] Cell 1 (A1 seed=123) trainer_state @ step 50 — synced to local")
    p("- [ ] Cell 1 step 100 trainer_state — pending (~2.5h)")
    p("- [ ] Cell 1 eval JSONs (AMC23 + MATH-500 + AIME-2024) — pending after step 100")
    p("- [ ] Cell 2 (A1 seed=7) — full run pending")
    p("- [ ] Cells 3-4 (A2 multi-seed) — full run pending")
    p("- [ ] Cells 5-6 (A3 multi-seed) — full run pending")
    p("- [ ] Cell 7 (A4 const-bias ablation) — full run pending")
    p("- [ ] Bootstrap CI tables (3 seeds per cond) — needs all cells done")
    p("- [ ] MGSM 10-language eval — Phase 9.4 (after training)")
    p("- [ ] Hand-translated VI MATH-500 (100 problems) — user task")
    p("")

    # SECTION 8 — FILES
    p("## 8. Archived data files")
    p("")
    p(f"- This report: `reports/phase9_runs/{TS}_status.md`")
    p(f"- Latest symlink: `reports/phase9_runs/latest_status.md`")
    p(f"- CSV per cell: `reports/phase9_runs/csv/{{cell}}_metrics.csv`")
    p("- Trainer states (live mirror): `results/grpo/{cell}/checkpoint-{N}/trainer_state.json`")
    p("- Eval JSONs (when produced): `results/eval/{cell}_{benchmark}*.json`")
    p("")

    return "\n".join(lines)


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[archive] {NOW.isoformat()}")
    print("[archive] rsync from Vast.ai...")
    ok, msg = rsync_all()
    print(f"[archive] rsync: {msg}")

    print("[archive] SSH probe...")
    probe = ssh_probe()
    print(f"[archive] probe: PROC={probe.get('PROC')} ETIME={probe.get('ETIME')} "
          f"GPU={probe.get('GPU')} EVAL_CT={probe.get('EVAL_CT')}")

    print("[archive] parsing trainer states + evals...")
    cells = collect_cells()
    evals = collect_evals()
    econ = estimate_cost_eta(probe, cells)
    print(f"[archive] cells={len(cells)} evals={len(evals)} cost=${econ['cost_so_far']:.2f}")

    md = render_markdown(probe, cells, evals, econ)
    out_path = ARCHIVE / f"{TS}_status.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[archive] wrote {out_path}")

    # Also update latest_status.md (regular file copy — not symlink for cross-platform safety)
    latest = ARCHIVE / "latest_status.md"
    latest.write_text(md, encoding="utf-8")
    print(f"[archive] updated {latest}")


if __name__ == "__main__":
    main()
