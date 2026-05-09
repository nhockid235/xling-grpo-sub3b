"""Data integrity pipeline — apply the project verification discipline.

Run after every Phase 9 milestone (cell completes, eval finishes). Performs:

1. Rsync ALL results artifacts from Vast.ai to local (no data loss).
2. Compute SHA256 checksums of every result file.
3. Generate `reports/phase9_runs/manifest.csv` with one row per artifact.
4. Verify trainer_state.json files parse and contain expected keys.
5. Verify eval JSONs parse and contain pass_at_1 / responses[].
6. Cross-reference paper claims (numbers in main.tex) against eval JSONs.
7. Update `VERIFICATION.md` with current status.

The script is idempotent: re-running just refreshes the manifest and
verifies new artifacts. Existing checksums are preserved.

Usage:
    python3 scripts/data_integrity_pipeline.py
    python3 scripts/data_integrity_pipeline.py --verify-only   # skip rsync
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/Users/vudang/PythonLab/Papper/xling-grpo-sub3b")
VAST_HOST = "root@202.122.49.242"
VAST_PORT = "22569"

ARTIFACT_DIRS = [
    "results/grpo",
    "results/eval",
]
LOG_DIR = "reports/phase9_runs/raw_logs"
ARCHIVE_DIR = ROOT / "reports/phase9_runs"
MANIFEST_FILE = ARCHIVE_DIR / "manifest.csv"
VERIFICATION_FILE = ROOT / "VERIFICATION.md"


def run(cmd: list[str], capture: bool = True, timeout: int = 600) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage1_rsync_all() -> None:
    print("[1/6] rsync all results from Vast.ai...")
    for src in ARTIFACT_DIRS:
        cmd = [
            "rsync", "-az", "-e", f"ssh -p {VAST_PORT}",
            f"{VAST_HOST}:/workspace/xling-grpo-sub3b/{src}/",
            str(ROOT / src) + "/",
        ]
        rc, out = run(cmd, timeout=900)
        if rc != 0:
            print(f"  WARN rsync {src} returned {rc}")
            print(f"  output: {out[-500:]}")
    LOG_DEST = ROOT / LOG_DIR
    LOG_DEST.mkdir(parents=True, exist_ok=True)
    rc, out = run([
        "rsync", "-az", "-e", f"ssh -p {VAST_PORT}",
        f"{VAST_HOST}:/workspace/phase9_logs/",
        str(LOG_DEST) + "/",
    ], timeout=300)
    if rc != 0:
        print(f"  WARN log rsync returned {rc}")
    print("  done.")


def stage2_checksum_all() -> dict[str, dict]:
    print("[2/6] compute SHA256 of every result file...")
    manifest: dict[str, dict] = {}
    for d in ARTIFACT_DIRS + [LOG_DIR]:
        base = ROOT / d
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if f.name == ".gitkeep":
                continue
            rel = f.relative_to(ROOT).as_posix()
            try:
                h = sha256_file(f)
                size = f.stat().st_size
            except Exception as e:
                manifest[rel] = {"error": str(e)}
                continue
            manifest[rel] = {
                "sha256": h,
                "size_bytes": size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
    print(f"  {len(manifest)} artifacts checksummed.")
    return manifest


def stage3_write_manifest(manifest: dict[str, dict]) -> None:
    print("[3/6] write manifest.csv...")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_FILE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "sha256", "size_bytes", "modified"])
        for path in sorted(manifest):
            entry = manifest[path]
            w.writerow([
                path,
                entry.get("sha256", entry.get("error", "?")),
                entry.get("size_bytes", ""),
                entry.get("modified", ""),
            ])
    print(f"  wrote {MANIFEST_FILE.relative_to(ROOT)}")


def stage4_verify_trainer_states() -> dict[str, dict]:
    print("[4/6] verify trainer_state.json files...")
    out: dict[str, dict] = {}
    for f in (ROOT / "results/grpo").rglob("trainer_state.json"):
        rel = f.relative_to(ROOT).as_posix()
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            out[rel] = {"valid": False, "reason": f"parse_error: {e}"}
            continue
        if "global_step" not in d or "log_history" not in d:
            out[rel] = {"valid": False, "reason": "missing_required_keys"}
            continue
        history = d["log_history"]
        out[rel] = {
            "valid": True,
            "global_step": d.get("global_step"),
            "max_steps": d.get("max_steps"),
            "n_logs": len(history),
            "last_reward": history[-1].get("reward") if history else None,
            "last_kl": history[-1].get("kl") if history else None,
        }
    valid = sum(1 for v in out.values() if v.get("valid"))
    print(f"  {valid}/{len(out)} trainer states valid.")
    return out


def stage5_verify_eval_jsons() -> dict[str, dict]:
    print("[5/6] verify eval JSONs...")
    out: dict[str, dict] = {}
    for f in (ROOT / "results/eval").rglob("*.json"):
        if f.name == ".gitkeep":
            continue
        rel = f.relative_to(ROOT).as_posix()
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            out[rel] = {"valid": False, "reason": f"parse_error: {e}"}
            continue
        required = {"benchmark", "n_samples", "pass_at_1"}
        missing = required - d.keys()
        out[rel] = {
            "valid": not missing,
            "benchmark": d.get("benchmark"),
            "n_samples": d.get("n_samples"),
            "pass_at_1": d.get("pass_at_1"),
            "maj_at_4": d.get("maj_at_4"),
            "maj_at_8": d.get("maj_at_8"),
            "responses_present": isinstance(d.get("responses"), list) and len(d.get("responses", [])) > 0,
            "missing_keys": list(missing) if missing else [],
        }
    valid = sum(1 for v in out.values() if v.get("valid"))
    print(f"  {valid}/{len(out)} eval JSONs valid.")
    return out


_NUM_RE = re.compile(r"(\d+\.\d+)")


def stage6_paper_claim_crosscheck(eval_results: dict[str, dict]) -> list[dict]:
    """Best-effort cross-reference: extract percentages from main.tex,
    flag those NOT matching any eval JSON pass_at_1."""
    print("[6/6] cross-reference paper claims to eval JSONs...")
    out: list[dict] = []
    paper = ROOT / "paper/main.tex"
    if not paper.exists():
        print("  paper/main.tex not found, skip.")
        return out

    eval_pass_set: set[float] = set()
    for v in eval_results.values():
        if isinstance(v.get("pass_at_1"), (int, float)):
            eval_pass_set.add(round(v["pass_at_1"] * 100, 1))

    for i, line in enumerate(paper.read_text().splitlines(), start=1):
        if "%" not in line:
            continue
        for m in re.finditer(r"(\d+\.\d+)\s*\\?%", line):
            val = float(m.group(1))
            if 5.0 <= val <= 99.0:
                matched = any(abs(val - p) <= 0.5 for p in eval_pass_set)
                out.append({
                    "line": i,
                    "value_pct": val,
                    "matched_eval_json": matched,
                    "context": line.strip()[:120],
                })
    unmatched = [c for c in out if not c["matched_eval_json"]]
    print(f"  {len(out)} numerical claims found; {len(unmatched)} not matching eval JSONs.")
    if unmatched:
        for c in unmatched[:5]:
            print(f"    L{c['line']}: {c['value_pct']}% — {c['context']}")
    return out


def update_verification_md(
    manifest: dict[str, dict],
    trainer_results: dict[str, dict],
    eval_results: dict[str, dict],
    crosscheck: list[dict],
) -> None:
    """Append timestamped block to VERIFICATION.md showing current state."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = [
        "",
        f"## Pipeline run @ {ts}",
        "",
        f"- Total artifacts checksummed: **{len(manifest)}**",
        f"- Trainer states valid: **{sum(1 for v in trainer_results.values() if v.get('valid'))}/{len(trainer_results)}**",
        f"- Eval JSONs valid: **{sum(1 for v in eval_results.values() if v.get('valid'))}/{len(eval_results)}**",
        f"- Numerical claims in paper: {len(crosscheck)} (unmatched: {sum(1 for c in crosscheck if not c['matched_eval_json'])})",
        "",
        "### Eval JSONs present",
        "",
    ]
    for path, info in sorted(eval_results.items()):
        if info.get("valid"):
            p1 = info.get("pass_at_1")
            p1s = f"{p1:.4f}" if isinstance(p1, (int, float)) else "?"
            block.append(f"- `{path}` — pass@1={p1s}, n={info.get('n_samples')}, responses={info.get('responses_present')}")
    block.append("")
    block.append("### Trainer states present")
    block.append("")
    for path, info in sorted(trainer_results.items()):
        if info.get("valid"):
            block.append(f"- `{path}` — step {info.get('global_step')}/{info.get('max_steps')}, n_logs={info.get('n_logs')}")
    block.append("")
    block.append(f"Manifest checksum file: `{MANIFEST_FILE.relative_to(ROOT)}`")
    block.append("")

    if VERIFICATION_FILE.exists():
        existing = VERIFICATION_FILE.read_text()
    else:
        existing = ""
    VERIFICATION_FILE.write_text(existing + "\n".join(block))
    print(f"  appended verification block to {VERIFICATION_FILE.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="Skip rsync stage")
    args = parser.parse_args()

    if not args.verify_only:
        stage1_rsync_all()
    manifest = stage2_checksum_all()
    stage3_write_manifest(manifest)
    trainer_results = stage4_verify_trainer_states()
    eval_results = stage5_verify_eval_jsons()
    crosscheck = stage6_paper_claim_crosscheck(eval_results)
    update_verification_md(manifest, trainer_results, eval_results, crosscheck)
    print()
    print("[done] data integrity pipeline complete.")


if __name__ == "__main__":
    main()
