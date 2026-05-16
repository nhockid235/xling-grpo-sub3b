"""Eval runner — vLLM batch generation + scoring orchestration.

Usage:
    python src/eval/runner.py \\
        --checkpoint results/grpo/qwen15b_en_42/checkpoint-500 \\
        --benchmarks gsm8k math500 aime2024 mgsm msvamp \\
        --config configs/eval.yaml \\
        --output_dir results/eval/

Output JSON file per benchmark (per language for multilingual):
    {output_dir}/{run_id}_{benchmark}.json                 (single-lang)
    {output_dir}/{run_id}_{benchmark}_{lang}.json          (multilingual)

Schema follows CLAUDE.md § Logging schema. ALWAYS includes `responses` array.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from src.utils.io import load_config
from src.utils.seed import seed_everything

# Allowed benchmark names + module names
_BENCHMARK_MODULES = {
    "gsm8k": "src.eval.gsm8k",
    "math500": "src.eval.math500",
    "aime2024": "src.eval.aime",
    "amc23": "src.eval.amc23",                # W2 gating
    "mgsm": "src.eval.mgsm",
    "msvamp": "src.eval.msvamp",
    "math500_vi_hand": "src.eval.math500_vi_hand",  # Phase 9: hand-translated VI
}

_MULTILINGUAL = {"mgsm", "msvamp"}


def _import_vllm():
    """Try importing vLLM with friendly error message if missing."""
    try:
        from vllm import LLM, SamplingParams  # type: ignore

        return LLM, SamplingParams
    except ImportError as e:
        msg = (
            "vLLM is required for runner.py. Install with `pip install vllm==0.7.2`.\n"
            "On macOS no GPU? Use unit tests in tests/test_eval_adapters.py instead.\n"
            f"Original error: {e}"
        )
        raise ImportError(msg) from e


def _try_load_tokenizer(checkpoint: Path):
    """Best-effort load HuggingFace tokenizer for chat template support."""
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=False)
    except Exception:
        return None


def _build_sampling_params(SamplingParams, gen_cfg: dict[str, Any], seed: int):
    """Build vLLM SamplingParams from config dict."""
    return SamplingParams(
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        max_tokens=gen_cfg.get("max_tokens", 2048),
        stop=gen_cfg.get("stop", None),
        seed=seed,
    )


def _build_sampling_params_maj(SamplingParams, gen_cfg: dict[str, Any], seed: int):
    """Stochastic sampling params cho maj@K (temperature > 0)."""
    return SamplingParams(
        temperature=gen_cfg.get("temperature_maj", 0.7),
        top_p=gen_cfg.get("top_p_maj", 0.95),
        max_tokens=gen_cfg.get("max_tokens", 2048),
        stop=gen_cfg.get("stop", None),
        seed=seed,
    )


def _output_path(output_dir: Path, run_id: str, benchmark: str, language: str | None) -> Path:
    if language is not None:
        return output_dir / f"{run_id}_{benchmark}_{language}.json"
    return output_dir / f"{run_id}_{benchmark}.json"


def _save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def run_benchmark(
    benchmark: str,
    *,
    model: Any,
    cfg: dict[str, Any],
    sampling_params: Any,
    sampling_params_maj: Any,
    run_id: str,
    model_path: Path,
    output_dir: Path,
    seed: int,
    chat_template_tokenizer: Any | None,
) -> list[Path]:
    """Dispatch one benchmark, write output JSON(s). Returns list of paths written."""
    module = importlib.import_module(_BENCHMARK_MODULES[benchmark])
    bench_cfg = cfg.get("benchmarks", {}).get(benchmark, {})
    gen_cfg = cfg.get("generation", {})
    n_samples = bench_cfg.get("n_samples")

    common_kwargs = dict(
        model=model,
        n_samples=n_samples,
        seed=seed,
        sampling_params=sampling_params,
        run_id=run_id,
        model_path=str(model_path),
        max_tokens=gen_cfg.get("max_tokens", 2048),
        temperature=gen_cfg.get("temperature", 0.0),
        chat_template_tokenizer=chat_template_tokenizer,
    )

    written: list[Path] = []

    if benchmark in _MULTILINGUAL:
        languages = bench_cfg.get("languages", [])
        for lang in languages:
            result = module.evaluate(language=lang, **common_kwargs)
            out = _output_path(output_dir, run_id, benchmark, lang)
            _save(result, out)
            written.append(out)
    elif benchmark == "aime2024":
        result = module.evaluate(
            n_seeds_for_maj8=bench_cfg.get("n_seeds_for_maj8", 8),
            sampling_params_maj=sampling_params_maj,
            **common_kwargs,
        )
        out = _output_path(output_dir, run_id, benchmark, None)
        _save(result, out)
        written.append(out)
    elif benchmark == "amc23":
        result = module.evaluate(
            n_seeds_for_maj=bench_cfg.get("n_seeds_for_maj", 4),
            sampling_params_maj=sampling_params_maj,   # Phase 9.1: fix maj@4=0 bug
            **common_kwargs,
        )
        out = _output_path(output_dir, run_id, benchmark, None)
        _save(result, out)
        written.append(out)
    else:
        result = module.evaluate(**common_kwargs)
        out = _output_path(output_dir, run_id, benchmark, None)
        _save(result, out)
        written.append(out)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval runner")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(_BENCHMARK_MODULES.keys()),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/eval/"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID; if not set, derived from checkpoint dirname (parent of checkpoint-N).",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    cfg = load_config(args.config)

    # Derive run_id từ checkpoint path nếu không truyền explicit.
    # E.g. results/grpo/qwen15b_en_42/checkpoint-500 → "qwen15b_en_42"
    run_id = args.run_id
    if run_id is None:
        ckpt = args.checkpoint
        if ckpt.name.startswith("checkpoint-"):
            run_id = ckpt.parent.name
        else:
            run_id = ckpt.name

    LLM, SamplingParams = _import_vllm()

    vllm_cfg = cfg.get("vllm", {})
    gen_cfg = cfg.get("generation", {})

    eval_path = args.checkpoint
    if (eval_path / "adapter_config.json").exists():
        from src.trainers.checkpoint_utils import merge_lora_if_needed
        print(f"[runner] LoRA adapter detected; merging before vLLM load", file=sys.stderr)
        eval_path = merge_lora_if_needed(args.checkpoint)
        print(f"[runner] Using merged weights at {eval_path}", file=sys.stderr)

    print(f"[runner] Loading vLLM model from {eval_path}", file=sys.stderr)
    model = LLM(
        model=str(eval_path),
        dtype=vllm_cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.85),
        max_model_len=vllm_cfg.get("max_model_len", 4096),
        enforce_eager=vllm_cfg.get("enforce_eager", False),
        trust_remote_code=vllm_cfg.get("trust_remote_code", False),
        seed=args.seed,
    )

    sampling_params = _build_sampling_params(SamplingParams, gen_cfg, args.seed)
    sampling_params_maj = _build_sampling_params_maj(SamplingParams, gen_cfg, args.seed)

    tok = _try_load_tokenizer(eval_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for bench in args.benchmarks:
        print(f"[runner] Running benchmark: {bench}", file=sys.stderr)
        paths = run_benchmark(
            bench,
            model=model,
            cfg=cfg,
            sampling_params=sampling_params,
            sampling_params_maj=sampling_params_maj,
            run_id=run_id,
            model_path=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            chat_template_tokenizer=tok,
        )
        for p in paths:
            print(f"[runner]   wrote {p}", file=sys.stderr)


if __name__ == "__main__":
    main()
