"""AMC23 eval adapter.

HF: ``knoveleng/AMC-23`` (40 problems, single ``train`` split, AoPS 2023 AMC 12A+12B).
Schema: ``{id, problem, question, answer, url}``. Matches Open-RS evaluate.py.

License unspecified on Hub metadata. Cite as "AoPS wiki, 2023 AMC 12A/12B,
redistributed via knoveleng/AMC-23".

Metrics: pass@1 (greedy) + maj@4 (4 stochastic rollouts).
"""

from __future__ import annotations

from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    majority_vote,
    make_metadata,
    math_verify_match,
    numeric_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts

HF_ID = "knoveleng/AMC-23"
SPLIT = "train"


def _amc_match(pred: str | None, gold: str | None) -> bool:
    """AMC answers are usually numeric (int/decimal), occasionally LaTeX. Try numeric match first, then math-verify."""
    if numeric_match(pred, gold):
        return True
    return math_verify_match(pred, gold)


def _maybe_with_seed(sp: Any, base_seed: int, k: int) -> Any:
    """Clone SamplingParams with seed = base_seed + k (mirrors aime.py)."""
    if sp is None:
        return None
    try:
        import copy

        sp_new = copy.copy(sp)
        if hasattr(sp_new, "seed"):
            sp_new.seed = base_seed + k
        return sp_new
    except Exception:
        return sp


def evaluate(
    model: Any,
    n_samples: int | None = None,
    n_seeds_for_maj: int = 4,
    seed: int = 42,
    sampling_params: Any | None = None,
    sampling_params_maj: Any | None = None,
    *,
    run_id: str = "unknown",
    model_path: str = "unknown",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    chat_template_tokenizer: Any | None = None,
    system_prompt: str | None = None,
    dataset: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """AMC23 eval -- pass@1 (greedy) + maj@N (N stochastic seeds).

    Args:
        model: vLLM ``LLM`` or mock exposing ``.generate(prompts, sampling_params)``.
        n_samples: dev limit. None = full 40.
        n_seeds_for_maj: number of rollouts for maj@N. Set to 1 to skip maj.
        seed: base seed for stochastic rollouts.
        sampling_params: greedy params (temperature 0).
        sampling_params_maj: stochastic params (temperature > 0).
        run_id, model_path: metadata.
        chat_template_tokenizer: HF tokenizer for chat template (optional).
        dataset: injectable for test mocks; None = load from HF.

    Returns:
        Dict matching the eval JSON schema, with key ``maj_at_{N}``.
    """
    metadata = make_metadata(model_path, max_tokens, temperature, seed)
    metadata["hf_dataset"] = HF_ID
    metadata["split"] = SPLIT
    metadata["n_seeds_for_maj"] = n_seeds_for_maj

    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset(HF_ID, split=SPLIT)

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "amc23", "en", metadata)

    problems = [r["problem"] for r in records]
    golds = [str(r["answer"]) for r in records]
    n = len(records)

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )

    # ---- Pass@1: greedy ----
    completions_greedy = vllm_generate(model, prompts, sampling_params=sampling_params)
    n_correct = sum(
        1 for c, g in zip(completions_greedy, golds)
        if _amc_match(extract_prediction(c), g)
    )
    pass_at_1 = n_correct / n if n > 0 else 0.0

    # ---- Maj@N: K stochastic samples per problem ----
    maj_at_n: float | None = None
    if n_seeds_for_maj and n_seeds_for_maj > 1 and model is not None:
        all_runs: list[list[str]] = []
        for k in range(n_seeds_for_maj):
            sp_k = _maybe_with_seed(sampling_params_maj, base_seed=seed, k=k)
            comps_k = vllm_generate(model, prompts, sampling_params=sp_k)
            all_runs.append(comps_k)

        n_correct_maj = 0
        for i, gold in enumerate(golds):
            preds_for_i = [extract_prediction(all_runs[k][i]) for k in range(n_seeds_for_maj)]
            voted = majority_vote(preds_for_i)
            if _amc_match(voted, gold):
                n_correct_maj += 1
        maj_at_n = n_correct_maj / n if n > 0 else 0.0

    result = {
        "run_id": run_id,
        "benchmark": "amc23",
        "language": "en",
        "n_samples": n,
        "pass_at_1": pass_at_1,
        f"maj_at_{n_seeds_for_maj}": maj_at_n,
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions_greedy),
        "responses": completions_greedy,
        "metadata": metadata,
    }
    return result
