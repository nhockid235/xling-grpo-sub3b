"""AIME 2024 eval adapter.

HF: Maxwell-Jia/AIME_2024 (30 records, AIME I + II). Single "train" split.
Schema fields are CAPITALIZED: Problem, Solution, Answer, ID. Lowercase access raises KeyError.
Metrics: pass@1 (greedy) + maj@8 (8 stochastic samples, majority vote)."""

from __future__ import annotations

from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    majority_vote,
    make_metadata,
    math_verify_match,
    normalize_number_string,
    numeric_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts


def _aime_match(pred: str | None, gold: str) -> bool:
    """AIME answers are integers 0-999. Try numeric match first, fall back to math-verify."""
    if numeric_match(pred, gold):
        return True
    return math_verify_match(pred, gold)


def evaluate(
    model: Any,
    n_samples: int | None = None,
    n_seeds_for_maj8: int = 8,
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
    """AIME 2024 eval -- pass@1 (greedy) + maj@8 (8 stochastic seeds).

    `sampling_params_maj` should use temperature > 0 (e.g., 0.7) so the 8
    samples diverge.
    """
    metadata = make_metadata(model_path, max_tokens, temperature, seed)

    if dataset is None:
        from datasets import load_dataset

        # AIME_2024 only has "train" split
        dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "aime2024", None, metadata)

    # CAPITALIZED field names — DO NOT lowercase
    problems = [r["Problem"] for r in records]
    golds = [normalize_number_string(str(r["Answer"])) for r in records]

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )

    # ---- Pass@1: greedy ----
    completions_greedy = vllm_generate(model, prompts, sampling_params=sampling_params)
    n_correct = 0
    for completion, gold in zip(completions_greedy, golds):
        pred = extract_prediction(completion)
        if _aime_match(pred, gold):
            n_correct += 1
    n = len(records)
    pass_at_1 = n_correct / n if n > 0 else 0.0

    # ---- Maj@8: K stochastic samples per problem, majority vote ----
    maj_at_8: float | None = None
    if n_seeds_for_maj8 and n_seeds_for_maj8 > 0 and model is not None:
        # Generate K completions per prompt; vLLM supports `n` in SamplingParams,
        all_runs: list[list[str]] = []
        for k in range(n_seeds_for_maj8):
            sp_k = _maybe_with_seed(sampling_params_maj, base_seed=seed, k=k)
            comps_k = vllm_generate(model, prompts, sampling_params=sp_k)
            all_runs.append(comps_k)

        # Per-problem: collect K predictions, majority vote, score.
        n_correct_maj = 0
        for i, gold in enumerate(golds):
            preds_for_i = [extract_prediction(all_runs[k][i]) for k in range(n_seeds_for_maj8)]
            voted = majority_vote(preds_for_i)
            if _aime_match(voted, gold):
                n_correct_maj += 1
        maj_at_8 = n_correct_maj / n if n > 0 else 0.0

    return {
        "run_id": run_id,
        "benchmark": "aime2024",
        "language": "en",
        "n_samples": n,
        "pass_at_1": pass_at_1,
        "maj_at_8": maj_at_8,
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions_greedy),
        "responses": completions_greedy,
        "metadata": metadata,
    }


def _maybe_with_seed(sp: Any, base_seed: int, k: int) -> Any:
    """Clone SamplingParams with deterministic per-k seed."""
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
