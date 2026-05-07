"""GSM8K eval adapter.

HF: openai/gsm8k, config=main, split=test (1.32K records).
Schema: {question, answer}. Gold extraction: '#### (.+)'. Pred extraction:
<answer> → \\boxed{} → last number, then numeric match (string compare sau khi
strip commas/units).
"""

from __future__ import annotations

from typing import Any

from src.eval._common import (
    avg_response_tokens,
    empty_result,
    extract_prediction,
    make_metadata,
    numeric_match,
    vllm_generate,
)
from src.eval.prompts import build_prompts
from src.utils.parsing import extract_gsm8k_gold


def evaluate(
    model: Any,
    n_samples: int | None = None,
    seed: int = 42,
    sampling_params: Any | None = None,
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
    """Run GSM8K eval. Returns dict matching CLAUDE.md eval JSON schema.

    Args:
        model: vLLM `LLM` instance (or mock w/ `.generate`). Pass None for dry-run.
        n_samples: limit number of samples; None = all.
        seed: deterministic shuffle seed (not strictly needed for ordered eval).
        sampling_params: vLLM SamplingParams instance.
        run_id: e.g. "qwen15b_en_42".
        dataset: optional pre-loaded list[dict] or HF Dataset for testing.
            Mỗi record có keys ``question`` và ``answer``.

    Returns:
        Dict matching CLAUDE.md § Logging schema.
    """
    metadata = make_metadata(model_path, max_tokens, temperature, seed)

    if dataset is None:
        from datasets import load_dataset  # lazy

        dataset = load_dataset("openai/gsm8k", "main", split="test")

    records = list(dataset)
    if n_samples is not None and n_samples < len(records):
        records = records[:n_samples]

    if not records:
        return empty_result(run_id, "gsm8k", None, metadata)

    problems = [r["question"] for r in records]
    golds = [extract_gsm8k_gold(r["answer"]) for r in records]

    prompts = build_prompts(
        problems,
        system_prompt=system_prompt,
        chat_template_tokenizer=chat_template_tokenizer,
    )
    completions = vllm_generate(model, prompts, sampling_params=sampling_params)

    n_correct = 0
    for completion, gold in zip(completions, golds):
        pred = extract_prediction(completion)
        if numeric_match(pred, gold):
            n_correct += 1

    n = len(records)
    return {
        "run_id": run_id,
        "benchmark": "gsm8k",
        "language": "en",
        "n_samples": n,
        "pass_at_1": n_correct / n if n > 0 else 0.0,
        "maj_at_8": None,
        "lang_consistency_rate": None,
        "avg_response_tokens": avg_response_tokens(completions),
        "responses": completions,
        "metadata": metadata,
    }
