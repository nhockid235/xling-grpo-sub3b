"""Prompt templates for eval -- Open-RS lighteval style."""

from __future__ import annotations

from typing import Any

# Open-RS verbatim eval template (lighteval MATH_QUERY_TEMPLATE).
EVAL_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression that "
    "solves the problem. Think step by step before answering.\n\n"
    "{Question}"
)

DEFAULT_SYSTEM_PROMPT = None


def build_prompt(
    problem: str,
    system_prompt: str | None = None,
    chat_template_tokenizer: Any | None = None,
) -> str:
    """Build a prompt for vLLM `LLM.generate(...)` (Open-RS lighteval style).

    Args:
        chat_template_tokenizer: HF tokenizer for chat template.

    Returns:
        Prompt string.
    """
    user_content = EVAL_QUERY_TEMPLATE.format(Question=problem)

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]

    if chat_template_tokenizer is not None:
        return chat_template_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"
    return f"User: {user_content}\n\nAssistant:"


def build_prompts(
    problems: list[str],
    system_prompt: str | None = None,
    chat_template_tokenizer: Any | None = None,
) -> list[str]:
    """Vectorized version for batch generation."""
    return [
        build_prompt(p, system_prompt=system_prompt, chat_template_tokenizer=chat_template_tokenizer)
        for p in problems
    ]
