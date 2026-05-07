"""Prompt construction helpers — shared giữa các benchmark adapters.

Tất cả benchmarks dùng cùng một system prompt (English) để test cross-lingual
transfer: model được train trên EN nhưng eval trên VI/ZH/FR/...; nếu đổi system
prompt theo ngôn ngữ thì không còn đo được "transfer of EN-trained reasoning".

Khi tokenizer được cung cấp (HuggingFace AutoTokenizer), ta apply chat template
để output tương thích với checkpoint tinh chỉnh từ instruct base. Khi không có
tokenizer (e.g., vLLM offline với base model), ta fallback về plain string với
markers an toàn.
"""

from __future__ import annotations

from typing import Any

# Open-RS verbatim eval template (lighteval MATH_QUERY_TEMPLATE).
# Phase 8.2 (2026-05-06): KHÁC TRAINING — eval không yêu cầu <think>/<answer>,
# chỉ yêu cầu \\boxed{ANSWER} ở dòng cuối. Critical để match Open-RS reported numbers.
EVAL_QUERY_TEMPLATE = (
    "Solve the following math problem efficiently and clearly.  "
    "The last line of your response should be of the following format: "
    "'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' "
    "(without quotes) where ANSWER is just the final number or expression that "
    "solves the problem. Think step by step before answering.\n\n"
    "{Question}"
)

# DEFAULT_SYSTEM_PROMPT giữ làm None → caller embed template vào user message.
DEFAULT_SYSTEM_PROMPT = None


def build_prompt(
    problem: str,
    system_prompt: str | None = None,
    chat_template_tokenizer: Any | None = None,
) -> str:
    """Build prompt cho vLLM `LLM.generate(...)` — Open-RS lighteval style.

    Open-RS không dùng system prompt cho eval. Template được embed thẳng vào
    user message với `{Question}` placeholder.

    Args:
        problem: nội dung bài toán.
        system_prompt: nếu provided, sẽ dùng làm system message. Default None
            (Open-RS style — không có system).
        chat_template_tokenizer: HF tokenizer cho chat template.

    Returns:
        Prompt string.
    """
    # Embed Open-RS eval template vào user content
    user_content = EVAL_QUERY_TEMPLATE.format(Question=problem)

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        # Open-RS style: chỉ user message
        messages = [{"role": "user", "content": user_content}]

    if chat_template_tokenizer is not None:
        return chat_template_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback plain-text khi không có tokenizer (unit test mock).
    if system_prompt:
        return f"System: {system_prompt}\n\nUser: {user_content}\n\nAssistant:"
    return f"User: {user_content}\n\nAssistant:"


def build_prompts(
    problems: list[str],
    system_prompt: str | None = None,
    chat_template_tokenizer: Any | None = None,
) -> list[str]:
    """Vectorized version cho batch generation."""
    return [
        build_prompt(p, system_prompt=system_prompt, chat_template_tokenizer=chat_template_tokenizer)
        for p in problems
    ]
