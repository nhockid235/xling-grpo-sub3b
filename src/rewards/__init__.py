"""Reward functions for GRPOTrainer.

All reward functions follow the TRL 0.14+ signature:
    def reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]

The returned list MUST have length == len(prompts). Returning a scalar / tensor /
numpy array silently breaks reward aggregation in TRL.

The registry maps name -> callable; configs/grpo_*.yaml references rewards by name."""

from typing import Callable

REWARD_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator to register a reward callable under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        REWARD_REGISTRY[name] = fn
        return fn
    return _wrap


def get_reward(name: str) -> Callable:
    """Look up a reward callable by name. Raises KeyError if not registered."""
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Reward '{name}' not registered. Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]


from src.rewards import correctness, format, length, tag, lang  # noqa: E402, F401

__all__ = ["REWARD_REGISTRY", "register", "get_reward"]
