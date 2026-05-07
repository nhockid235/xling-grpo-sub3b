"""Reward functions cho GRPOTrainer.

All reward functions follow TRL 0.14+ signature:
    def reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]

Return list MUST have length == len(prompts). Returning scalar / tensor / numpy
silently breaks reward aggregation in TRL.

Registry maps name → callable. configs/grpo_*.yaml references rewards by name.
"""

from typing import Callable

# Registry — populated when modules are imported. Phase 2 fills in implementations.
REWARD_REGISTRY: dict[str, Callable] = {}


def register(name: str):
    """Decorator để add reward function vào registry."""
    def _wrap(fn: Callable) -> Callable:
        REWARD_REGISTRY[name] = fn
        return fn
    return _wrap


def get_reward(name: str) -> Callable:
    """Lookup reward by name; raises KeyError nếu chưa registered."""
    if name not in REWARD_REGISTRY:
        raise KeyError(
            f"Reward '{name}' not registered. Available: {sorted(REWARD_REGISTRY)}"
        )
    return REWARD_REGISTRY[name]


# Import sub-modules để trigger @register decorators
from src.rewards import correctness, format, length, tag, lang  # noqa: E402, F401

__all__ = ["REWARD_REGISTRY", "register", "get_reward"]
