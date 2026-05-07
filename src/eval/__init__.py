"""Evaluation pipeline — vLLM-based generation + per-benchmark scoring.

Each benchmark module implements:
    def evaluate(model_path: str, language: str | None, n_samples: int | None, ...) -> dict

Returns JSON-serializable dict matching schema in CLAUDE.md § Logging schema.
"""
