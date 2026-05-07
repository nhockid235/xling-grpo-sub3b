# Model Card — LoRA Adapters

This card documents the three LoRA adapters released in
`results/training/grpo/` alongside the paper.

| Field | Value |
|---|---|
| Base model | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| Adapter type | LoRA r=16, α=32, dropout=0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training framework | TRL 0.15.2 GRPOTrainer with in-process vLLM 0.7.2 rollout |
| Training steps | 50 |
| Effective batch | 96 prompts/step (per-device 6 × grad_accum 16 on 1×A100 80GB) |
| Hardware | 1× NVIDIA A100-SXM4-80GB (Vast.ai cloud) |

## Three released checkpoints

### `reproduce_openrs_rs2_42/checkpoint-50/` — A1 arm

- **Training data:** `knoveleng/open-rs` (7K English math problems)
- **Rewards:** correctness (R1) + format (R2)
- **Use case:** reproduces Open-RS RS2 protocol on a single-A100 LoRA budget; provides a baseline for the cross-lingual / auxiliary-reward axes.

### `a2_vi_42/checkpoint-50/` — A2 arm

- **Training data:** `5CD-AI/Vietnamese-meta-math-MetaMathQA-40K-gg-translated` (5,203 records filtered from a 7K subset)
- **Rewards:** correctness (R1) + format (R2)
- **Use case:** isolates the effect of training language on cross-lingual math reasoning.

### `a3_enlang_42/checkpoint-50/` — A3 arm

- **Training data:** same as A1 (`knoveleng/open-rs` 7K)
- **Rewards:** correctness (R1) + format (R2) + fastText language-consistency (R5)
- **Use case:** isolates the effect of an auxiliary reward (R5 fires uniformly at 1.0 on English data).

## Evaluation results (single seed, pass@1)

| Benchmark | Base | A1 | A2 | A3 |
|---|---|---|---|---|
| AMC23 | 50.0% | 57.5% | 52.5% | 52.5% |
| MATH-500 | 59.4% | 58.8% | 60.2% | 60.6% |
| AIME-2024 | 26.7% | 10.0% | 16.7% | 23.3% |
| AIME-2024 (maj@8) | 33.3% | 30.0% | 33.3% | 36.7% |

See `paper/main.pdf` Table 1 + Figure 3 for context. All eval JSONs (with full
`responses[]` arrays) are released in `results/training/eval/`.

## Inference example

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# Pick one of the three released arms
model = PeftModel.from_pretrained(base, "results/training/grpo/a3_enlang_42/checkpoint-50/")

prompt = tok.apply_chat_template(
    [{"role": "user", "content": (
        "Solve the following math problem efficiently and clearly. "
        "The last line of your response should be of the following format: "
        "'Therefore, the final answer is: $\\boxed{ANSWER}$.' "
        "Think step by step.\n\n"
        "What is the sum of all positive integers less than 100 divisible by 7?")}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tok(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.0)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

- Single seed (seed=42) — multi-seed replication still pending.
- LoRA r=16 does not match Open-RS full-parameter setup; adapters likely
  have lower expressive capacity than reported Open-RS checkpoints.
- A2 dataset confounds language and distribution (5CD-AI MetaMathQA-VI ≠
  Open-RS translated to Vietnamese).
- AMC23 `maj@4` sampling implementation has a bug; pass@1 numbers reliable.
- Models inherit any bias / safety issues from DeepSeek-R1-Distill base.

## License

LoRA adapter weights: Apache-2.0 (matching the code license).
