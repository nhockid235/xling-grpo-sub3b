# Phase 0 — TRL API Surface Research

**Verify date:** 2026-05-05
**Researcher:** `general-purpose` subagent
**Source authority:** GitHub `huggingface/trl` (tags v0.13.0, v0.14.0+), PyPI, release notes
**Scope:** Verify exact signatures cho `GRPOTrainer`, `GRPOConfig`, `SFTTrainer` để pin version đúng

---

## Verbatim agent output

### CRITICAL FINDING: GRPO is NOT in TRL 0.13.0

**`trl==0.13.0` does NOT contain `GRPOTrainer` or `GRPOConfig`.** The `trl/trainer/` directory in tag `v0.13.0` has no `grpo_*.py` files. **GRPOTrainer was first introduced in TRL 0.14.0** (released **2025-01-29**). Pinning `trl==0.13.0` per `CLAUDE.md` will cause `ImportError: cannot import name 'GRPOTrainer'`.

**Recommended pin: `trl>=0.14.0,<0.15.0`** (or ideally `trl==0.15.x` for vLLM-stable GRPO). Update `CLAUDE.md` Code Standards section.

### 1. `GRPOTrainer` (from TRL 0.14.0 source — what you'll actually use)

```python
def __init__(
    self,
    model: Union[str, PreTrainedModel],
    reward_funcs: Union[RewardFunc, list[RewardFunc]],
    args: GRPOConfig = None,
    train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
    eval_dataset: Optional[...] = None,
    processing_class: Optional[PreTrainedTokenizerBase] = None,
    reward_processing_classes: Optional[...] = None,
    callbacks: Optional[list[TrainerCallback]] = None,
    optimizers: tuple[...] = (None, None),
    peft_config: Optional["PeftConfig"] = None,
)
```

```python
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
```

- `reward_funcs`: single callable OR `list[Callable]` (sum of rewards). Each callable signature: `def f(prompts, completions, **kwargs) -> list[float]` (one float per sample). Extra dataset columns are passed via `**kwargs`.
- `args`: must be `GRPOConfig` (not `TrainingArguments`).
- `peft_config`: yes, LoRA supported. Pass `LoraConfig(...)`.
- Dataset must contain a column named `"prompt"`.

### 2. `GRPOConfig` (key defaults)

```python
num_generations: Optional[int] = 8       # use 4 per CLAUDE.md
temperature: Optional[float] = 0.9       # CLAUDE.md says 1.0
max_prompt_length: Optional[int] = 512
max_completion_length: Optional[int] = 256   # bump to 2048
beta: float = 0.04                       # KL coefficient
learning_rate: float = 1e-6              # matches CLAUDE.md
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 8
use_vllm: Optional[bool] = False
vllm_device: Optional[str] = "auto"
vllm_gpu_memory_utilization: float = 0.9
```

No `group_size` field — use `num_generations`.

### 3. `SFTTrainer` (0.13.0)

Args: `model, args (SFTConfig), data_collator, train_dataset, eval_dataset, processing_class, peft_config, formatting_func, ...`. `SFTConfig` defaults: `dataset_text_field="text"`, `max_seq_length=None` (falls back to min(tokenizer, 1024)), `packing=False`. **No native `messages` column support** in 0.13.0 — pre-format with a `formatting_func` or `apply_chat_template`. The `tokenizer=` arg is deprecated; use `processing_class=`.

### 4. Compatibility (TRL 0.14.0 setup.py)

```
accelerate>=0.34.0
datasets>=2.21.0
transformers>=4.46.0
peft>=0.8.0   # extras
```

`transformers==4.46.3`, `accelerate==1.2.1`, `peft==0.14.0` from CLAUDE.md all satisfy. `vllm==0.7.2` coexists OK with TRL 0.14+ (TRL imports vLLM lazily under `use_vllm=True`); ensure same `torch==2.4.1` build.

### Gotchas khi viết code

- **Pin `trl>=0.14.0`, not 0.13.0** — GRPOTrainer literally does not exist in 0.13.0.
- **Reward functions must return `list[float]`**, length == len(prompts). Returning a scalar or a tensor silently breaks reward aggregation.
- **Reward function signature is keyword-only**: `def reward(prompts, completions, **kwargs)`. The trainer calls it as `reward_func(prompts=..., completions=..., **reward_kwargs)`.
- **Dataset must have `"prompt"` column** (not `"text"`/`"question"`). Pre-render chat template into the prompt string yourself.
- **`num_generations` default is 8**, but CLAUDE.md says 4 to fit A100 80GB at 3B — explicitly set `num_generations=4`.
- **`max_completion_length` default is 256**, way below CLAUDE.md's 2048 target — explicitly override or generations get truncated mid-CoT.
- **Use `processing_class=tokenizer`, not `tokenizer=tokenizer`** (deprecated) for both SFT and GRPO trainers.
- **SFTTrainer in 0.13.0 doesn't auto-handle `messages`** — apply chat template before passing to trainer, or use `formatting_func`.

---

## Phân tích & action items (main agent)

### Resolution của assumption gốc

CLAUDE.md ghi:
> `trl==0.13.0` (verify GRPOTrainer signature in this version)

→ Đã verify: **không có**. Phải đổi pin.

### Quy chuẩn import cho `src/trainers/`

```python
# src/trainers/grpo.py
from trl import GRPOTrainer, GRPOConfig  # only available in trl>=0.14.0
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# ...

trainer = GRPOTrainer(
    model=model_id_or_obj,
    reward_funcs=[r1_correctness, r2_format, r3_length, r4_tag, r5_lang],  # list, summed
    args=GRPOConfig(
        learning_rate=1e-6,
        num_generations=6,              # Open-RS Exp2 verified
        max_prompt_length=512,
        max_completion_length=3584,     # Open-RS Exp2 verified
        temperature=0.7,                # Open-RS verified
        beta=0.04,                      # KL, TRL default
        per_device_train_batch_size=6,
        gradient_accumulation_steps=16, # for 1×A100 to match 96 eff batch
        max_steps=500,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7,
        vllm_max_model_len=4608,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        warmup_ratio=0.1,
    ),
    train_dataset=ds,
    processing_class=tokenizer,        # NOT tokenizer=
    peft_config=None,                  # full-param for 1.5B; LoraConfig(r=16, ...) for 3B fallback
)
```

### Reward function template (canonical)

```python
# src/rewards/correctness.py
def r1_correctness(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    Math-Verify reward. Compare extracted answer vs gold.
    kwargs sẽ chứa các column khác từ dataset (e.g., 'answer' nếu có).
    Return: list of float, length == len(prompts), values in {0.0, 1.0}.
    """
    golds = kwargs.get("answer", [None] * len(prompts))
    return [
        1.0 if math_verify(extract_answer(c), g) else 0.0
        for c, g in zip(completions, golds)
    ]
```

### Test cho reward registry

```python
# tests/test_rewards_correctness.py
def test_r1_returns_list():
    out = r1_correctness(prompts=["p1", "p2"], completions=["c1", "c2"], answer=["1", "2"])
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(x, float) for x in out)
```

### Compatibility matrix (locked)

| Package | Version | Source justification |
|---|---|---|
| `python` | 3.11 | CLAUDE.md |
| `torch` | 2.4.1 | CLAUDE.md |
| `transformers` | 4.46.3 | CLAUDE.md, satisfies TRL 0.14 `>=4.46.0` |
| `trl` | `>=0.14.0,<0.16.0` | **CHANGED from 0.13.0** |
| `vllm` | 0.7.2 | CLAUDE.md, lazy import OK với TRL 0.14+ |
| `accelerate` | 1.2.1 | CLAUDE.md, satisfies `>=0.34.0` |
| `peft` | 0.14.0 | CLAUDE.md, satisfies `>=0.8.0` |
| `datasets` | `>=2.21.0` | TRL 0.14 requirement |
| `sympy` | `<1.13` | Math-Verify constraint |
| `fasttext` | latest | for R5 langID |

### Sources cited

- `https://github.com/huggingface/trl/tree/v0.13.0`
- `https://github.com/huggingface/trl/tree/v0.14.0`
- `https://github.com/huggingface/trl/releases/tag/v0.14.0`
- `https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/grpo_trainer.py`
- `https://raw.githubusercontent.com/huggingface/trl/v0.14.0/trl/trainer/grpo_config.py`
- `https://pypi.org/project/trl/0.13.0/`
- `https://pypi.org/project/trl/0.14.0/`

### Risk còn open

1. **TRL 0.15+ breaking changes:** TRL release cadence nhanh, có thể đã ra 0.15/0.16 với GRPOConfig field renames. Nếu pin range `<0.16.0` thì OK, nhưng cần verify tại Phase 2 trước khi viết trainer.
2. **vLLM 0.7.2 + TRL 0.14:** TRL 0.14 GRPOTrainer support vLLM nhưng API có thể đổi ở 0.15+ (chuyển sang vllm-server external). Nếu paper W3 chạy training fail vì vllm integration broken → fallback `use_vllm=False` (chậm hơn 3-4× nhưng chạy được).
