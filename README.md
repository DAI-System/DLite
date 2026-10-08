<p align="center">
  <img src="assets/dlite.svg" alt="DLite logo" width="480">
</p>

<h1 align="center">DLite inference</h1>

This repository is the inference-only release of DLite for Qwen3. It provides a
small Python API, a CLI, and a JSONL throughput benchmark. Model weights are
distributed separately and are never stored in this repository.

## Install

```bash
pip install -e .
```

FlashAttention 2 is optional. When installed, `attention_backend="auto"` uses
it; otherwise DLite falls back to SDPA.

## Python API

```python
from dlite import DLiteEngine

engine = DLiteEngine.from_pretrained(
    target_model="/path/to/Qwen3-8B",
    draft_model="/path/to/dlite-student",
    attention_backend="auto",  # auto, eager, sdpa, flash_attention_2
)
result = engine.generate("Explain speculative decoding in one paragraph.")
print(result.text)
print(result.stats)
```

The draft checkpoint must use `DLiteDraftModel`, `dlite_config`, architecture
version `dlite_v1`, and `sequential_head: "rnn"`.

## Repository layout

```text
dlite/
├── engine.py          # Public loading and generation API
├── cli.py             # `dlite` command
├── benchmark.py       # `dlite-benchmark` command
├── modeling/
│   ├── dlite.py       # Qwen3 DLite model and decoding loop
│   ├── qwen3.py       # Attention and decoder layers
│   └── sequential_head.py
└── utils/
    ├── hidden_states.py
    └── sampling.py
```

Model classes are available from `dlite.modeling`; normal users only need the
top-level `DLiteEngine` API.

## CLI

```bash
dlite \
  --target-model /path/to/Qwen3-8B \
  --draft-model /path/to/dlite-student \
  --prompt 'Write a CUDA optimization checklist.'
```

For stochastic decoding, set a positive `--temperature` and optionally use
`--verification-mode rejection`. Greedy decoding uses `match` verification.

## Benchmark

Create a JSONL file with one `prompt` string per line, then run:

```bash
dlite-benchmark \
  --target-model /path/to/Qwen3-8B \
  --draft-model /path/to/dlite-student \
  --prompts prompts.jsonl \
  --max-samples 100 \
  --output benchmark-results/run.json
```

The report includes generated-token throughput and mean accepted block length.
