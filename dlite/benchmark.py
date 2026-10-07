"""Small reproducible throughput benchmark for JSONL prompts."""

import argparse
import json
import time
from pathlib import Path

import torch

from .engine import DLiteEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DLite inference")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help='JSONL with a string field named "prompt"',
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--attention-backend",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--verification-mode", default="match", choices=["match", "rejection"]
    )
    parser.add_argument("--compile-sequential-head", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prompts = []
    with args.prompts.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= args.max_samples:
                break
    if not prompts:
        raise ValueError("No prompts were loaded.")

    engine = DLiteEngine.from_pretrained(
        args.target_model,
        args.draft_model,
        attention_backend=args.attention_backend,
    )
    total_tokens = 0
    accept_lengths = []
    started = time.perf_counter()
    for prompt in prompts:
        result = engine.generate(
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stochastic_verification_mode=args.verification_mode,
            compile_sequential_head=args.compile_sequential_head,
        )
        total_tokens += len(engine.tokenizer.encode(result.text))
        accept_lengths.extend(result.stats.get("accept_lengths", []))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = {
        "samples": len(prompts),
        "generated_tokens": total_tokens,
        "seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "mean_accept_length": (
            sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
