"""Command-line text generation with DLite."""

import argparse

import torch

from .engine import DLiteEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with DLite")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--attention-backend",
        default="auto",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--verification-mode", choices=["match", "rejection"], default="match"
    )
    parser.add_argument("--compile-sequential-head", action="store_true")
    parser.add_argument("--verify-block-size", type=int)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine = DLiteEngine.from_pretrained(
        args.target_model,
        args.draft_model,
        device=args.device,
        dtype=torch.bfloat16,
        attention_backend=args.attention_backend,
        trust_remote_code=args.trust_remote_code,
    )
    result = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        stochastic_verification_mode=args.verification_mode,
        compile_sequential_head=args.compile_sequential_head,
        verify_block_size=args.verify_block_size,
    )
    print(result.text)


if __name__ == "__main__":
    main()
