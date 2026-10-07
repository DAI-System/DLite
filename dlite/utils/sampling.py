"""Sampling and speculative-verification utilities."""

from __future__ import annotations

import time

import torch


STOCHASTIC_VERIFICATION_MODES = ("match", "rejection")


def cuda_sync_time(device: torch.device) -> float:
    """Synchronize CUDA when needed and return a wall-clock timestamp."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def sample_logits(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """Sample token IDs, using argmax for effectively zero temperature."""
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    batch_size, sequence_length, vocab_size = logits.shape
    probabilities = torch.softmax(
        logits.reshape(-1, vocab_size).float() / temperature,
        dim=-1,
    )
    return torch.multinomial(probabilities, num_samples=1).view(
        batch_size, sequence_length
    )


def validate_verification_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in STOCHASTIC_VERIFICATION_MODES:
        raise ValueError(
            f"Unknown stochastic verification mode {mode!r}; expected one of "
            f"{STOCHASTIC_VERIFICATION_MODES}."
        )
    return mode


def _logits_to_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).float()
    return torch.softmax(logits.float() / float(temperature), dim=-1)


def _sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    vocab_size = probs.shape[-1]
    return torch.multinomial(probs.reshape(-1, vocab_size), num_samples=1).reshape(
        *probs.shape[:-1]
    )


def _sample_residual(
    target_probs: torch.Tensor, draft_probs: torch.Tensor
) -> torch.Tensor:
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 1e-8,
        residual / residual_mass.clamp_min(1e-8),
        target_probs,
    )
    return _sample_from_probs(residual)


def rejection_sample_verify(
    *,
    proposed_tokens: torch.Tensor,
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float,
) -> tuple[int, torch.Tensor]:
    """Verify one batch-size-one block with speculative rejection sampling."""
    if temperature < 1e-5:
        raise ValueError("rejection sampling requires temperature > 0.")
    if proposed_tokens.ndim != 2 or proposed_tokens.shape[0] != 1:
        raise ValueError(
            "rejection sampling currently requires proposed_tokens shape [1, K]."
        )
    proposal_count = proposed_tokens.shape[1]
    if draft_logits.shape[:2] != (1, proposal_count):
        raise ValueError(
            "draft_logits must have shape [1, K, vocab] matching proposed_tokens."
        )
    if target_logits.shape[:2] != (1, proposal_count + 1):
        raise ValueError("target_logits must have shape [1, K + 1, vocab].")
    if draft_logits.shape[-1] != target_logits.shape[-1]:
        raise ValueError("draft and target logits must use the same vocabulary.")

    target_probs = _logits_to_probs(target_logits, temperature)
    if proposal_count == 0:
        return 0, _sample_from_probs(target_probs[:, 0, :])

    draft_probs = _logits_to_probs(draft_logits, temperature)
    token_index = proposed_tokens.unsqueeze(-1)
    selected_target = (
        target_probs[:, :proposal_count, :]
        .gather(dim=-1, index=token_index)
        .squeeze(-1)
    )
    selected_draft = draft_probs.gather(dim=-1, index=token_index).squeeze(-1)
    accept_probs = torch.minimum(
        torch.ones_like(selected_target),
        selected_target / selected_draft.clamp_min(1e-20),
    )
    accepted_prefix = (
        (torch.rand_like(accept_probs) < accept_probs).to(torch.int64).cumprod(dim=1)
    )
    accepted_count = int(accepted_prefix.sum().item())

    if accepted_count < proposal_count:
        next_token = _sample_residual(
            target_probs[:, accepted_count, :],
            draft_probs[:, accepted_count, :],
        )
    else:
        next_token = _sample_from_probs(target_probs[:, proposal_count, :])
    return accepted_count, next_token


__all__ = [
    "STOCHASTIC_VERIFICATION_MODES",
    "cuda_sync_time",
    "rejection_sample_verify",
    "sample_logits",
    "validate_verification_mode",
]
