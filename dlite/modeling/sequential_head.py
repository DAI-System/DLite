"""Low-rank sequential head for DLite block prediction.

The expensive DLite backbone produces all block-position hidden states in
parallel.  These heads reintroduce a light autoregressive dependency through
the previously generated token.  Training uses teacher-forced previous tokens;
inference samples the block from left to right.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


SEQUENTIAL_HEAD_TYPES = ("rnn",)


def _sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


class DLiteSequentialHead(nn.Module):
    """Low-rank serial vocabulary head with optional recurrent state."""

    def __init__(
        self,
        *,
        head_type: str,
        vocab_size: int,
        sequential_rank: int,
        hidden_size: int,
        max_prediction_length: int,
    ) -> None:
        super().__init__()
        self.head_type = str(head_type).lower()
        self.vocab_size = int(vocab_size)
        self.sequential_rank = int(sequential_rank)
        self.hidden_size = int(hidden_size)
        self.max_prediction_length = int(max_prediction_length)

        if self.head_type not in SEQUENTIAL_HEAD_TYPES:
            raise ValueError(
                f"Unknown sequential head type {self.head_type!r}; "
                f"expected one of {SEQUENTIAL_HEAD_TYPES}."
            )
        if self.sequential_rank <= 0:
            raise ValueError(
                f"sequential_rank must be positive, got {self.sequential_rank}."
            )
        if self.max_prediction_length <= 0:
            raise ValueError(
                "max_prediction_length must be positive, got "
                f"{self.max_prediction_length}."
            )
        self.prev_token_embedding = nn.Embedding(self.vocab_size, self.sequential_rank)
        self.output_proj = nn.Linear(self.sequential_rank, self.vocab_size, bias=False)
        self.state_proj = nn.Linear(2 * self.sequential_rank, 2 * self.sequential_rank)
        self.state_hidden_mlp = nn.Linear(
            self.sequential_rank + self.hidden_size,
            self.sequential_rank,
        )

    def project_logits(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Project low-rank head states to full-vocabulary logits."""
        return self.output_proj(latent_states)

    def _compute_step_latent(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        prev_embeddings = self.prev_token_embedding(prev_token_ids.long())
        if state is None:
            state = torch.zeros_like(prev_embeddings)
        mem_inputs = torch.cat([state, prev_embeddings], dim=-1)
        gate_raw, candidate_raw = self.state_proj(mem_inputs).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        new_state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        fused_inputs = torch.cat([new_state, hidden_states], dim=-1)
        return self.state_hidden_mlp(fused_inputs), new_state

    def _seed_rnn_state(
        self,
        initial_prev_token_ids: torch.Tensor,
        *,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Prime recurrent state with the token immediately before the anchor."""
        initial_prev_token_ids = initial_prev_token_ids.long()
        if initial_prev_token_ids.ndim != 1:
            raise ValueError(
                "initial_prev_token_ids must be 1-D for state seeding, got "
                f"{tuple(initial_prev_token_ids.shape)}."
            )
        dummy_hidden = reference.new_zeros(
            initial_prev_token_ids.shape[0],
            self.hidden_size,
        )
        _, state = self._compute_step_latent(
            prev_token_ids=initial_prev_token_ids,
            hidden_states=dummy_hidden,
            state=None,
        )
        return state

    def forward_teacher_forcing(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        initial_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return low-rank states for teacher-forced block predictions.

        Args:
            hidden_states: ``[..., prediction_length, hidden_size]``.
            prev_token_ids: ``[..., prediction_length]``; entry ``k`` is the
                ground-truth token immediately preceding prediction ``k``.
            initial_prev_token_ids: Optional ``[...]`` token ids at position
                ``anchor-1``. When provided, the
                recurrent state is primed with this token before the first
                supervised prediction step.
        """
        if hidden_states.shape[:-1] != prev_token_ids.shape:
            raise ValueError(
                "hidden_states and prev_token_ids leading shapes must match, "
                f"got {tuple(hidden_states.shape)} and "
                f"{tuple(prev_token_ids.shape)}."
            )
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"got {hidden_states.size(-1)}."
            )

        prediction_length = hidden_states.size(-2)
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured "
                f"maximum {self.max_prediction_length}."
            )
        batch_shape = hidden_states.shape[:-2]
        if initial_prev_token_ids is not None:
            if initial_prev_token_ids.shape != batch_shape:
                raise ValueError(
                    "initial_prev_token_ids must match hidden_states batch "
                    f"dimensions {batch_shape}, got "
                    f"{tuple(initial_prev_token_ids.shape)}."
                )
            flat_reference = hidden_states.reshape(-1, hidden_states.size(-1))
            state = self._seed_rnn_state(
                initial_prev_token_ids.reshape(-1),
                reference=flat_reference,
            ).view(*batch_shape, self.sequential_rank)
        else:
            state = torch.zeros(
                *batch_shape,
                self.sequential_rank,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        outputs: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids[..., position],
                hidden_states=hidden_states[..., position, :],
                state=state,
            )
            outputs.append(latent.unsqueeze(-2))
        if not outputs:
            return hidden_states.new_empty(
                *hidden_states.shape[:-2], 0, self.sequential_rank
            )
        return torch.cat(outputs, dim=-2)

    def forward_scheduled_sampling(
        self,
        *,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        initial_prev_token_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Run the serial head with sampled previous tokens during training.

        Position zero always consumes the real anchor token.  Later positions
        consume either the preceding ground-truth token or the preceding greedy
        prediction.  This matches autoregressive draft inference when the ratio
        is zero and avoids the large teacher-forcing exposure gap.
        """
        ratio = float(teacher_forcing_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0, 1].")
        if hidden_states.shape[:-1] != prev_token_ids.shape:
            raise ValueError(
                "hidden_states and prev_token_ids leading shapes must match"
            )

        batch_shape = hidden_states.shape[:-2]
        state = (
            self._seed_rnn_state(
                initial_prev_token_ids.reshape(-1),
                reference=hidden_states,
            ).view(*batch_shape, self.sequential_rank)
            if initial_prev_token_ids is not None
            else (hidden_states.new_zeros(*batch_shape, self.sequential_rank))
        )
        outputs: list[torch.Tensor] = []
        current_prev = prev_token_ids[..., 0]
        for position in range(hidden_states.size(-2)):
            latent, state = self._compute_step_latent(
                prev_token_ids=current_prev,
                hidden_states=hidden_states[..., position, :],
                state=state,
            )
            outputs.append(latent.unsqueeze(-2))
            if position + 1 < hidden_states.size(-2):
                with torch.no_grad():
                    predicted = self.project_logits(latent).argmax(dim=-1)
                    if ratio <= 0.0:
                        current_prev = predicted
                    elif ratio >= 1.0:
                        current_prev = prev_token_ids[..., position + 1]
                    else:
                        use_teacher = (
                            torch.rand_like(predicted, dtype=torch.float32) < ratio
                        )
                        current_prev = torch.where(
                            use_teacher,
                            prev_token_ids[..., position + 1],
                            predicted,
                        )
        if not outputs:
            return hidden_states.new_empty(
                *hidden_states.shape[:-2], 0, self.sequential_rank
            )
        return torch.cat(outputs, dim=-2)

    def sample_block_tokens(
        self,
        *,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        initial_prev_token_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Serially sample one DLite prediction block.

        Returns sampled token IDs and the final logits actually used to sample
        them.
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must have shape [batch, prediction_length, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )

        batch_size, prediction_length = hidden_states.shape[:2]
        if prediction_length > self.max_prediction_length:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured "
                f"maximum {self.max_prediction_length}."
            )
        state = (
            self._seed_rnn_state(
                initial_prev_token_ids.reshape(-1),
                reference=hidden_states,
            ).view(batch_size, self.sequential_rank)
            if initial_prev_token_ids is not None
            else (hidden_states.new_zeros(batch_size, self.sequential_rank))
        )
        prev_token_ids = first_prev_token_ids.long()
        sampled_tokens: list[torch.Tensor] = []
        final_logits: list[torch.Tensor] = []
        for position in range(prediction_length):
            latent, state = self._compute_step_latent(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_states[:, position, :],
                state=state,
            )
            step_logits = self.project_logits(latent)
            next_token_ids = _sample_tokens(step_logits, float(temperature))
            sampled_tokens.append(next_token_ids.unsqueeze(1))
            final_logits.append(step_logits.unsqueeze(1))
            prev_token_ids = next_token_ids

        if not sampled_tokens:
            return (
                torch.empty(
                    batch_size,
                    0,
                    dtype=torch.long,
                    device=hidden_states.device,
                ),
                hidden_states.new_empty(batch_size, 0, self.vocab_size),
            )
        return torch.cat(sampled_tokens, dim=1), torch.cat(final_logits, dim=1)


__all__ = [
    "DLiteSequentialHead",
    "SEQUENTIAL_HEAD_TYPES",
]
