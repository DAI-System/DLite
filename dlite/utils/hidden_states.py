"""Target hidden-state selection helpers."""

from __future__ import annotations

import torch


def build_target_layer_ids(
    num_transformer_layers: int, chs_num_layers: int
) -> list[int]:
    """Select evenly spaced target layers, including the first and last."""
    layer_count = int(num_transformer_layers)
    selection_count = int(chs_num_layers)
    if layer_count < 2:
        raise ValueError(f"DLite requires at least 2 target layers, got {layer_count}.")
    if not 2 <= selection_count <= layer_count:
        raise ValueError(
            f"chs_num_layers must be in [2, {layer_count}], got {selection_count}."
        )

    selected = {0, layer_count - 1}
    middle_count = selection_count - 2
    first_middle, last_middle = 1, layer_count - 2
    if middle_count > 0:
        span = last_middle - first_middle
        for index in range(middle_count):
            fraction = 0.5 if middle_count == 1 else index / (middle_count - 1)
            selected.add(first_middle + round(fraction * span))

    result = sorted(selected)
    if len(result) != selection_count:
        raise RuntimeError(
            f"Failed to select {selection_count} layers from depth "
            f"{layer_count}: {result}"
        )
    return result


def _embedding_offset(hidden_states: tuple | list, layer_count: int) -> int:
    state_count = len(hidden_states)
    if state_count == layer_count:
        return 0
    if state_count == layer_count + 1:
        return 1
    return int(state_count > layer_count)


def gather_pivot_hidden_states(
    hidden_states: tuple | list,
    target_layer_ids: list[int],
    token_index: int,
    num_transformer_layers: int,
) -> torch.Tensor:
    """Return selected pivot features with shape ``(B, 1, S, H)``."""
    offset = _embedding_offset(hidden_states, num_transformer_layers)
    selected = [
        hidden_states[layer_id + offset][:, token_index, :].unsqueeze(1)
        for layer_id in target_layer_ids
    ]
    return torch.stack(selected, dim=2)


__all__ = ["build_target_layer_ids", "gather_pivot_hidden_states"]
