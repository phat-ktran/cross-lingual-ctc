import torch
from typing import List


def retain_and_map_tokens_in_batch(
    targets, lengths, blank_token_idx: int, device: torch.device | str
):
    targets_list = targets.tolist()

    # Get unique tokens from all targets in the batch
    unique_tokens = set()
    for i, length in enumerate(lengths):
        for j in range(length):
            unique_tokens.add(targets_list[i][j])

    # Create select_idx with blank, star, and unique tokens
    select_idx = [blank_token_idx] + sorted(list(unique_tokens))

    # Create mapping from original token IDs to remapped IDs
    target_map = {}
    for i, t in enumerate(select_idx):
        target_map[t] = i

    # Remap targets
    remapped_targets = []
    for i, length in enumerate(lengths):
        remapped_target = [target_map[targets_list[i][j]] for j in range(length)]
        remapped_targets.append(remapped_target)

    # Update targets tensor with remapped values
    max_target_len = max(len(target) for target in remapped_targets)
    padded_remapped_targets = []
    for target in remapped_targets:
        padded = target + [0] * (max_target_len - len(target))
        padded_remapped_targets.append(padded)

    targets = torch.tensor(padded_remapped_targets, device=device)
    select_idx_tensor = torch.tensor(select_idx, device=device)

    return targets, select_idx_tensor


def encode_text_supervisions(
    targets: torch.Tensor, timestep: int, return_indices: bool = False
) -> torch.Tensor | List[torch.Tensor]:
    """
    Encodes text-based batch tensors into a supervision tensor format.

    This function is adapted for text recognition from a speech recognition
    equivalent. It creates a `supervision_segments` tensor of shape
    `(batch_size, 3)`, where the columns represent sequence index,
    start position (always 0), and the number of tokens (length).

    The batch items are re-ordered by length in descending order. The returned
    tensors are all guaranteed to be consistent with this new order.

    Args:
        targets: A tensor of shape (B, L) containing the tokenized sequences.
        lengths: A tensor of shape (B,) specifying the length of each sequence.

    Returns:
        A tuple containing:
        - supervision_segments (torch.Tensor): The re-ordered supervision
          tensor of shape (B, 3).
    """
    batch_size = targets.size(0)
    lengths = torch.full((batch_size,), timestep)

    # Create the supervision tensor columns
    # 0: sequence_idx -> [0, 1, 2, ..., B-1]
    # 1: start_position -> Always 0 for text
    # 2: num_tokens -> The length of each sequence
    supervision_segments = torch.stack(
        (
            torch.arange(batch_size, dtype=torch.int32),
            torch.zeros(batch_size, dtype=torch.int32),
            lengths.to(torch.int32).to("cpu"),
        ),
        dim=1,
    )

    # Sort by sequence length (column 2) in descending order
    indices = torch.argsort(supervision_segments[:, 2], descending=True)

    # Re-order the supervisions, the original targets, and the lengths
    supervision_segments = supervision_segments[indices]

    results = supervision_segments

    if return_indices:
        results = [supervision_segments, indices]

    return results
