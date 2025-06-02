# verl/train/utils.py
import torch
from typing import List, Any # Added Any for future placeholders

def build_loss_mask(input_ids: torch.Tensor, prompt_end_positions: List[int]) -> torch.Tensor:
    """
    Creates a loss mask for sequences in a batch.

    The mask is 0 for prompt tokens and 1 for response tokens.
    input_ids: Tensor of shape [B, T].
    prompt_end_positions: List of length B. Each element is the index of the first response token.
                          Indices < prompt_end_positions[b] are prompt tokens.
    Returns: Float tensor of shape [B, T].
    """
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids must be a torch.Tensor. Got {type(input_ids)}")
    if not isinstance(prompt_end_positions, list):
        raise TypeError(f"prompt_end_positions must be a list. Got {type(prompt_end_positions)}")

    batch_size, seq_length = input_ids.shape

    if len(prompt_end_positions) != batch_size:
        raise ValueError(
            f"Length of prompt_end_positions ({len(prompt_end_positions)}) "
            f"must match batch_size of input_ids ({batch_size})."
        )

    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32, device=input_ids.device)

    for i in range(batch_size):
        end_pos = prompt_end_positions[i]
        
        if not isinstance(end_pos, int):
            raise TypeError(f"Elements of prompt_end_positions must be integers. Found {type(end_pos)} at index {i}.")

        # end_pos is the index of the *first response token*.
        # So, if end_pos == 0, all tokens are response tokens.
        # If end_pos == seq_length, all tokens are prompt tokens (mask remains all zeros).
        # If end_pos > seq_length, it's an error.
        if not (0 <= end_pos <= seq_length): 
            raise ValueError(
                f"prompt_end_positions[{i}] is {end_pos}, which is out of range [0, {seq_length}]. "
                f"It should be the index of the first response token."
            )
        
        if end_pos < seq_length: # If end_pos == seq_length, means no response tokens, mask is all 0.
            loss_mask[i, end_pos:] = 1.0
            
    return loss_mask

# Placeholder for weight_sync_manager (Task 3.1)
# from contextlib import contextmanager
# import torch.nn as nn
# @contextmanager
# def weight_sync_manager(training_model: nn.Module, inference_server: Any):
#     pass

# Placeholder for assert_gpu_memory_freed (Task 3.2)
# def assert_gpu_memory_freed(threshold_bytes: int = 1000000):
#     pass
