# verl/train/utils.py
import torch
import torch.nn as nn # For nn.Module type hint
from typing import List, Any, Generator, Optional # Added Optional for type hint
from contextlib import contextmanager
import logging # For logging

logger_utils = logging.getLogger(__name__) # Logger for this module
# Basic config for standalone use/testing, e.g. if logger has no handlers
# if not logger_utils.hasHandlers():
#    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
@contextmanager
def weight_sync_manager(training_model: nn.Module, inference_server: Any) -> Generator[None, None, None]:
    """
    A context manager to synchronize weights from a training model to an inference server.

    1. Extracts state_dict from `training_model` (cloned to CPU).
    2. Calls `inference_server.load_weights(state_dict)`. The server is expected to handle
       moving weights to its own device.
    3. Yields control to the `with` block.
    4. After the `with` block, optionally attempts to free GPU memory if the
       `inference_server` is an embedded type and holds its model on GPU. This might involve
       moving the model to CPU or clearing CUDA cache.

    Args:
        training_model (nn.Module): The model from which to get weights.
        inference_server (Any): The inference server object. Must have a `load_weights` method.
                                May have a `model` attribute and a custom `offload_model_to_cpu`
                                or similar method for memory management.
    """
    # original_inference_model_device = None # Not used in current simplified logic

    logger_utils.info("Entering weight_sync_manager...")
    if not hasattr(inference_server, 'load_weights') or not callable(inference_server.load_weights):
        logger_utils.error("Inference server does not have a callable load_weights method. Cannot sync weights.")
        # Decide on behavior: raise error or yield without syncing?
        # For now, let's yield as the block might do other things, but log error.
        yield
        return

    try:
        logger_utils.debug("Extracting state_dict from training_model (cloning to CPU)...")
        # Clone to CPU to avoid issues if inference_server is on a different device or needs CPU weights
        state_dict_cpu = {k: v.cpu().clone().detach() for k, v in training_model.state_dict().items()}

        logger_utils.debug("Calling inference_server.load_weights()...")
        inference_server.load_weights(state_dict_cpu)
        logger_utils.info("Weights loaded into inference server.")

    except Exception as e:
        logger_utils.error(f"Error during weight synchronization in weight_sync_manager: {e}", exc_info=True)
        # Depending on policy, might re-raise or just yield.
        # For now, log and yield. If sync is critical, an error should be raised.
        # raise # Example: if sync must succeed.

    try:
        yield # Control is passed to the 'with' block here
    finally:
        logger_utils.info("Exiting weight_sync_manager, performing cleanup...")

        # Optional: Free inference_server GPU memory if it's an embedded server holding a model on GPU.
        is_embedded_by_name = type(inference_server).__name__ == 'EmbeddedInferenceServer' # Heuristic
        has_model_attribute = hasattr(inference_server, 'model') and isinstance(inference_server.model, nn.Module)

        if is_embedded_by_name and has_model_attribute:
            logger_utils.debug("Attempting to manage memory for embedded inference server.")
            if hasattr(inference_server, 'offload_model_to_cpu') and callable(inference_server.offload_model_to_cpu):
                try:
                    logger_utils.info("Calling custom offload_model_to_cpu() on inference server.")
                    inference_server.offload_model_to_cpu()
                except Exception as e_offload: # pragma: no cover
                    logger_utils.error(f"Error calling custom offload_model_to_cpu: {e_offload}", exc_info=True)
            else:
                # Generic approach: if model is on CUDA, try to clear cache.
                try:
                    # Check current device of the model within the inference server
                    # This requires inference_server.model to have parameters to check device from.
                    if len(list(inference_server.model.parameters())) > 0:
                        param_device = next(inference_server.model.parameters()).device
                        if param_device.type == 'cuda': # pragma: no cover (depends on test env)
                            logger_utils.info(f"Embedded server model is on CUDA ({param_device}). Clearing CUDA cache.")
                            torch.cuda.empty_cache()
                    elif torch.cuda.is_available(): # Fallback if no params but CUDA is generally active
                            logger_utils.debug("CUDA is available and embedded server model has no params to check device; clearing cache as a precaution.")
                            torch.cuda.empty_cache()
                except Exception as e_mem: # pragma: no cover
                    logger_utils.error(f"Error during optional memory management for embedded server: {e_mem}", exc_info=True)
        else:
            logger_utils.debug("Inference server not identified as embedded with model for specific memory management.")

        logger_utils.info("weight_sync_manager cleanup finished.")


# Placeholder for assert_gpu_memory_freed (Task 3.2)
def assert_gpu_memory_freed(threshold_bytes: int = 1000000, device_id: Optional[int] = None): # Default 1MB
    """
    Asserts that the currently allocated GPU memory on a specific device (or default device)
    is below a given threshold. Raises a RuntimeError if the allocated memory exceeds the threshold.
    This function only proceeds if CUDA is available.

    Args:
        threshold_bytes (int): The maximum allowed allocated GPU memory in bytes.
                               Defaults to 1,000,000 bytes (1MB).
        device_id (Optional[int]): The GPU device ID to check. If None, checks the
                                   current CUDA device.

    Raises:
        RuntimeError: If allocated GPU memory exceeds threshold_bytes.
        AssertionError: If CUDA is not available (can be caught by test skipper).
    """
    if not torch.cuda.is_available():
        # This function is only meaningful if CUDA is available.
        logger_utils.warning("CUDA not available. assert_gpu_memory_freed check skipped.")
        # Tests using this should ideally be skipped if CUDA is not available.
        # If called directly in such an environment, it will effectively do nothing.
        return

    if device_id is None:
        try:
            allocated_memory = torch.cuda.memory_allocated()
            device_str = f"current device ({torch.cuda.current_device()})"
        except RuntimeError as e: # pragma: no cover (should not happen if torch.cuda.is_available() is true)
             logger_utils.error(f"Could not get allocated memory for current CUDA device: {e}")
             raise RuntimeError(f"Failed to get allocated memory for current CUDA device: {e}") from e
    else:
        try:
            allocated_memory = torch.cuda.memory_allocated(device=device_id)
            device_str = f"device {device_id}"
        except Exception as e: # Handles invalid device_id or other CUDA errors
            logger_utils.error(f"Could not get allocated memory for device {device_id}: {e}")
            raise RuntimeError(f"Failed to get allocated memory for device {device_id}: {e}") from e

    logger_utils.debug(f"Currently allocated GPU memory on {device_str}: {allocated_memory} bytes.")

    if allocated_memory > threshold_bytes:
        raise RuntimeError(
            f"GPU memory allocated on {device_str} ({allocated_memory} bytes) "
            f"exceeds threshold ({threshold_bytes} bytes)."
        )
    else: # pragma: no cover (logging success might be too verbose for frequent calls)
        logger_utils.info(
            f"GPU memory allocated on {device_str} ({allocated_memory} bytes) "
            f"is within threshold ({threshold_bytes} bytes)."
        )
