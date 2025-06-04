# tests/test_gpu_memory.py
import pytest
import torch
from verl.train.utils import assert_gpu_memory_freed
import logging # For checking logs in some cases
import time # For potential small delays if needed, though not strictly in this version

# Check if CUDA is available for skipping tests
is_cuda_available = torch.cuda.is_available()

@pytest.mark.skipif(not is_cuda_available, reason="CUDA not available, skipping GPU memory tests.")
def test_memory_freed_passes_when_low_allocation():
    """
    Tests that assert_gpu_memory_freed passes when allocated memory is low.
    """
    try:
        # Ensure a clean state for this specific test part by clearing cache first
        torch.cuda.empty_cache()
        # Get baseline after cache clear. This might still not be zero due to CUDA context.
        baseline_allocated = torch.cuda.memory_allocated()

        # Allocate a tensor, perform operations, then delete and clear cache
        a = torch.randn(256, 256, device="cuda") # Approx 256KB
        b = a * 2
        c = b + a
        del a, b, c
        torch.cuda.empty_cache()

        # After clearing, memory should be at or very near the baseline.
        # We use a threshold that is baseline + a small buffer, or a fixed reasonable threshold
        # if baseline itself can vary slightly across environments.
        # The function's default threshold is 1MB.
        # Let's test against a threshold slightly above the baseline to ensure our operation didn't leak significantly.
        # For simplicity, if the function default (1MB) is used, it should pass if baseline is low.
        # We will use a larger threshold here as specified in the original task (10MB) for this test.
        threshold_for_pass = 10 * 1024 * 1024 # 10MB

        # If baseline itself is already > threshold_for_pass, this test is ill-defined.
        # However, assert_gpu_memory_freed checks current total, not relative.
        # This test essentially checks if current total is < 10MB after operations.
        if baseline_allocated < threshold_for_pass :
            assert_gpu_memory_freed(threshold_bytes=threshold_for_pass)
        else: # pragma: no cover
            pytest.skip(f"Baseline allocated memory ({baseline_allocated} bytes) already exceeds test threshold ({threshold_for_pass} bytes). Skipping test.")

    except Exception as e: # pragma: no cover
        pytest.fail(f"assert_gpu_memory_freed test failed unexpectedly: {e}")


@pytest.mark.skipif(not is_cuda_available, reason="CUDA not available, skipping GPU memory tests.")
def test_memory_not_freed_raises_runtimeerror():
    """
    Tests that assert_gpu_memory_freed raises RuntimeError when allocated memory is high.
    """
    large_tensor_for_test = None # Ensure it's defined for finally block
    try:
        torch.cuda.empty_cache()
        # Get current allocation before creating the large tensor
        # current_allocated_before_tensor = torch.cuda.memory_allocated()

        # Allocate a noticeable tensor, e.g., 20MB of floats (20 * 1024 * 1024 / 4 elements)
        num_elements = (20 * 1024 * 1024) // 4
        large_tensor_for_test = torch.randn(num_elements, device="cuda")
        # Perform an operation to ensure it's "used"
        _ = large_tensor_for_test * 2

        # Now, memory allocated should be at least 20MB + whatever was there before.
        # Set a threshold that is definitely lower than this (e.g., 1MB, the function's default).
        threshold_to_fail = 1 * 1024 * 1024 # 1MB

        # Sanity check for test logic: allocated memory should exceed this threshold
        current_total_allocated = torch.cuda.memory_allocated()
        assert current_total_allocated > threshold_to_fail, \
            f"Test setup issue: Current total allocated ({current_total_allocated}) is not greater than threshold ({threshold_to_fail}). Tensor might not have allocated as expected."

        with pytest.raises(RuntimeError, match="GPU memory allocated on"):
            assert_gpu_memory_freed(threshold_bytes=threshold_to_fail)

    finally:
        # Cleanup the tensor to free memory for other tests
        if large_tensor_for_test is not None:
            del large_tensor_for_test
        torch.cuda.empty_cache()


@pytest.mark.skipif(is_cuda_available, reason="Test only when CUDA is NOT available.")
def test_assert_gpu_memory_freed_no_cuda(caplog):
    """
    Tests that assert_gpu_memory_freed logs a warning if CUDA is not available.
    """
    with caplog.at_level(logging.WARNING):
        assert_gpu_memory_freed(threshold_bytes=100)

    assert "CUDA not available. assert_gpu_memory_freed check skipped." in caplog.text


@pytest.mark.skipif(not is_cuda_available, reason="CUDA not available for device ID test.")
def test_assert_gpu_memory_freed_specific_device():
    """Tests assert_gpu_memory_freed with a specific device ID."""
    # This test assumes a multi-GPU environment to be fully meaningful for testing different device_ids,
    # but will run on device 0 if only one GPU is present.
    device_id_to_test = 0
    try:
        # Test on device 0 (or the current default if only one GPU)
        torch.cuda.empty_cache(device=device_id_to_test)
        assert_gpu_memory_freed(threshold_bytes=20 * 1024 * 1024, device_id=device_id_to_test) # Generous 20MB threshold for existing allocations
    except RuntimeError as e: # pragma: no cover
        # This might fail if device 0 has persistent allocations beyond the threshold.
        pytest.fail(f"assert_gpu_memory_freed with device_id={device_id_to_test} failed: {e}")

    # Test with an invalid device ID (if more than 0 devices, otherwise this ID might be valid)
    # This check is more robust if we know there's only one GPU (device_count = 1, so invalid_id = 1)
    # Or if we can guarantee an invalid ID.
    num_gpus = torch.cuda.device_count()
    invalid_device_id = num_gpus # device IDs are 0-indexed, so num_gpus is always an invalid ID.

    if num_gpus >= 1 : # Only proceed if we can construct a definitively invalid ID relative to actual count
        with pytest.raises(RuntimeError, match=f"Failed to get allocated memory for device {invalid_device_id}"):
            assert_gpu_memory_freed(threshold_bytes=100, device_id=invalid_device_id)
    else: # pragma: no cover (should be skipped by top-level skipif if no CUDA)
        pass

```
