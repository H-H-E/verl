# tests/test_dummy_cuda.py
import pytest
import importlib

def is_torch_available():
    return importlib.util.find_spec("torch") is not None

@pytest.mark.skipif(not is_torch_available(), reason="PyTorch is not installed.")
def test_cuda_availability_and_operation():
    '''
    Tests if PyTorch can detect CUDA and performs a simple tensor operation on GPU.
    Skips if PyTorch is not installed.
    The test itself will pass if CUDA is not available but PyTorch is,
    allowing the CI job to pass on CPU-only runners where this test might still run.
    The CI workflow's GPU_CI job is specifically for validating CUDA execution.
    '''
    import torch
    if torch.cuda.is_available():
        # If CUDA is available, perform a simple operation
        try:
            tensor_on_gpu = torch.zeros(1).cuda()
            tensor_on_cpu = tensor_on_gpu.cpu()
            assert tensor_on_cpu.item() == 0.0, "CUDA tensor operation failed."
            print("CUDA is available and test operation succeeded.")
        except RuntimeError as e:
            pytest.fail(f"CUDA operation failed even though CUDA is reported as available: {e}")
    else:
        # If CUDA is not available, this test should not fail the suite.
        # It's more of an indicator for environments expected to have CUDA.
        # The GPU_CI job in the workflow is where we'd expect this to fully pass.
        print("CUDA is not available. Skipping CUDA-specific assertions.")
        pytest.skip("CUDA not available, skipping CUDA-specific assertions.")
