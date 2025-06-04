# tests/test_inference_server.py
import pytest
import subprocess
import time
import socket
import os # For checking log file existence
from pathlib import Path # For log file path construction

from verl.atropos_inference import start_vllm_server, stop_server, is_server_ready, start_sglang_server
# from verl.atropos_config import AtroposConfig # Not strictly needed for these tests unless using default ports from config

# Helper to check if a port is open
def is_port_in_use(port: int, host: str = "localhost") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5) # Set a short timeout to avoid long hangs
        try:
            # Try to connect to the port. If it's in use by a listening server, this might succeed or fail.
            # A more reliable way for "is in use" is to try to bind to it.
            s.bind((host, port)) # Try to bind to the port
            return False # If bind succeeds, port is not in use (available)
        except socket.error:
            return True # If bind fails, port is likely in use (unavailable)


# Check if torch is available to determine if GPU tests can be meaningful
try:
    import torch
    torch_available = True
    is_cuda_available = torch.cuda.is_available()
except ImportError:
    torch_available = False
    is_cuda_available = False

# Skip condition for tests that require a running vLLM/SGLang server and potentially a GPU
skip_heavy_tests_condition = not is_cuda_available
skip_heavy_tests_reason = "CUDA not available or torch not installed, skipping vLLM/SGLang lifecycle tests."

TEST_MODEL_NAME_VLLM = "h2oai/h2ogpt-oasst1-512-12b"
# Using a smaller, widely available model for SGLang as it's generally more flexible.
TEST_MODEL_NAME_SGLANG = "facebook/opt-125m"


@pytest.mark.skipif(skip_heavy_tests_condition, reason=skip_heavy_tests_reason)
def test_vllm_server_lifecycle():
    '''
    Tests the start, readiness check, and stop lifecycle of the vLLM server.
    This test is resource-intensive and may require a GPU.
    '''
    test_port = 8060 # Use a non-default port for testing

    if is_port_in_use(test_port):
        pytest.skip(f"Port {test_port} is already in use. Skipping test.")

    server_proc = None
    try:
        server_proc = start_vllm_server(
            model_name=TEST_MODEL_NAME_VLLM,
            port=test_port,
            tensor_parallel=1
        )
        assert server_proc is not None, "start_vllm_server should return a Popen object."
        # Give a brief moment for the process to potentially fail fast
        time.sleep(2)
        assert server_proc.poll() is None, "vLLM server process should be running (or starting)."

        expected_log_file = Path("logs") / f"vllm_server_{test_port}.log"
        assert expected_log_file.exists(), f"vLLM log file {expected_log_file} was not created."

        ready_url = f"http://localhost:{test_port}/v1/models"
        # Increased timeout as model loading can be very slow.
        assert is_server_ready(ready_url, timeout=300.0, poll_interval=10.0), \
            f"vLLM server did not become ready at {ready_url} within timeout. Check logs at {expected_log_file}."

    finally:
        if server_proc:
            stop_server(server_proc, server_name="vLLM Test Server")
            # Wait for server to stop and release port
            for _ in range(10): # Wait up to 10 seconds for port to be free
                if not is_port_in_use(test_port):
                    break
                time.sleep(1)
            assert not is_port_in_use(test_port), f"Port {test_port} should be freed after stopping vLLM server."

            if server_proc.poll() is None:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=5)
                except Exception:
                    pass
                pytest.fail("vLLM server process was not properly terminated by stop_server.")


@pytest.mark.skipif(skip_heavy_tests_condition, reason=skip_heavy_tests_reason)
def test_sglang_server_lifecycle():
    '''
    Tests the start, readiness check, and stop lifecycle of the SGLang server.
    This test is resource-intensive and may require a GPU.
    '''
    test_port = 8061

    if is_port_in_use(test_port):
        pytest.skip(f"Port {test_port} is already in use. Skipping test.")

    server_proc = None
    try:
        server_proc = start_sglang_server(
            model_name=TEST_MODEL_NAME_SGLANG,
            port=test_port,
            tensor_parallel_size=1
        )
        assert server_proc is not None, "start_sglang_server should return a Popen object."
        time.sleep(2)
        assert server_proc.poll() is None, "SGLang server process should be running (or starting)."

        expected_log_file = Path("logs") / f"sglang_server_{test_port}.log"
        assert expected_log_file.exists(), f"SGLang log file {expected_log_file} was not created."

        ready_url = f"http://localhost:{test_port}/v1/models"
        assert is_server_ready(ready_url, timeout=180.0, poll_interval=5.0), \
            f"SGLang server did not become ready at {ready_url} within timeout. Check logs at {expected_log_file}."

    finally:
        if server_proc:
            stop_server(server_proc, server_name="SGLang Test Server")
            for _ in range(10):
                if not is_port_in_use(test_port):
                    break
                time.sleep(1)
            assert not is_port_in_use(test_port), f"Port {test_port} should be freed after stopping SGLang server."

            if server_proc.poll() is None:
                try:
                    server_proc.kill()
                    server_proc.wait(timeout=5)
                except Exception:
                    pass
                pytest.fail("SGLang server process was not properly terminated by stop_server.")


@pytest.mark.skip(reason="Mock tests for server lifecycle not yet implemented. Real tests are GPU dependent.")
def test_vllm_server_lifecycle_mocked():
    pass
