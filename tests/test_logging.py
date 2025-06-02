# tests/test_logging.py
import pytest
import logging
from unittest import mock
import subprocess # For Popen spec if needed by mock
import requests # For mocking requests.get
import asyncio # For async test for AtroposDataset
import aiohttp # For mocking aiohttp.ClientSession

# Modules to test logging for
from verl.atropos_inference import start_vllm_server, is_server_ready, stop_server
# from verl.atropos_api_launcher import start_atropos_api, stop_atropos_api # Not directly tested here for specific DEBUG logs
# from verl.atropos_env_spawner import start_env_server, _is_env_registered, stop_env_server # _is_env_registered uses requests.get
from verl.datasets.atropos_dataset import AtroposDataset 

# --- Test for Inference Server Logging (is_server_ready) ---
@mock.patch('verl.atropos_inference.subprocess.Popen')
@mock.patch('verl.atropos_inference.requests.get') # Target where requests.get is used
def test_inference_server_readiness_logging(mock_requests_get, mock_popen, caplog):
    """
    Tests DEBUG logs during inference server start attempt and readiness check.
    Focuses on start_vllm_server and is_server_ready.
    """
    caplog.set_level(logging.DEBUG) # Capture DEBUG and above

    # Configure mock_popen for start_vllm_server
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 12345
    mock_popen.return_value = mock_proc
    
    # Mock the file object opened by start_vllm_server for logging stdout/stderr
    # Popen is called as: subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
    # We need to ensure the mock_popen call can handle this.
    # A simple way is to make side_effect check for the stdout argument if it's critical.
    # For this test, we primarily care that Popen is called and returns our mock_proc.
    # The actual file writing is not tested here, but the logging from start_vllm_server itself.

    # Configure mock_requests_get for is_server_ready
    mock_response_ok = mock.Mock(spec=requests.Response)
    mock_response_ok.status_code = 200
    mock_response_ok.text = '{"models": ["test-model"]}' 

    mock_requests_get.side_effect = [
        requests.exceptions.ConnectionError("Connection refused test"), # First call to is_server_ready fails
        mock_response_ok # Second call succeeds
    ]
    
    test_port = 9001
    model_name = "test-logging-model"
    server_ready_url = f"http://localhost:{test_port}/v1/models" # Example URL

    # Call start_vllm_server
    # Note: start_vllm_server itself opens a file for subprocess stdout/stderr.
    # We need to ensure our mock_popen doesn't break due to this.
    # A more robust mock_popen might use a lambda to inspect args or provide a dummy file mock if needed.
    # For now, assuming the default MagicMock for Popen handles the stdout/stderr args okay.
    # Let's refine the mock_popen to handle the file pointer argument for stdout
    mock_file_obj = mock.MagicMock(spec=open) # Mock the file object
    mock_file_obj.name = f"logs/vllm_server_{test_port}.log" # Give it a name for the check
    
    # Redefine mock_popen to use a side_effect that checks for file-like stdout
    original_popen_side_effect = mock_popen.side_effect 
    def popen_side_effect_for_file(command, stdout, stderr):
        if hasattr(stdout, 'write'): # Check if it's file-like
            mock_proc.stdout = stdout # Attach the passed file obj to our mock_proc for verif if needed
            return mock_proc
        # Fallback if Popen is called elsewhere without file-like stdout (should not happen for start_vllm_server)
        if original_popen_side_effect: # pragma: no cover
            return original_popen_side_effect(command, stdout, stderr)
        return mock_proc # Default return
    mock_popen.side_effect = popen_side_effect_for_file


    proc = start_vllm_server(model_name=model_name, port=test_port, tensor_parallel=1)
    assert proc is mock_proc # Ensure server "started"

    # Call is_server_ready
    ready = is_server_ready(server_ready_url, timeout=0.5, poll_interval=0.05) # Short timeout for quick test
    assert ready

    log_text = caplog.text
    
    # Check logs from start_vllm_server (mostly INFO)
    assert f"Starting vLLM server for model '{model_name}' on port {test_port}" in log_text
    assert f"vLLM server logs will be saved to: logs/vllm_server_{test_port}.log" in log_text
    assert f"vLLM server process started with PID: {mock_proc.pid}" in log_text
    
    # Check logs from is_server_ready
    assert f"Checking server readiness at {server_ready_url} (timeout: 0.5s)" in log_text # INFO
    assert f"Server at {server_ready_url} not yet responding (ConnectionError). Retrying..." in log_text # DEBUG
    assert f"Server at {server_ready_url} is ready (status 200 OK)." in log_text # INFO
    
    # Negative assertion: ensure a log specific to AtroposDataset is NOT here
    assert f"Received response from {server_ready_url}, Status=200" not in log_text 
    # (The log "Received response from..." with status is from AtroposDataset._fetch_once)

    # Clean up
    if proc: # pragma: no cover (proc is always mock_proc here)
        # Mock Popen's methods for stop_server to run without error
        proc.poll.return_value = None # Simulate running
        proc.terminate = mock.Mock()
        proc.kill = mock.Mock()
        proc.wait = mock.Mock()
        # If stop_server tries to close proc.stdout (the log_fp from start_vllm_server)
        # Ensure our mock_file_obj (which became proc.stdout via side_effect) can be closed.
        if not hasattr(proc.stdout, 'closed'): proc.stdout.closed = False
        if not hasattr(proc.stdout, 'close'): proc.stdout.close = mock.Mock()
        
        stop_server(proc, server_name="TestVLLMServer")


@pytest.mark.asyncio 
@mock.patch('verl.datasets.atropos_dataset.aiohttp.ClientSession') 
async def test_atropos_dataset_fetch_logging(mock_aiohttp_session, caplog):
    """Tests DEBUG logs from AtroposDataset._fetch_once."""
    caplog.set_level(logging.DEBUG)

    api_url = "http://mockdatasetapi:7000"
    batch_size = 2
    max_seq_len = 16

    # Configure mock response for aiohttp
    mock_aio_response = mock.AsyncMock(spec=aiohttp.ClientResponse) # Use aiohttp.ClientResponse for spec
    mock_aio_response.status = 200
    mock_aio_response.content_type = 'application/json'
    
    expected_data = [{"prompt": "p1", "response": "r1"}, {"prompt": "p2", "response": "r2"}]
    async def json_func(): return expected_data 
    mock_aio_response.json = json_func 
    # raise_for_status should not raise for 200
    mock_aio_response.raise_for_status = mock.Mock()


    # Configure the session context manager parts
    mock_session_instance = mock.AsyncMock()
    mock_session_instance.get.return_value.__aenter__.return_value = mock_aio_response 
    mock_aiohttp_session.return_value.__aenter__.return_value = mock_session_instance 

    # Use a very simple dummy tokenizer for this test
    class SimpleTestTokenizer:
        def __init__(self): pass
    
    dataset = AtroposDataset(api_url=api_url, batch_size=batch_size, tokenizer=SimpleTestTokenizer(), max_seq_len=max_seq_len)
    
    fetched_data = await dataset._fetch_once()
    assert fetched_data == expected_data

    log_text = caplog.text
    expected_fetch_url = f"{api_url}/batch?size={batch_size}"
    
    assert f"Fetching rollout batch: URL='{expected_fetch_url}', Requested BatchSize={batch_size}" in log_text
    assert f"Received response from {expected_fetch_url}, Status=200" in log_text
    assert f"Successfully fetched {len(expected_data)} groups from {expected_fetch_url} (requested size: {batch_size})." in log_text

    # Test error case: non-JSON response
    mock_aio_response.status = 200 # Still 200, but wrong content type
    mock_aio_response.content_type = 'text/html'
    # No need to mock .json() again as it shouldn't be called if content_type check works.

    caplog.clear() 
    fetched_data_error = await dataset._fetch_once()
    assert fetched_data_error == [] 
    assert f"Unexpected content type from {expected_fetch_url}: text/html" in caplog.text

    # Test error case: HTTP error status
    mock_aio_response.status = 500
    mock_aio_response.content_type = 'application/json' # Content type is fine, but status is bad
    # Configure raise_for_status to simulate its behavior for non-2xx codes
    mock_aio_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
        mock.Mock(), (), status=500, message="Internal Server Error", headers={}
    )
    
    caplog.clear()
    fetched_data_http_error = await dataset._fetch_once()
    assert fetched_data_http_error == []
    assert f"HTTP error fetching from {expected_fetch_url}: Status=500, Message='Internal Server Error'" in caplog.text
    mock_aio_response.raise_for_status.side_effect = None # Reset side effect
```
