# tests/test_rollout_speed.py
import pytest
import time
import os
import threading
import uvicorn
from fastapi import FastAPI
from typing import List, Dict, Any
import logging # For caplog and logging messages
import requests # For is_server_ready in fixture

# Components to test/use
from verl.datasets.atropos_dataset import AtroposDataset, DummyTokenizer
from verl.atropos_inference import is_server_ready # For robust server check in fixture

# --- Mock Atropos API Server ---
MOCK_API_PORT_SPEED_TEST = 8110
MOCK_API_HOST_SPEED_TEST = "localhost"

@pytest.fixture(scope="function")
def mock_rollout_api_for_speed_test():
    app = FastAPI()

    app.state.rollout_data_store: List[Dict[str, Any]] = []

    @app.get("/batch")
    async def get_rollout_batch(size: int):
        num_to_serve = min(size, len(app.state.rollout_data_store))
        data_to_return = app.state.rollout_data_store[:num_to_serve]
        # For continuous serving if dataset retries or fetches multiple times (though test expects one fetch)
        # app.state.rollout_data_store = app.state.rollout_data_store[num_to_serve:] + data_to_return
        return data_to_return

    @app.get("/health")
    async def health():
        return {"status": "healthy_speed_test_mock"}

    config = uvicorn.Config(app, host=MOCK_API_HOST_SPEED_TEST, port=MOCK_API_PORT_SPEED_TEST, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    ready_url = f"http://{MOCK_API_HOST_SPEED_TEST}:{MOCK_API_PORT_SPEED_TEST}/health"
    if not is_server_ready(ready_url, timeout=10.0, poll_interval=0.2): # pragma: no cover
        server.should_exit = True # Attempt to clean up if server failed
        thread.join(timeout=1)
        pytest.skip(f"Mock Speed Test API server on port {MOCK_API_PORT_SPEED_TEST} did not become ready.")

    yield app

    server.should_exit = True
    thread.join(timeout=5)

# --- Test Case ---
@pytest.mark.performance
@pytest.mark.parametrize("test_batch_size", [50])
def test_dataset_rollout_speed(mock_rollout_api_for_speed_test, test_batch_size: int, caplog):
    """
    Tests the speed of fetching and processing one batch from AtroposDataset.
    """
    caplog.set_level(logging.INFO)
    logger_test = logging.getLogger("TestRolloutSpeed") # Test specific logger

    api_url = f"http://{MOCK_API_HOST_SPEED_TEST}:{MOCK_API_PORT_SPEED_TEST}"

    simple_rollout_group = {"prompt": "hi", "response": "hello", "reward": 1.0}
    mock_rollout_api_for_speed_test.state.rollout_data_store = [
        simple_rollout_group for _ in range(test_batch_size)
    ]

    # Use DummyTokenizer from atropos_dataset.py, set small max_length for speed test
    # "hihello" is 7 chars. TestTokenizer adds EOS, so 8 tokens. Padded/truncated to 10.
    tokenizer = DummyTokenizer(max_length=10)

    dataset = AtroposDataset(
        api_url=api_url,
        batch_size=test_batch_size,
        tokenizer=tokenizer,
        max_seq_len=10
    )

    logger_test.info(f"Starting rollout speed test for batch_size={test_batch_size}...")
    start_time = time.perf_counter()

    batch_data = None
    try:
        dataset_iterator = iter(dataset)
        batch_data = next(dataset_iterator)
    except Exception as e: # pragma: no cover
        pytest.fail(f"Fetching batch from dataset failed: {e}")

    end_time = time.perf_counter()
    duration = end_time - start_time

    assert batch_data is not None and "input_ids" in batch_data, \
        "Dataset did not return valid batch data."
    assert isinstance(batch_data["input_ids"], torch.Tensor), \
        "input_ids in batch should be a torch.Tensor."
    assert batch_data["input_ids"].shape[0] == test_batch_size, \
        f"Expected batch size {test_batch_size}, got {batch_data['input_ids'].shape[0]}"

    default_threshold_seconds = 30.0
    threshold_str = os.environ.get("ROLLOUT_SPEED_THRESHOLD")
    if threshold_str: # pragma: no cover
        try:
            threshold_seconds = float(threshold_str)
            logger_test.info(f"Using custom ROLLOUT_SPEED_THRESHOLD: {threshold_seconds}s")
        except ValueError:
            logger_test.warning(f"Invalid ROLLOUT_SPEED_THRESHOLD value '{threshold_str}'. Using default {default_threshold_seconds}s.")
            threshold_seconds = default_threshold_seconds
    else:
        threshold_seconds = default_threshold_seconds

    logger_test.info(f"Rollout speed test for batch_size={test_batch_size}: {duration:.4f} seconds (Threshold: {threshold_seconds}s)")

    assert duration < threshold_seconds, \
        f"Rollout fetching and processing took {duration:.4f}s, exceeding threshold of {threshold_seconds}s for batch size {test_batch_size}."

```
