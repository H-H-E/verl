# tests/test_fetch_rollouts_integration.py
import pytest
import torch
from typing import Dict, Any, List
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
import time # For time.sleep

# Components to test/use
from verl.atropos_config import AtroposConfig
from verl.train.grpo_trainer import AtroposGrpoTrainer
# AtroposGrpoTrainer's __init__ uses dummy loaders for model and tokenizer,
# and AtroposDataset, which are defined in their respective modules.

# --- Mock Atropos API Server ---
MOCK_API_PORT_TRAINER_TEST = 8110
MOCK_API_HOST_TRAINER_TEST = "localhost"

@pytest.fixture(scope="function")
def mock_rollout_api_for_trainer():
    app = FastAPI()
    # Initialize with a structure that AtroposDataset expects (list of dicts)
    app.state.rollout_data_store: List[Dict[str, Any]] = []

    @app.get("/batch")
    async def get_rollout_batch(size: int):
        if not app.state.rollout_data_store:
            return []

        # Serve data based on size, but for this test, we usually configure one batch.
        # If test configures more data than 'size', it will be truncated here.
        data_to_serve = app.state.rollout_data_store[:size]
        # To simulate one-shot data for exhaustion test, clear after serving if needed by test logic.
        # Test logic below will explicitly clear it.
        return data_to_serve

    # Add a health endpoint for robust server start check
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    config = uvicorn.Config(app, host=MOCK_API_HOST_TRAINER_TEST, port=MOCK_API_PORT_TRAINER_TEST, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    health_url = f"http://{MOCK_API_HOST_TRAINER_TEST}:{MOCK_API_PORT_TRAINER_TEST}/health"
    server_ready = False
    start_time = time.monotonic()
    while time.monotonic() - start_time < 5: # 5s timeout for dummy server
        try:
            response = requests.get(health_url, timeout=0.5) # Using requests, need import
            if response.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    if not server_ready: # pragma: no cover
        server.should_exit = True
        thread.join(timeout=1)
        pytest.fail(f"Mock API server failed to start on port {MOCK_API_PORT_TRAINER_TEST}.")

    yield app

    server.should_exit = True
    thread.join(timeout=5)

# Need to import requests for the health check in the fixture
import requests

# --- Test Case ---
def test_dataset_connectivity_and_fetch_rollouts(mock_rollout_api_for_trainer):
    """
    Tests AtroposGrpoTrainer.fetch_rollouts() connectivity to AtroposDataset,
    which in turn connects to a mock Atropos API.
    Asserts that the fetched batch has the expected structure.
    """
    # 1. Configure AtroposConfig for the trainer
    config = AtroposConfig(
        model="dummy-model-for-trainer-test",
        environments=["test_env"],
        rollout_server_port=MOCK_API_PORT_TRAINER_TEST,
        batch_size=1,
        # Ensure AtroposConfig has default for kl_coef, lr, etc., or set them if needed by GRPO_Trainer's __init__
        # The DummyAtroposConfig should provide these. Let's assume it does.
        # Add max_seq_len, as AtroposGrpoTrainer uses it for AtroposDataset
        max_seq_len=32
    )
    # If AtroposConfig doesn't have max_seq_len as a field, setattr is needed.
    # Let's assume it's part of the AtroposConfig definition or we add it.
    # For this test, we are directly providing it to config.

    # 2. Prepare mock data for the API server to serve
    mock_api_data_group = {
        "prompt": "Prmpt",
        "response": "Rspns",
        "reward": 1.0,
        "token_advantages": [0.1, 0.2, 0.3, 0.2, 0.1], # Corresponds to "Rspns" (5 tokens)
        "logprobs": [-0.5, -0.4, -0.3, -0.2, -0.1]    # Corresponds to "Rspns" (5 tokens)
    }
    mock_rollout_api_for_trainer.state.rollout_data_store = [mock_api_data_group] * config.batch_size

    # 3. Instantiate AtroposGrpoTrainer
    try:
        trainer = AtroposGrpoTrainer(config)
        # Ensure the trainer's device is known for assertions
        assert trainer.device is not None, "Trainer device not set"
    except Exception as e: # pragma: no cover
        pytest.fail(f"AtroposGrpoTrainer initialization failed: {e}")

    # 4. Call fetch_rollouts()
    fetched_batch = trainer.fetch_rollouts()

    # 5. Assertions on the fetched batch
    assert isinstance(fetched_batch, dict), "fetch_rollouts should return a dictionary."
    assert bool(fetched_batch), "Fetched batch should not be empty with mock data configured."

    expected_keys = ["input_ids", "attention_mask", "old_logprobs", "advantages", "prompt_token_lengths"]
    for key in expected_keys:
        assert key in fetched_batch, f"Fetched batch is missing expected key: {key}"

    # Check tensor types and device
    assert isinstance(fetched_batch["input_ids"], torch.Tensor)
    assert fetched_batch["input_ids"].device == trainer.device
    assert isinstance(fetched_batch["attention_mask"], torch.Tensor)
    assert fetched_batch["attention_mask"].device == trainer.device
    assert isinstance(fetched_batch["old_logprobs"], torch.Tensor)
    assert fetched_batch["old_logprobs"].device == trainer.device
    assert isinstance(fetched_batch["advantages"], torch.Tensor)
    assert fetched_batch["advantages"].device == trainer.device

    assert isinstance(fetched_batch["prompt_token_lengths"], list)
    if config.batch_size > 0:
      assert len(fetched_batch["prompt_token_lengths"]) == config.batch_size
      assert isinstance(fetched_batch["prompt_token_lengths"][0], int)

    assert fetched_batch["input_ids"].shape == (config.batch_size, config.max_seq_len)
    assert fetched_batch["attention_mask"].shape == (config.batch_size, config.max_seq_len)
    assert fetched_batch["old_logprobs"].shape == (config.batch_size, config.max_seq_len)
    assert fetched_batch["advantages"].shape == (config.batch_size, config.max_seq_len)

    # Test exhaustion
    mock_rollout_api_for_trainer.state.rollout_data_store = [] # API now has no data
    exhausted_batch = trainer.fetch_rollouts()
    assert isinstance(exhausted_batch, dict), "fetch_rollouts should return dict on exhaustion."
    assert not bool(exhausted_batch), "Fetched batch should be empty after data exhaustion."

    if hasattr(trainer, 'cleanup'): # pragma: no cover (dummy trainer might not have real cleanup)
        trainer.cleanup()
