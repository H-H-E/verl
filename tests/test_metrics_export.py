# tests/test_metrics_export.py
import pytest
import torch
from typing import Dict, Any, List
import threading
import uvicorn
from fastapi import FastAPI
import time
import json
from pathlib import Path
import os # For changing CWD for test output dirs, or using tmp_path
import logging # For caplog
import requests # For fixture readiness check

# Components to test/use
from verl.atropos_config import AtroposConfig
from verl.train.grpo_trainer import AtroposGrpoTrainer 
from verl.atropos_inference import is_server_ready # For robust server check in fixture

# Ensure dummy components used by AtroposGrpoTrainer are accessible if not already imported by trainer

# --- Mock Atropos API Server ---
MOCK_API_PORT_METRICS_TEST = 8112 
MOCK_API_HOST_METRICS_TEST = "localhost"

@pytest.fixture(scope="function")
def mock_rollout_api_for_metrics_test(tmp_path: Path): 
    app = FastAPI()
    fixed_rollout_group = {
        "prompt": "MetricsP", "response": "MetricsR", "reward": 0.5, # Shortened
        "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7], # Length matching "MetricsR" + EOS (7 chars)
        "token_advantages": [0.4,0.5,0.6,0.7,0.8,0.9,1.0] 
    }
    
    @app.get("/batch")
    async def get_rollout_batch(size: int):
        return [fixed_rollout_group for _ in range(size)]

    @app.get("/health")
    async def health_check(): return {"status": "healthy_metrics_mock"}

    original_cwd = Path.cwd()
    os.chdir(tmp_path) 

    config_uvicorn = uvicorn.Config(app, host=MOCK_API_HOST_METRICS_TEST, port=MOCK_API_PORT_METRICS_TEST, log_level="warning")
    server = uvicorn.Server(config_uvicorn)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    ready_url = f"http://{MOCK_API_HOST_METRICS_TEST}:{MOCK_API_PORT_METRICS_TEST}/health"
    if not is_server_ready(ready_url, timeout=10.0, poll_interval=0.2): # pragma: no cover
        os.chdir(original_cwd) 
        server.should_exit = True # Attempt to stop server
        thread.join(timeout=1)
        pytest.skip(f"Mock Metrics API server on port {MOCK_API_PORT_METRICS_TEST} did not become ready.")
    
    yield f"http://{MOCK_API_HOST_METRICS_TEST}:{MOCK_API_PORT_METRICS_TEST}" 
    
    server.should_exit = True
    thread.join(timeout=5)
    os.chdir(original_cwd) 


# --- Test Cases ---
def test_tensorboard_and_jsonl_metrics_created(mock_rollout_api_for_metrics_test: str, tmp_path: Path, caplog):
    """
    Tests that AtroposGrpoTrainer creates TensorBoard event files and a JSONL metrics file
    with expected content after one training iteration. Test runs within tmp_path.
    """
    caplog.set_level(logging.INFO)

    api_url = mock_rollout_api_for_metrics_test 
    server_port = int(api_url.split(":")[-1]) 
    
    try:
        config = AtroposConfig(
            model="dummy-metrics-model", 
            environments=["metrics_test_env"],
            rollout_server_port=server_port,
            batch_size=1, 
            ppo_epochs=1, 
            lr=1e-4,      
            num_iterations=1, 
            clip_ratio=0.2, kl_coef=0.05, entropy_coef=0.001, tensor_parallel=1,
            max_seq_len=32, # Important for dataset and model ops
            reference_model=None # Explicitly ensure no ref model if kl_coef > 0 but not testing ref_model loading
        )
    except TypeError as e: # pragma: no cover (if AtroposConfig definition changed)
        pytest.fail(f"Failed to instantiate AtroposConfig, check fields: {e}")


    trainer = None # Define trainer outside try for access in finally
    try:
        trainer = AtroposGrpoTrainer(config)
        trainer.train(num_iterations=1) 
    except Exception as e: # pragma: no cover
        print("Logs during trainer execution for metrics test failure:", caplog.text)
        pytest.fail(f"Trainer execution for metrics test raised an exception: {e}")
    finally:
        if trainer and hasattr(trainer, 'cleanup'): # pragma: no cover
            trainer.cleanup() # Ensure summary writer is closed


    # --- Assertions for TensorBoard ---
    # All paths are now relative to tmp_path because of chdir in fixture
    runs_dir = Path("runs") # tmp_path / "runs"
    assert runs_dir.exists() and runs_dir.is_dir(), f"TensorBoard 'runs/' directory was not created in {tmp_path}."
    
    run_subdirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("grpo_")]
    assert len(run_subdirs) >= 1, f"Expected at least 1 run subdirectory in {runs_dir}, found {len(run_subdirs)}"
    specific_run_dir = run_subdirs[0] # Assume first one is ours for this test
    
    event_files = list(specific_run_dir.glob("events.out.tfevents.*"))
    assert len(event_files) >= 1, f"No TensorBoard event file found in {specific_run_dir}"
    assert event_files[0].stat().st_size > 0, f"TensorBoard event file {event_files[0]} is empty."
    
    # --- Assertions for JSONL Metrics File ---
    metrics_dir = Path("metrics") # tmp_path / "metrics"
    assert metrics_dir.exists() and metrics_dir.is_dir(), f"Metrics 'metrics/' directory was not created in {tmp_path}."
    
    metrics_file_path = metrics_dir / "logs.jsonl"
    assert metrics_file_path.exists() and metrics_file_path.is_file(), "JSONL metrics file 'logs.jsonl' was not created."
    
    logged_metrics = []
    try:
        with open(metrics_file_path, 'r') as f:
            for line in f:
                if line.strip(): 
                    logged_metrics.append(json.loads(line))
    except Exception as e_json_read: # pragma: no cover
        pytest.fail(f"Error reading or parsing JSONL metrics file {metrics_file_path}: {e_json_read}")
        
    assert len(logged_metrics) == 1, f"Expected 1 log entry in JSONL file for 1 iteration, found {len(logged_metrics)}"
    
    first_entry = logged_metrics[0]
    
    # Iteration in JSONL is 1-based from trainer's implementation
    assert first_entry.get("iteration") == 1, "JSONL 'iteration' should be 1 for the first iteration."
    assert "timestamp" in first_entry and isinstance(first_entry["timestamp"], str)

    expected_metric_keys = [
        "reward_mean_advantages", "total_loss", "ppo_objective", "kl_penalty", "kl_divergence_raw"
    ]
    for key in expected_metric_keys:
        assert key in first_entry, f"JSONL entry missing key: '{key}'"
        assert isinstance(first_entry[key], float), f"JSONL key '{key}' should be a float, got {type(first_entry[key])} (value: {first_entry[key]})"
        
    # Check that current working directory is restored by fixture (important for other tests)
    # This assertion is better placed in the fixture's teardown or a separate test if critical.
    # For now, we assume fixture handles CWD restoration.
    # The `os.chdir(original_cwd)` in fixture's teardown should cover this.
```
