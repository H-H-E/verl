# tests/test_full_e2e_cpu.py
import pytest
import subprocess
import sys
from pathlib import Path
import yaml # For creating config fixture
import time
import threading
import uvicorn
from fastapi import FastAPI
import os # For environment variable for E2E test flag
import logging # For caplog
import requests # For fixture readiness check

# Project layout for launch script
PROJECT_ROOT_E2E = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT_PATH_E2E = PROJECT_ROOT_E2E / "recipe" / "atropos" / "launch_atropos_verl.py"
DUMMY_ENV_SCRIPT_DIR_E2E = PROJECT_ROOT_E2E / "tests" / "dummy_env_scripts"

# Ensure the launch script itself exists, otherwise skip all tests in this file
if not LAUNCH_SCRIPT_PATH_E2E.exists(): # pragma: no cover
    pytest.skip(f"Launch script not found at {LAUNCH_SCRIPT_PATH_E2E}, skipping E2E tests.", allow_module_level=True)

# --- Mock Atropos API Server (for launch_atropos_verl.py to connect to) ---
MOCK_ATROPOS_API_PORT_E2E = 8120 # Must match e2e_cpu_config.yaml

@pytest.fixture(scope="function")
def mock_e2e_atropos_api():
    app = FastAPI()

    app.state.environments_registered = {}
    # Data for AtroposDataset. Note: prompt_token_lengths is added by dataset, not served here.
    app.state.rollout_data_to_serve = [{
        "prompt": "TestP", "response": "TestR", "reward": 1.0, # Shortened for small max_seq_len
        "logprobs": [-1.0, -1.0, -1.0, -1.0, -1.0] # Length matching "TestR"
    }]

    @app.get("/health")
    async def health():
        return {"status": "healthy_e2e_atropos_mock"}

    @app.get("/environments")
    async def get_environments():
        return {"environments": app.state.environments_registered}

    @app.get("/batch")
    async def get_batch(size: int):
        # Serve 'size' copies of the data. Trainer config uses batch_size=1.
        data_to_return = [app.state.rollout_data_to_serve[0] for _ in range(size)]
        return data_to_return

    config = uvicorn.Config(app, host="localhost", port=MOCK_ATROPOS_API_PORT_E2E, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready by polling health endpoint
    health_url = f"http://localhost:{MOCK_ATROPOS_API_PORT_E2E}/health"
    server_ready = False
    start_time = time.monotonic()
    while time.monotonic() - start_time < 10: # 10s timeout
        try:
            response = requests.get(health_url, timeout=0.5)
            if response.status_code == 200:
                server_ready = True
                break
        except requests.ConnectionError:
            time.sleep(0.1)
    if not server_ready: # pragma: no cover
        server.should_exit = True
        thread.join(timeout=1)
        pytest.skip(f"Mock E2E Atropos API server on port {MOCK_ATROPOS_API_PORT_E2E} did not become ready.")

    yield app

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def e2e_cpu_config_file(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp("e2e_fixtures_cpu") # Unique name
    config_content = {
        "model": "e2e-cpu-dummy-model",
        "environments": ["e2e_dummy_env"],
        "rollout_server_port": MOCK_ATROPOS_API_PORT_E2E,
        "inference_api_port": 8121,
        "use_sglang": False,
        "lr": 1e-3, # Higher LR for dummy model to learn "something"
        "num_iterations": 1,
        "batch_size": 1,
        "ppo_epochs": 1,
        "max_seq_len": 16,
        "clip_ratio": 0.2, "kl_coef": 0.0, "entropy_coef": 0.0, # Simplified loss for E2E
        "tensor_parallel": 1,
        "reference_model": None, # Ensure this is explicit
    }
    config_file_path = fixture_dir / "e2e_cpu_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_file_path

# The dummy_env_scripts/e2e_dummy_env_server.py is already created by a previous step.
# This fixture just confirms its directory exists for clarity if needed by other tests.
@pytest.fixture(scope="module", autouse=False) # Not autouse, test will specify path
def e2e_dummy_env_script_directory():
    DUMMY_ENV_SCRIPT_DIR_E2E.mkdir(exist_ok=True)
    # Check if the specific script exists, fail early if not
    expected_script = DUMMY_ENV_SCRIPT_DIR_E2E / "e2e_dummy_env_server.py"
    if not expected_script.exists(): # pragma: no cover
        pytest.fail(f"Required dummy environment script {expected_script} not found. Was it created in a previous step?")
    return DUMMY_ENV_SCRIPT_DIR_E2E


E2E_TEST_ENABLED = os.environ.get("RUN_E2E_TESTS", "false").lower() == "true"

@pytest.mark.skipif(not E2E_TEST_ENABLED, reason="Full E2E test, skipped by default (set RUN_E2E_TESTS=true to enable).")
@pytest.mark.e2e_test
def test_full_loop_cpu(e2e_cpu_config_file: Path, mock_e2e_atropos_api, e2e_dummy_env_script_directory: Path, caplog):
    caplog.set_level(logging.INFO)

    # Configure the mock Atropos API to mark the dummy env as "registered"
    mock_e2e_atropos_api.state.environments_registered = {"e2e_dummy_env": {"status":"registered"}} # Match structure _is_env_registered checks

    command = [
        sys.executable, str(LAUNCH_SCRIPT_PATH_E2E),
        "--config", str(e2e_cpu_config_file),
        "--env-script-path", str(e2e_dummy_env_script_directory)
    ]

    timeout_seconds = 60
    stdout_data = ""
    stderr_data = ""

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout_data, stderr_data = process.communicate(timeout=timeout_seconds)
        return_code = process.returncode
    except subprocess.TimeoutExpired: # pragma: no cover
        process.kill()
        stdout_data, stderr_data = process.communicate()
        pytest.fail(f"E2E test command timed out after {timeout_seconds}s. Command: {' '.join(command)}\nStdout:\n{stdout_data}\nStderr:\n{stderr_data}")
    except FileNotFoundError: # pragma: no cover
        pytest.fail(f"Launch script not found at {LAUNCH_SCRIPT_PATH_E2E}.")
    except Exception as e: # pragma: no cover
        pytest.fail(f"E2E test command failed with exception: {e}. Command: {' '.join(command)}")

    full_output = stdout_data + stderr_data
    if return_code != 0: # pragma: no cover
        print("E2E Test Full Output (on failure):")
        print(full_output)

    assert return_code == 0, f"E2E test script failed. RC: {return_code}.\nOutput:\n{full_output}"

    # Check for key log messages indicating successful stages
    assert "Configuration loaded successfully." in full_output
    assert "Atropos API server started" in full_output
    assert "Environment server 'e2e_dummy_env' initiated" in full_output # Based on launch script log
    assert "Initializing AtroposGrpoTrainer" in full_output
    assert "Attempting to fetch rollouts..." in full_output # From trainer
    assert "Successfully fetched batch." in full_output # From trainer

    # This depends on compute_loss being implemented and trainer actually running.
    # If compute_loss is still dummy 0.0, this will be "Loss: 0.0"
    assert "PPO Epoch 1/1, Loss:" in full_output

    assert "Training finished successfully." in full_output # From launch script
    assert "Initiating cleanup of all started resources..." in full_output
    assert "Stopping environment server PID" in full_output # Check for environment server stop
    assert "Stopping Atropos API server..." in full_output
    # Trainer's cleanup logs stopping its inference server.
    # For Embedded server: "Stopping Uvicorn server..." or "EmbeddedInferenceServer stopped."
    # For External (dummy): "ExternalInferenceWrapper: stop called."
    assert "Cleaning up GRPO trainer (handles its own inference server)..." in full_output
    assert ("EmbeddedInferenceServer stopped." in full_output or "ExternalInferenceWrapper: stop called." in full_output)
    assert "Cleanup finished." in full_output
