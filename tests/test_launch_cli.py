# tests/test_launch_cli.py
import pytest
import subprocess
import sys
from pathlib import Path
import yaml # To create the fixture file
import time # For the mock API server, though not directly in this test

# Define project root assuming tests are in tests/something.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
LAUNCH_SCRIPT_PATH = PROJECT_ROOT / "recipe" / "atropos" / "launch_atropos_verl.py"

# Ensure the launch script itself exists, otherwise skip all tests in this file
if not LAUNCH_SCRIPT_PATH.exists(): # pragma: no cover
    pytest.skip(f"Launch script not found at {LAUNCH_SCRIPT_PATH}, skipping all tests in this file.", allow_module_level=True)


@pytest.fixture(scope="module") 
def minimal_config_file(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp("fixtures")
    # This content should be comprehensive enough for AtroposConfig to load
    # and for the dry run to mention various components.
    config_content = {
        "model": "dummy-cli-model",
        "environments": ["gsm8k_dummy", "another_dummy_env"],
        "rollout_server_port": 8000,
        "inference_api_port": 8001,
        "use_sglang": False, 
        "reference_model": None, # Explicitly None
        "batch_size": 2, 
        "tensor_parallel": 1,
        "lr": 1e-5,
        "clip_ratio": 0.2,
        "ppo_epochs": 1, # Small for tests if it were a real run
        "kl_coef": 0.0,  # Explicitly disable for clarity
        "entropy_coef": 0.01,
        "num_iterations": 3, # For dry run log
        "max_seq_len": 64 # Assuming AtroposConfig or trainer setup might use this
    }
    config_file = fixture_dir / "minimal_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    return config_file


def test_dry_run_output(minimal_config_file: Path):
    """
    Tests the --dry-run mode of launch_atropos_verl.py.
    Checks for successful execution (exit code 0) and presence of key phrases in stdout/stderr.
    """
    command = [
        sys.executable, 
        str(LAUNCH_SCRIPT_PATH),
        "--config", str(minimal_config_file),
        "--dry-run",
        "--env-script-path", "dummy_env_path_for_dry_run" 
    ]

    try:
        # Setting a timeout for the subprocess run
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20) # Increased timeout slightly
    except FileNotFoundError: # pragma: no cover
        pytest.fail(f"Launch script not found at {LAUNCH_SCRIPT_PATH}. Ensure path is correct and script is executable if needed.")
    except subprocess.TimeoutExpired: # pragma: no cover
        pytest.fail(f"Dry run command timed out: {' '.join(command)}")


    assert result.returncode == 0, \
        f"Dry run failed with exit code {result.returncode}.\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}"

    output = result.stdout + result.stderr 

    # Positive assertions for key phrases
    assert "--- DRY RUN MODE ---" in output, "Missing start of dry run mode marker."
    assert "Config loaded:" in output, "Missing 'Config loaded' message."
    assert "dummy-cli-model" in output, "Config model 'dummy-cli-model' not mentioned."
    
    # Check for inference server plans (type and port)
    # The launch script logs "Inference Server: Type ... on port ... for model ... (managed by Trainer)."
    assert "Planned: Start Inference Server" in output or "Inference Server: Type" in output, "Missing Inference Server planning."
    assert "port 8001" in output, "Inference API port 8001 not mentioned."
    assert "Embedded" in output, "Expected 'Embedded' inference server type for use_sglang=False."

    assert "Planned: Start Atropos API Server" in output or "Atropos API Server: Port" in output, "Missing Atropos API Server planning."
    assert "port 8000" in output, "Rollout server port 8000 not mentioned."
    
    assert "Planned: Start Environment Server for 'gsm8k_dummy'" in output or \
           "Environment Server: For 'gsm8k_dummy'" in output, "Missing 'gsm8k_dummy' env planning."
    assert "Planned: Start Environment Server for 'another_dummy_env'" in output or \
           "Environment Server: For 'another_dummy_env'" in output, "Missing 'another_dummy_env' env planning."
    assert "dummy_env_path_for_dry_run" in output, "Specified env-script-path not mentioned."
           
    assert "Planned: Instantiate AtroposGrpoTrainer" in output or "AtroposGrpoTrainer: Model" in output, "Missing Trainer instantiation planning."
    assert "Planned: Call trainer.train(num_iterations=3)" in output or "num_iterations=3" in output, "Missing trainer.train planning with num_iterations."
    
    assert "Planned: Cleanup all resources on exit." in output or "Cleanup all resources on exit" in output, "Missing cleanup planning."
    assert "--- END DRY RUN ---" in output, "Missing end of dry run mode marker."
    
    # Negative assertions: check that no actual error messages are in output
    # These need to be specific enough not to match legitimate "error" in log levels or planning messages.
    # Convert output to lower for case-insensitive matching of error terms.
    output_lower = output.lower()
    assert "failed to start" not in output_lower, "Dry run should not attempt to start and fail services."
    # Be careful with generic "error" if normal log messages might contain it.
    # Check for specific error patterns if possible, or ensure logs are at INFO for dry run.
    # For now, let's assume "error" in a log line that isn't explicitly about "Error during..." is bad.
    # This is a heuristic. A more robust way is to check specific known error messages.
    # Example: if "critical" or "exception" appears outside of a planned log of an error handler.
    # The launch script logs "Error during..." for caught exceptions, which is fine.
    # But "Failed to start" or unhandled Python exceptions should not appear.
    assert "traceback (most recent call last)" not in output_lower, "Dry run should not produce Python tracebacks."
    assert "runtimeerror" not in output_lower, "Dry run should not result in RuntimeErrors being printed."
```
