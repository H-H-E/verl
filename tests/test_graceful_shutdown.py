# tests/test_graceful_shutdown.py
import pytest
import subprocess
import sys
from pathlib import Path
import yaml 
import time
import signal
import psutil # For checking if processes are alive
import os
import logging

# Project layout
PROJECT_ROOT_SD_TEST = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT_PATH_SD_TEST = PROJECT_ROOT_SD_TEST / "recipe" / "atropos" / "launch_atropos_verl.py"
DUMMY_ENV_SCRIPT_DIR_SD_TEST = PROJECT_ROOT_SD_TEST / "tests" / "dummy_env_scripts"

# Ensure the launch script itself exists, otherwise skip all tests in this file
if not LAUNCH_SCRIPT_PATH_SD_TEST.exists(): # pragma: no cover
    pytest.skip(f"Launch script not found at {LAUNCH_SCRIPT_PATH_SD_TEST}, skipping shutdown tests.", allow_module_level=True)

# Skip E2E test by default unless env var is set
E2E_TEST_ENABLED_SD = os.environ.get("RUN_E2E_TESTS", "false").lower() == "true"

@pytest.fixture(scope="module")
def shutdown_test_config_file(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp("shutdown_fixtures")
    config_content = {
        "model": "shutdown-test-dummy-model",
        "environments": ["shutdown_dummy_env"], # Corresponds to shutdown_dummy_env_server.py
        "rollout_server_port": 8130,
        "inference_api_port": 8131, 
        "use_sglang": False,
        "lr": 1e-4, 
        "num_iterations": 10, # Run enough iterations for sleep to be meaningful
        "batch_size": 1,    
        "ppo_epochs": 1,    
        "max_seq_len": 16,   
        "test_mode_iteration_sleep_s": 10, # Key for this test: sleep in trainer
        "clip_ratio": 0.2, 
        "kl_coef": 0.0, 
        "entropy_coef": 0.01, 
        "tensor_parallel": 1,
        "reference_model": None,
    }
    config_file_path = fixture_dir / "shutdown_test_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_file_path

# The shutdown_dummy_env_server.py was created in a previous step.
# This fixture just ensures its directory is known and script exists.
@pytest.fixture(scope="module", autouse=True)
def check_shutdown_dummy_env_script_exists():
    DUMMY_ENV_SCRIPT_DIR_SD_TEST.mkdir(exist_ok=True) # Ensure dir exists
    script_path = DUMMY_ENV_SCRIPT_DIR_SD_TEST / "shutdown_dummy_env_server.py"
    if not script_path.exists(): # pragma: no cover
        pytest.fail(f"Dummy environment script {script_path} not found. It should have been created in a prior step.")
    # No need to yield path, test uses DUMMY_ENV_SCRIPT_DIR_SD_TEST directly.

# No separate mock Atropos API server fixture is needed here because the
# launch_atropos_verl.py script itself is responsible for starting the
# Atropos API server (using start_atropos_api, which uses dummy components internally if not mocked).
# This test is for the launch script's overall behavior and cleanup.

@pytest.mark.skipif(not E2E_TEST_ENABLED_SD, reason="Graceful shutdown test is E2E, skipped by default (set RUN_E2E_TESTS=true).")
@pytest.mark.e2e_test 
def test_ctrl_c_cleanup(shutdown_test_config_file: Path, caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("TestCtrlCShutdown") # For test-specific logs

    command = [
        sys.executable, str(LAUNCH_SCRIPT_PATH_SD_TEST),
        "--config", str(shutdown_test_config_file),
        "--env-script-path", str(DUMMY_ENV_SCRIPT_DIR_SD_TEST)
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    # Heuristic: Try to capture PIDs if logged by the main script.
    # This is for optional deeper check, primary check is logs and exit code.
    child_pids = set()
    
    # Let the script run for a bit to start services and enter the trainer's sleep
    # Wait less than test_mode_iteration_sleep_s to interrupt during sleep.
    # Increased initial wait to allow all servers to potentially start and log PIDs.
    logger.info(f"Allowing launch script {process.pid} to run for ~12 seconds before sending SIGINT...")
    time.sleep(12) 

    # Check if main process is still running before sending signal
    if process.poll() is not None: # pragma: no cover
        stdout, stderr = process.communicate()
        full_output = stdout + stderr
        print(full_output)
        pytest.fail(f"Launch script terminated prematurely with code {process.returncode} before SIGINT could be sent.")

    logger.info(f"Sending SIGINT to launch script process PID {process.pid}...")
    process.send_signal(signal.SIGINT)

    try:
        stdout, stderr = process.communicate(timeout=25) # Allow ample time for cleanup
        return_code = process.returncode
    except subprocess.TimeoutExpired: # pragma: no cover
        logger.error("Launch script did not terminate after SIGINT and timeout. Killing.")
        process.kill()
        stdout, stderr = process.communicate()
        return_code = process.returncode # Likely non-zero after kill
        pytest.fail(f"Launch script timed out after SIGINT. RC: {return_code}\nStdout:\n{stdout}\nStderr:\n{stderr}")

    full_output = stdout + stderr
    if return_code != 0: # SIGINT should lead to exit 0 if handled gracefully by signal_handler's sys.exit(0)
        logger.error(f"Launch script output on non-zero exit ({return_code}):\n{full_output}")

    assert return_code == 0, \
        f"Launch script exited with code {return_code} after SIGINT, expected 0 for graceful shutdown."

    # Check logs for evidence of graceful shutdown
    assert "Signal SIGINT received. Starting graceful shutdown..." in full_output or \
           "KeyboardInterrupt received by main loop. Initiating graceful shutdown..." in full_output, \
           "Did not find SIGINT or KeyboardInterrupt handling message in logs."
    
    # Check for specific cleanup messages from cleanup_resources()
    assert "Cleaning up GRPO trainer" in full_output
    assert "Stopping environment server PID" in full_output # Check for at least one env server stop msg
    assert "Stopping Atropos API server..." in full_output
    # Check for inference server stop log (depends on type, but trainer cleanup should log it)
    assert ("EmbeddedInferenceServer stopped." in full_output or \
            "ExternalInferenceWrapper: stop called." in full_output or \
            "Cleaning up GRPO trainer (handles its own inference server)..." in full_output), \
           "Did not find log message for inference server cleanup."
    assert "Cleanup finished." in full_output, "Missing 'Cleanup finished.' log message."

    # Optional: Parse PIDs from output and check with psutil (if psutil is a project dependency)
    # Example of parsing (can be made more robust):
    import re
    parsed_pids = set()
    for line in full_output.splitlines():
        # Example: "Atropos API server started successfully (PID: 12345)."
        match_api = re.search(r"Atropos API server started .*\(PID: (\d+)\)", line)
        if match_api: parsed_pids.add(int(match_api.group(1)))
        # Example: "Environment server 'shutdown_dummy_env' initiated (PID: 12346)."
        match_env = re.search(r"Environment server '.*?' initiated \(PID: (\d+)\)", line)
        if match_env: parsed_pids.add(int(match_env.group(1)))
        # Note: EmbeddedInferenceServer runs in a thread, not a separate process managed by launch script globals.
        # ExternalInferenceWrapper *could* start a process, but its dummy doesn't.
        # The trainer itself is not a separate process from the launch script.

    logger.info(f"Parsed PIDs from launch script output: {parsed_pids}")
    if parsed_pids:
        time.sleep(1) # Give OS a moment to fully release PIDs after processes terminate
        for pid in parsed_pids:
            assert not psutil.pid_exists(pid), f"Child process PID {pid} was still running after cleanup."
        logger.info(f"Confirmed PIDs {parsed_pids} are no longer running.")
    else: # pragma: no cover
        logger.warning("Could not parse any child PIDs from launch script output to verify their termination with psutil.")
        # This is not a test failure, but a limitation of this PID checking approach if logs don't match.

    logger.info("Graceful shutdown test completed successfully.")

```
