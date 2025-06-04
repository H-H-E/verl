# tests/test_env_spawner.py
import pytest
import subprocess
from unittest import mock
import time
import socket
from pathlib import Path
import threading # For running dummy FastAPI server
import uvicorn # For dummy FastAPI server
from fastapi import FastAPI # For dummy FastAPI server
import sys # To get python executable for dummy script
from typing import Optional # For type hinting in dummy script factory

from verl.atropos_env_spawner import start_env_server, stop_env_server

# --- Fixtures and Helpers ---

LOGS_DIR_TEST = Path("logs") # Ensure logs dir is same as used by module
LOGS_DIR_TEST.mkdir(exist_ok=True)

DUMMY_ENV_SCRIPTS_DIR = Path("tests/dummy_env_scripts")
DUMMY_ENV_SCRIPTS_DIR.mkdir(exist_ok=True)

# Dummy Atropos API server using FastAPI
@pytest.fixture(scope="function")
def dummy_atropos_api_server():
    app = FastAPI()

    # State for the dummy API, accessible via app.state
    # Initialize with a structure that _is_env_registered can parse
    app.state.registered_environments = {"environments": []}

    @app.get("/environments")
    async def get_environments_endpoint(): # Renamed to avoid conflict with any potential var named get_environments
        return app.state.registered_environments

    port = 8090
    host = "localhost"

    # Check if port is free using a more reliable method for fixtures
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except socket.error:
        pytest.skip(f"Port {port} for dummy Atropos API is already in use.")
    finally:
        sock.close() # Release the port after checking

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready by polling a simple health endpoint
    health_url = f"http://{host}:{port}/health" # Add a health endpoint to dummy API for robust check
    @app.get("/health")
    async def health_check(): return {"status": "ok"}

    start_time = time.monotonic()
    server_ready = False
    while time.monotonic() - start_time < 10: # 10s timeout for dummy server
        try:
            response = requests.get(health_url, timeout=0.5)
            if response.status_code == 200:
                server_ready = True
                break
        except requests.ConnectionError:
            time.sleep(0.1)
    if not server_ready:
        # Try to stop server before failing test
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail(f"Dummy Atropos API server failed to start on port {port}.")

    yield app

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def dummy_env_script_factory():
    '''Factory to create dummy environment server scripts.'''
    created_scripts_paths = [] # Store Path objects
    def _create_script(env_name: str, content: Optional[str] = None):
        script_path = DUMMY_ENV_SCRIPTS_DIR / f"{env_name}_server.py"
        if content is None:
            content = f'''#!{sys.executable}
import sys
import time
print(f"Dummy {env_name}_server.py running with args: {{sys.argv}}", flush=True)
# Keep running for a bit to simulate a server
# In a real scenario, this script would start its own server (e.g. FastAPI/gRPC)
# and register with the Atropos API. For this test, we just need it to run.
time.sleep(60) # Keep alive for a while unless killed
print(f"Dummy {env_name}_server.py finished.", flush=True)
'''
        with open(script_path, "w") as f:
            f.write(content)
        script_path.chmod(0o755) # Make it executable
        created_scripts_paths.append(script_path)
        return script_path

    yield _create_script

    for script_p in created_scripts_paths:
        if script_p.exists():
            script_p.unlink()
    # Only remove dir if it's empty and was created by this session (harder to track, skip for now)
    # if DUMMY_ENV_SCRIPTS_DIR.exists() and not any(DUMMY_ENV_SCRIPTS_DIR.iterdir()):
    #     DUMMY_ENV_SCRIPTS_DIR.rmdir()


# --- Tests ---

def test_start_env_server_successful_registration(dummy_atropos_api_server, dummy_env_script_factory, caplog):
    '''
    Tests start_env_server with successful registration.
    '''
    caplog.set_level(logging.INFO) # Capture info logs for debugging if needed
    env_name = "testenv_success"
    dummy_env_script_factory(env_name)

    rollout_server_url = "http://localhost:8090" # Matches dummy_atropos_api_server
    inference_url = "http://localhost:8070/v1"

    # Configure dummy API to report env as registered (using one of the formats _is_env_registered checks)
    dummy_atropos_api_server.state.registered_environments["environments"] = [{"name": env_name, "status": "registered"}]

    env_proc = None
    try:
        env_proc = start_env_server(
            env_name=env_name,
            rollout_server_url=rollout_server_url,
            inference_url=inference_url,
            env_script_path=str(DUMMY_ENV_SCRIPTS_DIR),
            readiness_timeout=5,
            poll_interval=1
        )
        assert env_proc is not None, f"start_env_server should return Popen object. Logs: {caplog.text}"
        assert env_proc.poll() is None, f"Env server process should be running. Logs: {caplog.text}"

        expected_log_file = LOGS_DIR_TEST / f"env_{env_name}.log"
        assert expected_log_file.exists(), f"Log file {expected_log_file} not created. Logs: {caplog.text}"

    finally:
        if env_proc and env_proc.poll() is None:
            stop_env_server(env_proc, env_name=env_name)
            try: env_proc.wait(timeout=5)
            except subprocess.TimeoutExpired: env_proc.kill()

def test_start_env_server_registration_timeout(dummy_atropos_api_server, dummy_env_script_factory, caplog):
    '''
    Tests registration timeout.
    '''
    caplog.set_level(logging.INFO)
    env_name = "testenv_reg_timeout"
    dummy_env_script_factory(env_name)

    rollout_server_url = "http://localhost:8090"
    inference_url = "http://localhost:8070/v1"

    # Dummy API will NOT report this env_name as registered
    dummy_atropos_api_server.state.registered_environments["environments"] = [{"name": "other_env", "status": "registered"}]

    env_proc = None # To ensure it's defined for a potential finally block if process did start
    try:
        env_proc = start_env_server(
            env_name=env_name,
            rollout_server_url=rollout_server_url,
            inference_url=inference_url,
            env_script_path=str(DUMMY_ENV_SCRIPTS_DIR),
            readiness_timeout=2,
            poll_interval=0.5
        )
        assert env_proc is None, f"start_env_server should return None on registration timeout. Logs: {caplog.text}"
    finally: # Cleanup if process was started but registration failed
        if env_proc and env_proc.poll() is None:
             stop_env_server(env_proc, env_name=env_name) # stop_env_server handles None proc too
             try: env_proc.wait(timeout=1)
             except subprocess.TimeoutExpired: env_proc.kill()


def test_start_env_server_script_not_found(caplog):
    '''
    Tests FileNotFoundError for non-existent env script.
    '''
    caplog.set_level(logging.ERROR)
    env_name = "nonexistent_env_test"
    with pytest.raises(FileNotFoundError, match=f"Environment server script for '{env_name}' not found"):
        start_env_server(
            env_name=env_name,
            rollout_server_url="http://localhost:8090",
            inference_url="http://localhost:8070/v1",
            env_script_path=str(DUMMY_ENV_SCRIPTS_DIR), # Script won't be in this dir
            readiness_timeout=1
        )
    assert f"Environment server script for '{env_name}' not found" in caplog.text


@mock.patch('verl.atropos_env_spawner.subprocess.Popen')
def test_start_env_server_premature_exit(mock_popen, dummy_atropos_api_server, dummy_env_script_factory, caplog):
    '''
    Tests handling of premature exit of the env server process.
    '''
    caplog.set_level(logging.INFO)
    env_name = "testenv_premature_exit_script"
    dummy_env_script_factory(env_name) # Script is created

    mock_process = mock.Mock(spec=subprocess.Popen)
    mock_process.pid = 9999
    mock_process.poll.return_value = 1 # Simulate process exited with code 1 immediately
    mock_log_fp = mock.Mock(spec=open)
    mock_log_fp.closed = False
    mock_process.stdout = mock_log_fp
    mock_popen.return_value = mock_process

    rollout_server_url = "http://localhost:8090"
    inference_url = "http://localhost:8070/v1"

    env_proc = start_env_server(
        env_name=env_name,
        rollout_server_url=rollout_server_url,
        inference_url=inference_url,
        env_script_path=str(DUMMY_ENV_SCRIPTS_DIR),
        readiness_timeout=3, poll_interval=0.5
    )
    assert env_proc is None, f"start_env_server should return None if env process dies. Logs: {caplog.text}"
    mock_popen.assert_called_once() # Check that Popen was called to start the process
    assert f"Env server for '{env_name}' (PID: 9999) terminated prematurely" in caplog.text


def test_stop_env_server_calls_generic_stop():
    mock_proc = mock.Mock(spec=subprocess.Popen)
    mock_proc.pid = 1234; mock_proc.poll.return_value = None
    with mock.patch('verl.atropos_env_spawner.generic_stop_server') as mock_gss:
        stop_env_server(mock_proc, "test_stop_env")
        mock_gss.assert_called_once_with(mock_proc, server_name="Environment Server 'test_stop_env'")

def test_stop_env_server_fallback_if_no_generic_stop():
    mock_proc = mock.Mock(spec=subprocess.Popen); mock_proc.pid = 1234
    mock_proc.poll.return_value = None; mock_proc.stdout = mock.Mock(); mock_proc.stdout.closed = False
    with mock.patch('verl.atropos_env_spawner.helpers_imported_env', False):
        stop_env_server(mock_proc, "test_stop_fallback")
        mock_proc.terminate.assert_called_once()
        mock_proc.stdout.close.assert_called_once() # Check log file stream is closed
