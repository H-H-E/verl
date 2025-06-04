# tests/test_atropos_api.py
import pytest
import subprocess
from unittest import mock # For mocking
from pathlib import Path
import time # For sleep in port check
import socket # For port check

# Functions to test
from verl.atropos_api_launcher import start_atropos_api, stop_atropos_api
# For mocking, we target the name in the module where it's USED.
# So, if atropos_api_launcher.py uses 'is_server_ready' from itself (after import),
# we mock 'verl.atropos_api_launcher.is_server_ready'.

# Helper to check if a port is open
def is_port_in_use(port: int, host: str = "localhost") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1)
    try:
        s.bind((host, port))
        return False
    except socket.error:
        return True
    finally:
        s.close()


@pytest.fixture
def mock_popen_fixture():
    '''Fixture to mock subprocess.Popen and the process it returns.'''
    with mock.patch('subprocess.Popen') as mock_popen_constructor:
        mock_process = mock.Mock()  # Remove spec=subprocess.Popen to avoid attribute errors
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        mock_log_fp = mock.Mock(spec=open) # Mock the file pointer object
        mock_log_fp.closed = False
        def close_fp():
            mock_log_fp.closed = True
        mock_log_fp.close = mock.Mock(side_effect=close_fp)

        # In the code, Popen is called with stdout=log_fp, stderr=subprocess.STDOUT
        # So, proc.stdout would be this log_fp object.
        mock_process.stdout = mock_log_fp
        # stderr is not explicitly set on mock_process if it's STDOUT, Popen handles that.

        mock_popen_constructor.return_value = mock_process
        yield mock_popen_constructor, mock_process

# Test for FileNotFoundError when 'atropos' command is not found
@mock.patch('subprocess.Popen', side_effect=FileNotFoundError("'atropos' command not found"))
def test_start_atropos_api_command_not_found(mock_subproc_popen_fnf, caplog):
    '''
    Tests that start_atropos_api handles FileNotFoundError.
    '''
    test_port = 8088
    process = start_atropos_api(port=test_port)
    assert process is None, "start_atropos_api should return None when command is not found."
    assert "Atropos command 'atropos' not found" in caplog.text # Fix: Match actual log message

# Test for successful lifecycle with mocks
@mock.patch('verl.atropos_api_launcher.helpers_imported', True)
@mock.patch('verl.atropos_api_launcher.is_server_ready', return_value=True, create=True)
@mock.patch('verl.atropos_api_launcher.generic_stop_server', create=True) # Add create=True for functions that may not exist
def test_atropos_api_lifecycle_mocked(mock_generic_stop, mock_is_ready, mock_popen_fixture):
    '''
    Tests the mocked lifecycle (start, stop) of the Atropos API.
    '''
    mock_popen_constructor, mock_process_instance = mock_popen_fixture
    test_port = 8080

    # --- Test start_atropos_api ---
    api_proc = start_atropos_api(port=test_port, host="127.0.0.1", readiness_timeout=10)

    assert api_proc is mock_process_instance

    expected_command = ["atropos", "run-api", "--port", str(test_port), "--host", "127.0.0.1"]
    # Check that Popen was called. args_list[0][0] gives the first positional arg (the command list)
    # The actual call is mock_popen_constructor(expected_command, stdout=ANY, stderr=ANY)
    # We should check the first argument of the call
    called_command = mock_popen_constructor.call_args[0][0]
    assert called_command == expected_command

    expected_health_url = f"http://127.0.0.1:{test_port}/health"
    mock_is_ready.assert_called_once_with(expected_health_url, timeout=10)

    # --- Test stop_atropos_api ---
    # Ensure poll returns None initially (running), then 0 (stopped)
    mock_process_instance.poll.side_effect = [None, 0]

    stop_atropos_api(api_proc)
    # Check if generic_stop_server was called (assuming it was imported successfully)
    # This depends on 'helpers_imported' being True in atropos_api_launcher.
    # If we want to test the fallback, we'd need to manipulate that import.
    # For now, assume helper is used.
    if hasattr(mock_generic_stop, 'assert_called_once_with'): # Check if it's a mock
        mock_generic_stop.assert_called_once_with(api_proc, server_name="Atropos API Server")
    else:
        # This case implies generic_stop_server was not mocked, meaning it might not have been imported.
        # The test should be structured to explicitly test one path or the other.
        # For this test, we are directly mocking generic_stop_server via the decorator.
        pass


@mock.patch('verl.atropos_api_launcher.helpers_imported', True)
@mock.patch('verl.atropos_api_launcher.is_server_ready', return_value=False, create=True) # Mock readiness check failing
@mock.patch('verl.atropos_api_launcher.generic_stop_server', create=True) # Also mock stop server for the cleanup path
def test_start_atropos_api_readiness_fails(mock_generic_stop_cleanup, mock_is_ready_fail, mock_popen_fixture, caplog):
    '''
    Tests that if is_server_ready returns False, start_atropos_api stops the process and returns None.
    '''
    mock_popen_constructor, mock_process_instance = mock_popen_fixture
    test_port = 8081

    api_proc = start_atropos_api(port=test_port, readiness_timeout=0.1)
    assert api_proc is None, "start_atropos_api should return None if readiness check fails."

    # Check that is_server_ready was called
    expected_health_url = f"http://localhost:{test_port}/health" # Use default host "localhost"
    mock_is_ready_fail.assert_called_once_with(expected_health_url, timeout=0.1)

    # Check that an attempt was made to stop the process via generic_stop_server
    # This mock is for the call inside start_atropos_api's error handling.
    mock_generic_stop_cleanup.assert_called_once_with(mock_process_instance, server_name=f"Atropos API (port {test_port})")
    assert f"Atropos API server failed to become ready within 0.1s" in caplog.text


@mock.patch('subprocess.Popen') # Basic Popen mock, no fixture needed for this specific test
def test_stop_atropos_api_no_proc(mock_popen_constructor_basic):
    '''Test stop_atropos_api with None process.'''
    # We don't need generic_stop_server mocked here as it shouldn't be called for None proc
    stop_atropos_api(None)
    # No assertion needed, just checking it runs without error.
    # Can add a check that logger.info was called with "process is None" if logging is captured.

@mock.patch('verl.atropos_api_launcher.generic_stop_server', create=True)
def test_stop_atropos_api_proc_already_stopped(mock_generic_stop_stopped, mock_popen_fixture):
    '''Test stop_atropos_api if process is already stopped (poll() is not None).'''
    _ , mock_process_instance = mock_popen_fixture # We only need the instance here
    mock_process_instance.poll.return_value = 0 # Simulate process already stopped

    stop_atropos_api(mock_process_instance)
    mock_generic_stop_stopped.assert_not_called() # Should not call stop if already stopped
    # Can check logs too: "already stopped"

# To test the fallback in stop_atropos_api (if generic_stop_server import fails):
# This requires manipulating the import mechanism, which can be tricky.
# One way: patch 'verl.atropos_api_launcher.helpers_imported' to False.
@mock.patch('verl.atropos_api_launcher.helpers_imported', False)
def test_stop_atropos_api_fallback_logic(mock_popen_fixture):
    '''
    Test the fallback termination logic in stop_atropos_api when generic_stop_server is "not imported".
    '''
    _ , mock_process_instance = mock_popen_fixture

    # Simulate Popen methods for the fallback logic
    mock_process_instance.poll.side_effect = [None, 0] # Running, then stopped
    mock_process_instance.terminate.return_value = None
    mock_process_instance.kill.return_value = None
    mock_process_instance.wait.return_value = None # Simulate immediate stop after terminate

    stop_atropos_api(mock_process_instance)

    # Check that the Popen object's terminate method was called
    mock_process_instance.terminate.assert_called_once()
    mock_process_instance.kill.assert_not_called() # Should not be called if terminate succeeds quickly
    # Check that the log file stream (mock_process_instance.stdout) was closed
    assert mock_process_instance.stdout.closed, "Log file stream should be closed by fallback stop logic."

@mock.patch('verl.atropos_api_launcher.helpers_imported', False)
@mock.patch('time.sleep') # To speed up the test for timeout scenario
def test_stop_atropos_api_fallback_logic_timeout(mock_sleep, mock_popen_fixture):
    '''
    Test the fallback termination logic with SIGKILL after timeout.
    '''
    _ , mock_process_instance = mock_popen_fixture

    # Simulate Popen methods for the fallback logic with terminate timeout
    mock_process_instance.poll.side_effect = [None, None, 0] # Running, running (after term), then stopped (after kill)
    mock_process_instance.terminate.return_value = None
    mock_process_instance.kill.return_value = None
    mock_process_instance.wait.side_effect = [subprocess.TimeoutExpired(cmd="atropos", timeout=10), None] # Timeout on first wait, success on second

    stop_atropos_api(mock_process_instance)

    mock_process_instance.terminate.assert_called_once()
    mock_process_instance.kill.assert_called_once()
    assert mock_process_instance.stdout.closed, "Log file stream should be closed by fallback stop logic even on timeout."
