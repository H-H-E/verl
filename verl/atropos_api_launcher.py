# verl/atropos_api_launcher.py
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional # For type hinting Popen object

# Attempt to import is_server_ready and stop_server from atropos_inference
# This assumes they are in the same package and accessible via relative import.
try:
    from .atropos_inference import is_server_ready, stop_server as generic_stop_server
    helpers_imported = True
except ImportError:
    helpers_imported = False
    # Log this issue. For the functions to work correctly, these helpers are needed.
    # A more robust solution might involve a shared utils module if this becomes a common pattern.
    # logging.getLogger(__name__).warning(
    #     "Could not import helpers from .atropos_inference. "
    #     "stop_atropos_api and parts of start_atropos_api may have reduced functionality."
    # )
    # For the purpose of this task, we will proceed as if the import is expected to work.
    # If it fails at runtime, the user will see the ImportError.
    # The code below will check for `helpers_imported` or function existence in globals().
    pass


# Configure logging
LOGS_DIR_LAUNCHER = Path("logs")
LOGS_DIR_LAUNCHER.mkdir(exist_ok=True)

logger_launcher = logging.getLogger(__name__)
# Ensure basicConfig is called if no other logging configuration is set up by the application
# This is a simple way to see logs if this module is used standalone or early in an app.
if not logger_launcher.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def start_atropos_api(port: int, host: str = "localhost", readiness_timeout: float = 60.0) -> Optional[subprocess.Popen]:
    '''
    Launches the Atropos API server (atropos run-api) as a subprocess.
    Redirects stdout/stderr to 'logs/atropos_api_<port>.log'.
    Polls the /health endpoint until the server is ready.
    Returns the Popen object for the server process, or None if startup fails.
    '''
    log_file = LOGS_DIR_LAUNCHER / f"atropos_api_{port}.log"

    command = [
        "atropos", "run-api",
        "--port", str(port),
        "--host", host
    ]

    logger_launcher.info(f"Starting Atropos API server on port {port}.")
    logger_launcher.info(f"Command: {' '.join(command)}")
    logger_launcher.info(f"Atropos API server logs will be saved to: {log_file}")

    process: Optional[subprocess.Popen] = None
    log_fp = None # Define log_fp outside try block for broader scope in finally/except
    try:
        log_fp = open(log_file, 'w')
        process = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
        logger_launcher.info(f"Atropos API server process started with PID: {process.pid}.")

        if not helpers_imported or 'is_server_ready' not in globals():
            logger_launcher.error("is_server_ready helper not available. Cannot check server readiness. Assuming success after start.")
            # Not returning process here as readiness is critical. Or, adjust behavior.
            # For now, let's assume readiness check is essential.
            raise RuntimeError("is_server_ready is not available due to import issues.")

        health_url = f"http://{host}:{port}/health"
        if is_server_ready(health_url, timeout=readiness_timeout):
            logger_launcher.info(f"Atropos API server is ready at {health_url}.")
            # Do not close log_fp here; Popen object handles the stream.
            # stdout/stderr of the process are writing to it.
            return process
        else:
            logger_launcher.error(f"Atropos API server failed to become ready within {readiness_timeout}s.")
            if process:
                if helpers_imported and 'generic_stop_server' in globals():
                    generic_stop_server(process, server_name=f"Atropos API (port {port})")
                else:
                    logger_launcher.warning("generic_stop_server not available for cleanup. Using basic kill.")
                    process.kill() # Basic kill if helper is missing
                    process.wait(timeout=5)
            # log_fp is associated with the Popen object's streams;
            # generic_stop_server (if used) or Popen's own cleanup should handle it.
            # However, if we are here, Popen object might still be around, let's ensure log is closed if we manually kill
            # This part is tricky: if generic_stop_server closes it, this would be an error.
            # For now, assuming generic_stop_server handles proc.stdout.close()
            return None

    except FileNotFoundError:
        logger_launcher.error("Atropos command 'atropos' not found. Ensure Atropos is installed and in PATH.")
        if process and process.poll() is None:
            process.kill()
        # log_fp needs to be closed here as Popen might not have taken full ownership or failed early
        if log_fp and not log_fp.closed:
            log_fp.close()
        return None
    except Exception as e:
        logger_launcher.error(f"Failed to start or check Atropos API server: {e}")
        if process and process.poll() is None:
            if helpers_imported and 'generic_stop_server' in globals():
                generic_stop_server(process, server_name=f"Atropos API (port {port})")
            else:
                logger_launcher.warning("generic_stop_server not available for cleanup. Using basic kill.")
                process.kill()
        if log_fp and not log_fp.closed: # Ensure log_fp is closed on other exceptions too
            log_fp.close()
        return None

def stop_atropos_api(proc: Optional[subprocess.Popen]) -> None:
    '''
    Stops the Atropos API server process cleanly.
    Uses the generic_stop_server from atropos_inference if available,
    otherwise implements basic termination.
    '''
    if not proc:
        logger_launcher.info("Atropos API process is None (not started or already handled).")
        return

    if proc.poll() is not None:
        logger_launcher.info(f"Atropos API process (PID: {proc.pid}) already stopped.")
        return

    logger_launcher.info(f"Stopping Atropos API server (PID: {proc.pid})...")

    if helpers_imported and 'generic_stop_server' in globals():
        generic_stop_server(proc, server_name="Atropos API Server")
    else:
        logger_launcher.warning("generic_stop_server not available. Using basic termination for Atropos API.")
        # Fallback basic stop logic
        # Closing proc.stdout (which is the log_fp) should be done first if not handled by Popen itself on termination
        if hasattr(proc, 'stdout') and proc.stdout and not proc.stdout.closed:
            try:
                proc.stdout.close()
            except Exception as e_close:
                logger_launcher.error(f"Error closing Atropos API log stream: {e_close}")
        try:
            proc.terminate()
            proc.wait(timeout=10)
            logger_launcher.info(f"Atropos API server (PID: {proc.pid}) terminated.")
        except subprocess.TimeoutExpired:
            logger_launcher.warning(f"Atropos API server (PID: {proc.pid}) did not terminate after 10s. Sending SIGKILL...")
            proc.kill()
            try:
                proc.wait(timeout=5)
                logger_launcher.info(f"Atropos API server (PID: {proc.pid}) killed.")
            except Exception as e_kill_wait:
                 logger_launcher.error(f"Error waiting for Atropos API SIGKILL: {e_kill_wait}")
        except Exception as e_term: # Catch other potential errors during termination
            logger_launcher.error(f"Error during Atropos API server termination: {e_term}")
