# verl/atropos_env_spawner.py
import subprocess
import time
import logging
import requests # For polling /environments endpoint
from pathlib import Path
from typing import Optional, Dict, Any # For type hinting

# Attempt to import generic_stop_server from atropos_inference
try:
    from .atropos_inference import stop_server as generic_stop_server
    helpers_imported_env = True
except ImportError:
    helpers_imported_env = False
    # logging.getLogger(__name__).warning("Could not import generic_stop_server. stop_env_server will use basic termination.")
    pass # Fallback implemented in stop_env_server

LOGS_DIR_ENV = Path("logs")
LOGS_DIR_ENV.mkdir(exist_ok=True)

logger_env = logging.getLogger(__name__)
# Basic logging config if no handlers are configured by the main application
if not logger_env.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _is_env_registered(atropos_api_url: str, env_name: str, timeout: float = 5.0) -> bool:
    '''
    Polls the Atropos API's /environments endpoint to check if a specific environment is registered.
    Atropos API URL should be like http://localhost:8000
    '''
    environments_url = f"{atropos_api_url.rstrip('/')}/environments"
    logger_env.debug(f"Polling {environments_url} for registration of environment '{env_name}'.")
    
    try:
        response = requests.get(environments_url, timeout=timeout)
        response.raise_for_status() 
        
        registered_envs_data = response.json()
        logger_env.debug(f"Received from /environments: {registered_envs_data}")
        
        # Flexible checking for env_name and its "registered" status
        if isinstance(registered_envs_data, dict) and "environments" in registered_envs_data:
            envs = registered_envs_data["environments"]
            if isinstance(envs, list): 
                for entry in envs:
                    if isinstance(entry, dict) and entry.get("name") == env_name and entry.get("status", "").lower() == "registered":
                        return True
                    if isinstance(entry, str) and entry == env_name: # If it's just a list of names, presence implies registered
                        return True
            elif isinstance(envs, dict): 
                if envs.get(env_name, {}).get("status", "").lower() == "registered": # e.g. {"gsm8k": {"status": "registered"}}
                    return True
                elif envs.get(env_name, "").lower() == "registered": # e.g. {"gsm8k": "registered"}
                    return True
        elif isinstance(registered_envs_data, list): # If root is a list
             for entry in registered_envs_data:
                 if isinstance(entry, dict) and entry.get("name") == env_name and entry.get("status", "").lower() == "registered":
                     return True
                 if isinstance(entry, str) and entry == env_name:
                     return True
        
        logger_env.debug(f"Environment '{env_name}' not found or not 'registered' in response: {registered_envs_data}")
        return False
        
    except requests.exceptions.RequestException as e:
        logger_env.error(f"Error polling {environments_url} for '{env_name}': {e}")
        return False


def start_env_server(
    env_name: str,
    rollout_server_url: str, 
    inference_url: str,      
    env_script_path: str = "environments", 
    readiness_timeout: float = 60.0,
    poll_interval: float = 5.0,
    additional_env_args: Optional[Dict[str, Any]] = None
) -> Optional[subprocess.Popen]:
    '''
    Launches an Atropos environment server script.
    Polls Atropos API's /environments for registration.
    '''
    log_file = LOGS_DIR_ENV / f"env_{env_name}.log"
    
    script_executable_primary = Path(env_script_path) / f"{env_name}_server.py"
    script_executable_fallback = Path(env_script_path) / f"{env_name}.py"

    if script_executable_primary.exists():
        script_to_run = script_executable_primary
    elif script_executable_fallback.exists():
        script_to_run = script_executable_fallback
    else:
        logger_env.error(f"Environment server script not found: tried {script_executable_primary} and {script_executable_fallback}")
        raise FileNotFoundError(f"Environment server script for '{env_name}' not found in '{env_script_path}'.")

    command = [
        "python", str(script_to_run), "serve",
        "--rollout_server_url", rollout_server_url,
        "--openai.base_url", inference_url 
    ]

    if additional_env_args:
        for key, value in additional_env_args.items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])

    logger_env.info(f"Starting environment server for '{env_name}'.")
    logger_env.info(f"Command: {' '.join(command)}")
    logger_env.info(f"Logs for '{env_name}' server: {log_file}")

    process: Optional[subprocess.Popen] = None
    log_fp = None
    try:
        log_fp = open(log_file, 'w')
        process = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
        logger_env.info(f"Env server process for '{env_name}' started (PID: {process.pid}). Polling for registration...")

        start_poll_time = time.monotonic()
        is_registered = False
        while time.monotonic() - start_poll_time < readiness_timeout:
            if process.poll() is not None: # Check if process died
                logger_env.error(f"Env server for '{env_name}' (PID: {process.pid}) terminated prematurely (code: {process.returncode}).")
                break 
            if _is_env_registered(rollout_server_url, env_name):
                is_registered = True
                break
            logger_env.info(f"Env '{env_name}' not registered. Retrying in {poll_interval}s...")
            time.sleep(poll_interval)

        if is_registered:
            logger_env.info(f"Env '{env_name}' registered with Atropos API at {rollout_server_url}.")
            return process
        else:
            logger_env.error(f"Env '{env_name}' failed to register within {readiness_timeout}s or process died.")
            if process and process.poll() is None: 
                if helpers_imported_env and 'generic_stop_server' in globals():
                    generic_stop_server(process, server_name=f"Env {env_name}")
                else:
                    logger_env.warning(f"generic_stop_server not available. Using basic kill for Env {env_name}.")
                    process.kill()
            # log_fp is managed by Popen, but if error before return, ensure it's handled if needed.
            # If generic_stop_server is used, it handles proc.stdout.close().
            # If we kill manually and didn't use generic_stop_server, we might need to close log_fp.
            # Popen's own cleanup usually handles its streams when the process ends or Popen object is GC'd.
            return None

    except FileNotFoundError as fnf_error: 
        logger_env.error(f"Failed to start env server for '{env_name}' (FileNotFound): {fnf_error}")
        if log_fp and not log_fp.closed: log_fp.close()
        if process and process.poll() is None: process.kill()
        if str(script_to_run) in str(fnf_error):
             raise FileNotFoundError(f"Env server script '{script_to_run}' not found.") from fnf_error
        raise 
    except Exception as e:
        logger_env.error(f"Unexpected error starting/registering env '{env_name}': {e}")
        if log_fp and not log_fp.closed: log_fp.close()
        if process and process.poll() is None:
            if helpers_imported_env and 'generic_stop_server' in globals():
                generic_stop_server(process, server_name=f"Env {env_name}")
            else:
                logger_env.warning(f"generic_stop_server not available. Using basic kill for Env {env_name}.")
                process.kill()
        return None


def stop_env_server(proc: Optional[subprocess.Popen], env_name: Optional[str] = None) -> None:
    '''
    Stops an environment server process cleanly.
    '''
    server_display_name = f"Environment Server '{env_name if env_name else proc.pid if proc else 'N/A'}'"
    if not proc:
        logger_env.info(f"{server_display_name} process is None.")
        return
    if proc.poll() is not None:
        logger_env.info(f"{server_display_name} (PID: {proc.pid}) already stopped.")
        return
        
    logger_env.info(f"Stopping {server_display_name} (PID: {proc.pid})...")

    if helpers_imported_env and 'generic_stop_server' in globals():
        generic_stop_server(proc, server_name=server_display_name)
    else:
        logger_env.warning(f"generic_stop_server not available. Using basic termination for {server_display_name}.")
        if hasattr(proc, 'stdout') and proc.stdout and not proc.stdout.closed:
            try: proc.stdout.close()
            except Exception as e: logger_env.error(f"Error closing log for {server_display_name}: {e}")
        try:
            proc.terminate()
            proc.wait(timeout=10) 
            logger_env.info(f"{server_display_name} (PID: {proc.pid}) terminated.")
        except subprocess.TimeoutExpired:
            logger_env.warning(f"{server_display_name} (PID: {proc.pid}) did not terminate. Sending SIGKILL...")
            proc.kill()
            try: proc.wait(timeout=5)
            except Exception as e_k: logger_env.error(f"Error waiting for {server_display_name} SIGKILL: {e_k}")
            logger_env.info(f"{server_display_name} (PID: {proc.pid}) killed.")
        except Exception as e_t: logger_env.error(f"Error during {server_display_name} termination: {e_t}")
