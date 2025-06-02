# recipe/atropos/launch_atropos_verl.py
import argparse
import logging
import sys # For sys.exit, sys.path
import signal # For signal handling
import time # For time.sleep in main loop (if any, not currently)
from pathlib import Path
from typing import List, Optional, Any
import subprocess # For Popen type hint

# Assuming verl package is in PYTHONPATH
# This try-except block attempts to make the script runnable directly if verl is a sibling directory.
try:
    from verl.atropos_config import AtroposConfig, load_atropos_config
    from verl.atropos_inference import (
        start_vllm_server, 
        start_sglang_server, 
        stop_server as stop_external_server, 
        is_server_ready,
        EmbeddedInferenceServer
    )
    from verl.atropos_api_launcher import start_atropos_api, stop_atropos_api
    from verl.atropos_env_spawner import start_env_server, stop_env_server
    from verl.train.grpo_trainer import AtroposGrpoTrainer
except ImportError as e: # pragma: no cover
    current_dir = Path(__file__).resolve().parent
    # Assuming this script is at recipe/atropos/launch_atropos_verl.py
    # Then verl_root would be current_dir.parent.parent
    verl_root = current_dir.parent.parent 
    if str(verl_root) not in sys.path:
        sys.path.insert(0, str(verl_root))
    
    # Retry imports
    from verl.atropos_config import AtroposConfig, load_atropos_config
    from verl.atropos_inference import (
        start_vllm_server, start_sglang_server, 
        stop_server as stop_external_server, 
        is_server_ready, EmbeddedInferenceServer
    )
    from verl.atropos_api_launcher import start_atropos_api, stop_atropos_api
    from verl.atropos_env_spawner import start_env_server, stop_env_server
    from verl.train.grpo_trainer import AtroposGrpoTrainer


# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AtroposVeRLLauncher")

# --- Global list to keep track of started processes/servers for cleanup ---
# Using specific variables for clearer management
external_inference_proc: Optional[subprocess.Popen] = None # If script manages external vLLM/SGLang
# embedded_inference_server_instance is managed by grpo_trainer_instance
atropos_api_proc: Optional[subprocess.Popen] = None
env_server_procs: List[subprocess.Popen] = []
grpo_trainer_instance: Optional[AtroposGrpoTrainer] = None


def cleanup_resources():
    logger.info("Initiating cleanup of all started resources...")
    
    # Trainer cleanup should be called first as it might manage its own inference server.
    global grpo_trainer_instance
    if grpo_trainer_instance and hasattr(grpo_trainer_instance, 'cleanup'):
        logger.info("Cleaning up GRPO trainer (which should handle its inference server)...")
        try:
            grpo_trainer_instance.cleanup()
        except Exception as e: # pragma: no cover
            logger.error(f"Error during GRPO trainer cleanup: {e}", exc_info=True)
    
    # If this script specifically started an external inference server (not via trainer), stop it.
    # Currently, trainer's __init__ handles inference server startup.
    # global external_inference_proc # This would be used if launch script started it directly
    # if external_inference_proc:
    #     logger.info("Stopping externally managed inference server...")
    #     try:
    #         stop_external_server(external_inference_proc, "External Inference Server")
    #     except Exception as e:
    #         logger.error(f"Error stopping external inference server: {e}", exc_info=True)
    #     external_inference_proc = None

    # Stop environment servers (reverse order of start)
    global env_server_procs
    if env_server_procs:
        logger.info(f"Stopping {len(env_server_procs)} environment server(s)...")
        for proc in reversed(env_server_procs):
            env_name_for_log = f"PID {proc.pid}" # Actual env_name not stored here, use PID
            logger.info(f"Stopping environment server {env_name_for_log}...")
            try:
                stop_env_server(proc, env_name=env_name_for_log) 
            except Exception as e: # pragma: no cover
                logger.error(f"Error stopping environment server {env_name_for_log}: {e}", exc_info=True)
        env_server_procs = []

    # Stop Atropos API server
    global atropos_api_proc
    if atropos_api_proc:
        logger.info("Stopping Atropos API server...")
        try:
            stop_atropos_api(atropos_api_proc)
        except Exception as e: # pragma: no cover
            logger.error(f"Error stopping Atropos API server: {e}", exc_info=True)
        atropos_api_proc = None
        
    logger.info("Cleanup finished.")

def signal_handler(sig, frame): # pragma: no cover
    logger.warning(f"Signal {signal.Signals(sig).name} received. Starting graceful shutdown...")
    cleanup_resources()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Launch script for Atropos-VeRL GRPO training.")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML configuration file for AtroposConfig."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, prints planned steps and exits without starting any processes."
    )
    parser.add_argument(
        "--env-script-path",
        type=str,
        default="environments", # Default relative path
        help="Path to the directory containing environment server scripts (e.g., gsm8k_server.py)."
    )

    args = parser.parse_args()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler) 

    # Make global variables accessible for assignment
    global atropos_api_proc, env_server_procs, grpo_trainer_instance
    # external_inference_proc is not directly managed by this script in current design

    # Outer try ensures cleanup_resources is always called
    try:
        # 1. Load Configuration (moved outside the inner try for KeyboardInterrupt)
        logger.info(f"Loading configuration from: {args.config}")
        try:
            config: AtroposConfig = load_atropos_config(args.config) # Type hint for clarity
            logger.info(f"Configuration loaded successfully.") # Brief log for normal operation
            logger.debug(f"Full config details: {config}") # More detail at debug level
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to load configuration from '{args.config}': {e}", exc_info=True)
            sys.exit(1) # Exit if config fails, no resources to clean yet.

        # -- DRY RUN Logic --
        if args.dry_run: # pragma: no cover
            logger.info("--- DRY RUN MODE ---")
            logger.info(f"Config loaded: model={config.model}, envs={config.environments}, num_iterations={config.num_iterations}")
            logger.info(f"Planned: Start Inference Server (type: {'External (SGLang)' if config.use_sglang else 'Embedded'}) on port {config.inference_api_port} for model {config.model} (managed by Trainer).")
            logger.info(f"Planned: Start Atropos API Server on port {config.rollout_server_port}.")
            for env_name in config.environments:
                # Construct full path for clarity in dry run log for env scripts
                env_script_example_path = Path(args.env_script_path) / f"{env_name}_server.py or {env_name}.py"
                logger.info(f"Planned: Start Environment Server for '{env_name}' from script path example '{env_script_example_path}'. Connects to Atropos API and Inference Server.")
            logger.info(f"Planned: Instantiate AtroposGrpoTrainer with loaded config.")
            logger.info(f"Planned: Call trainer.train(num_iterations={config.num_iterations}).")
            logger.info(f"Planned: Cleanup all resources on exit.")
            logger.info("--- END DRY RUN ---")
            sys.exit(0)

        # Inner try for the main orchestration, specifically catching KeyboardInterrupt
        try:
            # --- Main Orchestration ---
            # Note: Inference server (Embedded or dummy External) is started within AtroposGrpoTrainer's __init__.
            logger.info("Inference server startup is managed by AtroposGrpoTrainer.")

            logger.info(f"Starting Atropos API server on port {config.rollout_server_port}...")
            atropos_api_proc = start_atropos_api(port=config.rollout_server_port, host="localhost") # Assume localhost for now
            if atropos_api_proc is None:
                raise RuntimeError("Atropos API server failed to start.") 
            logger.info(f"Atropos API server started successfully (PID: {atropos_api_proc.pid}).")

            if not config.environments: # pragma: no cover
                logger.warning("No environments specified in config. Skipping environment server startup.")
            else:
                logger.info(f"Starting {len(config.environments)} environment server(s) from path '{args.env_script_path}'...")
                atropos_api_url = f"http://localhost:{config.rollout_server_port}" 
                inference_service_url = f"http://localhost:{config.inference_api_port}/v1"

                for env_name in config.environments:
                    logger.info(f"Starting environment server for '{env_name}'...")
                    env_proc = start_env_server(
                        env_name=env_name,
                        rollout_server_url=atropos_api_url,
                        inference_url=inference_service_url,
                        env_script_path=args.env_script_path 
                    )
                    if env_proc is None: # start_env_server handles registration polling and returns None on failure
                        raise RuntimeError(f"Environment server '{env_name}' failed to start or register.")
                    env_server_procs.append(env_proc)
                    # Log message from start_env_server already confirms registration or failure.
                    logger.info(f"Environment server '{env_name}' initiated (PID: {env_proc.pid}).") 
            
            if config.environments: logger.info("All specified environment servers initiated.")

            logger.info("Instantiating AtroposGrpoTrainer...")
            grpo_trainer_instance = AtroposGrpoTrainer(config)
            
            logger.info(f"Starting trainer.train(num_iterations={config.num_iterations})...")
            grpo_trainer_instance.train(num_iterations=config.num_iterations)
            
            logger.info("Training finished successfully.")

        except KeyboardInterrupt: # pragma: no cover
            logger.warning("KeyboardInterrupt received by main loop. Initiating graceful shutdown...")
            # Cleanup is handled by the outer finally block.
            # The signal_handler also calls cleanup_resources. This explicit catch provides
            # a clear log message if KeyboardInterrupt is caught here.

        except Exception as e: # pragma: no cover
            logger.error(f"An error occurred during main orchestration: {e}", exc_info=True)
            # Fallthrough to finally for cleanup

    finally:
        cleanup_resources() # Ensure cleanup happens on normal exit, Ctrl+C, or other exceptions.
        logger.info("Launch script finished execution.")

if __name__ == "__main__": # pragma: no cover
    main()
```

**Self-correction during thought process:**
-   Added `--env-script-path` CLI argument to make the location of environment server scripts configurable, as hardcoding `"environments"` is brittle. Defaulted it to `"environments"`.
-   Clarified in `cleanup_resources` that the GRPO trainer is expected to clean up its own inference server. If an external inference server were managed directly by this script (which it isn't in the current design where trainer handles it), its stop logic would be here.
-   Ensured `AtroposConfig` type hint for `config` variable.
-   Made sure logger name for this script is distinct (e.g., "AtroposVeRLLauncher").
-   In `cleanup_resources`, when stopping env servers, used PID for logging if `env_name` isn't readily available with the `proc` object.
-   Added `pragma: no cover` to branches that are hard to reliably trigger in CI unit tests (e.g., specific import errors, some exception paths, signal handling).
-   The `AtroposGrpoTrainer`'s `__init__` method already attempts to start its inference server. The main `train` loop in the trainer also has a check. This launch script doesn't need to manage that directly unless the design changes for external servers started *before* the trainer.
-   The `atropos_api_url` and `inference_service_url` for environment servers assume `localhost`. This might need to be more configurable in a distributed setup, but is fine for a local launch script.
-   The `cleanup_resources` function now stops environment servers in reversed order of their presumed start, which is generally good practice.

This initial structure provides a command-line interface, configuration loading, process management for the Atropos API and environment servers, trainer instantiation, and robust cleanup. The actual success of starting environment servers will depend on the `env_script_path` and the content of those scripts.
