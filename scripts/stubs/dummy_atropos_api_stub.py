# scripts/stubs/dummy_atropos_api_stub.py
import uvicorn
from fastapi import FastAPI, Query
from typing import List, Dict, Any, Optional
import random
import argparse
import logging
import sys # For sys.stdout in basicConfig

# Configure basic logging for the stub
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout for visibility
)
logger = logging.getLogger("DummyAtroposAPIStub")

app = FastAPI()

# --- In-memory state for the stub ---
REGISTERED_ENVS_FOR_DEMO: Dict[str, Dict[str, str]] = {} # e.g. {"env_name": {"status": "registered"}}

# Counter for varying rollout data
fetch_counter = 0

def generate_dummy_rollout_group(env_name: str, iteration: int) -> Dict[str, Any]:
    """Generates a single rollout group with random reward."""
    # Make prompts and responses somewhat unique per call for variety
    prompt_text = f"DemoPrompt_for_{env_name}_id_{iteration}"
    response_text = f"GeneratedResponse_from_{env_name}_id_{iteration}"

    # Calculate dummy logprobs based on response length (char-level for simplicity)
    # AtroposDataset expects logprobs for response tokens.
    # DummyTokenizer is char-level. "GeneratedResponse..." + EOS token.
    # This needs to be consistent with how AtroposDataset's _to_tensors expects logprobs.
    # The _to_tensors logic assumes logprobs are for response tokens.
    num_response_tokens = len(response_text) + 1 # +1 for a potential EOS/separator by tokenizer

    return {
        "prompt": prompt_text,
        "response": response_text,
        "reward": round(random.uniform(0.1, 1.0), 2),
        "logprobs": [-round(random.uniform(0.5, 2.0), 2) for _ in range(num_response_tokens)],
        "token_advantages": None # Let dataset use scalar reward
    }

@app.on_event("startup")
async def startup_event():
    logger.info("Dummy Atropos API Stub: Startup sequence initiating...")
    # REGISTERED_ENVS_FOR_DEMO is now populated from CLI args in the main block
    # before uvicorn.run is called.
    # Log initial state if needed, but it's set by main.
    logger.info(f"Stub started. Expected environment from CLI: '{app.state.cli_args.expected_env_name if hasattr(app.state, 'cli_args') else 'N/A'}'")
    logger.info(f"Current registered environments viewable at /environments: {REGISTERED_ENVS_FOR_DEMO}")


@app.get("/health")
async def health_check():
    logger.debug("GET /health called")
    return {"status": "healthy_dummy_atropos_api"}

@app.get("/environments")
async def get_environments_endpoint(): # Renamed to avoid conflict with global
    logger.debug(f"GET /environments called, returning: {REGISTERED_ENVS_FOR_DEMO}")
    # Format for _is_env_registered: {"environments": {"env_name": {"status": "registered"}}}
    # or {"environments": [{"name": "env_name", "status": "registered"}]}
    # The stub sets REGISTERED_ENVS_FOR_DEMO to {"env_name": "registered"}
    # So, the response should be {"environments": {"env_name": "registered"}}
    # The _is_env_registered checks for envs.get(env_name, {}).get("status", "").lower() == "registered"
    # or envs.get(env_name, "").lower() == "registered"
    # So, format like REGISTERED_ENVS_FOR_DEMO = {"echo_demo_env": "registered"} is fine.
    return {"environments": REGISTERED_ENVS_FOR_DEMO}


@app.get("/batch")
async def get_batch_rollouts(size: int = Query(1, ge=1)):
    global fetch_counter
    logger.info(f"GET /batch called with size={size}. Current fetch_counter: {fetch_counter}")

    batch_to_serve = []
    env_name_for_rollout = "default_demo_env" # Fallback
    if hasattr(app.state, 'cli_args') and app.state.cli_args.expected_env_name:
         env_name_for_rollout = app.state.cli_args.expected_env_name
    else: # pragma: no cover (should be set by main)
        logger.warning("app.state.cli_args.expected_env_name not found, using default for rollouts.")


    for _ in range(size):
        batch_to_serve.append(generate_dummy_rollout_group(env_name_for_rollout, fetch_counter))
        fetch_counter += 1

    logger.info(f"Serving batch of {len(batch_to_serve)} new rollout groups for env '{env_name_for_rollout}'.")
    return batch_to_serve


if __name__ == "__main__": # pragma: no cover
    parser = argparse.ArgumentParser(description="Dummy Atropos API Stub for VeRL Demo")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the stub API on.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind the stub API to.")
    parser.add_argument(
        "--expected-env-name",
        type=str,
        default="echo_demo_env",
        help="The environment name that this stub should report as 'registered'."
    )
    cli_args = parser.parse_args()

    # Make args available to app state for endpoints to use
    app.state.cli_args = cli_args

    # Set the environment specified by CLI arg as "registered"
    # The _is_env_registered check looks for envs.get(env_name, {}).get("status", "").lower() == "registered"
    # OR envs.get(env_name, "").lower() == "registered"
    # So, using {"status": "registered"} as value is more robust.
    if cli_args.expected_env_name:
        REGISTERED_ENVS_FOR_DEMO[cli_args.expected_env_name] = {"status": "registered"}
        logger.info(f"Stub configured: '{cli_args.expected_env_name}' will be reported with status 'registered'.")
    else:
        logger.warning("No --expected-env-name provided. /environments might not satisfy demo script's env registration check.")

    logger.info(f"Starting Dummy Atropos API Stub on http://{cli_args.host}:{cli_args.port}")
    uvicorn.run(app, host=cli_args.host, port=cli_args.port, log_level="info")

```
