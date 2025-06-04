# scripts/stubs/dummy_echo_env_server.py
import argparse
import time
import sys
import logging
from pathlib import Path # For Path(__file__).name

# Configure basic logging for the dummy environment server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to stdout
)
# Create a logger specific to this server instance if needed, or use root.
# Using a named logger is good practice.
logger = logging.getLogger(f"DummyEchoEnvServer[{Path(__file__).name}]")

def main():
    parser = argparse.ArgumentParser(description="Dummy Echo Environment Server for VeRL Demo")

    parser.add_argument(
        "action",
        choices=["serve"],
        help="The action to perform (only 'serve' is relevant for Atropos envs)."
    )
    parser.add_argument(
        "--rollout_server_url",
        type=str,
        required=True,
        help="URL of the Atropos API server (rollout collector)."
    )
    parser.add_argument(
        "--openai.base_url",
        dest='openai_base_url',
        type=str,
        required=True,
        help="Base URL of the OpenAI-compatible inference API."
    )
    # Example of another potential argument that might be passed by start_env_server's additional_env_args
    parser.add_argument(
        "--custom_env_param",
        type=str,
        default="not_set",
        help="A custom parameter for this dummy environment."
    )

    try:
        args = parser.parse_args()
    except SystemExit as e: # pragma: no cover (argparse exits on error)
        logger.error(f"Argument parsing failed with exit code {e.code}. Arguments: {sys.argv}")
        sys.exit(e.code)


    if args.action == "serve":
        logger.info("Starting up...")
        logger.info(f"  Action: {args.action}")
        logger.info(f"  Rollout Server URL (Atropos API): {args.rollout_server_url}")
        logger.info(f"  Inference API URL (Model): {args.openai_base_url}")
        logger.info(f"  Custom Env Param: {args.custom_env_param}")

        logger.info("Dummy Echo Environment Server is now 'running' (simulated).")
        logger.info("This server will keep running and printing a heartbeat until terminated by the main launch script.")

        heartbeat_count = 0
        try:
            while True:
                # In a real environment, this loop would interact with Atropos and Inference APIs.
                time.sleep(30) # Keep the process alive, sleep.
                heartbeat_count += 1
                logger.info(f"Heartbeat {heartbeat_count}. Still alive and simulating work.")
        except KeyboardInterrupt: # pragma: no cover
            logger.info("Received KeyboardInterrupt. Shutting down gracefully.")
        except SystemExit: # pragma: no cover
            logger.info("Received SystemExit. Shutting down.")
        except Exception as e: # pragma: no cover
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        finally:
            logger.info("Exiting.")
    else: # pragma: no cover (should be caught by argparse choices)
        logger.error(f"Unknown action: {args.action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
