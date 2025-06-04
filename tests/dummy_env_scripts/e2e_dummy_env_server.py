# tests/dummy_env_scripts/e2e_dummy_env_server.py
import argparse
import time
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="E2E Dummy Environment Server")
    parser.add_argument("action", choices=["serve"], help="Server action")
    parser.add_argument("--rollout_server_url", type=str, help="URL of the Atropos API server")
    # Use a more robust way to handle arguments with dots or that might be optional
    parser.add_argument("--openai.base_url", type=str, dest="openai_base_url", help="URL of the OpenAI-compatible inference API")
    parser.add_argument("--some_other_arg", type=str, default="default_val", help="Another dummy arg")

    args = parser.parse_args()

    # Use a unique identifier for this server instance for logging if multiple are run
    # pid = os.getpid() # Not strictly needed for this dummy version

    print(f"E2E Dummy Environment Server ({sys.argv[0]}) started. Action: {args.action}", flush=True)
    print(f"Received --rollout_server_url: {args.rollout_server_url}", flush=True)
    print(f"Received --openai.base_url: {args.openai_base_url}", flush=True)
    print(f"Received --some_other_arg: {args.some_other_arg}", flush=True)

    print("E2E Dummy Environment Server running and ready to simulate work...", flush=True)

    # The launch script will manage termination via SIGTERM/SIGINT.
    # This loop simulates a server staying alive.
    try:
        while True:
            time.sleep(1) # Keep alive, check for interruptions infrequently
    except KeyboardInterrupt: # pragma: no cover
        print("E2E Dummy Environment Server received KeyboardInterrupt, shutting down.", flush=True)
    except SystemExit: # pragma: no cover
        print("E2E Dummy Environment Server received SystemExit, shutting down.", flush=True)
    finally:
        print("E2E Dummy Environment Server finished.", flush=True)

if __name__ == "__main__":
    main()
