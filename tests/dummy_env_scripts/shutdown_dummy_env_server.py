# tests/dummy_env_scripts/shutdown_dummy_env_server.py
import argparse
import time
import sys
import os # Not strictly needed here, but often useful in scripts

def main():
    parser = argparse.ArgumentParser(description="Shutdown Test Dummy Environment Server")
    parser.add_argument("action", choices=["serve"], help="Server action (must be 'serve')")
    parser.add_argument("--rollout_server_url", type=str, help="URL of the Atropos API server")
    # Use dest to handle argument names with dots if they come from Atropos
    parser.add_argument("--openai.base_url", type=str, dest="openai_base_url", help="URL of the OpenAI-compatible inference API")

    args = parser.parse_args()

    print(f"Shutdown Dummy Environment Server ({Path(__file__).name}) started. PID: {os.getpid()}", flush=True)
    print(f"Action: {args.action}", flush=True)
    print(f"Rollout Server URL: {args.rollout_server_url}", flush=True)
    print(f"Inference API URL: {args.openai_base_url}", flush=True)

    print("Shutdown Dummy Environment Server running and simulating work...", flush=True)

    try:
        # Loop indefinitely, simulating a server process that needs to be killed.
        # The launch script is responsible for terminating this process.
        while True:
            time.sleep(5) # Keep alive, print a heartbeat or status periodically if verbose
            # print(f"Shutdown Dummy Env Server (PID: {os.getpid()}) still alive...", flush=True)
    except KeyboardInterrupt: # pragma: no cover
        print(f"Shutdown Dummy Env Server (PID: {os.getpid()}) received KeyboardInterrupt, exiting.", flush=True)
    except SystemExit: # pragma: no cover
        print(f"Shutdown Dummy Env Server (PID: {os.getpid()}) received SystemExit, exiting.", flush=True)
    finally:
        print(f"Shutdown Dummy Env Server (PID: {os.getpid()}) finished.", flush=True)

if __name__ == "__main__":
    # Add pathlib import for Path(__file__).name
    from pathlib import Path
    main()
