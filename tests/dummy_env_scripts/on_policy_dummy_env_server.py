# tests/dummy_env_scripts/on_policy_dummy_env_server.py
import argparse
import time
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="On-Policy Test Dummy Environment Server")
    parser.add_argument("action", choices=["serve"], help="Server action (must be 'serve')")
    parser.add_argument("--rollout_server_url", type=str, help="URL of the Atropos API server")
    parser.add_argument("--openai.base_url", type=str, dest="openai_base_url", help="URL of the OpenAI-compatible inference API")

    args = parser.parse_args()

    script_name = Path(__file__).name
    pid = os.getpid()

    print(f"{script_name} (PID: {pid}) started. Action: {args.action}", flush=True)
    print(f"{script_name} (PID: {pid}) Rollout Server URL: {args.rollout_server_url}", flush=True)
    print(f"{script_name} (PID: {pid}) Inference API URL: {args.openai_base_url}", flush=True)

    print(f"{script_name} (PID: {pid}) running and simulating work...", flush=True)

    try:
        while True:
            time.sleep(5)
            # print(f"{script_name} (PID: {pid}) still alive...", flush=True) # Optional heartbeat
    except KeyboardInterrupt: # pragma: no cover
        print(f"{script_name} (PID: {pid}) received KeyboardInterrupt, exiting.", flush=True)
    except SystemExit: # pragma: no cover
        print(f"{script_name} (PID: {pid}) received SystemExit, exiting.", flush=True)
    finally:
        print(f"{script_name} (PID: {pid}) finished.", flush=True)

if __name__ == "__main__":
    main()
