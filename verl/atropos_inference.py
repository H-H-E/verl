# verl/atropos_inference.py
import subprocess
import time
import os
import signal
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import requests
import threading # For running Uvicorn in a separate thread
import uvicorn # For serving FastAPI app
from fastapi import FastAPI, Request # Add FastAPI and Request
from pydantic import BaseModel # For request body validation (optional but good practice)
import torch # Assuming torch.device and torch.Tensor are used


# Configure logging
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True) # Ensure logs directory exists

# Basic logger for this module
logger = logging.getLogger(__name__)


def start_vllm_server(model_name: str, port: int, tensor_parallel: int, host: str = "localhost") -> subprocess.Popen:
    '''
    Launches a vLLM server as a subprocess.
    Manages logging of stdout and stderr to 'logs/vllm_<port>.log'.
    '''
    vllm_log_file = LOGS_DIR / f"vllm_server_{port}.log"
    command = [
        "vllm", "serve", model_name,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--host", host
    ]
    logger.info(f"Starting vLLM server for model '{model_name}' on port {port} with TP={tensor_parallel}.")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"vLLM server logs will be saved to: {vllm_log_file}")
    log_fp = None
    try:
        log_fp = open(vllm_log_file, 'w')
        proc = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
        logger.info(f"vLLM server process started with PID: {proc.pid}.")
    except FileNotFoundError:
        logger.error("vLLM command not found. Ensure vLLM is installed and in PATH.")
        if log_fp:
             log_fp.close()
        raise
    except Exception as e:
        logger.error(f"Failed to start vLLM server: {e}")
        if log_fp:
             log_fp.close()
        raise
    return proc


def stop_server(proc: subprocess.Popen, server_name: str = "Server") -> None:
    '''
    Stops a server process gracefully (SIGTERM) and then forcefully (SIGKILL) if needed.
    Also closes the log file associated with the process if it's available on the Popen object.
    '''
    if not proc or proc.poll() is not None:
        logger.info(f"{server_name} process (PID: {proc.pid if proc else 'N/A'}) already stopped or not started.")
        return
    logger.info(f"Stopping {server_name} process (PID: {proc.pid})...")
    if hasattr(proc, 'stdout') and proc.stdout and not proc.stdout.closed:
        try:
            proc.stdout.close()
            logger.info(f"Closed log file for {server_name} (PID: {proc.pid}).")
        except Exception as e:
            logger.error(f"Error closing log file for {server_name} (PID: {proc.pid}): {e}")
    try:
        proc.terminate()
        logger.info(f"Sent SIGTERM to {server_name} (PID: {proc.pid}). Waiting for termination...")
    except ProcessLookupError:
        logger.warning(f"{server_name} (PID: {proc.pid}) not found. Already terminated?")
        return
    except Exception as e:
        logger.error(f"Error sending SIGTERM to {server_name} (PID: {proc.pid}): {e}")
    try:
        proc.wait(timeout=5)
        logger.info(f"{server_name} (PID: {proc.pid}) terminated gracefully.")
    except subprocess.TimeoutExpired:
        logger.warning(f"{server_name} (PID: {proc.pid}) did not terminate after 5s. Sending SIGKILL...")
        try:
            proc.kill()
            proc.wait(timeout=5)
            logger.info(f"{server_name} (PID: {proc.pid}) killed.")
        except ProcessLookupError:
             logger.warning(f"{server_name} (PID: {proc.pid}) not found during SIGKILL. Already terminated?")
        except Exception as e_kill:
            logger.error(f"Error sending SIGKILL to {server_name} (PID: {proc.pid}): {e_kill}")
    except Exception as e_wait:
        logger.error(f"Error waiting for {server_name} (PID: {proc.pid}) termination: {e_wait}")


def start_sglang_server(
    model_name: str,
    port: int,
    host: str = "localhost",
    log_level: str = "info",
    tensor_parallel_size: int = 1,
    additional_args: Optional[Dict[str, Any]] = None
) -> subprocess.Popen:
    '''
    Launches an SGLang server as a subprocess.
    Manages logging of stdout and stderr to 'logs/sglang_server_<port>.log'.
    '''
    sglang_log_file = LOGS_DIR / f"sglang_server_{port}.log"
    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--port", str(port),
        "--host", host,
        "--log-level", log_level,
        "--tp-size", str(tensor_parallel_size)
    ]
    if additional_args:
        for key, value in additional_args.items():
            command_key = f"--{key.replace('_', '-')}"
            command.extend([command_key, str(value)])
    logger.info(f"Starting SGLang server for model '{model_name}' on port {port} with TP={tensor_parallel_size}.")
    logger.info(f"Command: {' '.join(command)}")
    logger.info(f"SGLang server logs will be saved to: {sglang_log_file}")
    log_fp = None
    try:
        log_fp = open(sglang_log_file, 'w')
        proc = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT)
        logger.info(f"SGLang server process started with PID: {proc.pid}.")
    except FileNotFoundError:
        logger.error("Failed to start SGLang server: 'python' command not found or sglang not installed correctly.")
        if log_fp:
             log_fp.close()
        raise
    except Exception as e:
        logger.error(f"Failed to start SGLang server: {e}")
        if log_fp:
             log_fp.close()
        raise
    return proc


def is_server_ready(url: str, timeout: float = 60.0, poll_interval: float = 2.0) -> bool:
    '''
    Polls a given URL until it returns a 200 OK status or a timeout is reached.
    '''
    start_time = time.monotonic()
    logger.info(f"Checking server readiness at {url} (timeout: {timeout}s)")
    while time.monotonic() - start_time < timeout:
        try:
            response = requests.get(url, timeout=poll_interval)
            if response.status_code == 200:
                logger.info(f"Server at {url} is ready (status 200 OK).")
                return True
            else:
                logger.debug(f"Server at {url} responded with status {response.status_code}. Retrying...")
        except requests.exceptions.ConnectionError:
            logger.debug(f"Server at {url} not yet responding (ConnectionError). Retrying...")
        except requests.exceptions.Timeout:
            logger.debug(f"Request to {url} timed out. Retrying...")
        except requests.exceptions.RequestException as e:
            logger.warning(f"An unexpected error occurred while polling {url}: {e}. Retrying...")
        time.sleep(poll_interval)
    logger.warning(f"Timeout: Server at {url} was not ready within {timeout} seconds.")
    return False


# Dummy/mock model and loader for now
class DummyModelForEmbeddedServer(torch.nn.Module):
    def __init__(self, model_name="dummy"):
        super().__init__()
        self.model_name = model_name
        self.dummy_param = torch.nn.Parameter(torch.randn(1))

    def generate(self, input_text: str, **kwargs):
        return f"Generated text for '{input_text}' by {self.model_name}"

    def to(self, device):
        self.device = device # Store device
        return super().to(device)

def load_pretrained_model_dummy(model_name: str, device: torch.device):
    logger.info(f"(Dummy) Loading pretrained model: {model_name} to device: {str(device)}")
    model = DummyModelForEmbeddedServer(model_name)
    return model.to(device)


# Pydantic models for OpenAI-like chat completion
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # temperature: Optional[float] = 0.7
    # max_tokens: Optional[int] = 256

class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-dummy"
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time())) # Use field for dynamic default
    model: str
    choices: List[ChatCompletionResponseChoice]


class EmbeddedInferenceServer:
    def __init__(self, model_name: str, port: int, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.port = port
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing EmbeddedInferenceServer for model '{self.model_name}' on port {self.port} using device '{self.device}'")
        self.model = load_pretrained_model_dummy(self.model_name, self.device)
        self.app = FastAPI()
        self._register_routes()
        self.server_thread: Optional[threading.Thread] = None
        self.uvicorn_server: Optional[uvicorn.Server] = None

    def _register_routes(self):
        @self.app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            logger.debug(f"Received chat completion request: {request.dict()}")
            prompt = "No user message found"
            for msg in reversed(request.messages):
                if msg.role == "user":
                    prompt = msg.content
                    break
            generated_text = self.model.generate(input_text=prompt)
            response_message = ChatMessage(role="assistant", content=generated_text)
            choice = ChatCompletionResponseChoice(message=response_message)
            # Make sure 'created' is set dynamically if not using default_factory in Pydantic model
            return ChatCompletionResponse(
                created=int(time.time()), # Explicitly set if not using default_factory
                model=self.model_name,
                choices=[choice]
            )

        @self.app.get("/v1/models")
        async def get_models():
            logger.debug("Received request for /v1/models")
            return {
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "organization-owner",
                    "permission": []
                }]
            }

        @self.app.get("/health")
        async def health_check():
            logger.debug("Received request for /health")
            return {"status": "healthy"}

    def start(self):
        if self.server_thread is not None and self.server_thread.is_alive():
            logger.warning("Server is already running.")
            return
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        self.uvicorn_server = uvicorn.Server(config)
        self.server_thread = threading.Thread(target=self.uvicorn_server.run, daemon=True)
        self.server_thread.start()
        logger.info(f"EmbeddedInferenceServer started on http://0.0.0.0:{self.port} in a background thread.")

    def stop(self):
        if self.uvicorn_server is not None:
            logger.info("Attempting to stop Uvicorn server...")
            self.uvicorn_server.should_exit = True
            if self.server_thread is not None and self.server_thread.is_alive():
                logger.info("Waiting for server thread to join...")
                self.server_thread.join(timeout=10)
                if self.server_thread.is_alive():
                    logger.warning("Server thread did not exit cleanly after 10s.")
                else:
                    logger.info("Server thread joined.")
            self.uvicorn_server = None
            self.server_thread = None
            logger.info("EmbeddedInferenceServer stopped.")
        else:
            logger.info("EmbeddedInferenceServer was not running or already stopped.")

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        logger.info(f"Loading new weights into embedded model '{self.model_name}'.")
        try:
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            logger.info("Successfully loaded new weights.")
        except Exception as e:
            logger.error(f"Error loading weights into embedded model: {e}")
            raise
# (Keep existing placeholders for other tasks if any)
