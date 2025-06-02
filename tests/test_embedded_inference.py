# tests/test_embedded_inference.py
import pytest
import requests # For making HTTP requests to the embedded server
import time
import socket # For checking port status
from pathlib import Path
import torch # For device and state_dict

# Assuming verl.atropos_inference and EmbeddedInferenceServer are accessible
# Adjust path if necessary based on project structure for imports
from verl.atropos_inference import EmbeddedInferenceServer, is_server_ready

# Helper to check if a port is open (can be reused or defined in a common test utils)
def is_port_in_use(port: int, host: str = "localhost") -> bool:
    # Set a timeout to prevent indefinite blocking.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.1) # Short timeout for the check
    try:
        # Try to bind to the port. If it succeeds, the port is not in use.
        s.bind((host, port))
        return False
    except socket.error:
        # If bind fails, the port is likely in use.
        return True
    finally:
        s.close()


@pytest.fixture(scope="function")
def test_port_embedded():
    # Using a fixed port for simplicity in this example.
    # For parallel tests, dynamic port allocation would be better.
    return 8070 

@pytest.fixture(scope="function")
def embedded_server(test_port_embedded: int):
    '''Fixture to create, start, and stop an EmbeddedInferenceServer instance.'''
    
    model_name = "dummy-embedded-model" 
    
    if is_port_in_use(test_port_embedded):
        # If port is in use, attempt to manually connect to see if it's a leftover server.
        # This is just for debugging, pytest.skip is the main action.
        try:
            requests.get(f"http://localhost:{test_port_embedded}/health", timeout=0.5)
            print(f"Warning: Port {test_port_embedded} seems to be in use by a responsive server.")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Port {test_port_embedded} is in use but server not responsive to /health.")
        pytest.skip(f"Port {test_port_embedded} is already in use for embedded server test. Skipping.")

    device = torch.device("cpu") # Use CPU for these tests

    server = EmbeddedInferenceServer(model_name=model_name, port=test_port_embedded, device=device)
    
    server_ready = False
    try:
        server.start()
        # Check /health first as it's simpler
        if is_server_ready(f"http://localhost:{test_port_embedded}/health", timeout=15.0, poll_interval=0.5):
            server_ready = True
        # Fallback to /v1/models if /health check wasn't enough or to be sure
        if not server_ready and is_server_ready(f"http://localhost:{test_port_embedded}/v1/models", timeout=15.0, poll_interval=0.5):
            server_ready = True
        
        assert server_ready, f"Embedded server did not become ready on port {test_port_embedded}."
        yield server 
    finally:
        server.stop()
        # Wait a moment for the OS to release the port
        released = False
        for _ in range(50): # Try for up to 5 seconds
            if not is_port_in_use(test_port_embedded):
                released = True
                break
            time.sleep(0.1)
        assert released, f"Port {test_port_embedded} should be free after stopping embedded server."


def test_start_and_respond(embedded_server: EmbeddedInferenceServer, test_port_embedded: int):
    '''
    Tests starting the EmbeddedInferenceServer, polling /v1/models, 
    sending a request to /v1/chat/completions, and checking the response.
    '''
    # Server is started and readiness checked by the fixture.
    
    # 1. Check /v1/models endpoint
    models_url = f"http://localhost:{test_port_embedded}/v1/models"
    try:
        response = requests.get(models_url, timeout=5)
        response.raise_for_status() 
        models_data = response.json()
        assert "data" in models_data
        assert len(models_data["data"]) > 0
        assert models_data["data"][0]["id"] == embedded_server.model_name
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to get /v1/models: {e}")

    # 2. Send a POST request to /v1/chat/completions
    chat_url = f"http://localhost:{test_port_embedded}/v1/chat/completions"
    chat_payload = {
        "model": embedded_server.model_name,
        "messages": [
            {"role": "user", "content": "Hello, world!"}
        ]
    }
    
    try:
        response = requests.post(chat_url, json=chat_payload, timeout=5)
        response.raise_for_status()
        chat_response_data = response.json()

        assert "choices" in chat_response_data
        assert len(chat_response_data["choices"]) > 0
        choice = chat_response_data["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert len(choice["message"]["content"]) > 0 
        assert f"Generated text for 'Hello, world!' by {embedded_server.model_name}" in choice["message"]["content"]
        assert "created" in chat_response_data # Check for timestamp
        assert isinstance(chat_response_data["created"], int)

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Failed to post to /v1/chat/completions: {e}")

def test_load_weights_on_embedded_server(embedded_server: EmbeddedInferenceServer):
    '''Tests the load_weights functionality of the EmbeddedInferenceServer.'''
    # Get the current state_dict (or a part of it)
    original_param_value = embedded_server.model.dummy_param.clone()

    # Create a new state_dict with a modified value for the dummy parameter
    new_state_dict = embedded_server.model.state_dict()
    # Modify a parameter
    with torch.no_grad(): # Ensure no gradient tracking during modification
        new_state_dict['dummy_param'] = torch.randn_like(original_param_value) 
        # Ensure it's different
        while torch.equal(new_state_dict['dummy_param'], original_param_value):
            new_state_dict['dummy_param'] = torch.randn_like(original_param_value)
    
    modified_param_value = new_state_dict['dummy_param'].clone()

    # Load the new state_dict
    try:
        embedded_server.load_weights(new_state_dict)
    except Exception as e:
        pytest.fail(f"load_weights raised an exception: {e}")

    # Verify that the model's parameter has changed
    assert not torch.equal(embedded_server.model.dummy_param, original_param_value), \
        "Model parameter should have changed after loading new weights."
    assert torch.equal(embedded_server.model.dummy_param, modified_param_value), \
        "Model parameter should match the new value loaded from state_dict."

    # Ensure the model is still on the correct device
    assert str(embedded_server.model.dummy_param.device) == str(embedded_server.device), \
        f"Model parameter is on device {embedded_server.model.dummy_param.device} but should be on {embedded_server.device}."
