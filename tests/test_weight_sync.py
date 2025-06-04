# tests/test_weight_sync.py
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
import logging
from unittest import mock # For mocking torch.cuda.empty_cache

# Function and class to test (or simulate)
from verl.train.utils import weight_sync_manager

# --- Dummy Components for Testing ---

class SimpleTestModel(nn.Module):
    def __init__(self, in_features=5, out_features=5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Track device via parameter device, typical for nn.Module
        # self.device_tracker = torch.device("cpu") # Not needed explicitly like this

    def forward(self, x): # pragma: no cover
        return self.linear(x)

    # .to() method is inherited and handles device movement.
    # We can check self.linear.weight.device to know where the model is.

class MockEmbeddedInferenceServer:
    def __init__(self, model_name: str, port: int, device: torch.device):
        self.model_name = model_name
        self.port = port
        self.device = device
        self.model = SimpleTestModel().to(self.device)
        self.logger = logging.getLogger("MockEmbeddedInferenceServer")
        if not self.logger.hasHandlers(): # pragma: no cover
            logging.basicConfig(level=logging.DEBUG)
        self.offload_called = False
        self.weights_loaded_count = 0 # To check if load_weights was called

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.logger.info(f"MockServer loading weights. Incoming state_dict on device: {next(iter(state_dict.values())).device}")
        # Ensure state_dict values are on CPU as per weight_sync_manager's behavior
        for k, v in state_dict.items():
            assert v.device == torch.device("cpu"), "State_dict tensors should be on CPU when passed to load_weights."

        self.model.load_state_dict(state_dict)
        self.model.to(self.device) # Ensure model is on its designated device
        self.weights_loaded_count +=1
        self.logger.info(f"MockServer model device after load_weights: {next(self.model.parameters()).device}")

    def offload_model_to_cpu(self):
        self.logger.info("MockServer offload_model_to_cpu called.")
        self.model.to(torch.device("cpu"))
        self.offload_called = True

    def start(self): pass # pragma: no cover
    def stop(self): pass # pragma: no cover


# --- Test Cases ---

def test_state_dict_transfer_to_embedded_server():
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_device = torch.device("cpu") # Test with server on CPU

    training_model = SimpleTestModel().to(train_device)
    inference_server = MockEmbeddedInferenceServer("test-server", 9999, server_device)
    initial_server_weight = inference_server.model.linear.weight.data.clone()

    with torch.no_grad():
        training_model.linear.weight.data[0,0] += 1.0
    modified_training_weight_cpu = training_model.linear.weight.data.clone().cpu()

    assert inference_server.weights_loaded_count == 0
    with weight_sync_manager(training_model, inference_server):
        assert inference_server.weights_loaded_count == 1
        server_weight_inside_context = inference_server.model.linear.weight.data.clone()
        assert torch.equal(server_weight_inside_context.cpu(), modified_training_weight_cpu), \
            "Server model weights incorrect *inside* context manager."
        assert next(inference_server.model.parameters()).device == server_device, \
             f"Server model should be on its designated device ({server_device}) inside context."

    assert inference_server.weights_loaded_count == 1 # Should not change after exiting
    server_weight_after_context = inference_server.model.linear.weight.data.clone()
    assert torch.equal(server_weight_after_context.cpu(), modified_training_weight_cpu), \
        "Server model weights did not persist correctly *after* exiting context manager."
    assert not torch.equal(initial_server_weight.cpu()[0,0], modified_training_weight_cpu[0,0]), \
        "Test setup error: initial and modified weights were somehow the same."


def test_weight_sync_manager_no_load_weights_method(caplog):
    train_device = torch.device("cpu")
    training_model = SimpleTestModel().to(train_device)
    class ServerWithoutLoadWeights: pass
    inference_server_no_method = ServerWithoutLoadWeights()

    did_yield = False
    with caplog.at_level(logging.ERROR):
        with weight_sync_manager(training_model, inference_server_no_method):
            did_yield = True
    assert did_yield, "weight_sync_manager should still yield even if load_weights is missing."
    assert "Inference server does not have a callable load_weights method" in caplog.text


def test_weight_sync_manager_load_weights_exception(caplog):
    train_device = torch.device("cpu")
    training_model = SimpleTestModel().to(train_device)
    class ServerWithFailingLoadWeights:
        def load_weights(self, state_dict: Dict[str, torch.Tensor]):
            raise RuntimeError("Simulated load_weights failure")
    inference_server_failing = ServerWithFailingLoadWeights()

    did_yield = False
    with caplog.at_level(logging.ERROR):
        with weight_sync_manager(training_model, inference_server_failing):
            did_yield = True
    assert did_yield, "weight_sync_manager should yield even if load_weights fails."
    assert "Error during weight synchronization" in caplog.text
    assert "Simulated load_weights failure" in caplog.text


def test_memory_cleanup_custom_offload():
    train_device = torch.device("cpu")
    server_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_model = SimpleTestModel().to(train_device)
    inference_server = MockEmbeddedInferenceServer("test-offload", 9998, server_device)

    assert next(inference_server.model.parameters()).device == server_device
    assert not inference_server.offload_called

    with weight_sync_manager(training_model, inference_server):
        pass

    assert inference_server.offload_called, "offload_model_to_cpu should have been called."
    assert next(inference_server.model.parameters()).device == torch.device("cpu"), \
        "Model should be on CPU after custom offload."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for GPU memory test.")
@mock.patch('torch.cuda.empty_cache')
def test_memory_cleanup_cuda_empty_cache(mock_empty_cache):
    train_device = torch.device("cpu")
    server_device = torch.device("cuda")

    training_model = SimpleTestModel().to(train_device)

    # Define a class with the exact name "EmbeddedInferenceServer" for the heuristic check
    # And ensure it does NOT have the custom offload method.
    class EmbeddedInferenceServer(nn.Module): # Making it an nn.Module to easily host a model
        def __init__(self, model_name: str, port: int, device: torch.device):
            super().__init__() # Important for nn.Module
            self.model_name = model_name
            self.port = port
            self.device = device
            self.model = SimpleTestModel().to(self.device)
            self.logger = logging.getLogger("TestEmbeddedInferenceServerNoOffload")
            if not self.logger.hasHandlers(): logging.basicConfig(level=logging.DEBUG) # pragma: no cover

        def load_weights(self, state_dict: Dict[str, torch.Tensor]):
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

        # NO offload_model_to_cpu method here

    inference_server_cuda = EmbeddedInferenceServer("test-cuda-cleanup", 9997, server_device)

    assert next(inference_server_cuda.model.parameters()).device == server_device

    with weight_sync_manager(training_model, inference_server_cuda):
        pass

    mock_empty_cache.assert_called_once()

```
