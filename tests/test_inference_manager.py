import pytest
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from verl.inference_manager import InferenceManager
from tests.test_grpo_trainer import TinyPolicy


@pytest.fixture
def mock_server():
    """Mock server that behaves like EmbeddedInferenceServer."""
    server = Mock()
    server.start = Mock()
    server.stop = Mock() 
    server.load_weights = Mock()
    server.port = 8080
    server.model_name = "test-model"
    return server


@pytest.fixture 
def inference_manager(mock_server):
    """InferenceManager with mocked server."""
    manager = InferenceManager()
    manager._servers = {"policy": mock_server}
    return manager


def test_start_policy_server():
    """Test that start_policy_server creates and starts a server."""
    manager = InferenceManager()
    
    with patch('verl.inference_manager.EmbeddedInferenceServer') as MockServer:
        mock_instance = Mock()
        MockServer.return_value = mock_instance
        
        endpoint = manager.start_policy_server(
            model_name="test-model",
            port=8080,
            device="cpu"
        )
        
        # Verify server was created and started
        MockServer.assert_called_once_with("test-model", 8080, torch.device("cpu"))
        mock_instance.start.assert_called_once()
        
        # Verify endpoint returned
        assert endpoint == "http://localhost:8080"
        assert manager._servers["policy"] == mock_instance


def test_stop_policy_server(inference_manager, mock_server):
    """Test that stop_policy_server stops and removes server."""
    inference_manager.stop_policy_server()
    
    mock_server.stop.assert_called_once()
    assert "policy" not in inference_manager._servers


def test_update_policy_weights(inference_manager, mock_server):
    """Test that update_policy_weights calls load_weights on server."""
    # Create dummy state dict
    model = TinyPolicy(vocab_size=10, hidden=8)
    state_dict = model.state_dict()
    
    inference_manager.update_policy_weights(state_dict)
    
    mock_server.load_weights.assert_called_once_with(state_dict)


def test_is_policy_server_ready(inference_manager, mock_server):
    """Test policy server readiness check."""
    with patch('verl.inference_manager.requests.get') as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        is_ready = inference_manager.is_policy_server_ready()
        
        assert is_ready is True
        mock_get.assert_called_once_with("http://localhost:8080/health", timeout=5)


def test_is_policy_server_ready_no_server():
    """Test readiness check when no server is running."""
    manager = InferenceManager()
    assert manager.is_policy_server_ready() is False


def test_get_policy_endpoint(inference_manager, mock_server):
    """Test getting policy server endpoint."""
    endpoint = inference_manager.get_policy_endpoint()
    assert endpoint == "http://localhost:8080"


def test_get_policy_endpoint_no_server():
    """Test getting endpoint when no server exists."""
    manager = InferenceManager()
    assert manager.get_policy_endpoint() is None 