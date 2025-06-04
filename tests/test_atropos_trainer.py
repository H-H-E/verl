import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from verl.atropos_trainer import AtroposTrainer
from tests.test_grpo_trainer import TinyPolicy


@pytest.fixture
def mock_atropos_dataset():
    """Mock AtroposDataset that returns dummy batches."""
    dataset = Mock()
    
    # Create a dummy batch generator 
    def batch_generator():
        for i in range(3):  # Yield 3 batches then stop
            yield {
                "input_ids": torch.randint(0, 10, (2, 8)),
                "attention_mask": torch.ones(2, 8),
                "advantages": torch.randn(2, 8),
                "old_logprobs": torch.randn(2, 8),
            }
    
    dataset.__iter__ = Mock(return_value=iter(batch_generator()))
    return dataset


@pytest.fixture 
def mock_inference_manager():
    """Mock InferenceManager for testing."""
    manager = Mock()
    manager.start_policy_server.return_value = "http://localhost:8080"
    manager.start_reference_server.return_value = "http://localhost:8081"
    manager.is_policy_server_ready.return_value = True
    manager.get_policy_endpoint.return_value = "http://localhost:8080"
    manager.get_reference_endpoint.return_value = "http://localhost:8081"
    return manager


def test_atropos_trainer_initialization():
    """Test AtroposTrainer can be initialized with required config."""
    config = {
        "model_name": "test-model",
        "policy_port": 8080,
        "reference_port": 8081,
        "atropos_api_url": "http://localhost:9000",
        "batch_size": 2,
        "max_seq_len": 512,
        "learning_rate": 1e-5,
        "kl_coeff": 0.1,
    }
    
    trainer = AtroposTrainer(config)
    
    assert trainer.config == config
    assert trainer.policy_model is None  # Not loaded until train() 
    assert trainer.reference_model is None


@patch('verl.atropos_trainer.InferenceManager')
@patch('verl.atropos_trainer.AtroposDataset')
def test_train_step_execution(mock_dataset_class, mock_manager_class, mock_atropos_dataset, mock_inference_manager):
    """Test that train() method executes training steps correctly."""
    
    # Setup mocks
    mock_dataset_class.return_value = mock_atropos_dataset
    mock_manager_class.return_value = mock_inference_manager
    
    config = {
        "model_name": "test-model",
        "policy_port": 8080,
        "reference_port": 8081,
        "atropos_api_url": "http://localhost:9000",
        "batch_size": 2,
        "max_seq_len": 8,
        "learning_rate": 1e-4,
        "kl_coeff": 0.1,
        "num_epochs": 1,
        "update_frequency": 2,  # Update weights every 2 steps
    }
    
    trainer = AtroposTrainer(config)
    
    with patch('verl.atropos_trainer.TinyPolicy') as mock_policy:
        # Mock the policy models
        policy_instance = TinyPolicy(vocab_size=10, hidden=8)
        ref_instance = TinyPolicy(vocab_size=10, hidden=8)
        mock_policy.side_effect = [policy_instance, ref_instance]
        
        # Mock optimizer
        with patch('torch.optim.AdamW') as mock_optimizer:
            optimizer_instance = Mock()
            mock_optimizer.return_value = optimizer_instance
            
            # Mock GRPO step
            with patch('verl.atropos_trainer.run_grpo_step') as mock_grpo:
                mock_grpo.return_value = 0.5  # Mock loss value
                
                # Run training
                trainer.train()
                
                # Verify servers were started
                mock_inference_manager.start_policy_server.assert_called_once_with(
                    model_name="test-model", port=8080, device="cpu"
                )
                mock_inference_manager.start_reference_server.assert_called_once_with(
                    model_name="test-model", port=8081, device="cpu"
                )
                
                # Verify GRPO step was called (3 batches from mock dataset)
                assert mock_grpo.call_count == 3
                
                # Verify weight updates happened (every 2 steps = once for 3 batches)
                assert mock_inference_manager.update_policy_weights.call_count == 1


def test_trainer_cleanup_on_shutdown():
    """Test that trainer properly cleans up resources."""
    config = {
        "model_name": "test-model",
        "policy_port": 8080,
        "reference_port": 8081,
        "atropos_api_url": "http://localhost:9000",
        "batch_size": 2,
        "max_seq_len": 512,
        "learning_rate": 1e-5,
        "kl_coeff": 0.1,
    }
    
    trainer = AtroposTrainer(config)
    
    # Mock the inference manager
    mock_manager = Mock()
    trainer.inference_manager = mock_manager
    
    trainer.shutdown()
    
    mock_manager.shutdown_all.assert_called_once()


def test_config_validation():
    """Test that AtroposTrainer validates required config parameters."""
    incomplete_config = {
        "model_name": "test-model",
        # Missing required fields
    }
    
    with pytest.raises(KeyError):
        trainer = AtroposTrainer(incomplete_config)
        trainer.train()  # Should fail during validation 