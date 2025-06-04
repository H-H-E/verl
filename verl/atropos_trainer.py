import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from verl.inference_manager import InferenceManager
from verl.datasets.atropos_dataset import AtroposDataset
from verl.grpo_trainer import run_grpo_step
from tests.test_grpo_trainer import TinyPolicy  # Use dummy model for now


class AtroposTrainer:
    """Main trainer class that orchestrates Atropos integration with VeRL.
    
    This class implements the full online RL training pipeline:
    1. Starts policy and reference inference servers
    2. Connects to Atropos API for rollout data
    3. Performs GRPO training steps
    4. Updates policy weights on inference servers
    5. Manages the complete training lifecycle
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize AtroposTrainer with configuration.
        
        Args:
            config: Training configuration dict containing:
                - model_name: HuggingFace model name or path
                - policy_port: Port for policy inference server
                - reference_port: Port for reference inference server  
                - atropos_api_url: URL of Atropos rollout API
                - batch_size: Training batch size
                - max_seq_len: Maximum sequence length
                - learning_rate: Learning rate for optimizer
                - kl_coeff: KL divergence coefficient
                - num_epochs: Number of training epochs (optional, default 1)
                - update_frequency: Weight update frequency (optional, default 1)
                - device: Device to use (optional, default auto-detect)
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.inference_manager = InferenceManager()
        self.policy_model: Optional[nn.Module] = None
        self.reference_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.dataset: Optional[AtroposDataset] = None
        
        # Training state
        self.step_count = 0
        
    def _validate_config(self) -> None:
        """Validate that all required config parameters are present."""
        required_keys = [
            "model_name", "policy_port", "reference_port", 
            "atropos_api_url", "batch_size", "max_seq_len",
            "learning_rate", "kl_coeff"
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Missing required config parameter: {key}")
                
    def _setup_models(self) -> None:
        """Initialize policy and reference models."""
        self.logger.info("Setting up models...")
        
        # For now, use TinyPolicy as placeholder - in real implementation
        # this would load the actual model specified in config
        vocab_size = 32000  # Typical vocab size, configurable
        hidden_size = 768   # Typical hidden size, configurable
        
        self.policy_model = TinyPolicy(vocab_size, hidden_size)
        self.reference_model = TinyPolicy(vocab_size, hidden_size)
        
        # Load same initial weights into reference model and freeze it
        self.reference_model.load_state_dict(self.policy_model.state_dict())
        for param in self.reference_model.parameters():
            param.requires_grad_(False)
            
        # Setup optimizer for policy model
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=self.config["learning_rate"]
        )
        
        self.logger.info("Models setup complete")
        
    def _setup_inference_servers(self) -> None:
        """Start policy and reference inference servers."""
        self.logger.info("Starting inference servers...")
        
        device = self.config.get("device", "cpu")
        
        # Start policy server
        policy_endpoint = self.inference_manager.start_policy_server(
            model_name=self.config["model_name"],
            port=self.config["policy_port"],
            device=device
        )
        self.logger.info(f"Policy server started at {policy_endpoint}")
        
        # Start reference server
        ref_endpoint = self.inference_manager.start_reference_server(
            model_name=self.config["model_name"],
            port=self.config["reference_port"],
            device=device
        )
        self.logger.info(f"Reference server started at {ref_endpoint}")
        
        # Wait for servers to be ready
        if not self.inference_manager.is_policy_server_ready():
            raise RuntimeError("Policy server failed to become ready")
            
    def _setup_dataset(self) -> None:
        """Initialize Atropos dataset for rollout data."""
        self.logger.info("Setting up Atropos dataset...")
        
        # For testing, use a simple dummy tokenizer
        # In real implementation, this would be the actual model tokenizer
        from verl.datasets.atropos_dataset import DummyTokenizer
        tokenizer = DummyTokenizer(
            pad_token_id=0, 
            max_length=self.config["max_seq_len"]
        )
        
        self.dataset = AtroposDataset(
            api_url=self.config["atropos_api_url"],
            batch_size=self.config["batch_size"],
            tokenizer=tokenizer,
            max_seq_len=self.config["max_seq_len"]
        )
        
        self.logger.info("Atropos dataset setup complete")
        
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one GRPO training step.
        
        Args:
            batch: Training batch from Atropos
            
        Returns:
            Loss value
        """
        loss = run_grpo_step(
            model=self.policy_model,
            optimizer=self.optimizer,
            batch=batch,
            ref_model=self.reference_model,
            kl_coeff=self.config["kl_coeff"]
        )
        
        self.step_count += 1
        
        # Update inference server weights periodically
        update_freq = self.config.get("update_frequency", 1)
        if self.step_count % update_freq == 0:
            self.logger.info(f"Updating policy weights (step {self.step_count})")
            self.inference_manager.update_policy_weights(
                self.policy_model.state_dict()
            )
            
        return loss
        
    def train(self) -> None:
        """Run the complete training pipeline.
        
        This implements the full online RL training loop:
        1. Validate configuration
        2. Setup models and servers
        3. Connect to Atropos for data
        4. Execute training steps with GRPO
        5. Update policy weights on inference servers
        """
        self.logger.info("Starting Atropos training pipeline...")
        
        try:
            # Setup phase
            self._validate_config()
            self._setup_models()
            self._setup_inference_servers()
            self._setup_dataset()
            
            # Training phase
            num_epochs = self.config.get("num_epochs", 1)
            
            for epoch in range(num_epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                epoch_losses = []
                
                for batch in self.dataset:
                    loss = self._training_step(batch)
                    epoch_losses.append(loss)
                    
                    self.logger.debug(f"Step {self.step_count}: loss = {loss:.4f}")
                    
                avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                self.logger.info(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.shutdown()
            
    def shutdown(self) -> None:
        """Clean up resources and stop inference servers."""
        self.logger.info("Shutting down Atropos trainer...")
        
        if hasattr(self, 'inference_manager') and self.inference_manager:
            self.inference_manager.shutdown_all()
            
        self.logger.info("Shutdown complete") 