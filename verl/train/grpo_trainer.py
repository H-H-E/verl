# verl/train/grpo_trainer.py
import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Iterator, List # Added List for type hints
import subprocess # For ExternalInferenceWrapper placeholder's Popen attribute
import logging 

# Configuration and previously defined components
from verl.atropos_config import AtroposConfig
from verl.datasets.atropos_dataset import AtroposDataset, DummyTokenizer # Use DummyTokenizer for now
from verl.atropos_inference import EmbeddedInferenceServer 
# ExternalInferenceWrapper is not yet defined, so create a placeholder for it.
# Helper model/tokenizer loaders (placeholders for now)

# Add new imports for loss functions and mask builder
from verl.train.utils import build_loss_mask
from verl.train.losses import advantage_weighted_loss, kl_penalty_loss
# Ensure torch is imported if not already at the top for torch.Tensor hints
# import torch # Already imported at the top
# from typing import Dict, Any, Optional, Iterator, List # Ensure all are present (already at top)

# --- Placeholders / Dummies for external dependencies ---
class VeRLTrainer: # Dummy base class
    def __init__(self, common_args: Optional[Dict[str, Any]] = None):
        self.common_args = common_args if common_args is not None else {}
        self.device = self.common_args.get(
            "device", 
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.logger = logging.getLogger(self.__class__.__name__) # Basic logger
        # Ensure logger has a handler if used in a context where root logger isn't configured
        if not self.logger.hasHandlers(): # pragma: no cover
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.info(f"VeRLTrainer (dummy) initialized on device: {self.device}")

class DummyTrainingModel(torch.nn.Module): # Placeholder for actual model
    def __init__(self, model_name="dummy_training_model"):
        super().__init__()
        self.model_name = model_name
        self.linear = torch.nn.Linear(10, 10) 
        self.config_data = {'vocab_size': 50304} # Example vocab size for GPT2-like models
        # Mock config attribute often found in HF models
        class MockConfig:
            def __init__(self, data):
                self.__dict__.update(data)
        self.config = MockConfig(self.config_data)


    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        # Return random logits that require grad if input_ids requires grad (which it won't by default)
        # To ensure loss.backward() works, the model output needs to require grad if params require grad.
        # This dummy model's output will require grad if self.linear.weight requires grad (which it does by default).
        dummy_hidden_state = torch.randn(batch_size, seq_len, 10, device=input_ids.device)
        return self.linear(dummy_hidden_state) # This should produce grad-requiring output if linear has grad-req params

    def to(self, device):
        # self.device = device # Keep track of device if model needs it
        return super().to(device)

    def generate(self, input_text: str, **kwargs): # Dummy generate for EmbeddedInferenceServer's model
         return f"Generated: {input_text} by {self.model_name}"

def load_tokenizer_dummy(model_name_or_path: str):
    logger_lt = logging.getLogger("load_tokenizer_dummy")
    if not logger_lt.hasHandlers(): logging.basicConfig(level=logging.INFO) # pragma: no cover
    logger_lt.info(f"(Dummy) Loading tokenizer for: {model_name_or_path}")
    return DummyTokenizer() # Using the one from atropos_dataset

def load_model_dummy(model_name_or_path: str, device: torch.device):
    logger_lm = logging.getLogger("load_model_dummy")
    if not logger_lm.hasHandlers(): logging.basicConfig(level=logging.INFO) # pragma: no cover
    logger_lm.info(f"(Dummy) Loading model: {model_name_or_path} to device: {device}")
    model = DummyTrainingModel(model_name_or_path)
    return model.to(device)

class ExternalInferenceWrapper: 
    def __init__(self, model_name: str, port: int, tensor_parallel: int, config: AtroposConfig):
        self.model_name = model_name
        self.port = port
        self.tensor_parallel = tensor_parallel 
        self.config_ref = config 
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers(): logging.basicConfig(level=logging.INFO) # pragma: no cover
        self.logger.info(f"ExternalInferenceWrapper (dummy) for {model_name} on port {port} initialized.")
        self.server_process: Optional[subprocess.Popen] = None 

    def load_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.logger.info("(Dummy) ExternalInferenceWrapper: load_weights called.")

    def start(self): 
        self.logger.info("(Dummy) ExternalInferenceWrapper: start called. (Would start vLLM/SGLang here).")
        # In a real scenario, this would use start_vllm_server or start_sglang_server
        # from verl.atropos_inference and store self.server_process.
    
    def stop(self): 
        self.logger.info("(Dummy) ExternalInferenceWrapper: stop called.")
        if self.server_process and self.server_process.poll() is None: # pragma: no cover
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired: 
                self.server_process.kill()


# --- AtroposGrpoTrainer ---
class AtroposGrpoTrainer(VeRLTrainer):
    def __init__(self, config: AtroposConfig):
        super().__init__() 

        self.config = config
        self.logger.info(f"Initializing AtroposGrpoTrainer with config: {self.config}")
        # self.device is inherited from VeRLTrainer and set in its __init__

        self.tokenizer = load_tokenizer_dummy(config.model)
        self.training_model = load_model_dummy(config.model, self.device)
        # Ensure model is in training mode initially
        self.training_model.train()


        self.ref_model: Optional[torch.nn.Module] = None
        if config.reference_model:
            self.ref_model = load_model_dummy(config.reference_model, self.device)
            if self.ref_model: self.ref_model.eval() # Set reference model to eval mode

        if config.use_sglang: 
            self.inference_server = ExternalInferenceWrapper(
                model_name=config.model, 
                port=config.inference_api_port,
                tensor_parallel=config.tensor_parallel,
                config=config 
            )
            # self.inference_server.start() # Actual start might be deferred or managed by launch script
        else:
            # For EmbeddedInferenceServer, the model it uses is derived from self.training_model's weights.
            # So, it needs a way to access them or be the same instance if not careful.
            # The current EmbeddedInferenceServer loads its own dummy model.
            # This needs to be reconciled: does Embedded server get a *copy* or share weights?
            # For now, it loads its own based on model_name. Weight sync is key.
            self.inference_server = EmbeddedInferenceServer(
                model_name=config.model, # This will load another DummyTrainingModel instance
                port=config.inference_api_port,
                device=self.device 
            )
            # self.inference_server.start() # Actual start might be deferred or managed by launch script
        
        # Start server only when trainer.train() or a specific setup method is called?
        # For now, let's assume __init__ doesn't auto-start it to give more control.
        # The prompt implies it should start: self.inference_server.start() was there.
        # Let's keep it but note that practical use might require more control.
        # self.inference_server.start() # Deferred to a 'setup' or 'train' call typically.
                                     # For this task, the prompt had it, so I'll keep it commented as a note.
                                     # The original prompt's ExternalInferenceWrapper also calls start().
                                     # The prompt for AtroposGrpoTrainer.__init__ has self.inference_server.start()
                                     # so I will add it back based on that specific instruction.
        if hasattr(self.inference_server, 'start'): # Check if dummy has start
            try:
                self.inference_server.start()
            except Exception as e: # pragma: no cover
                self.logger.error(f"Failed to start inference server during init: {e}")


        dataset_max_seq_len = 512 
        if hasattr(config, 'max_seq_len') and isinstance(config.max_seq_len, int) and config.max_seq_len > 0:
             dataset_max_seq_len = config.max_seq_len
        
        self.dataset = AtroposDataset(
            api_url=f"http://localhost:{config.rollout_server_port}", # Assuming rollout server is on localhost
            batch_size=config.batch_size,
            tokenizer=self.tokenizer,
            max_seq_len=dataset_max_seq_len 
        )
        self._dataset_iter: Optional[Iterator[Dict[str, torch.Tensor]]] = None

        # Optimizer for the training model
        self.optimizer = optim.AdamW(self.training_model.parameters(), lr=config.lr)
        
        self.logger.info("AtroposGrpoTrainer initialized successfully.")

    # --- Method Placeholders ---
    def sync_weights(self):
        """
        Copies the state_dict from self.training_model to the inference server.
        The inference_server is expected to have a `load_weights` method.
        """
        self.logger.info("Synchronizing weights from training model to inference server...")
        if not hasattr(self.inference_server, 'load_weights') or not callable(self.inference_server.load_weights):
            self.logger.warning(f"Inference server of type {type(self.inference_server).__name__} does not have a callable load_weights method. Skipping weight sync.")
            return

        try:
            state_dict = self.training_model.state_dict()
            # Consider deepcopy if models are complex or issues arise:
            # import copy
            # state_dict_cpu = {k: v.cpu().clone().detach() for k, v in self.training_model.state_dict().items()}
            # self.inference_server.load_weights(state_dict_cpu)
            
            self.inference_server.load_weights(state_dict)
            self.logger.info("Successfully synchronized weights to inference server.")
        except Exception as e:
            self.logger.error(f"Error during weight synchronization: {e}", exc_info=True)

    def fetch_rollouts(self) -> Dict[str, torch.Tensor]:
        """
        Fetches a batch of rollouts from the AtroposDataset.
        Initializes the dataset iterator if it's not already created.
        Moves all tensors in the fetched batch to the trainer's device.
        Returns an empty dictionary if the dataset is exhausted or an error occurs.
        """
        self.logger.debug("Attempting to fetch rollouts...")
        if self._dataset_iter is None:
            self.logger.info("Creating new AtroposDataset iterator for fetch_rollouts.")
            self._dataset_iter = iter(self.dataset)
        
        try:
            # next() will raise StopIteration if dataset is exhausted
            batch = next(self._dataset_iter) 
            
            if not batch: # Should not happen if next() succeeds, but as a safeguard
                self.logger.warning("next(dataset_iter) returned an empty batch (None or empty dict).")
                # Consider this as end of data or an issue
                self._dataset_iter = None # Reset to allow re-iteration if train loop continues & dataset supports it
                return {}

            device_batch = {}
            for key, tensor_val in batch.items(): 
                if isinstance(tensor_val, torch.Tensor):
                    device_batch[key] = tensor_val.to(self.device)
                else:
                    # Store non-tensor metadata as is (e.g., if dataset yields prompt_end_positions as a list)
                    device_batch[key] = tensor_val 
            
            self.logger.debug(f"Successfully fetched batch. Keys: {list(device_batch.keys())}")
            if "input_ids" in device_batch and isinstance(device_batch["input_ids"], torch.Tensor): # Check if tensor before shape
                 self.logger.debug(f"Batch input_ids shape: {device_batch['input_ids'].shape}")
            return device_batch
            
        except StopIteration:
            self.logger.info("AtroposDataset iterator exhausted.")
            self._dataset_iter = None # Reset iterator
            return {} # Signal end of data
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during fetch_rollouts: {e}", exc_info=True)
            self._dataset_iter = None # Reset iterator on error too
            return {}


    def _get_prompt_end_positions(self, batch: Dict[str, Any]) -> List[int]: # Value can be list or tensor
        """
        Extracts or determines the end position of prompts for each sequence in the batch.
        This is equivalent to the start index of the response.
        """
        if "prompt_token_lengths" in batch:
            prompt_lengths = batch["prompt_token_lengths"]
            if isinstance(prompt_lengths, torch.Tensor):
                # Convert to list of ints, ensure it's on CPU for list conversion
                return prompt_lengths.cpu().tolist() 
            elif isinstance(prompt_lengths, list) and all(isinstance(x, int) for x in prompt_lengths):
                return prompt_lengths
            else:
                self.logger.error(f"Batch contains 'prompt_token_lengths' but it's not a Tensor or List[int]. Type: {type(prompt_lengths)}")
        
        # Fallback or error if not found - this indicates a data pipeline issue.
        self.logger.error("CRITICAL: 'prompt_token_lengths' not found in batch from AtroposDataset. Loss mask will be incorrect.")
        if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
            # Return a dummy list (e.g., all zeros, meaning full sequence is response)
            # This is safer than assuming a small prompt like [1] if we don't know.
            # Or raise an error. For now, returning all zeros to avoid crashes but highlight issue.
            return [0] * batch["input_ids"].size(0) 
        
        raise ValueError("Cannot determine prompt_end_positions: 'prompt_token_lengths' missing and no 'input_ids' in batch.")


    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor: 
        self.logger.debug("compute_loss called (placeholder).")
        # Actual implementation will use self.training_model, self.ref_model,
        # and functions from verl.train.losses (advantage_weighted_loss, kl_penalty_loss)
        # For now, returning a dummy loss that requires grad.
        # Example: output = self.training_model(batch["input_ids"], attention_mask=batch["attention_mask"])
        # return output.mean() 
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_one_iteration(self) -> bool:
        self.logger.debug("train_one_iteration called.")
        self.sync_weights() # Ensure this is called
        
        batch = self.fetch_rollouts()
        if not batch: 
            self.logger.info("No data from fetch_rollouts. Skipping training step for this iteration.")
            return False 

        self.training_model.train() # Ensure model is in training mode for this iteration
        
        # Add conditional sleep for testing graceful shutdown
        # This sleep should ideally be before the optimization loop or at a point
        # where interrupting the trainer is meaningful for testing cleanup.
        # Placing it before PPO epochs loop.
        if hasattr(self.config, 'test_mode_iteration_sleep_s') and self.config.test_mode_iteration_sleep_s > 0:
            self.logger.info(f"TEST MODE: Sleeping for {self.config.test_mode_iteration_sleep_s}s in train_one_iteration.")
            import time # Make sure time is imported in this file
            time.sleep(self.config.test_mode_iteration_sleep_s)

        # PPO typically involves multiple optimization epochs on the same batch of rollouts
        for ppo_epoch in range(self.config.ppo_epochs): # Corrected variable name from prompt
            # In a more complex setup, batch might be further divided into mini-batches here.
            # For now, assume compute_loss processes the entire fetched batch.
            
            loss = self.compute_loss(batch) # This needs to use all relevant parts of the batch
            
            self.optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping (common in PPO)
            # torch.nn.utils.clip_grad_norm_(self.training_model.parameters(), max_norm=1.0) # Example
            self.optimizer.step()
            
            self.logger.debug(f"PPO Epoch {ppo_epoch+1}/{self.config.ppo_epochs}, Loss: {loss.item()}") # Corrected variable name
        
        return True # Indicate that training occurred for this iteration


    def train(self, num_iterations: Optional[int] = None):
        if num_iterations is None:
            num_iterations = self.config.num_iterations 
        
        self.logger.info(f"Starting GRPO training for {num_iterations} iterations.")
        
        # Start inference server before training loop if not already started by __init__
        # (or if it was stopped previously)
        if hasattr(self.inference_server, 'start') and \
           (not hasattr(self.inference_server, 'is_alive') or not self.inference_server.is_alive()): # Simplified check
            try:
                self.logger.info("Attempting to start inference server before training loop.")
                self.inference_server.start()
                # Optionally, add a readiness check here for the inference server
            except Exception as e: # pragma: no cover
                self.logger.error(f"Failed to start inference server before training: {e}")
                return # Cannot proceed if inference server (needed for rollouts via AtroposDataset) fails

        for i in range(num_iterations):
            self.logger.info(f"Iteration {i+1}/{num_iterations}")
            
            iteration_trained = self.train_one_iteration()
            
            # Hypothetical config to stop if dataset runs out
            stop_on_empty = getattr(self.config, 'stop_on_empty_rollout', False) 
            if not iteration_trained and stop_on_empty:
                 self.logger.info("Stopping training early due to empty rollouts and 'stop_on_empty_rollout' is True.")
                 break
            
            # Optional: Log metrics, save checkpoints, etc.
            # if (i + 1) % self.config.save_interval == 0: self.save_checkpoint()

        self.logger.info("GRPO training finished.")

    def cleanup(self): 
        self.logger.info("Cleaning up AtroposGrpoTrainer resources...")
        if hasattr(self.inference_server, 'stop'):
            self.inference_server.stop()
        self.logger.info("Cleanup complete.")
