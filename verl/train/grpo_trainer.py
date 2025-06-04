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
import time # For test_mode_iteration_sleep_s, already added in prev step, ensuring it's there. Will be consolidated by linter.

# Add new imports for loss functions and mask builder
from verl.train.utils import build_loss_mask
from verl.train.losses import advantage_weighted_loss, kl_penalty_loss
from torch.utils.tensorboard import SummaryWriter # Add this
import json # Add this
from datetime import datetime # Add this
from pathlib import Path # Add this if not already fully used (it is, but good to be explicit for this change)

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

        # Metrics Export Setup
        run_name = f"grpo_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{self.config.model.replace('/', '_')}"
        self.log_dir = Path("runs") / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writer = SummaryWriter(log_dir=str(self.log_dir))

        self.metrics_dir = Path("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.metrics_dir / "logs.jsonl"

        self.logger.info(f"TensorBoard logs will be saved to: {self.log_dir}")
        self.logger.info(f"JSONL metrics will be saved to: {self.metrics_file}")
        self.logger.info("AtroposGrpoTrainer initialized successfully for metrics.") # Updated log msg

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


    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]: # Return dict of losses
        """
        Computes the GRPO loss for a given batch of data.
        The batch is expected to contain 'input_ids', 'attention_mask',
        'old_logprobs', 'advantages', and 'prompt_token_lengths'.
        """
        self.logger.debug("Computing loss for batch...")

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        old_logprobs = batch.get("old_logprobs")
        advantages = batch.get("advantages")

        if not all(isinstance(t, torch.Tensor) for t in [input_ids, attention_mask, old_logprobs, advantages]):
            self.logger.error("Batch is missing required tensor keys (input_ids, attention_mask, old_logprobs, advantages) or they are not tensors. Cannot compute loss.")
            # Return dict with zero losses that require grad to prevent crashes in the training loop
            return {"total_loss": torch.tensor(0.0, device=self.device, requires_grad=True, dtype=torch.float32),
                    "ppo_objective": torch.tensor(0.0, device=self.device),
                    "kl_penalty": torch.tensor(0.0, device=self.device),
                    "kl_divergence_raw": torch.tensor(0.0, device=self.device)}

        # 1. Get current model logits
        try:
            curr_logits = self.training_model(input_ids, attention_mask=attention_mask) # Shape: [B, T, V]
        except Exception as e: # pragma: no cover
            self.logger.error(f"Error during training_model forward pass: {e}", exc_info=True)
            raise

        # 2. Determine action_ids
        action_ids = input_ids

        # 3. Get prompt_end_positions and build loss_mask
        prompt_end_positions = self._get_prompt_end_positions(batch)
        try:
            loss_m = build_loss_mask(input_ids, prompt_end_positions)
            loss_m = loss_m.to(self.device)
        except Exception as e: # pragma: no cover
            self.logger.error(f"Error building loss mask: {e}", exc_info=True)
            self.logger.warning("Falling back to a loss_mask of all ones due to error.")
            loss_m = torch.ones_like(input_ids, dtype=torch.float32, device=self.device)

        # 4. Compute PPO objective (policy loss + entropy bonus)
        try:
            # advantage_weighted_loss returns the combined PPO objective value (policy_loss - entropy_term)
            policy_objective_val = advantage_weighted_loss(
                curr_logits=curr_logits,
                action_ids=action_ids,
                old_logprobs=old_logprobs,
                advantages=advantages,
                loss_mask=loss_m,
                clip_ratio=self.config.clip_ratio,
                entropy_coef=self.config.entropy_coef
            )
            self.logger.debug(f"Advantage-weighted PPO objective (policy loss - entropy bonus): {policy_objective_val.item()}")
        except Exception as e: # pragma: no cover
            self.logger.error(f"Error computing advantage_weighted_loss: {e}", exc_info=True)
            raise

        # 5. Compute KL penalty if reference model exists
        kl_div_val = torch.tensor(0.0, device=self.device)
        kl_penalty_val = torch.tensor(0.0, device=self.device)
        if self.ref_model is not None and self.config.kl_coef > 0:
            self.logger.debug("Computing KL penalty with reference model...")
            try:
                with torch.no_grad():
                    ref_logits = self.ref_model(input_ids, attention_mask=attention_mask)

                kl_div_val = kl_penalty_loss(curr_logits, ref_logits.detach())
                self.logger.debug(f"Raw KL divergence: {kl_div_val.item()}")

                kl_penalty_val = self.config.kl_coef * kl_div_val
                self.logger.debug(f"KL penalty ({self.config.kl_coef} * KL): {kl_penalty_val.item()}")
            except Exception as e: # pragma: no cover
                self.logger.error(f"Error computing KL penalty: {e}", exc_info=True)

        total_loss = policy_objective_val + kl_penalty_val
        self.logger.debug(f"Total computed loss for batch: {total_loss.item()}")

        return {
            "total_loss": total_loss,
            "ppo_objective": policy_objective_val,
            "kl_penalty": kl_penalty_val,
            "kl_divergence_raw": kl_div_val
        }


    def train_one_iteration(self) -> Optional[Dict[str, float]]: # Return dict of avg losses for logging
        self.logger.debug(f"Starting train_one_iteration for iteration {getattr(self, 'current_iteration_num', 'N/A') + 1}")
        self.sync_weights() # Ensure this is called

        batch = self.fetch_rollouts()
        if not batch:
            self.logger.info("No data from fetch_rollouts. Skipping training step for this iteration.")
            return None # Return None to signal no training happened

        self.training_model.train() # Ensure model is in training mode for this iteration

        if hasattr(self.config, 'test_mode_iteration_sleep_s') and self.config.test_mode_iteration_sleep_s > 0: # pragma: no cover
            self.logger.info(f"TEST MODE: Sleeping for {self.config.test_mode_iteration_sleep_s}s in train_one_iteration.")
            time.sleep(self.config.test_mode_iteration_sleep_s)

        # Store losses from each PPO epoch for this batch
        # Initialize with lists to append to.
        epoch_losses: Dict[str, List[float]] = {"total_loss": [], "ppo_objective": [], "kl_penalty": [], "kl_divergence_raw": []}

        for ppo_epoch_num in range(self.config.ppo_epochs):
            loss_components = self.compute_loss(batch)
            total_loss = loss_components["total_loss"]

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.logger.debug(f"Iter {getattr(self, 'current_iteration_num', 'N/A') + 1}/PPO Epoch {ppo_epoch_num+1}: TotalLoss={total_loss.item():.4f}, PPO_Obj={loss_components['ppo_objective'].item():.4f}, KLpen={loss_components['kl_penalty'].item():.4f}")
            for key, value_tensor in loss_components.items():
                # Ensure key exists in epoch_losses, which it should from initialization
                epoch_losses[key].append(value_tensor.item()) # Store scalar loss value

        # Average losses over PPO epochs for this batch
        avg_losses = {key: sum(values)/len(values) if values else 0.0 for key, values in epoch_losses.items()}

        advantages_tensor = batch.get("advantages")
        # Ensure input_ids is available for building loss mask for advantages mean calculation
        input_ids_for_mask = batch.get("input_ids")
        avg_reward_or_advantage_val = 0.0
        if advantages_tensor is not None and input_ids_for_mask is not None:
            # We need prompt_end_positions to build the correct mask for advantages.
            # _get_prompt_end_positions needs the full batch dict.
            prompt_end_pos_for_adv = self._get_prompt_end_positions(batch)
            loss_mask_for_adv_mean = build_loss_mask(input_ids_for_mask, prompt_end_pos_for_adv).to(self.device)

            masked_advantages = advantages_tensor * loss_mask_for_adv_mean
            num_active_elements = loss_mask_for_adv_mean.sum()
            if num_active_elements > 0:
                avg_reward_or_advantage = masked_advantages.sum() / num_active_elements
                avg_reward_or_advantage_val = avg_reward_or_advantage.item()

        avg_losses["reward_mean_advantages"] = avg_reward_or_advantage_val

        return avg_losses


    def train(self, num_iterations: Optional[int] = None):
        if num_iterations is None:
            num_iterations = self.config.num_iterations

        self.logger.info(f"Starting GRPO training for {num_iterations} iterations.")

        if hasattr(self.inference_server, 'start') and \
           (not hasattr(self.inference_server, 'is_alive') or not getattr(self.inference_server, 'is_alive', lambda: True)()): # More robust check for is_alive
            try:
                self.logger.info("Attempting to start inference server before training loop.")
                self.inference_server.start()
            except Exception as e: # pragma: no cover
                self.logger.error(f"Failed to start inference server before training: {e}", exc_info=True)
                return

        for i in range(num_iterations):
            self.current_iteration_num = i # Store for logging
            self.logger.info(f"Iteration {i+1}/{num_iterations}")

            iteration_metrics = self.train_one_iteration()

            if iteration_metrics is None:
                 self.logger.info("Dataset iterator exhausted or error in fetching. Stopping training.")
                 break

            # Log to TensorBoard
            self.summary_writer.add_scalar("rollout/reward_mean_advantages", iteration_metrics.get("reward_mean_advantages",0.0), i)
            self.summary_writer.add_scalar("train/total_loss", iteration_metrics.get("total_loss",0.0), i)
            self.summary_writer.add_scalar("train/ppo_objective", iteration_metrics.get("ppo_objective",0.0), i)
            self.summary_writer.add_scalar("train/kl_penalty", iteration_metrics.get("kl_penalty",0.0), i)
            self.summary_writer.add_scalar("train/kl_divergence_raw", iteration_metrics.get("kl_divergence_raw",0.0), i)

            # Append to JSONL
            log_entry = {
                "iteration": i + 1, # Use 1-based iteration for logging
                "timestamp": datetime.now().isoformat(),
                "reward_mean_advantages": iteration_metrics.get("reward_mean_advantages",0.0),
                "total_loss": iteration_metrics.get("total_loss",0.0),
                "ppo_objective": iteration_metrics.get("ppo_objective",0.0),
                "kl_penalty": iteration_metrics.get("kl_penalty",0.0),
                "kl_divergence_raw": iteration_metrics.get("kl_divergence_raw",0.0)
            }
            try:
                with open(self.metrics_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e_json: # pragma: no cover
                self.logger.error(f"Failed to write metrics to JSONL file {self.metrics_file}: {e_json}")

            stop_on_empty = getattr(self.config, 'stop_on_empty_rollout', False)
            if iteration_metrics is None and stop_on_empty: # This check is now redundant due to break above
                 self.logger.info("Stopping training early due to empty rollouts and 'stop_on_empty_rollout' is True.")
                 break # Should have already broken if iteration_metrics is None

        self.logger.info("GRPO training finished.")
        self.cleanup()

    def cleanup(self):
        self.logger.info("Cleaning up AtroposGrpoTrainer resources...")
        if hasattr(self.inference_server, 'stop') and callable(self.inference_server.stop):
            self.logger.info(f"Stopping inference server: {type(self.inference_server).__name__}")
            self.inference_server.stop()

        if hasattr(self, 'summary_writer') and self.summary_writer: # Check if summary_writer exists
            self.summary_writer.close()
            self.logger.info("TensorBoard SummaryWriter closed.")
        self.logger.info("Cleanup complete.")
