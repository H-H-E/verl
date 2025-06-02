# tests/test_on_policy.py
import pytest
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional # Added Optional
import threading
import uvicorn
from fastapi import FastAPI 
import time
import os
import numpy as np # For cosine similarity/distance
from pathlib import Path
import yaml # For config fixture
import logging # For caplog
import requests # For fixture readiness check

# Components to test/use
from verl.atropos_config import AtroposConfig
from verl.train.grpo_trainer import AtroposGrpoTrainer, DummyTrainingModel 
from verl.atropos_inference import EmbeddedInferenceServer, is_server_ready
# DummyTokenizer is imported by AtroposGrpoTrainer which imports AtroposDataset

# --- Project Path Setup ---
PROJECT_ROOT_ON_POLICY_TEST = Path(__file__).resolve().parent.parent
LAUNCH_SCRIPT_PATH_ON_POLICY_TEST = PROJECT_ROOT_ON_POLICY_TEST / "recipe" / "atropos" / "launch_atropos_verl.py" # Not used directly by this test
DUMMY_ENV_SCRIPT_DIR_ON_POLICY_TEST = PROJECT_ROOT_ON_POLICY_TEST / "tests" / "dummy_env_scripts"


# --- Mock Atropos API Server (serves data for the trainer's AtroposDataset) ---
MOCK_API_PORT_ON_POLICY_TEST = 8113 
MOCK_API_HOST_ON_POLICY_TEST = "localhost"

@pytest.fixture(scope="function")
def mock_rollout_api_for_on_policy_test(tmp_path: Path): # Changed from tmp_path: Path to avoid conflict if not used for CWD
    app = FastAPI()
    fixed_rollout_group = {
        "prompt": "OnPolicyPrompt", 
        "response": "OnPolicyResp", 
        "reward": 1.0, 
        # Provide dummy logprobs and token_advantages to match AtroposDataset's expectations for _to_tensors
        "logprobs": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Len for "OnPolicyResp" + EOS (dummy tokenizer)
        "token_advantages": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # Positive advantages for response
    }
    
    @app.get("/batch")
    async def get_rollout_batch(size: int):
        return [fixed_rollout_group for _ in range(size)]

    @app.get("/health")
    async def health_check(): return {"status": "healthy_onpolicy_mock_api"}

    # No CWD change here, as trainer will write to default ./runs and ./metrics if not in tmp_path test
    # For this specific test, we are not checking those files, but the policy change.

    config = uvicorn.Config(app, host=MOCK_API_HOST_ON_POLICY_TEST, port=MOCK_API_PORT_ON_POLICY_TEST, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    
    ready_url = f"http://{MOCK_API_HOST_ON_POLICY_TEST}:{MOCK_API_PORT_ON_POLICY_TEST}/health"
    if not is_server_ready(ready_url, timeout=10.0, poll_interval=0.2): # pragma: no cover
        server.should_exit = True
        thread.join(timeout=1)
        pytest.skip(f"Mock On-Policy API server on port {MOCK_API_PORT_ON_POLICY_TEST} did not become ready.")
    
    yield f"http://{MOCK_API_HOST_ON_POLICY_TEST}:{MOCK_API_PORT_ON_POLICY_TEST}"
    
    server.should_exit = True
    thread.join(timeout=5)

@pytest.fixture(scope="module")
def on_policy_test_config_file(tmp_path_factory):
    fixture_dir = tmp_path_factory.mktemp("on_policy_fixtures")
    config_content = {
        "model": "on-policy-dummy-model",
        "environments": ["on_policy_dummy_env"], # Matches on_policy_dummy_env_server.py
        "rollout_server_port": MOCK_API_PORT_ON_POLICY_TEST,
        "inference_api_port": 8114, # Port for trainer's EmbeddedInferenceServer
        "use_sglang": False,
        "lr": 1e-2, # High LR for faster change
        "num_iterations": 2, # We run two iterations manually in the test
        "batch_size": 1, 
        "ppo_epochs": 1,
        "max_seq_len": 20, # "OnPolicyPrompt" is 14 chars, "OnPolicyResp" is 12. Total 26. Needs to be > 26 + EOS.
                           # DummyTokenizer is char-level. Let's make prompt shorter for smaller max_seq_len.
                           # Prompt "TestP" (5), Response "TestR" (5). Total 10 + EOS = 11. max_seq_len=16 is fine.
        "entropy_coef": 0.0, 
        "kl_coef": 0.0,
        "clip_ratio": 0.2, "tensor_parallel": 1, "reference_model": None,
        # test_mode_iteration_sleep_s: 0 # No sleep needed for this test
    }
    config_file_path = fixture_dir / "on_policy_test_config.yaml"
    with open(config_file_path, 'w') as f:
        yaml.dump(config_content, f)
    return config_file_path

# Ensure the dummy env script (on_policy_dummy_env_server.py) exists
@pytest.fixture(scope="module", autouse=True)
def check_on_policy_dummy_env_script_exists(): # pragma: no cover
    DUMMY_ENV_SCRIPT_DIR_ON_POLICY_TEST.mkdir(exist_ok=True)
    script_path = DUMMY_ENV_SCRIPT_DIR_ON_POLICY_TEST / "on_policy_dummy_env_server.py"
    if not script_path.exists():
        pytest.fail(f"Dummy environment script {script_path} not found. It should have been created by a prior step.")

# Skip E2E test by default unless env var is set
E2E_TEST_ENABLED_ON_POLICY = os.environ.get("RUN_E2E_TESTS", "false").lower() == "true"

@pytest.mark.skipif(not E2E_TEST_ENABLED_ON_POLICY, reason="On-policy E2E test, skipped by default (set RUN_E2E_TESTS=true).")
@pytest.mark.e2e_test 
def test_on_policy_updates_change_distribution(
    mock_rollout_api_for_on_policy_test: str, # Provides API URL, server is running
    on_policy_test_config_file: Path, 
    caplog,
    tmp_path # For trainer's own runs/metrics dirs if needed, though not checked here
):
    caplog.set_level(logging.INFO)
    logger_test = logging.getLogger("TestOnPolicy")

    # Change CWD to tmp_path so trainer outputs (runs/, metrics/) go there.
    original_cwd = Path.cwd()
    os.chdir(tmp_path)

    try:
        config = load_atropos_config(on_policy_test_config_file)
        # Override rollout_server_port to use the one from the fixture if different (it matches here)
        config.rollout_server_port = MOCK_API_PORT_ON_POLICY_TEST
        
        # Update prompt in mock API to match a shorter one for max_seq_len
        # This requires access to the 'app' object of the mock API server,
        # which the current fixture setup doesn't provide directly.
        # Modifying fixture to return app or making fixed_rollout_group in test.
        # For now, ensure config.max_seq_len is adequate for "OnPolicyPrompt" + "OnPolicyResp"
        # "OnPolicyPrompt" = 14, "OnPolicyResp" = 12. EOS = 1. Total = 27.
        # Set max_seq_len in config to e.g. 32
        config.max_seq_len = 32 
        
        # Modify mock API data to use shorter prompt/response for this config
        mock_rollout_api_for_on_policy_test_app = mock_rollout_api_for_on_policy_test # This IS the app fixture
        shorter_rollout_group = {
            "prompt": "TestP", "response": "TestR", "reward": 1.0,
            "logprobs": [-1.0] * (len("TestR") + 1), # For "TestR" + EOS
            "token_advantages": [1.0] * (len("TestR") + 1)
        }
        mock_rollout_api_for_on_policy_test_app.state.rollout_data_store = [shorter_rollout_group]


        trainer = AtroposGrpoTrainer(config)
        assert isinstance(trainer.inference_server, EmbeddedInferenceServer)
        assert isinstance(trainer.training_model, DummyTrainingModel)
        assert isinstance(trainer.inference_server.model, DummyTrainingModel)

        fixed_prompt_text = shorter_rollout_group["prompt"] # Use the same prompt as in rollouts
        
        # Tokenize prompt using trainer's tokenizer (DummyTokenizer)
        # This needs to match how AtroposDataset tokenizes for input_ids
        tokenized_output = trainer.tokenizer.batch_encode_plus(
            [fixed_prompt_text], add_special_tokens=True, padding="max_length",
            truncation=True, max_length=config.max_seq_len, return_tensors="pt"
        )
        prompt_input_ids = tokenized_output["input_ids"].to(trainer.device)
        # We need logits for the first token *after* the prompt.
        # The DummyTokenizer is char-level. "TestP" -> T,e,s,t,P (len 5). EOS added by batch_encode_plus.
        # If add_special_tokens=True for batch_encode_plus, it adds EOS at end of "TestP".
        # We need logits for the token at index len(tokenized_prompt_no_eos).
        # The current DummyTokenizer adds EOS if add_special_tokens=True to encode().
        # batch_encode_plus calls encode(text, add_special_tokens=True).
        # This means prompt_input_ids for "TestP" will be [T,e,s,t,P,EOS,pad,...]
        # We are interested in the distribution of the token immediately following the prompt.
        # This means we need logits from the model for the prompt "TestP" (without EOS),
        # then look at the distribution for the next token.
        
        # Get prompt tokens without final EOS for querying the model
        query_prompt_tokens = trainer.tokenizer.encode(fixed_prompt_text, add_special_tokens=False)["input_ids"]
        query_input_ids = torch.tensor([query_prompt_tokens], dtype=torch.long, device=trainer.device)


        # Sync initial weights (training_model -> inference_server.model)
        trainer.sync_weights() 
        with torch.no_grad():
            logits_p0_full = trainer.inference_server.model(query_input_ids) 
        # Get logits for the *next* token prediction after the prompt
        dist_p0 = F.softmax(logits_p0_full[:, -1, :], dim=-1).cpu().numpy().flatten() 

        # Perform one GRPO iteration
        logger_test.info("Running first train_one_iteration...")
        metrics_iter1 = trainer.train_one_iteration()
        assert metrics_iter1 is not None, "First training iteration failed to return metrics."

        trainer.sync_weights() 
        with torch.no_grad():
            logits_p1_full = trainer.inference_server.model(query_input_ids)
        dist_p1 = F.softmax(logits_p1_full[:, -1, :], dim=-1).cpu().numpy().flatten()

        if np.all(dist_p0 == dist_p1): # Check if they are exactly the same
             pytest.fail(f"Policy distribution p0 and p1 are identical. p0: {dist_p0[:10]}, p1: {dist_p1[:10]}. This might happen if compute_loss is still a placeholder or LR is too small.")

        cosine_similarity_p0_p1 = np.dot(dist_p0, dist_p1) / (np.linalg.norm(dist_p0) * np.linalg.norm(dist_p1))
        cosine_distance_p0_p1 = 1 - cosine_similarity_p0_p1
        logger_test.info(f"P0 (sample): {dist_p0[:5]}")
        logger_test.info(f"P1 (sample): {dist_p1[:5]}")
        logger_test.info(f"Cosine Distance (P0, P1): {cosine_distance_p0_p1:.6f}")
        assert cosine_distance_p0_p1 > 1e-4, \
            f"Policy distribution did not change significantly after one iteration. Cosine distance: {cosine_distance_p0_p1}"

        # Perform a second GRPO iteration
        logger_test.info("Running second train_one_iteration...")
        metrics_iter2 = trainer.train_one_iteration()
        assert metrics_iter2 is not None, "Second training iteration failed to return metrics."
        
        trainer.sync_weights()
        with torch.no_grad():
            logits_p2_full = trainer.inference_server.model(query_input_ids)
        dist_p2 = F.softmax(logits_p2_full[:, -1, :], dim=-1).cpu().numpy().flatten()

        if np.all(dist_p1 == dist_p2): # pragma: no cover
             pytest.fail(f"Policy distribution p1 and p2 are identical after second iteration. This might indicate issues.")

        cosine_similarity_p1_p2 = np.dot(dist_p1, dist_p2) / (np.linalg.norm(dist_p1) * np.linalg.norm(dist_p2))
        cosine_distance_p1_p2 = 1 - cosine_similarity_p1_p2
        logger_test.info(f"P2 (sample): {dist_p2[:5]}")
        logger_test.info(f"Cosine Distance (P1, P2): {cosine_distance_p1_p2:.6f}")
        assert cosine_distance_p1_p2 > 1e-4, \
            f"Policy distribution did not change significantly after second iteration. Cosine distance p1-p2: {cosine_distance_p1_p2}"

    finally:
        if 'trainer' in locals() and trainer and hasattr(trainer, 'cleanup'): # pragma: no cover
            trainer.cleanup()
        os.chdir(original_cwd) # Restore CWD
```
