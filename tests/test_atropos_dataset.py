# tests/test_atropos_dataset.py
import pytest
import asyncio
import torch
from torch.utils.data import DataLoader # To test iteration
from typing import List, Dict, Any, Optional
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # For request validation in dummy server (optional)
import time # For time.sleep

# Assuming AtroposDataset is in verl.datasets.atropos_dataset
from verl.datasets.atropos_dataset import AtroposDataset


# --- Mock HTTP Server (similar to one in test_env_spawner) ---
MOCK_API_PORT_DATASET = 8100
MOCK_API_HOST = "localhost"

@pytest.fixture(scope="function")
def mock_atropos_rollout_api_server():
    app = FastAPI()
    
    app.state.rollout_data = [] 
    app.state.serve_empty_once = False 
    app.state.error_once = False 

    @app.get("/batch")
    async def get_batch_endpoint(size: int): 
        if app.state.error_once:
            app.state.error_once = False
            raise HTTPException(status_code=500, detail="Simulated server error")

        if app.state.serve_empty_once:
            app.state.serve_empty_once = False 
            return [] 
            
        if not app.state.rollout_data:
            return [] 
        
        return app.state.rollout_data[:size]

    config = uvicorn.Config(app, host=MOCK_API_HOST, port=MOCK_API_PORT_DATASET, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.5) 
    yield app 
    server.should_exit = True
    thread.join(timeout=5)


# --- Dummy Tokenizer for Tests ---
class TestTokenizer:
    def __init__(self, vocab: Dict[str, int], pad_token_id: int, eos_token_id: int, max_len: int = 10):
        self.vocab = vocab
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id 
        self.max_len = max_len

    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, List[int]]: # Modified to return dict
        tokens = [self.vocab.get(char, self.vocab["<unk>"] if "<unk>" in self.vocab else self.pad_token_id) for char in text]
        if add_special_tokens and text:
            tokens.append(self.eos_token_id)
        return {"input_ids": tokens} # Return dict as expected by _to_tensors's use of encode

    def batch_encode_plus(self, texts: List[str], add_special_tokens=True, padding='max_length', truncation=True, max_length=None, return_tensors="pt", return_attention_mask=True):
        if max_length is None:
            max_length = self.max_len
        
        all_input_ids_list = []
        all_attention_masks_list = []

        for text in texts:
            encoded_output = self.encode(text, add_special_tokens=add_special_tokens)
            encoded_tokens = encoded_output["input_ids"]
            
            if truncation and len(encoded_tokens) > max_length:
                encoded_tokens = encoded_tokens[:max_length]
            
            current_length = len(encoded_tokens)
            attention_mask = [1] * current_length
            
            if padding == 'max_length' and current_length < max_length:
                pad_length = max_length - current_length
                encoded_tokens.extend([self.pad_token_id] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            all_input_ids_list.append(encoded_tokens)
            if return_attention_mask:
                all_attention_masks_list.append(attention_mask)
        
        if return_tensors == "pt":
            batch = {"input_ids": torch.tensor(all_input_ids_list, dtype=torch.long)}
            if return_attention_mask:
                batch["attention_mask"] = torch.tensor(all_attention_masks_list, dtype=torch.long)
            return batch
        else: 
            raise ValueError("TestTokenizer expects return_tensors='pt'")

    def get_prompt_length(self, prompt_text: str) -> int:
        # This should return the length of tokenized prompt *without* special tokens like EOS
        # as per assumption in _to_tensors for prompt_token_len_val.
        encoded_output = self.encode(prompt_text, add_special_tokens=False)
        return len(encoded_output["input_ids"])


# --- Test Cases ---
def test_fetch_and_tensor_shapes(mock_atropos_rollout_api_server):
    api_url = f"http://{MOCK_API_HOST}:{MOCK_API_PORT_DATASET}"
    batch_size = 2
    max_seq_len_test = 12 

    vocab = {"P": 0, "Q": 1, "A": 2, "1": 3, "2": 4, "E": 5, "S":6, "<unk>": 7} 
    pad_token_id = vocab["P"]
    eos_token_id = vocab["E"]
    tokenizer = TestTokenizer(vocab, pad_token_id, eos_token_id, max_len=max_seq_len_test)

    token_adv_sample1 = [1.1, 1.2] 
    scalar_reward_sample2 = -1.0
    mock_data = [
        {"prompt": "Q1", "response": "A1", "reward": 10.0, "token_advantages": token_adv_sample1},
        {"prompt": "Q2S", "response": "A2", "reward": scalar_reward_sample2}
    ]
    mock_atropos_rollout_api_server.state.rollout_data = mock_data
    
    dataset = AtroposDataset(api_url=api_url, batch_size=batch_size, tokenizer=tokenizer, max_seq_len=max_seq_len_test)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    
    batch_tensors = next(iter(dataloader))

    assert batch_tensors["input_ids"].shape == (batch_size, max_seq_len_test)
    assert batch_tensors["attention_mask"].shape == (batch_size, max_seq_len_test)
    assert batch_tensors["advantages"].shape == (batch_size, max_seq_len_test)

    advantages = batch_tensors["advantages"]
    attention_mask = batch_tensors["attention_mask"]

    # Sample 1: "Q1", "A1", token_advantages=[1.1, 1.2]
    # Prompt "Q1" -> tokenized by TestTokenizer.encode("Q1", add_special_tokens=False) -> {"input_ids": [1,3]} -> len=2
    prompt_len_s1 = tokenizer.get_prompt_length("Q1") 
    # Response "A1" -> tokenized by TestTokenizer.encode("A1", add_special_tokens=False) -> {"input_ids": [2,3]} -> len=2
    # This is the length of response tokens that should have advantages.
    
    for j in range(max_seq_len_test):
        if j < prompt_len_s1: 
            assert advantages[0, j] == 0.0, f"S1: Adv for prompt token {j} should be 0"
        elif j < prompt_len_s1 + len(token_adv_sample1) and attention_mask[0,j] == 1: 
            response_token_idx = j - prompt_len_s1
            assert advantages[0, j] == token_adv_sample1[response_token_idx], f"S1: Adv for response token {j} mismatch"
        elif attention_mask[0,j] == 1: # Other actual tokens (e.g. EOS after response)
             assert advantages[0,j] == 0.0, f"S1: Adv for other token {j} (e.g. EOS) should be 0"
        else: # Padding part
            assert advantages[0, j] == 0.0, f"S1: Adv for padding token {j} should be 0"

    # Sample 2: "Q2S", "A2", reward=-1.0
    prompt_len_s2 = tokenizer.get_prompt_length("Q2S") # "Q2S" -> [1,4,6] -> len=3
    num_response_tokens_s2 = len(tokenizer.encode("A2", add_special_tokens=False)["input_ids"]) # "A2" -> [2,4] -> len=2

    for j in range(max_seq_len_test):
        if j < prompt_len_s2: 
            assert advantages[1, j] == 0.0, f"S2: Adv for prompt token {j} should be 0"
        elif j < prompt_len_s2 + num_response_tokens_s2 and attention_mask[1,j] == 1: 
            assert advantages[1, j] == scalar_reward_sample2, f"S2: Adv for response token {j} should be scalar reward"
        else: 
            assert advantages[1, j] == 0.0, f"S2: Adv for padding/other token {j} (e.g. EOS) should be 0"
    
    mock_atropos_rollout_api_server.state.serve_empty_once = True
    dataset_for_empty_test = AtroposDataset(api_url=api_url, batch_size=batch_size, tokenizer=tokenizer, max_seq_len=max_seq_len_test)
    dataloader_for_empty = DataLoader(dataset_for_empty_test, batch_size=None, num_workers=0)
    empty_batch_count = 0
    for _ in dataloader_for_empty: # pragma: no cover
        empty_batch_count +=1 
    assert empty_batch_count == 0


def test_missing_advantages_fallback(mock_atropos_rollout_api_server):
    api_url = f"http://{MOCK_API_HOST}:{MOCK_API_PORT_DATASET}"
    batch_size = 2 
    max_seq_len_test = 10

    vocab = {"P": 0, "Q": 1, "A": 2, "X": 3, "Y": 4, "E": 5, "<unk>": 6} 
    pad_token_id = vocab["P"]
    eos_token_id = vocab["E"]
    tokenizer = TestTokenizer(vocab, pad_token_id, eos_token_id, max_len=max_seq_len_test)

    token_adv_s1 = [0.5, 0.6]
    scalar_reward_s2 = -0.5
    mock_data = [
        {"prompt": "QX", "response": "AY", "reward": 100.0, "token_advantages": token_adv_s1},
        {"prompt": "QY", "response": "AX", "reward": scalar_reward_s2} 
    ]
    mock_atropos_rollout_api_server.state.rollout_data = mock_data
    
    dataset = AtroposDataset(api_url=api_url, batch_size=batch_size, tokenizer=tokenizer, max_seq_len=max_seq_len_test)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    batch_tensors = next(iter(dataloader))

    advantages = batch_tensors["advantages"]
    attention_mask = batch_tensors["attention_mask"]

    # Check Sample 1 (uses token_advantages)
    # Prompt "QX" -> [Q,X] (len 2). Response "AY" -> [A,Y] (len 2).
    prompt_len_s1 = tokenizer.get_prompt_length("QX") 
    for j in range(max_seq_len_test):
        if j < prompt_len_s1:
            assert advantages[0, j] == 0.0, f"S1 Fallback: Adv for prompt token {j} should be 0"
        elif j < prompt_len_s1 + len(token_adv_s1) and attention_mask[0,j] == 1:
            response_token_idx = j - prompt_len_s1
            assert advantages[0, j] == token_adv_s1[response_token_idx], f"S1 Fallback: Adv for response token {j} mismatch"
        else: 
            assert advantages[0, j] == 0.0, f"S1 Fallback: Adv for padding/other token {j} should be 0"
            
    # Check Sample 2 (uses scalar reward fallback)
    # Prompt "QY" -> [Q,Y] (len 2). Response "AX" -> [A,X] (len 2).
    prompt_len_s2 = tokenizer.get_prompt_length("QY") 
    num_response_tokens_s2 = len(tokenizer.encode("AX", add_special_tokens=False)["input_ids"])

    for j in range(max_seq_len_test):
        if j < prompt_len_s2:
            assert advantages[1, j] == 0.0, f"S2 Fallback: Adv for prompt token {j} should be 0"
        elif j < prompt_len_s2 + num_response_tokens_s2 and attention_mask[1,j] == 1: 
            assert advantages[1, j] == scalar_reward_s2, f"S2 Fallback: Adv for response token {j} should be scalar reward"
        else: 
            assert advantages[1, j] == 0.0, f"S2 Fallback: Adv for padding/other token {j} should be 0"
