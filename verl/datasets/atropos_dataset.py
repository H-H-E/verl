# verl/datasets/atropos_dataset.py
import asyncio
import aiohttp # For async HTTP requests
import torch
from torch.utils.data import IterableDataset
from typing import List, Dict, Any, Optional # For type hinting
import logging # For logging

# Dummy Tokenizer for placeholder - replace with actual tokenizer later
class DummyTokenizer:
    def __init__(self, pad_token_id=0, max_length=512):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        # Minimal vocab for testing. A real tokenizer would have a large vocab.
        self.vocab = {"<pad>": 0, "<unk>": 1, "Q": 2, "A": 3, "1": 4, ":":5, "s":6, "p":7, "r":8, "o":9, "m":10, "t":11, "e":12, "n":13, "c":14}

    def encode(self, text: str, add_special_tokens=True, truncation=True, max_length=None, padding='max_length', return_tensors=None, return_attention_mask=True):
        tokens = []
        for char_token in list(text): # Simple char-level tokenization for dummy
            tokens.append(self.vocab.get(char_token, self.vocab["<unk>"]))
        
        effective_max_length = max_length if max_length is not None else self.max_length
        
        attention_mask = [1] * len(tokens)

        if len(tokens) < effective_max_length and padding == 'max_length':
            pad_len = effective_max_length - len(tokens)
            tokens.extend([self.pad_token_id] * pad_len)
            attention_mask.extend([0] * pad_len)
        elif len(tokens) > effective_max_length and truncation:
            tokens = tokens[:effective_max_length]
            attention_mask = attention_mask[:effective_max_length]

        if return_tensors == "pt":
            output = {
                "input_ids": torch.tensor([tokens], dtype=torch.long),
            }
            if return_attention_mask:
                output["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
            return output
        # If not returning tensors, return list of token IDs and optionally the mask
        if return_attention_mask:
            return {"input_ids": tokens, "attention_mask": attention_mask}
        return {"input_ids": tokens}


    def batch_encode_plus(self, texts: List[str], add_special_tokens=True, truncation=True, max_length=None, padding='max_length', return_tensors="pt", return_attention_mask=True):
        all_input_ids = []
        all_attention_masks = []
        
        effective_max_length = max_length if max_length is not None else self.max_length

        for text in texts:
            # Use the single encode method
            encoded_output = self.encode(
                text, 
                add_special_tokens=add_special_tokens, 
                truncation=truncation, 
                max_length=effective_max_length, 
                padding=padding, 
                return_tensors=None, # Get lists first
                return_attention_mask=return_attention_mask
            )
            all_input_ids.append(encoded_output["input_ids"])
            if return_attention_mask:
                all_attention_masks.append(encoded_output["attention_mask"])

        if return_tensors == "pt":
            batch = {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long)
            }
            if return_attention_mask:
                batch["attention_mask"] = torch.tensor(all_attention_masks, dtype=torch.long)
            return batch
        
        # Fallback if not returning tensors
        if return_attention_mask:
            return [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(all_input_ids, all_attention_masks)]
        return [{"input_ids": ids} for ids in all_input_ids]


class AtroposDataset(IterableDataset):
    def __init__(self, api_url: str, batch_size: int, tokenizer: Any, max_seq_len: int = 512):
        self.api_url = api_url.rstrip("/")
        self.batch_size = batch_size
        self.tokenizer = tokenizer 
        self.max_seq_len = max_seq_len

        self.logger = logging.getLogger(__name__)
        if not self.logger.hasHandlers(): # Ensure logger outputs something if not configured by main app
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    async def _fetch_once(self) -> List[Dict[str, Any]]:
        fetch_url = f"{self.api_url}/batch?size={self.batch_size}"
        self.logger.debug(f"Fetching data from {fetch_url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url) as response:
                    response.raise_for_status() 
                    data = await response.json()
                    # Assuming data is a list of rollout groups
                    if not isinstance(data, list):
                        self.logger.error(f"Expected a list from {fetch_url}, got {type(data)}. Data: {str(data)[:100]}")
                        return [] # Return empty on unexpected format
                    self.logger.debug(f"Successfully fetched {len(data)} groups.")
                    return data
        except aiohttp.ClientError as e:
            self.logger.error(f"AIOHTTP client error fetching from {fetch_url}: {e}")
            return [] 
        except Exception as e:
            self.logger.error(f"Unexpected error fetching from {fetch_url}: {e}")
            return []

    def _to_tensors(self, data: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not data:
            self.logger.warning("_to_tensors received empty data list.")
            return {}

        batch_texts = []
        # prompt_lengths_char = [] # Not strictly needed for this version of _to_tensors
        
        for group in data:
            prompt = group.get("prompt", "")
            response = group.get("response", "")
            # For tokenization, usually concatenate prompt and response.
            # The TestTokenizer used in tests adds EOS if add_special_tokens=True.
            # If a real tokenizer handles prompt/response separation differently (e.g. specific chat template),
            # this part or the tokenizer calls would need adjustment.
            text_sequence = prompt + response 
            batch_texts.append(text_sequence)
            # prompt_lengths_char.append(len(prompt)) # Original char length, for reference or debugging


        # Tokenize all text sequences in a batch
        try:
            tokenized_batch = self.tokenizer.batch_encode_plus(
                batch_texts,
                add_special_tokens=True, # Crucial: determines if EOS is added by TestTokenizer
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
                return_attention_mask=True
            )
        except Exception as e:
            self.logger.error(f"Error during batch tokenization: {e}")
            return {}

        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        current_batch_size = input_ids.size(0)

        advantages = torch.zeros(current_batch_size, self.max_seq_len, dtype=torch.float32)
        old_logprobs = torch.zeros(current_batch_size, self.max_seq_len, dtype=torch.float32)

        for i in range(current_batch_size):
            group = data[i]
            
            # Determine prompt token length.
            # This relies on self.tokenizer.encode behaving consistently for prompt-only vs part of combined sequence.
            # For TestTokenizer: encode(prompt, add_special_tokens=False) will give raw token count for prompt.
            prompt_text = group.get("prompt", "")
            prompt_only_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_token_len_val = len(prompt_only_tokens["input_ids"]) # Get the list of IDs then its length

            actual_seq_len = int(attention_mask[i].sum()) # Number of non-padding tokens in the current sequence

            token_advantages_data = group.get("token_advantages") # List of floats for response tokens
            reward_data = group.get("reward") # Scalar float
            logprobs_data = group.get("logprobs") # List of floats (assumed for response tokens primarily)

            for j in range(actual_seq_len): # Iterate through actual tokens (non-padding)
                if j < prompt_token_len_val: # Token is part of the prompt
                    advantages[i, j] = 0.0
                    # old_logprobs for prompt tokens:
                    # If logprobs_data is for the whole sequence (prompt+response)
                    if logprobs_data and j < len(logprobs_data): # Check if logprobs_data is long enough
                        old_logprobs[i, j] = float(logprobs_data[j])
                    else: # Otherwise, 0 or ignore_index for prompt logprobs
                        old_logprobs[i, j] = 0.0 
                else: # Token is part of the response
                    response_token_index = j - prompt_token_len_val
                    
                    # Set advantages for response tokens
                    if token_advantages_data and response_token_index < len(token_advantages_data):
                        advantages[i, j] = float(token_advantages_data[response_token_index])
                    elif reward_data is not None:
                        advantages[i, j] = float(reward_data)
                    else:
                        advantages[i, j] = 0.0 # Default if no reward/token_advantages info for response

                    # Set old_logprobs for response tokens
                    if logprobs_data:
                        # Option 1: logprobs_data is for the whole sequence (prompt+response)
                        # if j < len(logprobs_data):
                        #    old_logprobs[i, j] = float(logprobs_data[j])
                        # Option 2: logprobs_data is only for response tokens
                        if response_token_index < len(logprobs_data):
                            old_logprobs[i, j] = float(logprobs_data[response_token_index])
                        # Option 3: logprobs_data is for the whole sequence but token_advantages was also present
                        # (implies logprobs_data might be structured differently or we prioritize one source)
                        # The prompt specified "logprobs (assumed to be a list of log probabilities for tokens in the sequence, or at least response tokens)"
                        # Let's assume for now: if logprobs_data is given, it's for response tokens if token_advantages is also given,
                        # otherwise it might be for the whole sequence. This logic can be ambiguous.
                        # A simpler, common case: logprobs are provided *only* for the response tokens.
                        # The current code with `response_token_index < len(logprobs_data)` implements this for response.
                        else: # If logprobs_data is shorter than current response token index
                            old_logprobs[i, j] = 0.0 # Or ignore_index
                    else: # No logprobs_data provided
                        old_logprobs[i, j] = 0.0 # Or ignore_index
                        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "advantages": advantages,
            "old_logprobs": old_logprobs
        }

    def __iter__(self):
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running(): # pragma: no cover
                 self.logger.debug("Event loop is already running in this thread.")
        except RuntimeError: # No event loop in this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.logger.debug("Created new event loop for this thread.")
        
        while True:
            try:
                data = loop.run_until_complete(self._fetch_once())
            except Exception as e: 
                self.logger.error(f"Error in event loop running _fetch_once: {e}")
                break 

            if not data:
                self.logger.info("No data received from _fetch_once, stopping iteration.")
                break 
            
            try:
                tensors = self._to_tensors(data)
                if not tensors: 
                    self.logger.warning("Skipping batch due to _to_tensors returning empty.")
                    continue 
                yield tensors
            except Exception as e:
                self.logger.error(f"Error in _to_tensors: {e}. Skipping batch.")
                continue
