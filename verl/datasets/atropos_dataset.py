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
        # self.api_url and self.batch_size are from __init__
        fetch_url = f"{self.api_url}/batch?size={self.batch_size}"
        # Ensure logger is present, initialized in __init__

        self.logger.debug(f"Fetching rollout batch: URL='{fetch_url}', Requested BatchSize={self.batch_size}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url) as response:
                    status_code = response.status # Get status code
                    self.logger.debug(f"Received response from {fetch_url}, Status={status_code}")

                    response.raise_for_status() # Raises an error for bad status codes (4xx or 5xx)

                    # Ensure response is valid JSON before parsing
                    if 'application/json' not in response.content_type:
                        self.logger.error(f"Unexpected content type from {fetch_url}: {response.content_type}. Expected application/json.")
                        return [] # Or handle as error appropriate for your case

                    data = await response.json()

                    if not isinstance(data, list):
                        self.logger.error(f"Unexpected data format from {fetch_url}. Expected a list of groups, got {type(data)}.")
                        # Potentially log part of the data if small and safe for debugging
                        # self.logger.debug(f"Problematic data sample (first 100 chars): {str(data)[:100]}")
                        return []

                    self.logger.debug(f"Successfully fetched {len(data)} groups from {fetch_url} (requested size: {self.batch_size}).")
                    return data
        except aiohttp.ClientResponseError as e_resp: # More specific exception for HTTP errors after raise_for_status
            self.logger.error(f"HTTP error fetching from {fetch_url}: Status={e_resp.status}, Message='{e_resp.message}', Headers='{e_resp.headers}'")
            return []
        except aiohttp.ClientError as e_client: # Other client errors (connection, timeout etc.)
            self.logger.error(f"AIOHTTP client error fetching from {fetch_url}: {e_client}")
            return []
        except Exception as e_general: # Catch-all for other unexpected errors (e.g., JSON parsing if not caught by content_type check)
            self.logger.error(f"Unexpected error fetching or parsing data from {fetch_url}: {e_general}", exc_info=True)
            return []

    def _to_tensors(self, data: List[Dict[str, Any]]) -> Dict[str, Any]: # Return Any for dict value type
        if not data:
            self.logger.warning("_to_tensors received empty data list.")
            return {}

        batch_texts = []
        # Store prompt token lengths to be included in the output batch
        batch_prompt_token_lengths: List[int] = []

        for group in data:
            prompt = group.get("prompt", "")
            response = group.get("response", "")
            text_sequence = prompt + response
            batch_texts.append(text_sequence)

            # Calculate prompt_token_len using the tokenizer
            # This should be the length of the prompt part *as it appears in the tokenized text_sequence*
            # For simplicity and consistency with current _to_tensors, tokenize prompt separately without special tokens
            # that are only added to the full sequence.
            # This calculation must be very robust for real models and tokenizers.
            try:
                # If tokenizer is HF, encode_plus might be better for just getting tokens.
                # For DummyTokenizer or TestTokenizer, .encode() is fine.
                if hasattr(self.tokenizer, 'encode') and callable(self.tokenizer.encode):
                    # Assuming encode returns a list of token IDs or a dict with "input_ids"
                    prompt_only_encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
                    if isinstance(prompt_only_encoded, dict) and "input_ids" in prompt_only_encoded:
                        # This case was added to TestTokenizer in a recent test update
                        prompt_token_len_val = len(prompt_only_encoded["input_ids"])
                    elif isinstance(prompt_only_encoded, list):
                         prompt_token_len_val = len(prompt_only_encoded)
                    else:
                        self.logger.warning(f"Unexpected output from tokenizer.encode for prompt: {prompt_only_encoded}. Defaulting prompt length to 0.")
                        prompt_token_len_val = 0
                else: # Fallback if tokenizer doesn't have a simple .encode()
                    self.logger.warning("Tokenizer does not have a simple .encode() method for prompt length calculation. Defaulting to 0.")
                    prompt_token_len_val = 0
            except Exception as e_tok:
                self.logger.error(f"Error tokenizing prompt '{prompt}' separately: {e_tok}. Defaulting prompt length to 0.")
                prompt_token_len_val = 0
            batch_prompt_token_lengths.append(prompt_token_len_val)

        try:
            tokenized_batch = self.tokenizer.batch_encode_plus(
                batch_texts, add_special_tokens=True, padding="max_length",
                truncation=True, max_length=self.max_seq_len,
                return_tensors="pt", return_attention_mask=True
            )
        except Exception as e:
            self.logger.error(f"Error during batch tokenization: {e}", exc_info=True)
            return {}

        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        current_batch_size = input_ids.size(0)

        advantages = torch.zeros(current_batch_size, self.max_seq_len, dtype=torch.float32)
        old_logprobs = torch.zeros(current_batch_size, self.max_seq_len, dtype=torch.float32)

        for i in range(current_batch_size):
            group = data[i]
            prompt_token_len_val = batch_prompt_token_lengths[i] # Use pre-calculated length
            actual_seq_len = int(attention_mask[i].sum())

            token_advantages_data = group.get("token_advantages")
            reward_data = group.get("reward")
            logprobs_data = group.get("logprobs")

            for j in range(actual_seq_len):
                if j < prompt_token_len_val:
                    advantages[i, j] = 0.0
                    if logprobs_data and j < len(logprobs_data):
                        old_logprobs[i, j] = float(logprobs_data[j])
                    else:
                        old_logprobs[i, j] = 0.0
                else:
                    response_token_index = j - prompt_token_len_val
                    if token_advantages_data and response_token_index < len(token_advantages_data):
                        advantages[i, j] = float(token_advantages_data[response_token_index])
                    elif reward_data is not None:
                        advantages[i, j] = float(reward_data)
                    else:
                        advantages[i, j] = 0.0

                    if logprobs_data:
                        # This logic for logprobs needs to be clear: are they for all tokens or just response?
                        # Assuming logprobs_data, if present, covers the response tokens primarily.
                        # If its length matches response token count:
                        # num_response_tokens_expected = actual_seq_len - prompt_token_len_val
                        # if len(logprobs_data) == num_response_tokens_expected and response_token_index < len(logprobs_data):
                        #    old_logprobs[i,j] = float(logprobs_data[response_token_index])
                        # elif j < len(logprobs_data): # Or if it's for the whole sequence
                        #    old_logprobs[i,j] = float(logprobs_data[j])
                        # For now, simplified: assume logprobs are for response if token_advantages is not present (as per previous logic)
                        # This part is still a bit ambiguous from requirements.
                        # Using the logic from previous version of _to_tensors for old_logprobs:
                        if response_token_index < len(logprobs_data): # Assumes logprobs are for response if provided
                             old_logprobs[i, j] = float(logprobs_data[response_token_index])
                        elif j < len(logprobs_data) and not token_advantages_data: # Fallback if logprobs might be full seq
                             old_logprobs[i, j] = float(logprobs_data[j])
                        else:
                             old_logprobs[i, j] = 0.0
                    else:
                        old_logprobs[i, j] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "advantages": advantages,
            "old_logprobs": old_logprobs,
            "prompt_token_lengths": batch_prompt_token_lengths # Add this to the output
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
