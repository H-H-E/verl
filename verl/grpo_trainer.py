import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional


def _logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Helper that returns log-probabilities of the selected `labels` tokens.
    logits: (B, T, V)
    labels: (B, T)
    returns: (B, T)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    # gather log_probs at label indices
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def run_grpo_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    ref_model: Optional[torch.nn.Module] = None,
    kl_coeff: float = 0.1,
    max_grad_norm: Optional[float] = 1.0,
) -> float:
    """Performs one GRPO update step.

    Args:
        model: The policy model to update.
        optimizer: Its optimizer.
        batch: Dict containing keys at least:
            * input_ids (B, T)
            * attention_mask (B, T)
            * advantages (B, T)
            * old_logprobs (B, T)
        ref_model: Frozen reference model for KL term. If None, KL term is 0.
        kl_coeff: Coefficient for KL penalty.
        max_grad_norm: If set, gradients are clipped.

    Returns:
        Scalar loss value (float).
    """
    model.train()
    optimizer.zero_grad()

    input_ids = batch["input_ids"]  # (B, T)
    attention_mask = batch["attention_mask"]  # (B, T)
    advantages = batch["advantages"]  # (B, T)
    old_logprobs = batch["old_logprobs"]  # (B, T)

    # Forward pass
    outputs = model(input_ids)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs  # (B, T, V)

    new_logprobs = _logprobs_from_logits(logits, input_ids)

    # Importance sampling ratio
    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)

    # Policy gradient loss (negative expected advantage-weighted logprob ratio)
    pg_loss = -(ratio * advantages * attention_mask).sum() / attention_mask.sum()

    # KL penalty if reference model provided
    if ref_model is not None:
        with torch.no_grad():
            ref_logits = ref_model(input_ids)
            if isinstance(ref_logits, tuple):
                ref_logits = ref_logits[0]
        ref_logprobs = _logprobs_from_logits(ref_logits, input_ids)
        kl_div = (ref_logprobs - new_logprobs) * attention_mask
        kl_div = kl_div.sum() / attention_mask.sum()
    else:
        kl_div = torch.tensor(0.0, device=input_ids.device)

    loss = pg_loss + kl_coeff * kl_div

    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return loss.item() 