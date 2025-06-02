# verl/train/losses.py
import torch
import torch.nn.functional as F # For log_softmax and softmax

def advantage_weighted_loss(
    curr_logits: torch.Tensor,    # Shape: [B, T, V] - Logits from the current policy
    action_ids: torch.Tensor,     # Shape: [B, T] - IDs of actions taken (for which old_logprobs are given)
    old_logprobs: torch.Tensor,   # Shape: [B, T] - Log-probabilities of action_ids under the old policy
    advantages: torch.Tensor,     # Shape: [B, T] - Per-token advantages
    loss_mask: torch.Tensor,      # Shape: [B, T] - Mask with 1s for positions to include in loss (e.g., response tokens)
    clip_ratio: float,            # PPO clipping parameter (e.g., 0.2)
    entropy_coef: float = 0.0     # Weight for the entropy bonus
) -> torch.Tensor:
    """
    Computes the PPO-style advantage-weighted loss using the clipped surrogate objective.

    Args:
        curr_logits: Logits from the current policy, shape [B, T, V].
        action_ids: IDs of actions taken, shape [B, T].
        old_logprobs: Log-probabilities of action_ids under the old policy, shape [B, T].
        advantages: Per-token advantages, shape [B, T].
        loss_mask: Mask for loss calculation (1 for response, 0 for prompt/padding), shape [B, T].
        clip_ratio: PPO clipping parameter.
        entropy_coef: Coefficient for the entropy bonus.

    Returns:
        A scalar tensor representing the final computed loss.
    """

    device = curr_logits.device
    action_ids = action_ids.to(device)
    old_logprobs = old_logprobs.to(device)
    advantages = advantages.to(device)
    loss_mask = loss_mask.to(device)

    # Calculate current log-probabilities for the taken actions
    # log_softmax along the vocabulary dimension (V)
    log_probs_all_actions = F.log_softmax(curr_logits, dim=-1) 
    # Gather the log-probabilities of the specific actions taken
    curr_logprobs = torch.gather(log_probs_all_actions, dim=-1, index=action_ids.unsqueeze(-1)).squeeze(-1)

    # Calculate the importance ratio
    # r = exp(curr_logprobs - old_logprobs)
    # Detach old_logprobs as they are constants from the behavior policy
    ratios = torch.exp(curr_logprobs - old_logprobs.detach())

    # PPO Clipped Surrogate Objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    
    # The PPO loss is the negative of the objective function (we minimize loss)
    # Taking the element-wise minimum ensures a pessimistic bound.
    policy_loss_per_token = -torch.min(surr1, surr2)

    # Apply loss mask
    masked_policy_loss = policy_loss_per_token * loss_mask
    
    # Normalize by the number of active (masked) tokens
    num_active_tokens = loss_mask.sum()
    if num_active_tokens > 0:
        mean_policy_loss = masked_policy_loss.sum() / num_active_tokens
    else:
        # Avoid division by zero if mask is all zeros (e.g., all prompt batch or empty batch)
        mean_policy_loss = torch.tensor(0.0, device=device, dtype=curr_logits.dtype)
        
    # Entropy Bonus (optional)
    if entropy_coef > 0:
        # Calculate entropy of the current policy's action distribution
        probs_for_entropy = F.softmax(curr_logits, dim=-1)
        # Add a small epsilon to prevent log(0) for numerical stability, though log_softmax handles -inf
        # log_probs_for_entropy = F.log_softmax(curr_logits, dim=-1) 
        # However, if probs_for_entropy can be exactly 0, probs * log_probs can be NaN if log_probs is -inf.
        # So, it's safer to compute log_probs from probs with epsilon or use log_softmax output carefully.
        # PyTorch's Categorical distribution entropy uses: -sum(probs * log_probs)
        # Let's ensure log_probs are from log_softmax for stability.
        log_probs_for_entropy = F.log_softmax(curr_logits, dim=-1) # Re-use or ensure it's available
        
        entropy_per_token = -torch.sum(probs_for_entropy * log_probs_for_entropy, dim=-1) # Sum over vocab dim
        
        masked_entropy = entropy_per_token * loss_mask
        
        if num_active_tokens > 0:
            mean_masked_entropy = masked_entropy.sum() / num_active_tokens
        else:
            mean_masked_entropy = torch.tensor(0.0, device=device, dtype=curr_logits.dtype)
    else:
        mean_masked_entropy = torch.tensor(0.0, device=device, dtype=curr_logits.dtype)

    # Total loss
    total_loss = mean_policy_loss - (entropy_coef * mean_masked_entropy)
            
    return total_loss

# Placeholder for kl_penalty_loss (Task 2.4) - Now implementing it.
def kl_penalty_loss(
    curr_logits: torch.Tensor,    # Shape: [B, T, V] - Logits from the current policy
    ref_logits: torch.Tensor     # Shape: [B, T, V] - Logits from the reference policy
) -> torch.Tensor:
    """
    Computes the KL divergence penalty between the current policy and a reference policy.
    KL(current_policy || reference_policy) per token, then averaged over all tokens.

    Args:
        curr_logits: Logits from the current policy, shape [B, T, V].
        ref_logits: Logits from the reference policy, shape [B, T, V].
                     These should be detached from the computation graph if the reference model is fixed.

    Returns:
        A scalar tensor representing the mean KL divergence per token.
    """
    if curr_logits.shape != ref_logits.shape:
        raise ValueError(
            f"curr_logits shape {curr_logits.shape} must match ref_logits shape {ref_logits.shape}."
        )

    # It's good practice to ensure ref_logits don't contribute to gradients if it's a fixed reference
    ref_logits_detached = ref_logits.detach()

    curr_log_probs = F.log_softmax(curr_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits_detached, dim=-1) # Use detached ref_logits

    # curr_probs = F.softmax(curr_logits, dim=-1) # Can be calculated from curr_log_probs too
    curr_probs = torch.exp(curr_log_probs)

    # KL divergence D_KL(P || Q) = sum P(x) * (log P(x) - log Q(x))
    # Here, P is current policy, Q is reference policy.
    kl_div_per_element = curr_probs * (curr_log_probs - ref_log_probs)
    
    # Sum over the vocabulary dimension (V) to get KL divergence per token position
    kl_div_per_token = torch.sum(kl_div_per_element, dim=-1)

    # Average over all tokens in the batch (batch_size * seq_len)
    # This assumes KL penalty is applied to all tokens, including prompt/padding.
    # If a mask is needed, it should be passed as an argument.
    mean_kl_div = kl_div_per_token.mean()
            
    return mean_kl_div
