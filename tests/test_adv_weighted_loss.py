# tests/test_adv_weighted_loss.py
import pytest
import torch
import torch.nn.functional as F
from verl.train.losses import advantage_weighted_loss

def _get_dummy_inputs(batch_size, seq_len, vocab_size, device="cpu", requires_grad=False):
    curr_logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=torch.float32)
    if requires_grad:
        curr_logits.requires_grad_(True)

    action_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    # Create somewhat realistic old_logprobs (e.g., from a slightly different policy)
    # This ensures ratios are not always 1.
    with torch.no_grad():
        simulated_old_logits = curr_logits.detach() + torch.randn_like(curr_logits) * 0.5
        realistic_old_logprobs = torch.gather(
            F.log_softmax(simulated_old_logits, dim=-1),
            dim=-1,
            index=action_ids.unsqueeze(-1)
        ).squeeze(-1)

    old_logprobs = realistic_old_logprobs
    advantages = torch.randn(batch_size, seq_len, device=device, dtype=torch.float32)
    loss_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32) # Default to all ones

    return curr_logits, action_ids, old_logprobs, advantages, loss_mask

def test_no_clipping_and_basic_loss():
    B, T, V = 1, 2, 3
    device = "cpu"
    # Construct logits such that curr_logprobs will equal old_logprobs (ratio = 1)
    curr_logits = torch.rand(B, T, V, device=device, dtype=torch.float32)
    action_ids = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    with torch.no_grad():
        curr_logprobs_values = torch.gather(F.log_softmax(curr_logits, dim=-1), -1, action_ids.unsqueeze(-1)).squeeze(-1)

    old_logprobs = curr_logprobs_values.clone().detach() # Makes ratio = 1
    advantages = torch.tensor([[1.0, 2.0]], device=device, dtype=torch.float32)
    loss_mask = torch.tensor([[1.0, 1.0]], device=device, dtype=torch.float32)
    clip_ratio = 0.2
    entropy_coef = 0.0

    # When ratio = 1, surr1 = advantages, surr2 = clamp(1, 0.8, 1.2)*advantages = advantages
    # So, policy_loss_per_token = -advantages
    # Mean policy loss = sum(-advantages * loss_mask) / sum(loss_mask)
    expected_policy_loss_sum = (-advantages * loss_mask).sum()
    num_active_tokens = loss_mask.sum()
    expected_loss = expected_policy_loss_sum / num_active_tokens

    computed_loss = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss, expected_loss)

def test_clipping_behavior_positive_advantage():
    B, T, V = 1, 1, 3
    device = "cpu"
    # Manually set logits and old_logprobs to create a specific ratio
    # curr_logprobs for action 0: log(softmax([1, 0.1, 0.01]))[0] where raw logits are e.g. [0, -2.3, -4.6]
    # Let curr_logits lead to curr_logprob_action0 = log(0.7) approx -0.35
    # Let old_logprob_action0 = log(0.7 / 1.5) = log(0.7) - log(1.5) approx -0.35 - 0.40 = -0.75
    # This makes ratio = exp(curr - old) = exp(log(1.5)) = 1.5
    curr_logits = torch.tensor([[[0.0, -2.3025, -4.605]]], device=device, dtype=torch.float32) # approx softmax [0.7, 0.1, 0.01]
    action_ids = torch.tensor([[0]], device=device, dtype=torch.long)

    with torch.no_grad():
        curr_logprobs_val = torch.gather(F.log_softmax(curr_logits, dim=-1), -1, action_ids.unsqueeze(-1)).squeeze(-1)

    ratio_val = 1.5 # Target ratio
    old_logprobs = curr_logprobs_val - torch.log(torch.tensor(ratio_val, device=device))

    advantages = torch.tensor([[10.0]], device=device, dtype=torch.float32) # Positive advantage
    loss_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    clip_ratio = 0.2 # Clip range [0.8, 1.2]
    entropy_coef = 0.0

    # ratio = 1.5, advantages = 10.0
    # surr1 = 1.5 * 10.0 = 15.0
    # surr2 = clamp(1.5, 0.8, 1.2) * 10.0 = 1.2 * 10.0 = 12.0
    # policy_loss_per_token = -min(15.0, 12.0) = -12.0
    # mean_policy_loss = -12.0 / 1 = -12.0
    expected_loss_val = torch.tensor(-12.0, device=device, dtype=torch.float32)

    computed_loss = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss, expected_loss_val)

def test_entropy_bonus_calculation():
    B, T, V = 1, 2, 3
    device = "cpu"
    # curr_logits = 0 means uniform distribution after softmax (probs = [1/3, 1/3, 1/3])
    curr_logits = torch.zeros(B, T, V, device=device, dtype=torch.float32)
    action_ids = torch.zeros(B, T, device=device, dtype=torch.long) # Doesn't matter for this test
    old_logprobs = torch.zeros(B, T, device=device, dtype=torch.float32) # Ratio will be non-zero, but adv is 0
    advantages = torch.zeros(B, T, device=device, dtype=torch.float32) # Makes policy loss part zero
    loss_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    clip_ratio = 0.2
    entropy_coef = 0.5

    # For uniform distribution over V classes, entropy is log(V)
    expected_entropy_per_token = torch.log(torch.tensor(V, device=device, dtype=torch.float32))
    # Policy loss is 0 because advantages are 0
    # Total loss = 0 - entropy_coef * mean_masked_entropy
    # mean_masked_entropy = sum(entropy_per_token * loss_mask) / sum(loss_mask)
    # Since loss_mask is all 1s, mean_masked_entropy = expected_entropy_per_token
    expected_loss = -entropy_coef * expected_entropy_per_token

    computed_loss = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss, expected_loss)

def test_loss_mask_application():
    B, T, V = 1, 4, 2
    device = "cpu"
    # Use requires_grad=True to ensure intermediate tensors also have it if needed by autograd for the sub-problem
    curr_logits, action_ids, old_logprobs, advantages, _ = _get_dummy_inputs(B,T,V,device, requires_grad=True)
    loss_mask = torch.tensor([[0., 0., 1., 1.]], device=device, dtype=torch.float32) # Mask first two tokens
    clip_ratio = 0.2
    entropy_coef = 0.1

    loss_with_mask = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )

    # Calculate expected loss only on the unmasked part (last two tokens)
    unmasked_logits_part = curr_logits[:, 2:, :].clone()
    unmasked_action_ids_part = action_ids[:, 2:].clone()
    unmasked_old_logprobs_part = old_logprobs[:, 2:].clone()
    unmasked_advantages_part = advantages[:, 2:].clone()
    # For the sub-problem, the mask is all ones
    unmasked_loss_mask_part = torch.ones_like(unmasked_advantages_part, dtype=torch.float32)

    expected_loss_for_unmasked_part = advantage_weighted_loss(
        unmasked_logits_part, unmasked_action_ids_part,
        unmasked_old_logprobs_part, unmasked_advantages_part,
        unmasked_loss_mask_part,
        clip_ratio, entropy_coef
    )
    assert torch.isclose(loss_with_mask, expected_loss_for_unmasked_part)

def test_zero_active_tokens_in_mask():
    B, T, V = 1, 2, 3
    device = "cpu"
    curr_logits, action_ids, old_logprobs, advantages, _ = _get_dummy_inputs(B,T,V,device)
    loss_mask = torch.zeros(B, T, device=device, dtype=torch.float32) # Mask all tokens
    clip_ratio = 0.2
    entropy_coef = 0.1

    computed_loss = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )
    # Expected loss is 0.0 because num_active_tokens is 0 for both policy and entropy parts
    assert torch.isclose(computed_loss, torch.tensor(0.0, device=device, dtype=computed_loss.dtype))

def test_loss_requires_grad():
    B, T, V = 1, 2, 3
    # Get inputs, ensuring curr_logits requires grad
    curr_logits, action_ids, old_logprobs, advantages, loss_mask = _get_dummy_inputs(B,T,V, requires_grad=True)

    loss = advantage_weighted_loss(curr_logits, action_ids, old_logprobs, advantages, loss_mask, 0.2, 0.01)
    assert loss.requires_grad, "Loss should require grad if curr_logits requires grad."

    # Test backward pass
    try:
        loss.backward()
        assert curr_logits.grad is not None, "Gradients should be computed for curr_logits."
    except RuntimeError as e:
        pytest.fail(f"Backward pass failed: {e}")

def test_clipping_behavior_negative_advantage():
    B, T, V = 1, 1, 3
    device = "cpu"
    # Ratio > 1 + clip_ratio (e.g., 1.5)
    curr_logits = torch.tensor([[[0.0, -2.3025, -4.605]]], device=device, dtype=torch.float32)
    action_ids = torch.tensor([[0]], device=device, dtype=torch.long)
    with torch.no_grad():
        curr_logprobs_val = torch.gather(F.log_softmax(curr_logits, dim=-1), -1, action_ids.unsqueeze(-1)).squeeze(-1)
    ratio_val = 1.5
    old_logprobs = curr_logprobs_val - torch.log(torch.tensor(ratio_val, device=device))

    advantages = torch.tensor([[-10.0]], device=device, dtype=torch.float32) # Negative advantage
    loss_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    clip_ratio = 0.2 # Clip range [0.8, 1.2]
    entropy_coef = 0.0

    # ratio = 1.5, advantages = -10.0
    # surr1 = 1.5 * -10.0 = -15.0
    # surr2 = clamp(1.5, 0.8, 1.2) * -10.0 = 1.2 * -10.0 = -12.0
    # With negative advantages, min(surr1, surr2) = min(-15.0, -12.0) = -15.0 (this is where PPO maximizes the objective)
    # policy_loss_per_token = -(-15.0) = 15.0
    expected_loss_val = torch.tensor(15.0, device=device, dtype=torch.float32)

    computed_loss = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss, expected_loss_val), f"Expected {expected_loss_val}, got {computed_loss}"

def test_clipping_behavior_ratio_less_than_1_minus_clip():
    B, T, V = 1, 1, 3
    device = "cpu"
    # Ratio < 1 - clip_ratio (e.g., 0.5)
    curr_logits = torch.tensor([[[0.0, -2.3025, -4.605]]], device=device, dtype=torch.float32)
    action_ids = torch.tensor([[0]], device=device, dtype=torch.long)
    with torch.no_grad():
        curr_logprobs_val = torch.gather(F.log_softmax(curr_logits, dim=-1), -1, action_ids.unsqueeze(-1)).squeeze(-1)
    ratio_val = 0.5
    old_logprobs = curr_logprobs_val - torch.log(torch.tensor(ratio_val, device=device))

    advantages_pos = torch.tensor([[10.0]], device=device, dtype=torch.float32) # Positive advantage
    advantages_neg = torch.tensor([[-10.0]], device=device, dtype=torch.float32) # Negative advantage
    loss_mask = torch.ones(B, T, device=device, dtype=torch.float32)
    clip_ratio = 0.2 # Clip range [0.8, 1.2]
    entropy_coef = 0.0

    # Case 1: Positive Advantage
    # ratio = 0.5, advantages = 10.0
    # surr1 = 0.5 * 10.0 = 5.0
    # surr2 = clamp(0.5, 0.8, 1.2) * 10.0 = 0.8 * 10.0 = 8.0
    # policy_loss_per_token = -min(5.0, 8.0) = -5.0
    expected_loss_pos_adv = torch.tensor(-5.0, device=device, dtype=torch.float32)
    computed_loss_pos_adv = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages_pos, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss_pos_adv, expected_loss_pos_adv)

    # Case 2: Negative Advantage
    # ratio = 0.5, advantages = -10.0
    # surr1 = 0.5 * -10.0 = -5.0
    # surr2 = clamp(0.5, 0.8, 1.2) * -10.0 = 0.8 * -10.0 = -8.0
    # policy_loss_per_token = -min(-5.0, -8.0) = -(-8.0) = 8.0
    expected_loss_neg_adv = torch.tensor(8.0, device=device, dtype=torch.float32)
    computed_loss_neg_adv = advantage_weighted_loss(
        curr_logits, action_ids, old_logprobs, advantages_neg, loss_mask, clip_ratio, entropy_coef
    )
    assert torch.isclose(computed_loss_neg_adv, expected_loss_neg_adv)
