# tests/test_kl_penalty.py
import pytest
import torch
import torch.nn.functional as F
from verl.train.losses import kl_penalty_loss

def test_identical_logits():
    B, T, V = 2, 5, 10
    device = "cpu"
    curr_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
    # ref_logits are identical to curr_logits
    ref_logits = curr_logits.clone() # .detach() is handled inside the function for ref_logits if needed

    kl_loss = kl_penalty_loss(curr_logits, ref_logits)

    # KL divergence between identical distributions should be 0
    assert torch.isclose(kl_loss, torch.tensor(0.0, device=device, dtype=torch.float32), atol=1e-6), \
        f"KL loss should be near zero for identical logits, but got {kl_loss.item()}"

def test_shifted_logits_produces_nonzero_kl():
    B, T, V = 2, 5, 10
    device = "cpu"
    curr_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
    ref_logits = curr_logits.clone() # Start with identical ref_logits

    # Modify curr_logits to make them different from ref_logits
    curr_logits_shifted = curr_logits.clone()
    shift_index = V // 2 # Pick an arbitrary vocabulary index to shift
    curr_logits_shifted[..., shift_index] += 5.0 # Make a significant shift

    kl_loss = kl_penalty_loss(curr_logits_shifted, ref_logits)

    # KL divergence should be positive for different distributions
    assert kl_loss.item() > 0, \
        f"KL loss should be positive for different logits, but got {kl_loss.item()}"
    # It should also not be NaN or Inf
    assert not torch.isnan(kl_loss).item() and not torch.isinf(kl_loss).item(), \
        f"KL loss should be a finite number, but got {kl_loss.item()}"

def test_kl_divergence_value_simple_case():
    device = "cpu"
    # Reference distribution P_ref: p(0)=0.5, p(1)=0.5
    # Achieved by ref_logits = [[0.0, 0.0]] -> softmax -> [[0.5, 0.5]]
    ref_logits = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, device=device)

    # Current distribution P_curr: p(0)=0.1, p(1)=0.9
    # To get this, we need log(0.1/0.9) = log(1/9) = -log(9) for the second logit if first is 0.
    # Or, if curr_logits = [L1, L2], then exp(L1)/(exp(L1)+exp(L2)) = 0.1, exp(L2)/(exp(L1)+exp(L2)) = 0.9
    # Let L1 = 0. Then exp(L2)/(1+exp(L2)) = 0.9 => exp(L2) = 0.9 + 0.9exp(L2) => 0.1exp(L2) = 0.9 => exp(L2)=9 => L2=log(9)
    curr_logits_l2 = torch.log(torch.tensor(9.0, device=device))
    curr_logits = torch.tensor([[[0.0, curr_logits_l2.item()]]], dtype=torch.float32, device=device)

    # Expected KL(P_curr || P_ref) = P_curr(0)*log(P_curr(0)/P_ref(0)) + P_curr(1)*log(P_curr(1)/P_ref(1))
    # P_curr(0)=0.1, P_ref(0)=0.5
    # P_curr(1)=0.9, P_ref(1)=0.5
    expected_kl = (0.1 * torch.log(torch.tensor(0.1/0.5, device=device)) +
                   0.9 * torch.log(torch.tensor(0.9/0.5, device=device)))

    kl_loss = kl_penalty_loss(curr_logits, ref_logits)

    assert torch.isclose(kl_loss, expected_kl, atol=1e-5), \
        f"Expected KL {expected_kl.item()}, but got {kl_loss.item()}"

def test_kl_shape_mismatch():
    B, T, V = 2, 5, 10
    device = "cpu"
    curr_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)

    # Wrong vocabulary size
    ref_logits_wrong_shape_v = torch.randn(B, T, V + 1, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="must match ref_logits shape"):
        kl_penalty_loss(curr_logits, ref_logits_wrong_shape_v)

    # Wrong sequence length
    ref_logits_wrong_shape_t = torch.randn(B, T - 1, V, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="must match ref_logits shape"):
        kl_penalty_loss(curr_logits, ref_logits_wrong_shape_t)

    # Wrong batch size
    ref_logits_wrong_shape_b = torch.randn(B - 1, T, V, device=device, dtype=torch.float32)
    with pytest.raises(ValueError, match="must match ref_logits shape"):
        kl_penalty_loss(curr_logits, ref_logits_wrong_shape_b)


def test_kl_loss_requires_grad_curr_logits():
    B, T, V = 1, 2, 3
    device = "cpu"
    curr_logits = torch.randn(B,T,V, device=device, dtype=torch.float32, requires_grad=True)
    # ref_logits should not require grad, or if they do, they will be detached inside kl_penalty_loss
    ref_logits_no_grad = torch.randn(B,T,V, device=device, dtype=torch.float32)

    kl_loss = kl_penalty_loss(curr_logits, ref_logits_no_grad)

    assert kl_loss.requires_grad, "KL loss should require gradients wrt curr_logits."

    # Test if gradient actually flows to curr_logits
    # Need to make sure kl_loss is not zero, otherwise grad might be None or zero.
    # If curr_logits and ref_logits_no_grad are different (which they are by randn), kl_loss > 0.

    try:
        kl_loss.backward()
        assert curr_logits.grad is not None, "curr_logits should have gradients after backward() call."
        # Check that ref_logits_no_grad does not have gradients
        assert ref_logits_no_grad.grad is None, "ref_logits_no_grad should not have gradients."
    except RuntimeError as e:
        pytest.fail(f"Backward pass failed: {e}")

def test_kl_loss_ref_logits_grad_detached():
    B, T, V = 1, 2, 3
    device = "cpu"
    curr_logits = torch.randn(B,T,V, device=device, dtype=torch.float32, requires_grad=True)
    # ref_logits created with requires_grad=True to test if it's properly detached
    ref_logits_with_grad = torch.randn(B,T,V, device=device, dtype=torch.float32, requires_grad=True)

    kl_loss = kl_penalty_loss(curr_logits, ref_logits_with_grad)

    assert kl_loss.requires_grad, "KL loss should require gradients wrt curr_logits."

    try:
        kl_loss.backward()
        assert curr_logits.grad is not None, "curr_logits should have gradients."
        assert ref_logits_with_grad.grad is None, "ref_logits_with_grad should NOT have gradients due to detach inside function."
    except RuntimeError as e:
        pytest.fail(f"Backward pass failed: {e}")

# Test with different devices if CUDA is available
def test_kl_loss_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test for kl_penalty_loss")

    device = torch.device("cuda")
    B, T, V = 2, 5, 10

    curr_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
    ref_logits = curr_logits.clone()

    kl_loss = kl_penalty_loss(curr_logits, ref_logits)

    assert kl_loss.device == device, "Output tensor should be on the same device as input."
    assert torch.isclose(kl_loss, torch.tensor(0.0, device=device, dtype=torch.float32), atol=1e-6)

```
