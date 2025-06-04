import torch
import torch.nn as nn
from verl.grpo_trainer import run_grpo_step


class TinyPolicy(nn.Module):
    """A tiny causal language-model-like module for testing."""
    def __init__(self, vocab_size: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):  # (B, T)
        h = self.embed(input_ids)
        return self.lm_head(h)  # (B, T, V)


def _dummy_batch(batch_size: int = 2, seq_len: int = 4, vocab: int = 10):
    torch.manual_seed(0)
    input_ids = torch.randint(low=0, high=vocab, size=(batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    advantages = torch.randn(batch_size, seq_len)
    old_logprobs = torch.randn(batch_size, seq_len)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "advantages": advantages,
        "old_logprobs": old_logprobs,
    }


def test_run_grpo_step_updates_params():
    batch = _dummy_batch()
    model = TinyPolicy(vocab_size=10, hidden=8)
    ref_model = TinyPolicy(vocab_size=10, hidden=8)
    ref_model.load_state_dict(model.state_dict())  # identical to start
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Snapshot parameters
    before = [p.detach().clone() for p in model.parameters()]

    loss_val = run_grpo_step(model, optimizer, batch, ref_model=ref_model, kl_coeff=0.5)

    assert isinstance(loss_val, float)

    # Ensure at least one param changed
    changed = any(not torch.allclose(b, p, atol=1e-6) for b, p in zip(before, model.parameters()))
    assert changed, "Model parameters did not update after GRPO step"


def test_optimizer_step_called():
    batch = _dummy_batch()
    model = TinyPolicy(vocab_size=10, hidden=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Monkey-patch optimizer.step to track call
    called = {"step": False}

    def _wrapped_step(*args, **kwargs):
        called["step"] = True
        return original_step(*args, **kwargs)

    original_step = optimizer.step
    optimizer.step = _wrapped_step  # type: ignore

    run_grpo_step(model, optimizer, batch, ref_model=None)
    assert called["step"], "optimizer.step() was not called inside run_grpo_step" 