# tests/test_loss_mask.py
import pytest
import torch
from verl.train.utils import build_loss_mask

def test_mask_simple_case():
    input_ids = torch.tensor([[10, 11, 12, 13],
                              [20, 21, 22, 23]], dtype=torch.long)
    # For first sample, prompt is [10, 11], response is [12, 13]. First response token index = 2.
    # For second sample, prompt is [20], response is [21, 22, 23]. First response token index = 1.
    prompt_end_positions = [2, 1]
    expected_mask = torch.tensor([[0., 0., 1., 1.],
                                  [0., 1., 1., 1.]], dtype=torch.float32)
    loss_mask = build_loss_mask(input_ids, prompt_end_positions)
    assert torch.equal(loss_mask, expected_mask)

def test_mask_all_prompt():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    # All tokens are prompt, so the first response token would be at index 4 (end of sequence).
    prompt_end_positions = [4]
    expected_mask = torch.tensor([[0., 0., 0., 0.]], dtype=torch.float32)
    loss_mask = build_loss_mask(input_ids, prompt_end_positions)
    assert torch.equal(loss_mask, expected_mask)

def test_mask_all_response():
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    # All tokens are response, so the first response token is at index 0.
    prompt_end_positions = [0]
    expected_mask = torch.tensor([[1., 1., 1., 1.]], dtype=torch.float32)
    loss_mask = build_loss_mask(input_ids, prompt_end_positions)
    assert torch.equal(loss_mask, expected_mask)

def test_mask_mixed_batch():
    input_ids = torch.tensor([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]], dtype=torch.long)
    # Sample 1: prompt [1,2], response [3,4,5]. First response token index = 2.
    # Sample 2: prompt [], response [6,7,8,9,10]. First response token index = 0.
    # Sample 3: prompt [11,12,13,14,15], response []. First response token index = 5.
    prompt_end_positions = [2, 0, 5]
    expected_mask = torch.tensor([
        [0., 0., 1., 1., 1.],
        [1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 0.]
    ], dtype=torch.float32)
    loss_mask = build_loss_mask(input_ids, prompt_end_positions)
    assert torch.equal(loss_mask, expected_mask)

def test_invalid_positions_out_of_range_positive():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long) # seq_length = 3
    # Position 4 is invalid because valid indices are 0, 1, 2, and 3 (for all prompt).
    prompt_end_positions = [4]
    with pytest.raises(ValueError, match="out of range \\[0, 3\\]"):
        build_loss_mask(input_ids, prompt_end_positions)

def test_invalid_positions_out_of_range_negative():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    prompt_end_positions = [-1]
    with pytest.raises(ValueError, match="out of range \\[0, 3\\]"): # The message might vary slightly based on Python/PyTorch version
        build_loss_mask(input_ids, prompt_end_positions)

def test_mismatched_lengths_prompt_positions():
    input_ids = torch.tensor([[1,2],[3,4]], dtype=torch.long) # batch_size = 2
    prompt_end_positions = [1] # length 1, does not match batch_size
    with pytest.raises(ValueError, match="must match batch_size of input_ids"):
        build_loss_mask(input_ids, prompt_end_positions)

def test_invalid_input_types():
    # Test wrong type for input_ids
    with pytest.raises(TypeError, match="input_ids must be a torch.Tensor"):
        build_loss_mask([[1,2,3]], [1]) # input_ids as list of lists

    input_ids_good = torch.tensor([[1,2,3]], dtype=torch.long)

    # Test wrong type for prompt_end_positions
    with pytest.raises(TypeError, match="prompt_end_positions must be a list"):
        build_loss_mask(input_ids_good, "not_a_list")

    # Test wrong type for elements within prompt_end_positions
    with pytest.raises(TypeError, match="Elements of prompt_end_positions must be integers"):
        build_loss_mask(input_ids_good, [1.0]) # float instead of int

def test_empty_batch():
    input_ids = torch.empty((0, 5), dtype=torch.long) # Batch size 0
    prompt_end_positions = []
    # Expect an empty mask of shape (0, 5)
    expected_mask = torch.empty((0, 5), dtype=torch.float32)
    loss_mask = build_loss_mask(input_ids, prompt_end_positions)
    assert torch.equal(loss_mask, expected_mask)

def test_different_devices():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        input_ids_cuda = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
        prompt_end_positions = [1]
        expected_mask_cuda = torch.tensor([[0., 1., 1.]], dtype=torch.float32, device=device)

        loss_mask = build_loss_mask(input_ids_cuda, prompt_end_positions)
        assert torch.equal(loss_mask, expected_mask_cuda)
        assert loss_mask.device == device, "Mask should be on the same device as input_ids"
    else:
        pytest.skip("CUDA not available, skipping device test for build_loss_mask")
