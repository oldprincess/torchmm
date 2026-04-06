import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_add_cuda_matches_cpu_reference():
    lhs = torch.tensor([[[[1, 0, 0], [2, 0, 0], [3, 0, 0]]]], dtype=torch.int64)
    rhs = torch.tensor([[[[4, 0, 0]], [[5, 0, 0]]]], dtype=torch.int64)

    out = torchmm.wide_add(lhs.cuda(), rhs.cuda()).cpu()
    expected = torchmm.wide_add(lhs, rhs)

    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_add_cuda_adapts_common_logical_dims():
    lhs = torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.int64)
    rhs = torch.tensor(
        [[[10, 0], [20, 0], [30, 0]], [[40, 0], [50, 0], [60, 0]]],
        dtype=torch.int64,
    )

    out = torchmm.wide_add(lhs.cuda(), rhs.cuda()).cpu()
    expected = torchmm.wide_add(lhs, rhs)

    assert out.shape == (2, 3, 2)
    assert torch.equal(out, expected)
