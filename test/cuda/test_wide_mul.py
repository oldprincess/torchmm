import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_mul_cuda_matches_cpu_reference():
    lhs = torch.tensor([[[[1, 0, 0], [2, 0, 0], [-3, -1, -1]]]], dtype=torch.int64)
    rhs = torch.tensor([[[[4, 0, 0]], [[-5, -1, -1]]]], dtype=torch.int64)

    out = torchmm.wide_mul(lhs.cuda(), rhs.cuda()).cpu()
    expected = torchmm.wide_mul(lhs, rhs)

    assert torch.equal(out, expected)
