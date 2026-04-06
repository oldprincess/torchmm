import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_eq_cuda_matches_cpu_reference():
    lhs = torch.tensor([[[[-2, -1], [-1, -1], [3, 0], [4, 0]]]], dtype=torch.int64)
    rhs = torch.tensor([[[[-2, -1], [0, 0], [1, 0], [4, 0]]]], dtype=torch.int64)

    out = torchmm.wide_eq(lhs.cuda(), rhs.cuda()).cpu()
    expected = torchmm.wide_eq(lhs, rhs)

    assert torch.equal(out, expected)
