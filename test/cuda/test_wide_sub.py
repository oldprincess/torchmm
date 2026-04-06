import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_sub_cuda_matches_cpu_reference():
    lhs = torch.tensor([[[[9, 0, 0], [2, 0, 0], [3, 0, 0]]]], dtype=torch.int64)
    rhs = torch.tensor([[[[4, 0, 0]], [[5, 0, 0]]]], dtype=torch.int64)

    out = torchmm.wide_sub(lhs.cuda(), rhs.cuda()).cpu()
    expected = torchmm.wide_sub(lhs, rhs)

    assert torch.equal(out, expected)
