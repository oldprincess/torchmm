import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_neg_cuda_matches_cpu_reference():
    x = torch.tensor([[[[1, 0, 0], [-2, -1, -1], [8, 0, 0]]]], dtype=torch.int64)

    out = torchmm.wide_neg(x.cuda()).cpu()
    expected = torchmm.wide_neg(x)

    assert torch.equal(out, expected)
