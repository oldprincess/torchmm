import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_shr_cuda_matches_cpu_reference():
    x = torch.tensor([[[[1, 0, 0], [-1, -1, -1], [8, 0, 0]]]], dtype=torch.int64)

    out = torchmm.wide_shr(x.cuda(), 65).cpu()
    expected = torchmm.wide_shr(x, 65)

    assert torch.equal(out, expected)
