import pytest
import torch

import torchmm

from test.wide_test_utils import limb_tensor_to_ints, reshape_wide, truncate_signed


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_wide_neg_cpu_matches_python_ints(limbs):
    values = [0, 1, -1, -(2**63), 2**62 - 1]
    x = reshape_wide(values, (1, 1, len(values)), limbs)

    out = torchmm.wide_neg(x)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == [truncate_signed(-value, limbs) for value in values]
