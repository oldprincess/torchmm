import pytest
import torch

import torchmm

from test.wide_test_utils import expected_shift_left, limb_tensor_to_ints, reshape_wide


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
@pytest.mark.parametrize("shift", [0, 1, 63, 64, 127, 128])
def test_wide_shl_cpu_truncates_to_wide_width(limbs, shift):
    values = [0, 1, -1, -(2**63), 2**62 - 1]
    x = reshape_wide(values, (1, 1, len(values)), limbs)

    out = torchmm.wide_shl(x, shift)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == expected_shift_left(values, limbs, shift)
