import pytest
import torch

import torchmm

from test.wide_test_utils import expected_binary, limb_tensor_to_ints, reshape_wide


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_wide_sub_cpu_matches_python_ints(limbs):
    values = [(0, 1), (-1, 2)]
    x = reshape_wide([lhs for lhs, _ in values], (1, 1, 2), limbs)
    y = reshape_wide([rhs for _, rhs in values], (1, 1, 2), limbs)

    out = torchmm.wide_sub(x, y)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == expected_binary(values, limbs, lambda lhs, rhs: lhs - rhs)
