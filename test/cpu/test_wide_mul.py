import pytest
import torch

import torchmm

from test.wide_test_utils import expected_binary, limb_tensor_to_ints, reshape_wide


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_wide_mul_cpu_truncates_signed_products(limbs):
    values = [
        (0, 0),
        (1, -1),
        (-1, -1),
        (2**63 - 1, 2),
        (-(2**63), -1),
    ]
    x = reshape_wide([lhs for lhs, _ in values], (1, 1, len(values)), limbs)
    y = reshape_wide([rhs for _, rhs in values], (1, 1, len(values)), limbs)

    out = torchmm.wide_mul(x, y)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == expected_binary(values, limbs, lambda lhs, rhs: lhs * rhs)


@pytest.mark.parametrize("limbs", [1, 2, 4])
def test_wide_mul_cpu_supports_noncontiguous_inputs(limbs):
    base_x_values = list(range(24))
    base_y_values = [3 - value for value in range(24)]
    base_x = reshape_wide(base_x_values, (3, 2, 4), limbs)
    base_y = reshape_wide(base_y_values, (3, 2, 4), limbs)
    x = base_x.permute(1, 0, 2, 3)
    y = base_y.permute(1, 0, 2, 3)

    out = torchmm.wide_mul(x, y)

    assert not x.is_contiguous()
    assert not y.is_contiguous()
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == expected_binary(
        zip(limb_tensor_to_ints(x), limb_tensor_to_ints(y)),
        limbs,
        lambda lhs, rhs: lhs * rhs,
    )
