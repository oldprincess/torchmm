import pytest
import torch

import torchmm

from test.wide_test_utils import expected_binary, limb_tensor_to_ints, reshape_wide


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_wide_add_cpu_matches_python_ints(limbs):
    values = [(-(2**63), 1), (-1, 2)]
    x = reshape_wide([lhs for lhs, _ in values], (1, 1, 2), limbs)
    y = reshape_wide([rhs for _, rhs in values], (1, 1, 2), limbs)

    out = torchmm.wide_add(x, y)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == x.shape
    assert limb_tensor_to_ints(out) == expected_binary(values, limbs, lambda lhs, rhs: lhs + rhs)


@pytest.mark.parametrize("limbs", [1, 2, 4])
def test_wide_add_cpu_supports_3d_broadcast(limbs):
    x_values = [-(2**63), -1, 0, 1, 2, 3]
    y_values = [5, -7, 11, -(2**62)]
    x = reshape_wide(x_values, (2, 1, 3), limbs)
    y = reshape_wide(y_values, (1, 4, 1), limbs)

    out = torchmm.wide_add(x, y)

    expected_pairs = []
    for i in range(2):
        for j in range(4):
            for k in range(3):
                expected_pairs.append((x_values[i * 3 + k], y_values[j]))

    assert out.shape == (2, 4, 3, limbs)
    assert limb_tensor_to_ints(out) == expected_binary(
        expected_pairs, limbs, lambda lhs, rhs: lhs + rhs
    )


def test_wide_add_cpu_adapts_common_logical_dims():
    x = reshape_wide([1, 2, 3], (3,), 2)
    y = reshape_wide([10, 20, 30, 40, 50, 60], (2, 3), 2)

    out = torchmm.wide_add(x, y)

    assert out.shape == (2, 3, 2)
    assert limb_tensor_to_ints(out) == [11, 22, 33, 41, 52, 63]


def test_wide_tensor_add_adapts_common_logical_dims():
    x = torchmm.WideTensor(reshape_wide([1, 2, 3], (3,), 2))
    y = torchmm.WideTensor(reshape_wide([10, 20, 30, 40, 50, 60], (2, 3), 2))

    out = x + y

    assert isinstance(out, torchmm.WideTensor)
    assert out.tensor.shape == (2, 3, 2)
    assert limb_tensor_to_ints(out.tensor) == [11, 22, 33, 41, 52, 63]
