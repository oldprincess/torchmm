import pytest
import torch

import torchmm

from test.wide_test_utils import limb_tensor_to_ints, reshape_wide


@pytest.mark.parametrize("limbs", [1, 2, 4])
def test_wide_bmm_cpu_matches_torch_bmm_with_batch_broadcast(limbs):
    lhs_values = [1, 2, 3, 4, 5, 6]
    rhs_values = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    x = reshape_wide(lhs_values, (1, 2, 3), limbs)
    y = reshape_wide(rhs_values, (2, 3, 2), limbs)

    out = torchmm.wide_bmm(x, y)

    lhs_batches = torch.tensor(lhs_values, dtype=torch.int64).reshape(1, 2, 3)
    rhs_batches = torch.tensor(rhs_values, dtype=torch.int64).reshape(2, 3, 2)
    expected_dense = torch.bmm(lhs_batches.expand(2, -1, -1), rhs_batches)

    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
    assert out.shape == (2, 2, 2, limbs)
    assert limb_tensor_to_ints(out) == expected_dense.reshape(-1).tolist()


def test_wide_bmm_cpu_adapts_vector_matrix_cases():
    vector = reshape_wide([1, 2, 3], (3,), 2)
    rhs_matrix = reshape_wide([4, 5, 6, 7, 8, 9], (3, 2), 2)
    lhs_matrix = reshape_wide([4, 5, 6, 7, 8, 9], (2, 3), 2)

    left = torchmm.wide_bmm(vector, rhs_matrix)
    right = torchmm.wide_bmm(lhs_matrix, vector)

    assert left.shape == (2, 2)
    assert limb_tensor_to_ints(left) == [40, 46]
    assert right.shape == (2, 2)
    assert limb_tensor_to_ints(right) == [32, 50]


def test_wide_tensor_bmm_adapts_vector_matrix_cases():
    vector = torchmm.WideTensor(reshape_wide([1, 2, 3], (3,), 2))
    rhs_matrix = torchmm.WideTensor(reshape_wide([4, 5, 6, 7, 8, 9], (3, 2), 2))

    out = vector.bmm(rhs_matrix)

    assert isinstance(out, torchmm.WideTensor)
    assert out.tensor.shape == (2, 2)
    assert limb_tensor_to_ints(out.tensor) == [40, 46]
