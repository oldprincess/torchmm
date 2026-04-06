import torch

import torchmm

from test.wide_test_utils import bool_tensor_to_list, reshape_wide


def test_wide_lt_cpu_matches_python_ints():
    lhs = reshape_wide([-2, -1, 3, 4], (1, 1, 4), 2)
    rhs = reshape_wide([-2, 0, 1, 4], (1, 1, 4), 2)

    out = torchmm.wide_lt(lhs, rhs)

    assert out.dtype == torch.bool
    assert out.device.type == "cpu"
    assert out.shape == lhs.shape[:-1]
    assert bool_tensor_to_list(out) == [False, True, False, False]
