import pytest
import torch

import torchmm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_bmm_cuda_matches_cpu_reference_for_two_limbs():
    x = torch.tensor([[[[1, 0], [2, 0], [3, 0]]]], dtype=torch.int64)
    y = torch.tensor(
        [[[[4, 0], [5, 0]], [[6, 0], [7, 0]], [[8, 0], [9, 0]]]],
        dtype=torch.int64,
    )

    out = torchmm.wide_bmm(x.cuda(), y.cuda()).cpu()
    expected = torchmm.wide_bmm(x, y)

    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_bmm_cuda_matches_cpu_reference_for_three_limbs():
    x = torch.tensor([[[[1, 0, 0], [2, 0, 0], [3, 0, 0]]]], dtype=torch.int64)
    y = torch.tensor(
        [[[[4, 0, 0], [5, 0, 0]], [[6, 0, 0], [7, 0, 0]], [[8, 0, 0], [9, 0, 0]]]],
        dtype=torch.int64,
    )

    out = torchmm.wide_bmm(x.cuda(), y.cuda()).cpu()
    expected = torchmm.wide_bmm(x, y)

    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wide_bmm_cuda_adapts_vector_matrix_cases():
    vector = torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.int64)
    rhs_matrix = torch.tensor(
        [[[4, 0], [5, 0]], [[6, 0], [7, 0]], [[8, 0], [9, 0]]],
        dtype=torch.int64,
    )
    lhs_matrix = torch.tensor(
        [[[4, 0], [5, 0], [6, 0]], [[7, 0], [8, 0], [9, 0]]],
        dtype=torch.int64,
    )

    left = torchmm.wide_bmm(vector.cuda(), rhs_matrix.cuda()).cpu()
    right = torchmm.wide_bmm(lhs_matrix.cuda(), vector.cuda()).cpu()

    assert torch.equal(left, torchmm.wide_bmm(vector, rhs_matrix))
    assert torch.equal(right, torchmm.wide_bmm(lhs_matrix, vector))
