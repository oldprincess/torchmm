import torch

import torchmm


def test_ints_to_limb_tensor_encodes_signed_values():
    out = torchmm.ints_to_limb_tensor([0, 1, -1], limbs=2)

    expected = torch.tensor(
        [[0, 0], [1, 0], [-1, -1]],
        dtype=torch.int64,
    )
    assert torch.equal(out, expected)

