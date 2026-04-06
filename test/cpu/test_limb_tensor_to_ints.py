import torch

import torchmm


def test_limb_tensor_to_ints_decodes_signed_values():
    tensor = torch.tensor(
        [[0, 0], [1, 0], [-1, -1]],
        dtype=torch.int64,
    )

    assert torchmm.limb_tensor_to_ints(tensor) == [0, 1, -1]
