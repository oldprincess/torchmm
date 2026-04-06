import torch


def ints_to_limb_tensor(values, limbs: int, device=None) -> torch.Tensor:
    if limbs <= 0:
        raise ValueError("limbs must be a positive integer")

    values = list(values)
    if not values:
        return torch.empty((0, limbs), dtype=torch.int64, device=device)

    width = 64 * limbs
    modulus = 1 << width
    limb_mask = (1 << 64) - 1

    rows = []
    for value in values:
        unsigned = int(value) % modulus
        row = []
        for _ in range(limbs):
            limb = unsigned & limb_mask
            if limb >= 1 << 63:
                limb -= 1 << 64
            row.append(limb)
            unsigned >>= 64
        rows.append(row)

    return torch.tensor(rows, dtype=torch.int64, device=device)


def limb_tensor_to_ints(tensor: torch.Tensor):
    if tensor.dtype != torch.int64:
        raise ValueError("WideTensor storage must use torch.int64")
    if tensor.ndim < 1:
        raise ValueError("WideTensor storage must have rank >= 1")

    limbs = tensor.shape[-1]
    if limbs <= 0:
        return []

    width = 64 * limbs
    modulus = 1 << width
    sign_bit = 1 << (width - 1)
    limb_mask = (1 << 64) - 1

    ints = []
    for row in tensor.reshape(-1, limbs).tolist():
        value = 0
        for limb_index, limb in enumerate(row):
            value |= (int(limb) & limb_mask) << (64 * limb_index)
        if value >= sign_bit:
            value -= modulus
        ints.append(value)
    return ints

