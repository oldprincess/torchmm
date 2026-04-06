import torch


def ints_to_limb_tensor(values, limbs: int, device="cpu") -> torch.Tensor:
    if limbs <= 0:
        raise ValueError("limbs must be positive")

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


def reshape_wide(values, shape, limbs: int, device="cpu") -> torch.Tensor:
    return ints_to_limb_tensor(values, limbs=limbs, device=device).reshape(*shape, limbs)


def limb_tensor_to_ints(tensor: torch.Tensor):
    if tensor.dtype != torch.int64:
        raise ValueError("wide tensors must use torch.int64 storage")
    if tensor.ndim < 1:
        raise ValueError("wide tensors must have rank >= 1")

    limbs = tensor.shape[-1]
    if limbs <= 0:
        return []

    width = 64 * limbs
    modulus = 1 << width
    sign_bit = 1 << (width - 1)
    limb_mask = (1 << 64) - 1

    values = []
    for row in tensor.reshape(-1, limbs).tolist():
        value = 0
        for limb_index, limb in enumerate(row):
            value |= (int(limb) & limb_mask) << (64 * limb_index)
        if value >= sign_bit:
            value -= modulus
        values.append(value)
    return values


def bool_tensor_to_list(tensor: torch.Tensor):
    return tensor.reshape(-1).tolist()


def truncate_signed(value: int, limbs: int) -> int:
    width = 64 * limbs
    modulus = 1 << width
    sign_bit = 1 << (width - 1)
    value &= modulus - 1
    if value >= sign_bit:
        value -= modulus
    return value


def expected_binary(values, limbs: int, op):
    return [truncate_signed(op(lhs, rhs), limbs) for lhs, rhs in values]


def expected_shift_left(values, limbs: int, shift: int):
    return [truncate_signed(value << shift, limbs) for value in values]


def expected_shift_right(values, shift: int):
    return [value >> shift for value in values]

