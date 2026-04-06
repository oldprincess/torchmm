import torch

import torchmm._C as _ext
from torchmm.utils import ints_to_limb_tensor, limb_tensor_to_ints


def _validate_wide_storage(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype != torch.int64:
        raise ValueError("WideTensor storage must use torch.int64")
    if tensor.ndim < 2 or tensor.ndim > 4:
        raise ValueError("WideTensor storage must have rank 2-4: [..., limbs]")
    return tensor


def _normalize_wide_storage(tensor: torch.Tensor):
    tensor = _validate_wide_storage(tensor)
    logical_shape = tuple(tensor.shape[:-1])
    normalized = tensor.reshape(*((1,) * (4 - tensor.ndim)), *tensor.shape)
    return normalized, logical_shape


def _normalize_wide_out(out, logical_shape, limbs):
    if out is None:
        return None

    out = _validate_wide_storage(out)
    if tuple(out.shape[:-1]) != tuple(logical_shape):
        raise ValueError("out must have matching logical shape")
    if out.shape[-1] != limbs:
        raise ValueError("out must have matching limb count")
    return out.reshape(*((1,) * (4 - out.ndim)), *out.shape)


def _restore_wide_storage(tensor: torch.Tensor, logical_shape):
    return tensor.reshape(*logical_shape, tensor.shape[-1])


def _restore_bool_tensor(tensor: torch.Tensor, logical_shape):
    return tensor.reshape(*logical_shape)


def _wide_binary_op(op_name: str, x: torch.Tensor, y: torch.Tensor, out=None):
    x, x_shape = _normalize_wide_storage(x)
    y, y_shape = _normalize_wide_storage(y)
    logical_shape = torch.broadcast_shapes(x_shape, y_shape)
    out_tensor = _normalize_wide_out(out, logical_shape, x.shape[-1])
    result = getattr(_ext, op_name)(x, y, out_tensor)
    if out is not None:
        return out
    return _restore_wide_storage(result, logical_shape)


def _wide_unary_op(op_name: str, x: torch.Tensor, *args, out=None):
    x, logical_shape = _normalize_wide_storage(x)
    out_tensor = _normalize_wide_out(out, logical_shape, x.shape[-1])
    result = getattr(_ext, op_name)(x, *args, out_tensor)
    if out is not None:
        return out
    return _restore_wide_storage(result, logical_shape)


def _wide_compare_op(op_name: str, x: torch.Tensor, y: torch.Tensor):
    x, x_shape = _normalize_wide_storage(x)
    y, y_shape = _normalize_wide_storage(y)
    logical_shape = torch.broadcast_shapes(x_shape, y_shape)
    result = getattr(_ext, op_name)(x, y)
    return _restore_bool_tensor(result, logical_shape)


def _prepare_wide_bmm_operand(tensor: torch.Tensor, is_rhs: bool):
    normalized, logical_shape = _normalize_wide_storage(tensor)
    if len(logical_shape) == 1 and is_rhs:
        normalized = normalized.transpose(1, 2)
    return normalized, logical_shape


def _wide_bmm_output_shape(x_shape, y_shape):
    x_ndim = len(x_shape)
    y_ndim = len(y_shape)

    if x_ndim == 1 and y_ndim == 1:
        return (1,)
    if x_ndim == 1:
        return y_shape[:-2] + (y_shape[-1],)
    if y_ndim == 1:
        return x_shape[:-2] + (x_shape[-2],)
    return torch.broadcast_shapes(x_shape[:-2], y_shape[:-2]) + (x_shape[-2], y_shape[-1])


_ints_to_limb_tensor = ints_to_limb_tensor
_limb_tensor_to_ints = limb_tensor_to_ints


class WideTensor:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = _validate_wide_storage(tensor)

    @property
    def shape(self):
        return self.tensor.shape[:-1]

    @property
    def limbs(self) -> int:
        return self.tensor.shape[-1]

    @property
    def device(self):
        return self.tensor.device

    @property
    def dtype(self):
        return self.tensor.dtype

    def _coerce_operand(self, other):
        if isinstance(other, WideTensor):
            return other.tensor

        if not isinstance(other, torch.Tensor):
            other = torch.as_tensor(other, device=self.device)

        other = other.to(device=self.device, dtype=torch.int64)
        if 2 <= other.ndim <= 4 and other.shape[-1] == self.limbs:
            return _validate_wide_storage(other)

        dense = torch.broadcast_to(other, self.shape)
        storage = torch.where(
            dense.unsqueeze(-1) < 0,
            torch.full((*self.shape, self.limbs), -1, dtype=torch.int64, device=self.device),
            torch.zeros((*self.shape, self.limbs), dtype=torch.int64, device=self.device),
        )
        storage[..., 0] = dense
        return storage

    def _binary_op(self, other, op):
        return WideTensor(op(self.tensor, self._coerce_operand(other)))

    def _compare_op(self, other, op):
        return op(self.tensor, self._coerce_operand(other))

    def __add__(self, other):
        return self._binary_op(other, wide_add)

    def __radd__(self, other):
        return WideTensor(wide_add(self._coerce_operand(other), self.tensor))

    def __sub__(self, other):
        return self._binary_op(other, wide_sub)

    def __rsub__(self, other):
        return WideTensor(wide_sub(self._coerce_operand(other), self.tensor))

    def __mul__(self, other):
        return self._binary_op(other, wide_mul)

    def __rmul__(self, other):
        return WideTensor(wide_mul(self._coerce_operand(other), self.tensor))

    def __neg__(self):
        return WideTensor(wide_neg(self.tensor))

    def __lshift__(self, shift):
        return WideTensor(wide_shl(self.tensor, shift))

    def __rshift__(self, shift):
        return WideTensor(wide_shr(self.tensor, shift))

    def __eq__(self, other):
        return self._compare_op(other, wide_eq)

    def __ge__(self, other):
        return self._compare_op(other, wide_ge)

    def __gt__(self, other):
        return self._compare_op(other, wide_gt)

    def __le__(self, other):
        return self._compare_op(other, wide_le)

    def __lt__(self, other):
        return self._compare_op(other, wide_lt)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return WideTensor(self.tensor.reshape(*shape, self.limbs))

    def bmm(self, other):
        return WideTensor(wide_bmm(self.tensor, self._coerce_operand(other)))

    @classmethod
    def concat(cls, tensors, dim=0):
        tensors = list(tensors)
        if not tensors:
            raise ValueError("WideTensor.concat requires at least one tensor")

        storage = []
        first = tensors[0]
        if not isinstance(first, WideTensor):
            raise TypeError("WideTensor.concat expects WideTensor inputs")

        limbs = first.limbs
        for tensor in tensors:
            if not isinstance(tensor, WideTensor):
                raise TypeError("WideTensor.concat expects WideTensor inputs")
            if tensor.limbs != limbs:
                raise ValueError("All WideTensor inputs must use the same limb count")
            storage.append(tensor.tensor)

        return cls(torch.cat(storage, dim=dim))

    def __repr__(self):
        return f"WideTensor(shape={tuple(self.shape)}, limbs={self.limbs}, device={self.device})"


def wide_add(x, y, out=None):
    return _wide_binary_op("wide_add", x, y, out=out)


def wide_sub(x, y, out=None):
    return _wide_binary_op("wide_sub", x, y, out=out)


def wide_mul(x, y, out=None):
    return _wide_binary_op("wide_mul", x, y, out=out)


def wide_neg(x, out=None):
    return _wide_unary_op("wide_neg", x, out=out)


def wide_shl(x, shift, out=None):
    return _wide_unary_op("wide_shl", x, shift, out=out)


def wide_shr(x, shift, out=None):
    return _wide_unary_op("wide_shr", x, shift, out=out)


def wide_bmm(x, y, out=None):
    x, x_shape = _prepare_wide_bmm_operand(x, is_rhs=False)
    y, y_shape = _prepare_wide_bmm_operand(y, is_rhs=True)
    logical_shape = _wide_bmm_output_shape(x_shape, y_shape)
    out_tensor = _normalize_wide_out(out, logical_shape, x.shape[-1])
    result = _ext.wide_bmm(x, y, out_tensor)
    if out is not None:
        return out
    return _restore_wide_storage(result, logical_shape)


def wide_eq(x, y):
    return _wide_compare_op("wide_eq", x, y)


def wide_ge(x, y):
    return _wide_compare_op("wide_ge", x, y)


def wide_le(x, y):
    return _wide_compare_op("wide_le", x, y)


def wide_gt(x, y):
    return _wide_compare_op("wide_gt", x, y)


def wide_lt(x, y):
    return _wide_compare_op("wide_lt", x, y)
