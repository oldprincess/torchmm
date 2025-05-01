import functools
import torch
from torchmm._C import *


def matmul(
    x: torch.Tensor, y: torch.Tensor, *, out: torch.Tensor = None
) -> torch.Tensor:
    if x.is_cuda and x.dtype in [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ]:
        if out is None:
            out = torch.empty(0, device="cuda", dtype=x.dtype)
        x = x.contiguous()
        y = y.contiguous()
        if len(x.shape) == 1 and len(y.shape) == 1:
            return cuda_i_bmm(
                x.reshape(1, 1, x.shape[0]),
                y.reshape(1, y.shape[0], 1),
                out,
            ).reshape(1)[0]
        elif len(x.shape) == 1 and len(y.shape) == 2:
            return cuda_i_bmm(
                x.reshape(1, 1, x.shape[0]),
                y.reshape(1, y.shape[0], y.shape[1]),
                out,
            ).reshape(y.shape[1])
        elif len(x.shape) == 2 and len(y.shape) == 1:
            return cuda_i_bmm(
                x.reshape(1, x.shape[0], x.shape[1]),
                y.reshape(1, y.shape[0], 1),
                out,
            ).reshape(x.shape[0])
        elif len(x.shape) == 2 and len(y.shape) == 2:
            return cuda_i_bmm(
                x.reshape(1, x.shape[0], x.shape[1]),
                y.reshape(1, y.shape[0], y.shape[1]),
                out,
            ).reshape(x.shape[0], y.shape[1])
        elif len(x.shape) == 1 and len(y.shape) >= 3:
            return cuda_i_bmm(
                x.reshape(1, 1, x.shape[0]),
                y.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, y.shape[:-2]),
                    y.shape[-2],
                    y.shape[-1],
                ),
                out,
            ).reshape(*y.shape[:-2], y.shape[-1])
        elif len(x.shape) == 2 and len(y.shape) >= 3:
            return cuda_i_bmm(
                x.reshape(1, x.shape[0], x.shape[1]),
                y.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, y.shape[:-2]),
                    y.shape[-2],
                    y.shape[-1],
                ),
                out,
            ).reshape(*y.shape[:-2], x.shape[-2], y.shape[-1])
        elif len(x.shape) >= 3 and len(y.shape) == 1:
            return cuda_i_bmm(
                x.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, x.shape[:-2]),
                    x.shape[-2],
                    x.shape[-1],
                ),
                y.reshape(
                    1,
                    y.shape[0],
                    1,
                ),
                out,
            ).reshape(*x.shape[:-2], x.shape[-2])
        elif len(x.shape) >= 3 and len(y.shape) == 2:
            return cuda_i_bmm(
                x.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, x.shape[:-2]),
                    x.shape[-2],
                    x.shape[-1],
                ),
                y.reshape(
                    1,
                    y.shape[0],
                    y.shape[1],
                ),
                out,
            ).reshape(*x.shape[:-2], x.shape[-2], y.shape[-1])
        elif len(x.shape) >= 3 and len(y.shape) >= 3:
            if x.shape[:-2] != y.shape[:-2]:
                raise RuntimeError(
                    "Unsupported tensor shape: {} {}".format(x.shape, y.shape)
                )
            return cuda_i_bmm(
                x.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, x.shape[:-2]),
                    x.shape[-2],
                    x.shape[-1],
                ),
                y.reshape(
                    functools.reduce(lambda _x, _y: _x * _y, y.shape[:-2]),
                    y.shape[-2],
                    y.shape[-1],
                ),
                out,
            ).reshape(*x.shape[:-2], x.shape[-2], y.shape[-1])
        else:
            raise RuntimeError(
                "Unsupported tensor shape: {} {}".format(x.shape, y.shape)
            )
    else:
        return torch.matmul(x, y, out=out)
