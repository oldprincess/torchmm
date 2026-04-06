# Examples

This directory contains one minimal example per public `wide_*` function.

## Run

From the repository root:

```bash
python examples/wide_add.py
python examples/wide_bmm.py
```

## Helpers

The examples use:

- `torchmm.ints_to_limb_tensor(...)`
- `torchmm.limb_tensor_to_ints(...)`

These helpers convert between Python integers and torchmm wide storage with shape `[..., limbs]`.

## Files

- `wide_add.py`
- `wide_sub.py`
- `wide_mul.py`
- `wide_neg.py`
- `wide_shl.py`
- `wide_shr.py`
- `wide_bmm.py`
- `wide_eq.py`
- `wide_ge.py`
- `wide_gt.py`
- `wide_le.py`
- `wide_lt.py`
- `wide_tensor_operators.py`

All examples run on CPU. If CUDA is available, the same `wide_*` APIs can also be used with CUDA tensors.

`wide_tensor_operators.py` shows the Python operator overloads on `torchmm.WideTensor`, including `+`, `-`, `*`, unary `-`, shifts, and comparisons.
