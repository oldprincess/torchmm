from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


x = torchmm.ints_to_limb_tensor([2, -3, 4], limbs=2).reshape(3, 2)
y = torchmm.ints_to_limb_tensor([5, 6, -7], limbs=2).reshape(3, 2)
out = torchmm.wide_mul(x, y)

print("decoded:", torchmm.limb_tensor_to_ints(out))

