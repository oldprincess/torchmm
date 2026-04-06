from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


x = torchmm.ints_to_limb_tensor([1, 2, 3], limbs=2).reshape(3, 2)
y = torchmm.ints_to_limb_tensor([1, 3, 2], limbs=2).reshape(3, 2)

print(torchmm.wide_ge(x, y))

