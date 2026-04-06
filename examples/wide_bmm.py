from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


x = torchmm.ints_to_limb_tensor([1, 2, 3], limbs=2).reshape(3, 2)
y = torchmm.ints_to_limb_tensor([4, 5, 6, 7, 8, 9], limbs=2).reshape(3, 2, 2)
out = torchmm.wide_bmm(x, y)

print("out:", out)
print("decoded:", torchmm.limb_tensor_to_ints(out))

