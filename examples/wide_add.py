from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


x = torchmm.ints_to_limb_tensor([1, 2, 3], limbs=2).reshape(3, 2)
y = torchmm.ints_to_limb_tensor([10, 20, 30, 40, 50, 60], limbs=2).reshape(2, 3, 2)
out = torchmm.wide_add(x, y)

print("x:", x)
print("y:", y)
print("out:", out)
print("decoded:", torchmm.limb_tensor_to_ints(out))

