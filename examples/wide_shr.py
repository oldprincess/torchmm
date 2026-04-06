from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


x = torchmm.ints_to_limb_tensor([32, -64, 128], limbs=2).reshape(3, 2)
out = torchmm.wide_shr(x, 3)

print("decoded:", torchmm.limb_tensor_to_ints(out))

