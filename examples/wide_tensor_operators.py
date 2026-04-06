from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import torchmm


lhs = torchmm.WideTensor(torchmm.ints_to_limb_tensor([1, 2, 3], limbs=2).reshape(3, 2))
rhs = torchmm.WideTensor(torchmm.ints_to_limb_tensor([10, 20, 30], limbs=2).reshape(3, 2))

print("lhs + rhs:", torchmm.limb_tensor_to_ints((lhs + rhs).tensor))
print("rhs - lhs:", torchmm.limb_tensor_to_ints((rhs - lhs).tensor))
print("lhs * rhs:", torchmm.limb_tensor_to_ints((lhs * rhs).tensor))
print("-lhs:", torchmm.limb_tensor_to_ints((-lhs).tensor))
print("lhs << 2:", torchmm.limb_tensor_to_ints((lhs << 2).tensor))
print("rhs >> 1:", torchmm.limb_tensor_to_ints((rhs >> 1).tensor))
print("lhs == rhs:", (lhs == rhs).tolist())
print("lhs < rhs:", (lhs < rhs).tolist())
print("rhs >= lhs:", (rhs >= lhs).tolist())
