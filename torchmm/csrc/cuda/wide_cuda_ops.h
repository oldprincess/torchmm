#ifndef TORCHMM_CSRC_CUDA_WIDE_CUDA_OPS_H
#define TORCHMM_CSRC_CUDA_WIDE_CUDA_OPS_H

#include <torch/types.h>

namespace torchmm::cuda {

torch::Tensor wide_cuda_add(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_sub(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_mul(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_neg(const torch::Tensor         &x,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_shl(const torch::Tensor         &x,
                            int64_t                      shift,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_shr(const torch::Tensor         &x,
                            int64_t                      shift,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_bmm(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out);

torch::Tensor wide_cuda_eq(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_cuda_ge(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_cuda_le(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_cuda_gt(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_cuda_lt(const torch::Tensor &x, const torch::Tensor &y);

} // namespace torchmm::cuda

#endif
