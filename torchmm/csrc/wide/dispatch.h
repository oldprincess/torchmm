#ifndef TORCHMM_CSRC_WIDE_DISPATCH_H
#define TORCHMM_CSRC_WIDE_DISPATCH_H

#include <torch/types.h>

namespace torchmm::wide {

torch::Tensor wide_add(const torch::Tensor         &x,
                       const torch::Tensor         &y,
                       c10::optional<torch::Tensor> out);

torch::Tensor wide_sub(const torch::Tensor         &x,
                       const torch::Tensor         &y,
                       c10::optional<torch::Tensor> out);

torch::Tensor wide_mul(const torch::Tensor         &x,
                       const torch::Tensor         &y,
                       c10::optional<torch::Tensor> out);

torch::Tensor wide_neg(const torch::Tensor         &x,
                       c10::optional<torch::Tensor> out);

torch::Tensor wide_lshift(const torch::Tensor         &x,
                          int64_t                      shift,
                          c10::optional<torch::Tensor> out);

torch::Tensor wide_lshift(const torch::Tensor         &x,
                          const torch::Tensor         &shift,
                          c10::optional<torch::Tensor> out);

torch::Tensor wide_rshift(const torch::Tensor         &x,
                          int64_t                      shift,
                          c10::optional<torch::Tensor> out);

torch::Tensor wide_rshift(const torch::Tensor         &x,
                          const torch::Tensor         &shift,
                          c10::optional<torch::Tensor> out);

torch::Tensor wide_bmm(const torch::Tensor         &x,
                       const torch::Tensor         &y,
                       c10::optional<torch::Tensor> out);

torch::Tensor wide_eq(const torch::Tensor &x, const torch::Tensor &y);

torch::Tensor wide_ge(const torch::Tensor &x, const torch::Tensor &y);

torch::Tensor wide_le(const torch::Tensor &x, const torch::Tensor &y);

torch::Tensor wide_gt(const torch::Tensor &x, const torch::Tensor &y);

torch::Tensor wide_lt(const torch::Tensor &x, const torch::Tensor &y);

} // namespace torchmm::wide

#endif


