#ifndef TORCHMM_CSRC_WIDE_COMMON_H
#define TORCHMM_CSRC_WIDE_COMMON_H

#include <torch/types.h>

#include <array>

namespace torchmm::wide {

using WideShape = std::array<int64_t, 4>;

void validate_storage_tensor(const torch::Tensor &tensor, const char *name);
void validate_limb_count(const torch::Tensor &tensor, const char *name);
void validate_binary_inputs(const torch::Tensor &x, const torch::Tensor &y);
void validate_shift_input(const torch::Tensor &x, int64_t shift);
void validate_bmm_inputs(const torch::Tensor &x, const torch::Tensor &y);
void validate_optional_output(const c10::optional<torch::Tensor> &out,
                              int64_t                           limbs);
WideShape    get_shape4(const torch::Tensor &tensor);
WideShape    compute_broadcast_shape(const torch::Tensor &x,
                                     const torch::Tensor &y);
WideShape    compute_bmm_output_shape(const torch::Tensor &x,
                                      const torch::Tensor &y);
torch::Tensor make_output_tensor(const WideShape                     &shape,
                                 const torch::Tensor                &like,
                                 const c10::optional<torch::Tensor> &out);

} // namespace torchmm::wide

#endif



