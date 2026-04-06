#ifndef TORCHMM_CSRC_CUDA_WIDE_CUTLASS_PROBE_H
#define TORCHMM_CSRC_CUDA_WIDE_CUTLASS_PROBE_H

#include <torch/types.h>

namespace torchmm::cuda {

torch::Tensor wide_cuda_bmm_probe(const torch::Tensor &x, const torch::Tensor &y);

} // namespace torchmm::cuda

#endif
