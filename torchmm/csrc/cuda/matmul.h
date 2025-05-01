#ifndef TORCHMM_SCRS_CUDA_MATMUL_H
#define TORCHMM_SCRS_CUDA_MATMUL_H

#include <torch/extension.h>

namespace torchmm::cuda {

torch::Tensor& i_bmm(const torch::Tensor& in1,
                     const torch::Tensor& in2,
                     torch::Tensor&       out);

}; // namespace torchmm::cuda

#endif