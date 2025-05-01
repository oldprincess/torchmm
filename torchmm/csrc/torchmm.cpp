#include <torch/extension.h>

#include "cuda/matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_i_bmm",          //
          &torchmm::cuda::i_bmm, //
          py::arg("in1"),      //
          py::arg("in2"),      //
          py::arg("out"),      //
          R"(
bmm implementation on CUDA.

Args:
    in1 (torch.Tensor):     the first batch of matrices to be multiplied
    in2 (torch.Tensor):     the second batch of matrices to be multiplied
    out (torch.Tensor):     the output of matrices

Returns:
    torch.Tensor:     the output of matrices

Raises:
    ValueError:   If the input argument is invalid.
    RuntimeError: If the operation failed to execute.
        )");
}