#include <torch/extension.h>

#ifdef TORCHMM_WITH_CUDA
#include "cuda/matmul.h"
#include "cuda/wide_cuda_ops.h"
#include "cuda/wide_cutlass_probe.h"
#endif
#include "wide/dispatch.h"

namespace {

template <typename Fn>
void bind_wide_binary(py::module_ &m, const char *name, Fn fn)
{
    m.def(name, fn, py::arg("x"), py::arg("y"), py::arg("out") = c10::nullopt);
}

template <typename Fn>
void bind_wide_compare(py::module_ &m, const char *name, Fn fn)
{
    m.def(name, fn, py::arg("x"), py::arg("y"));
}

using WideShiftFn = torch::Tensor (*)(const torch::Tensor &, int64_t, c10::optional<torch::Tensor>);

void bind_wide_shift(py::module_ &m, const char *name, WideShiftFn fn)
{
    m.def(name,
          py::overload_cast<const torch::Tensor &, int64_t, c10::optional<torch::Tensor>>(fn),
          py::arg("x"), py::arg("shift"), py::arg("out") = c10::nullopt);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#ifdef TORCHMM_WITH_CUDA
    m.def("cuda_i_bmm", &torchmm::cuda::i_bmm, py::arg("in1"), py::arg("in2"), py::arg("out"),
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
    m.def("wide_cuda_bmm_probe", &torchmm::cuda::wide_cuda_bmm_probe, py::arg("x"), py::arg("y"));
    bind_wide_binary(m, "wide_cuda_add", &torchmm::cuda::wide_cuda_add);
    bind_wide_binary(m, "wide_cuda_sub", &torchmm::cuda::wide_cuda_sub);
    bind_wide_binary(m, "wide_cuda_mul", &torchmm::cuda::wide_cuda_mul);
    m.def("wide_cuda_neg", &torchmm::cuda::wide_cuda_neg, py::arg("x"), py::arg("out") = c10::nullopt);
    m.def("wide_cuda_shl", &torchmm::cuda::wide_cuda_shl, py::arg("x"), py::arg("shift"),
          py::arg("out") = c10::nullopt);
    m.def("wide_cuda_shr", &torchmm::cuda::wide_cuda_shr, py::arg("x"), py::arg("shift"),
          py::arg("out") = c10::nullopt);
    bind_wide_binary(m, "wide_cuda_bmm", &torchmm::cuda::wide_cuda_bmm);
    bind_wide_compare(m, "wide_cuda_eq", &torchmm::cuda::wide_cuda_eq);
    bind_wide_compare(m, "wide_cuda_ge", &torchmm::cuda::wide_cuda_ge);
    bind_wide_compare(m, "wide_cuda_le", &torchmm::cuda::wide_cuda_le);
    bind_wide_compare(m, "wide_cuda_gt", &torchmm::cuda::wide_cuda_gt);
    bind_wide_compare(m, "wide_cuda_lt", &torchmm::cuda::wide_cuda_lt);
#endif

    bind_wide_binary(m, "wide_add", &torchmm::wide::wide_add);
    bind_wide_binary(m, "wide_sub", &torchmm::wide::wide_sub);
    bind_wide_binary(m, "wide_mul", &torchmm::wide::wide_mul);
    m.def("wide_neg", &torchmm::wide::wide_neg, py::arg("x"), py::arg("out") = c10::nullopt);
    bind_wide_shift(m, "wide_shl", static_cast<WideShiftFn>(&torchmm::wide::wide_lshift));
    bind_wide_shift(m, "wide_shr", static_cast<WideShiftFn>(&torchmm::wide::wide_rshift));
    bind_wide_shift(m, "wide_lshift", static_cast<WideShiftFn>(&torchmm::wide::wide_lshift));
    bind_wide_shift(m, "wide_rshift", static_cast<WideShiftFn>(&torchmm::wide::wide_rshift));
    bind_wide_binary(m, "wide_bmm", &torchmm::wide::wide_bmm);
    bind_wide_compare(m, "wide_eq", &torchmm::wide::wide_eq);
    bind_wide_compare(m, "wide_ge", &torchmm::wide::wide_ge);
    bind_wide_compare(m, "wide_le", &torchmm::wide::wide_le);
    bind_wide_compare(m, "wide_gt", &torchmm::wide::wide_gt);
    bind_wide_compare(m, "wide_lt", &torchmm::wide::wide_lt);
}
