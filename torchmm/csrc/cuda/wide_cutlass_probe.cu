#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>

#include <stdexcept>
#include <string>

#include "wide_cutlass_probe.h"
#include "wideint/wideint.hpp"

namespace torchmm::cuda {

namespace {

using wide_scalar = wideint::sint<2>;

void validate_probe_inputs(const torch::Tensor &x, const torch::Tensor &y)
{
    TORCH_CHECK(x.device().type() == torch::kCUDA, "wide_cuda_bmm_probe requires CUDA inputs");
    TORCH_CHECK(y.device().type() == torch::kCUDA, "wide_cuda_bmm_probe requires CUDA inputs");
    TORCH_CHECK(x.dtype() == torch::kInt64, "wide_cuda_bmm_probe requires torch.int64 inputs");
    TORCH_CHECK(y.dtype() == torch::kInt64, "wide_cuda_bmm_probe requires torch.int64 inputs");
    TORCH_CHECK(x.dim() == 4, "wide_cuda_bmm_probe expects x to have rank 4");
    TORCH_CHECK(y.dim() == 4, "wide_cuda_bmm_probe expects y to have rank 4");
    TORCH_CHECK(x.is_contiguous(), "wide_cuda_bmm_probe requires contiguous x");
    TORCH_CHECK(y.is_contiguous(), "wide_cuda_bmm_probe requires contiguous y");
    TORCH_CHECK(x.size(-1) == 2, "wide_cuda_bmm_probe currently supports exactly 2 limbs");
    TORCH_CHECK(y.size(-1) == 2, "wide_cuda_bmm_probe currently supports exactly 2 limbs");
    TORCH_CHECK(x.size(2) == y.size(1), "wide_cuda_bmm_probe requires x.shape[2] == y.shape[1]");
    TORCH_CHECK(
        x.size(0) == y.size(0) || x.size(0) == 1 || y.size(0) == 1,
        "wide_cuda_bmm_probe requires broadcastable batch dimensions"
    );
    static_assert(sizeof(wide_scalar) == sizeof(std::int64_t) * 2);
    static_assert(alignof(wide_scalar) <= alignof(std::int64_t));
}

} // namespace

torch::Tensor wide_cuda_bmm_probe(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_probe_inputs(x, y);

    const int m = static_cast<int>(x.size(1));
    const int n = static_cast<int>(y.size(2));
    const int k = static_cast<int>(x.size(2));
    int batch_count = static_cast<int>(x.size(0));
    if (x.size(0) != y.size(0))
    {
        batch_count = static_cast<int>(std::max(x.size(0), y.size(0)));
    }

    int leading_dimension_1 = k;
    int leading_dimension_2 = n;
    int leading_dimension_3 = n;

    long long batch_stride_1 = x.size(0) == 1 ? 0 : static_cast<long long>(m) * k;
    long long batch_stride_2 = y.size(0) == 1 ? 0 : static_cast<long long>(k) * n;
    long long batch_stride_3 = static_cast<long long>(m) * n;

    auto out = torch::empty({batch_count, m, n, 2}, x.options());

    const auto *lhs_ptr = reinterpret_cast<const wide_scalar *>(x.const_data_ptr<int64_t>());
    const auto *rhs_ptr = reinterpret_cast<const wide_scalar *>(y.const_data_ptr<int64_t>());
    auto *out_ptr = reinterpret_cast<wide_scalar *>(out.mutable_data_ptr<int64_t>());

    using Gemm = cutlass::gemm::device::GemmBatched<
        wide_scalar,
        cutlass::layout::RowMajor,
        wide_scalar,
        cutlass::layout::RowMajor,
        wide_scalar,
        cutlass::layout::RowMajor>;

    Gemm gemm_op;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    cutlass::Status status = gemm_op(
        {
            {m, n, k},
            {lhs_ptr, leading_dimension_1},
            batch_stride_1,
            {rhs_ptr, leading_dimension_2},
            batch_stride_2,
            {nullptr, 0},
            0,
            {out_ptr, leading_dimension_3},
            batch_stride_3,
            {1, 0},
            batch_count,
        },
        nullptr,
        stream
    );
    if (status != cutlass::Status::kSuccess)
    {
        std::string err_msg = std::string("CUTLASS wide probe error: ");
        err_msg += cutlassGetStatusString(status);
        throw std::runtime_error(err_msg);
    }

    return out;
}

} // namespace torchmm::cuda
