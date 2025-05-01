#include <c10/cuda/CUDAStream.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_batched.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "matmul.h"

#define TORCHMM_ASSERT(expression)                                     \
    if (!(expression))                                                 \
    {                                                                  \
        std::printf("[TORCHMM] Assertion failed. %s:%d\n", __FILE__,   \
                    __LINE__);                                         \
        throw std::invalid_argument("Assertion failed: " #expression); \
    }

namespace torchmm::cuda {

torch::Tensor& i_bmm(const torch::Tensor& in1,
                     const torch::Tensor& in2,
                     torch::Tensor&       out)
{
    TORCHMM_ASSERT(in1.is_contiguous());
    TORCHMM_ASSERT(in1.device().type() == torch::kCUDA);
    TORCHMM_ASSERT(in1.sizes().size() == 3);
    TORCHMM_ASSERT(in2.is_contiguous());
    TORCHMM_ASSERT(in2.device().type() == torch::kCUDA);
    TORCHMM_ASSERT(in2.sizes().size() == 3);
    TORCHMM_ASSERT(in1.size(2) == in2.size(1));
    TORCHMM_ASSERT(in1.dtype() == in2.dtype());
    TORCHMM_ASSERT(out.is_contiguous());
    TORCHMM_ASSERT(out.device().type() == torch::kCUDA);
    TORCHMM_ASSERT(out.dtype() == in1.dtype());

    auto dtype = in1.dtype();

    // shape: (m x k) * (k * n) => (m, n)
    int m           = in1.size(1);
    int n           = in2.size(2);
    int k           = in1.size(2);
    int batch_count = in1.size(0);
    if (in1.size(0) != in2.size(0))
    {
        if (in1.size(0) == 1)
        {
            batch_count = in2.size(0);
        }
        else if (in2.size(0) == 1)
        {
            batch_count = in1.size(0);
        }
        else
        {
            throw std::invalid_argument("invalid shape");
        }
    }
    // storage in row major, leading dimensioin = num_cols
    int leading_dimension_1 = k;
    int leading_dimension_2 = n;
    int leading_dimension_3 = n;
    // batch stride = num_rows * num_cols
    long long batch_stride_1 = m * k;
    long long batch_stride_2 = k * n;
    long long batch_stride_3 = m * n;
    if (in1.size(0) == 1)
    {
        batch_stride_1 = 0;
    }
    if (in2.size(0) == 1)
    {
        batch_stride_2 = 0;
    }

    out.resize_({batch_count, m, n});

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_INTEGRAL_TYPES(dtype.toScalarType(), "cuda_bmm", [&] {
        using Gemm = cutlass::gemm::device::GemmBatched< //
            scalar_t,                                    //
            cutlass::layout::RowMajor,                   //
            scalar_t,                                    //
            cutlass::layout::RowMajor,                   //
            scalar_t,                                    //
            cutlass::layout::RowMajor                    //
            >;                                           //

        Gemm            gemm_op;
        cutlass::Status status = gemm_op({
            {m, n, k},                                               //
            {in1.const_data_ptr<scalar_t>(), leading_dimension_1},   //
            batch_stride_1,                                          //
            {in2.const_data_ptr<scalar_t>(), leading_dimension_2},   //
            batch_stride_2,                                          //
            {nullptr, 0},                                            //
            0,                                                       //
            {out.mutable_data_ptr<scalar_t>(), leading_dimension_3}, //
            batch_stride_3,                                          //
            {1, 0},                                                  //
            batch_count                                              //
        }, nullptr, stream);
        if (status != cutlass::Status::kSuccess)
        {
            std::string err_msg = std::string("CUTLASS error: ");
            err_msg += cutlassGetStatusString(status);
            throw std::runtime_error(err_msg);
        }
    });

    return out;
}

} // namespace torchmm::cuda