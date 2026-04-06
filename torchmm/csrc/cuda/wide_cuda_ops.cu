#include "wide_cuda_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAMacros.h>
#include <torch/types.h>

#include <array>
#include <cstdint>

#include "../wide/common.h"

namespace torchmm::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

enum class BinaryOpType : int
{
    Add = 0,
    Sub = 1,
    Mul = 2,
};

enum class CompareOpType : int
{
    Eq = 0,
    Ge = 1,
    Le = 2,
    Gt = 3,
    Lt = 4,
};

template <int N>
__device__ inline void load_limbs(const int64_t *src, std::uint64_t (&dst)[N])
{
    for (int limb = 0; limb < N; ++limb)
    {
        dst[limb] = static_cast<std::uint64_t>(src[limb]);
    }
}

template <int N>
__device__ inline void store_limbs(int64_t *dst, const std::uint64_t (&src)[N])
{
    for (int limb = 0; limb < N; ++limb)
    {
        dst[limb] = static_cast<int64_t>(src[limb]);
    }
}

template <int N>
__device__ inline void add_scalar(std::uint64_t (&acc)[N], int start, std::uint64_t value)
{
    for (int idx = start; idx < N && value != 0; ++idx)
    {
        const std::uint64_t previous = acc[idx];
        acc[idx] += value;
        value = acc[idx] < previous ? 1ULL : 0ULL;
    }
}

template <int N>
__device__ inline void add_limbs(const std::uint64_t (&lhs)[N],
                                 const std::uint64_t (&rhs)[N],
                                 std::uint64_t       (&out)[N])
{
    std::uint64_t carry = 0;
    for (int limb = 0; limb < N; ++limb)
    {
        const std::uint64_t sum1 = lhs[limb] + rhs[limb];
        const std::uint64_t carry1 = sum1 < lhs[limb] ? 1ULL : 0ULL;
        const std::uint64_t sum2 = sum1 + carry;
        const std::uint64_t carry2 = sum2 < sum1 ? 1ULL : 0ULL;
        out[limb] = sum2;
        carry = carry1 | carry2;
    }
}

template <int N>
__device__ inline void sub_limbs(const std::uint64_t (&lhs)[N],
                                 const std::uint64_t (&rhs)[N],
                                 std::uint64_t       (&out)[N])
{
    std::uint64_t borrow = 0;
    for (int limb = 0; limb < N; ++limb)
    {
        const std::uint64_t subtrahend = rhs[limb] + borrow;
        const bool wrapped = subtrahend < rhs[limb];
        out[limb] = lhs[limb] - subtrahend;
        borrow = (wrapped || lhs[limb] < subtrahend) ? 1ULL : 0ULL;
    }
}

template <int N>
__device__ inline void mul_limbs(const std::uint64_t (&lhs)[N],
                                 const std::uint64_t (&rhs)[N],
                                 std::uint64_t       (&out)[N])
{
    for (int limb = 0; limb < N; ++limb)
    {
        out[limb] = 0;
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j + i < N; ++j)
        {
            const int k = i + j;
            const std::uint64_t low = lhs[i] * rhs[j];
            const std::uint64_t high = __umul64hi(lhs[i], rhs[j]);
            add_scalar(out, k, low);
            if (k + 1 < N)
            {
                add_scalar(out, k + 1, high);
            }
        }
    }
}

template <int N>
__device__ inline void negate_limbs(const std::uint64_t (&src)[N], std::uint64_t (&out)[N])
{
    std::uint64_t carry = 1;
    for (int limb = 0; limb < N; ++limb)
    {
        const std::uint64_t inverted = ~src[limb];
        out[limb] = inverted + carry;
        carry = carry != 0 && out[limb] == 0 ? 1ULL : 0ULL;
    }
}

template <int N>
__device__ inline int compare_unsigned_limbs(const std::uint64_t (&lhs)[N],
                                             const std::uint64_t (&rhs)[N])
{
    for (int limb = N - 1; limb >= 0; --limb)
    {
        if (lhs[limb] < rhs[limb])
        {
            return -1;
        }
        if (lhs[limb] > rhs[limb])
        {
            return 1;
        }
    }
    return 0;
}

template <int N>
__device__ inline int compare_signed_limbs(const std::uint64_t (&lhs)[N],
                                           const std::uint64_t (&rhs)[N])
{
    const bool lhs_negative = (lhs[N - 1] >> 63) != 0;
    const bool rhs_negative = (rhs[N - 1] >> 63) != 0;
    if (lhs_negative != rhs_negative)
    {
        return lhs_negative ? -1 : 1;
    }
    return compare_unsigned_limbs(lhs, rhs);
}

template <int N>
__device__ inline void shift_left_limbs(const std::uint64_t (&src)[N],
                                        unsigned int              shift,
                                        std::uint64_t       (&out)[N])
{
    constexpr unsigned int limb_bits = 64;
    const unsigned int total_bits = static_cast<unsigned int>(N * limb_bits);
    if (shift >= total_bits)
    {
        for (int limb = 0; limb < N; ++limb)
        {
            out[limb] = 0;
        }
        return;
    }

    const unsigned int limb_shift = shift / limb_bits;
    const unsigned int bit_shift = shift % limb_bits;
    for (int i = 0; i < N; ++i)
    {
        std::uint64_t value = 0;
        if (static_cast<unsigned int>(i) >= limb_shift)
        {
            value = src[i - static_cast<int>(limb_shift)] << bit_shift;
            if (bit_shift != 0 && static_cast<unsigned int>(i) > limb_shift)
            {
                value |= src[i - static_cast<int>(limb_shift) - 1] >> (limb_bits - bit_shift);
            }
        }
        out[i] = value;
    }
}

template <int N>
__device__ inline void shift_right_limbs(const std::uint64_t (&src)[N],
                                         unsigned int              shift,
                                         std::uint64_t       (&out)[N])
{
    constexpr unsigned int limb_bits = 64;
    const unsigned int total_bits = static_cast<unsigned int>(N * limb_bits);
    const std::uint64_t fill = (src[N - 1] >> 63) != 0 ? ~0ULL : 0ULL;
    if (shift >= total_bits)
    {
        for (int limb = 0; limb < N; ++limb)
        {
            out[limb] = fill;
        }
        return;
    }

    const unsigned int limb_shift = shift / limb_bits;
    const unsigned int bit_shift = shift % limb_bits;
    for (int i = 0; i < N; ++i)
    {
        const int low_index = i + static_cast<int>(limb_shift);
        const std::uint64_t low = low_index < N ? src[low_index] : fill;
        std::uint64_t value = bit_shift == 0 ? low : (low >> bit_shift);
        if (bit_shift != 0)
        {
            const int high_index = low_index + 1;
            const std::uint64_t high = high_index < N ? src[high_index] : fill;
            value |= high << (limb_bits - bit_shift);
        }
        out[i] = value;
    }
}

template <int N, BinaryOpType Op>
__global__ void wide_binary_kernel(const int64_t *x,
                                   const int64_t *y,
                                   int64_t       *out,
                                   int64_t        elements)
{
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    const int64_t offset = linear * N;
    std::uint64_t lhs[N];
    std::uint64_t rhs[N];
    std::uint64_t value[N];
    load_limbs<N>(x + offset, lhs);
    load_limbs<N>(y + offset, rhs);

    if constexpr (Op == BinaryOpType::Add)
    {
        add_limbs(lhs, rhs, value);
    }
    else if constexpr (Op == BinaryOpType::Sub)
    {
        sub_limbs(lhs, rhs, value);
    }
    else
    {
        mul_limbs(lhs, rhs, value);
    }

    store_limbs<N>(out + offset, value);
}

template <int N>
__global__ void wide_neg_kernel(const int64_t *x, int64_t *out, int64_t elements)
{
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    const int64_t offset = linear * N;
    std::uint64_t src[N];
    std::uint64_t value[N];
    load_limbs<N>(x + offset, src);
    negate_limbs(src, value);
    store_limbs<N>(out + offset, value);
}

template <int N, bool LeftShift>
__global__ void wide_shift_kernel(const int64_t *x, int64_t *out, int64_t elements, unsigned int shift)
{
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    const int64_t offset = linear * N;
    std::uint64_t src[N];
    std::uint64_t value[N];
    load_limbs<N>(x + offset, src);
    if constexpr (LeftShift)
    {
        shift_left_limbs(src, shift, value);
    }
    else
    {
        shift_right_limbs(src, shift, value);
    }
    store_limbs<N>(out + offset, value);
}

template <int N, CompareOpType Op>
__global__ void wide_compare_kernel(const int64_t *x,
                                    const int64_t *y,
                                    bool          *out,
                                    int64_t        elements)
{
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= elements)
    {
        return;
    }

    const int64_t offset = linear * N;
    std::uint64_t lhs[N];
    std::uint64_t rhs[N];
    load_limbs<N>(x + offset, lhs);
    load_limbs<N>(y + offset, rhs);
    const int compare = compare_signed_limbs(lhs, rhs);

    if constexpr (Op == CompareOpType::Eq)
    {
        out[linear] = compare == 0;
    }
    else if constexpr (Op == CompareOpType::Ge)
    {
        out[linear] = compare >= 0;
    }
    else if constexpr (Op == CompareOpType::Le)
    {
        out[linear] = compare <= 0;
    }
    else if constexpr (Op == CompareOpType::Gt)
    {
        out[linear] = compare > 0;
    }
    else
    {
        out[linear] = compare < 0;
    }
}

template <int N>
__global__ void wide_bmm_kernel(const int64_t *x,
                                const int64_t *y,
                                int64_t       *out,
                                int64_t        batch,
                                int64_t        rows,
                                int64_t        reduction,
                                int64_t        cols)
{
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = batch * rows * cols;
    if (linear >= total)
    {
        return;
    }

    const int64_t plane = rows * cols;
    const int64_t b = linear / plane;
    const int64_t rem = linear % plane;
    const int64_t row = rem / cols;
    const int64_t col = rem % cols;

    std::uint64_t acc[N];
    for (int limb = 0; limb < N; ++limb)
    {
        acc[limb] = 0;
    }

    for (int64_t kk = 0; kk < reduction; ++kk)
    {
        const int64_t lhs_offset = (((b * rows) + row) * reduction + kk) * N;
        const int64_t rhs_offset = (((b * reduction) + kk) * cols + col) * N;
        std::uint64_t lhs[N];
        std::uint64_t rhs[N];
        std::uint64_t product[N];
        load_limbs<N>(x + lhs_offset, lhs);
        load_limbs<N>(y + rhs_offset, rhs);
        mul_limbs(lhs, rhs, product);
        add_limbs(acc, product, acc);
    }

    const int64_t out_offset = (((b * rows) + row) * cols + col) * N;
    store_limbs<N>(out + out_offset, acc);
}

inline torch::Tensor maybe_copy_back(const torch::Tensor                &result,
                                     const c10::optional<torch::Tensor> &out)
{
    if (out.has_value())
    {
        out->copy_(result);
        return *out;
    }
    return result;
}

inline int64_t element_count_from_shape(const torchmm::wide::WideShape &shape)
{
    return shape[0] * shape[1] * shape[2];
}

template <BinaryOpType Op>
torch::Tensor launch_binary(const torch::Tensor                &x,
                            const torch::Tensor                &y,
                            const c10::optional<torch::Tensor> &out)
{
    const auto shape = torchmm::wide::compute_broadcast_shape(x, y);
    auto lhs = x.expand({shape[0], shape[1], shape[2], shape[3]}).contiguous();
    auto rhs = y.expand({shape[0], shape[1], shape[2], shape[3]}).contiguous();
    auto result = torch::empty({shape[0], shape[1], shape[2], shape[3]}, x.options());

    const int64_t elements = element_count_from_shape(shape);
    const int blocks = static_cast<int>((elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    switch (shape[3])
    {
    case 1:
        wide_binary_kernel<1, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 2:
        wide_binary_kernel<2, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 3:
        wide_binary_kernel<3, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 4:
        wide_binary_kernel<4, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    default:
        TORCH_CHECK(false, "CUDA wide ops only support limb counts in [1, 4]");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return maybe_copy_back(result, out);
}

torch::Tensor launch_neg(const torch::Tensor                &x,
                         const c10::optional<torch::Tensor> &out)
{
    const auto shape = torchmm::wide::get_shape4(x);
    auto input = x.contiguous();
    auto result = torch::empty({shape[0], shape[1], shape[2], shape[3]}, x.options());

    const int64_t elements = element_count_from_shape(shape);
    const int blocks = static_cast<int>((elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    switch (shape[3])
    {
    case 1:
        wide_neg_kernel<1><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 2:
        wide_neg_kernel<2><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 3:
        wide_neg_kernel<3><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    case 4:
        wide_neg_kernel<4><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements);
        break;
    default:
        TORCH_CHECK(false, "CUDA wide_neg only supports limb counts in [1, 4]");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return maybe_copy_back(result, out);
}

template <bool LeftShift>
torch::Tensor launch_shift(const torch::Tensor                &x,
                           int64_t                             shift,
                           const c10::optional<torch::Tensor> &out)
{
    const auto shape = torchmm::wide::get_shape4(x);
    auto input = x.contiguous();
    auto result = torch::empty({shape[0], shape[1], shape[2], shape[3]}, x.options());

    const int64_t elements = element_count_from_shape(shape);
    const int blocks = static_cast<int>((elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    const unsigned int shift_u = static_cast<unsigned int>(shift);

    switch (shape[3])
    {
    case 1:
        wide_shift_kernel<1, LeftShift><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements, shift_u);
        break;
    case 2:
        wide_shift_kernel<2, LeftShift><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements, shift_u);
        break;
    case 3:
        wide_shift_kernel<3, LeftShift><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements, shift_u);
        break;
    case 4:
        wide_shift_kernel<4, LeftShift><<<blocks, kThreadsPerBlock, 0, stream>>>(
            input.data_ptr<int64_t>(), result.data_ptr<int64_t>(), elements, shift_u);
        break;
    default:
        TORCH_CHECK(false, "CUDA wide shifts only support limb counts in [1, 4]");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return maybe_copy_back(result, out);
}

template <CompareOpType Op>
torch::Tensor launch_compare(const torch::Tensor &x, const torch::Tensor &y)
{
    const auto shape = torchmm::wide::compute_broadcast_shape(x, y);
    auto lhs = x.expand({shape[0], shape[1], shape[2], shape[3]}).contiguous();
    auto rhs = y.expand({shape[0], shape[1], shape[2], shape[3]}).contiguous();
    auto result = torch::empty({shape[0], shape[1], shape[2]}, x.options().dtype(torch::kBool));

    const int64_t elements = element_count_from_shape(shape);
    const int blocks = static_cast<int>((elements + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    switch (shape[3])
    {
    case 1:
        wide_compare_kernel<1, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<bool>(), elements);
        break;
    case 2:
        wide_compare_kernel<2, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<bool>(), elements);
        break;
    case 3:
        wide_compare_kernel<3, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<bool>(), elements);
        break;
    case 4:
        wide_compare_kernel<4, Op><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<bool>(), elements);
        break;
    default:
        TORCH_CHECK(false, "CUDA wide comparisons only support limb counts in [1, 4]");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor launch_bmm(const torch::Tensor                &x,
                         const torch::Tensor                &y,
                         const c10::optional<torch::Tensor> &out)
{
    const auto shape = torchmm::wide::compute_bmm_output_shape(x, y);
    auto lhs = x.expand({shape[0], x.size(1), x.size(2), x.size(3)}).contiguous();
    auto rhs = y.expand({shape[0], y.size(1), y.size(2), y.size(3)}).contiguous();
    auto result = torch::empty({shape[0], shape[1], shape[2], shape[3]}, x.options());

    const int64_t total = element_count_from_shape(shape);
    const int blocks = static_cast<int>((total + kThreadsPerBlock - 1) / kThreadsPerBlock);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    switch (shape[3])
    {
    case 1:
        wide_bmm_kernel<1><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(),
            shape[0], shape[1], x.size(2), shape[2]);
        break;
    case 2:
        wide_bmm_kernel<2><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(),
            shape[0], shape[1], x.size(2), shape[2]);
        break;
    case 3:
        wide_bmm_kernel<3><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(),
            shape[0], shape[1], x.size(2), shape[2]);
        break;
    case 4:
        wide_bmm_kernel<4><<<blocks, kThreadsPerBlock, 0, stream>>>(
            lhs.data_ptr<int64_t>(), rhs.data_ptr<int64_t>(), result.data_ptr<int64_t>(),
            shape[0], shape[1], x.size(2), shape[2]);
        break;
    default:
        TORCH_CHECK(false, "CUDA wide_bmm only supports limb counts in [1, 4]");
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return maybe_copy_back(result, out);
}

} // namespace

torch::Tensor wide_cuda_add(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out)
{
    return launch_binary<BinaryOpType::Add>(x, y, out);
}

torch::Tensor wide_cuda_sub(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out)
{
    return launch_binary<BinaryOpType::Sub>(x, y, out);
}

torch::Tensor wide_cuda_mul(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out)
{
    return launch_binary<BinaryOpType::Mul>(x, y, out);
}

torch::Tensor wide_cuda_neg(const torch::Tensor         &x,
                            c10::optional<torch::Tensor> out)
{
    return launch_neg(x, out);
}

torch::Tensor wide_cuda_shl(const torch::Tensor         &x,
                            int64_t                      shift,
                            c10::optional<torch::Tensor> out)
{
    return launch_shift<true>(x, shift, out);
}

torch::Tensor wide_cuda_shr(const torch::Tensor         &x,
                            int64_t                      shift,
                            c10::optional<torch::Tensor> out)
{
    return launch_shift<false>(x, shift, out);
}

torch::Tensor wide_cuda_bmm(const torch::Tensor         &x,
                            const torch::Tensor         &y,
                            c10::optional<torch::Tensor> out)
{
    return launch_bmm(x, y, out);
}

torch::Tensor wide_cuda_eq(const torch::Tensor &x, const torch::Tensor &y)
{
    return launch_compare<CompareOpType::Eq>(x, y);
}

torch::Tensor wide_cuda_ge(const torch::Tensor &x, const torch::Tensor &y)
{
    return launch_compare<CompareOpType::Ge>(x, y);
}

torch::Tensor wide_cuda_le(const torch::Tensor &x, const torch::Tensor &y)
{
    return launch_compare<CompareOpType::Le>(x, y);
}

torch::Tensor wide_cuda_gt(const torch::Tensor &x, const torch::Tensor &y)
{
    return launch_compare<CompareOpType::Gt>(x, y);
}

torch::Tensor wide_cuda_lt(const torch::Tensor &x, const torch::Tensor &y)
{
    return launch_compare<CompareOpType::Lt>(x, y);
}

} // namespace torchmm::cuda
