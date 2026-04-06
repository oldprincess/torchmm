#include "dispatch.h"

#include <stdexcept>

#ifdef TORCHMM_WITH_CUDA
#include "../cuda/matmul.h"
#include "../cuda/wide_cuda_ops.h"
#endif
#include "common.h"

namespace torchmm::wide {

namespace {

void validate_out_shape(const c10::optional<torch::Tensor> &out,
                        const WideShape                    &shape,
                        const char                         *message)
{
    if (out.has_value())
    {
        TORCH_CHECK(get_shape4(*out) == shape, message);
    }
}

} // namespace

torch::Tensor wide_add_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_sub_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_mul_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_neg_cpu(const torch::Tensor &x, c10::optional<torch::Tensor> out);
torch::Tensor wide_shl_cpu(const torch::Tensor &x,
                           int64_t              shift,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_shr_cpu(const torch::Tensor &x,
                           int64_t              shift,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_bmm_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out);
torch::Tensor wide_eq_cpu(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_ge_cpu(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_le_cpu(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_gt_cpu(const torch::Tensor &x, const torch::Tensor &y);
torch::Tensor wide_lt_cpu(const torch::Tensor &x, const torch::Tensor &y);

torch::Tensor wide_add(const torch::Tensor &x,
                       const torch::Tensor &y,
                       c10::optional<torch::Tensor> out)
{
    validate_binary_inputs(x, y);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, compute_broadcast_shape(x, y), "out must have matching broadcast shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_add(x, y, out);
    }
#endif

    return wide_add_cpu(x, y, out);
}

torch::Tensor wide_sub(const torch::Tensor &x,
                       const torch::Tensor &y,
                       c10::optional<torch::Tensor> out)
{
    validate_binary_inputs(x, y);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, compute_broadcast_shape(x, y), "out must have matching broadcast shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_sub(x, y, out);
    }
#endif

    return wide_sub_cpu(x, y, out);
}

torch::Tensor wide_mul(const torch::Tensor &x,
                       const torch::Tensor &y,
                       c10::optional<torch::Tensor> out)
{
    validate_binary_inputs(x, y);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, compute_broadcast_shape(x, y), "out must have matching broadcast shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_mul(x, y, out);
    }
#endif

    return wide_mul_cpu(x, y, out);
}

torch::Tensor wide_neg(const torch::Tensor &x, c10::optional<torch::Tensor> out)
{
    validate_limb_count(x, "x");
    validate_optional_output(out, x.size(-1));

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_neg(x, out);
    }
#endif

    return wide_neg_cpu(x, out);
}

torch::Tensor wide_lshift(const torch::Tensor &x,
                          int64_t              shift,
                          c10::optional<torch::Tensor> out)
{
    validate_shift_input(x, shift);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, get_shape4(x), "out must have matching shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_shl(x, shift, out);
    }
#endif

    return wide_shl_cpu(x, shift, out);
}

torch::Tensor wide_rshift(const torch::Tensor &x,
                          int64_t              shift,
                          c10::optional<torch::Tensor> out)
{
    validate_shift_input(x, shift);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, get_shape4(x), "out must have matching shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_shr(x, shift, out);
    }
#endif

    return wide_shr_cpu(x, shift, out);
}

torch::Tensor wide_bmm(const torch::Tensor &x,
                       const torch::Tensor &y,
                       c10::optional<torch::Tensor> out)
{
    validate_bmm_inputs(x, y);
    validate_optional_output(out, x.size(-1));
    validate_out_shape(out, compute_bmm_output_shape(x, y), "out must have matching bmm shape");

#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_bmm(x, y, out);
    }
#endif

    return wide_bmm_cpu(x, y, out);
}

torch::Tensor wide_eq(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_eq(x, y);
    }
#endif
    return wide_eq_cpu(x, y);
}

torch::Tensor wide_ge(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_ge(x, y);
    }
#endif
    return wide_ge_cpu(x, y);
}

torch::Tensor wide_le(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_le(x, y);
    }
#endif
    return wide_le_cpu(x, y);
}

torch::Tensor wide_gt(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_gt(x, y);
    }
#endif
    return wide_gt_cpu(x, y);
}

torch::Tensor wide_lt(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
#ifdef TORCHMM_WITH_CUDA
    if (x.is_cuda())
    {
        return torchmm::cuda::wide_cuda_lt(x, y);
    }
#endif
    return wide_lt_cpu(x, y);
}

} // namespace torchmm::wide
