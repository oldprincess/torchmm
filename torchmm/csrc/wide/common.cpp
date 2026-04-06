#include "common.h"

#include <stdexcept>
#include <string>

namespace {

constexpr int64_t kMaxLimbs = 4;

void expect(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::invalid_argument(message);
    }
}

int64_t broadcast_dim(int64_t lhs, int64_t rhs, const char *name)
{
    if (lhs == rhs)
    {
        return lhs;
    }
    if (lhs == 1)
    {
        return rhs;
    }
    if (rhs == 1)
    {
        return lhs;
    }
    throw std::invalid_argument(std::string("cannot broadcast ") + name);
}

} // namespace

namespace torchmm::wide {

void validate_storage_tensor(const torch::Tensor &tensor, const char *name)
{
    expect(tensor.defined(), std::string(name) + " must be defined");
    expect(tensor.dtype() == torch::kInt64,
           std::string(name) + " must use torch.int64 storage");
    expect(tensor.dim() == 4,
           std::string(name)
               + " must have storage rank 4: [dim0, dim1, dim2, limbs]");
}

void validate_limb_count(const torch::Tensor &tensor, const char *name)
{
    validate_storage_tensor(tensor, name);
    int64_t limbs = tensor.size(-1);
    expect(limbs >= 1 && limbs <= kMaxLimbs,
           std::string(name) + " limb count must be in [1, 4]");
}

void validate_binary_inputs(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_limb_count(x, "x");
    validate_limb_count(y, "y");
    expect(x.device() == y.device(), "x and y must be on the same device");
    expect(x.size(-1) == y.size(-1), "x and y must have matching limb counts");
}

void validate_shift_input(const torch::Tensor &x, int64_t shift)
{
    validate_limb_count(x, "x");
    expect(shift >= 0, "shift must be non-negative");
}

void validate_bmm_inputs(const torch::Tensor &x, const torch::Tensor &y)
{
    validate_binary_inputs(x, y);
    expect(x.size(2) == y.size(1),
           "wide_bmm requires x.shape[2] == y.shape[1]");
    broadcast_dim(x.size(0), y.size(0), "batch dimensions");
}

void validate_optional_output(const c10::optional<torch::Tensor> &out,
                              int64_t                           limbs)
{
    if (!out.has_value())
    {
        return;
    }

    validate_limb_count(*out, "out");
    expect(out->size(-1) == limbs, "out must have matching limb count");
}

WideShape get_shape4(const torch::Tensor &tensor)
{
    return {tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)};
}

WideShape compute_broadcast_shape(const torch::Tensor &x, const torch::Tensor &y)
{
    return {
        broadcast_dim(x.size(0), y.size(0), "dim0"),
        broadcast_dim(x.size(1), y.size(1), "dim1"),
        broadcast_dim(x.size(2), y.size(2), "dim2"),
        x.size(3),
    };
}

WideShape compute_bmm_output_shape(const torch::Tensor &x, const torch::Tensor &y)
{
    return {
        broadcast_dim(x.size(0), y.size(0), "batch dimensions"),
        x.size(1),
        y.size(2),
        x.size(3),
    };
}

torch::Tensor make_output_tensor(const WideShape                     &shape,
                                 const torch::Tensor                &like,
                                 const c10::optional<torch::Tensor> &out)
{
    if (out.has_value())
    {
        expect(get_shape4(*out) == shape, "out must have the expected shape");
        expect(out->device() == like.device(), "out must be on the same device");
        return *out;
    }

    return torch::empty({shape[0], shape[1], shape[2], shape[3]}, like.options());
}

} // namespace torchmm::wide

