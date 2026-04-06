#include <stdexcept>
#include <utility>

#include "../wide/dispatch.h"
#include "wide_cpu_common.h"

namespace torchmm::wide {

template <std::size_t N>
torch::Tensor wide_add_dispatch(const torch::Tensor                &x,
                                const torch::Tensor                &y,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::binary_op_impl<N>(x, y, out, [](const auto &lhs, const auto &rhs) {
        return lhs + rhs;
    });
}

template <std::size_t N>
torch::Tensor wide_sub_dispatch(const torch::Tensor                &x,
                                const torch::Tensor                &y,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::binary_op_impl<N>(x, y, out, [](const auto &lhs, const auto &rhs) {
        return lhs - rhs;
    });
}

template <std::size_t N>
torch::Tensor wide_neg_dispatch(const torch::Tensor                &x,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::unary_op_impl<N>(x, out, [](const auto &value) { return -value; });
}

template <std::size_t N, class CompareOp>
torch::Tensor wide_compare_dispatch(const torch::Tensor &x, const torch::Tensor &y, CompareOp &&op)
{
    return cpu_detail::compare_op_impl<N>(x, y, std::forward<CompareOp>(op));
}

torch::Tensor wide_add_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_add_dispatch<1>(x, y, out);
    case 2: return wide_add_dispatch<2>(x, y, out);
    case 3: return wide_add_dispatch<3>(x, y, out);
    case 4: return wide_add_dispatch<4>(x, y, out);
    default: break;
    }

    throw std::invalid_argument("wide_add only supports limb counts in [1, 4]");
}

torch::Tensor wide_sub_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_sub_dispatch<1>(x, y, out);
    case 2: return wide_sub_dispatch<2>(x, y, out);
    case 3: return wide_sub_dispatch<3>(x, y, out);
    case 4: return wide_sub_dispatch<4>(x, y, out);
    default: break;
    }

    throw std::invalid_argument("wide_sub only supports limb counts in [1, 4]");
}

torch::Tensor wide_neg_cpu(const torch::Tensor &x, c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_neg_dispatch<1>(x, out);
    case 2: return wide_neg_dispatch<2>(x, out);
    case 3: return wide_neg_dispatch<3>(x, out);
    case 4: return wide_neg_dispatch<4>(x, out);
    default: break;
    }

    throw std::invalid_argument("wide_neg only supports limb counts in [1, 4]");
}

template <class CompareOp>
torch::Tensor wide_compare_cpu(const torch::Tensor &x, const torch::Tensor &y, CompareOp &&op)
{
    switch (x.size(-1))
    {
    case 1: return wide_compare_dispatch<1>(x, y, std::forward<CompareOp>(op));
    case 2: return wide_compare_dispatch<2>(x, y, std::forward<CompareOp>(op));
    case 3: return wide_compare_dispatch<3>(x, y, std::forward<CompareOp>(op));
    case 4: return wide_compare_dispatch<4>(x, y, std::forward<CompareOp>(op));
    default: break;
    }

    throw std::invalid_argument("wide comparisons only support limb counts in [1, 4]");
}

torch::Tensor wide_eq_cpu(const torch::Tensor &x, const torch::Tensor &y)
{
    return wide_compare_cpu(x, y, [](const auto &lhs, const auto &rhs) { return lhs == rhs; });
}

torch::Tensor wide_ge_cpu(const torch::Tensor &x, const torch::Tensor &y)
{
    return wide_compare_cpu(x, y, [](const auto &lhs, const auto &rhs) { return lhs >= rhs; });
}

torch::Tensor wide_le_cpu(const torch::Tensor &x, const torch::Tensor &y)
{
    return wide_compare_cpu(x, y, [](const auto &lhs, const auto &rhs) { return lhs <= rhs; });
}

torch::Tensor wide_gt_cpu(const torch::Tensor &x, const torch::Tensor &y)
{
    return wide_compare_cpu(x, y, [](const auto &lhs, const auto &rhs) { return lhs > rhs; });
}

torch::Tensor wide_lt_cpu(const torch::Tensor &x, const torch::Tensor &y)
{
    return wide_compare_cpu(x, y, [](const auto &lhs, const auto &rhs) { return lhs < rhs; });
}

} // namespace torchmm::wide
