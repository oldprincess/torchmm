#include <stdexcept>

#include "../wide/dispatch.h"
#include "wide_cpu_common.h"

namespace torchmm::wide {

template <std::size_t N>
torch::Tensor wide_shl_dispatch(const torch::Tensor                &x,
                                int64_t                             shift,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::shift_op_impl<N>(x, shift, out, [](const auto &value, unsigned int amount) {
        return value << amount;
    });
}

template <std::size_t N>
torch::Tensor wide_shr_dispatch(const torch::Tensor                &x,
                                int64_t                             shift,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::shift_op_impl<N>(x, shift, out, [](const auto &value, unsigned int amount) {
        return value >> amount;
    });
}

torch::Tensor wide_shl_cpu(const torch::Tensor &x,
                           int64_t              shift,
                           c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_shl_dispatch<1>(x, shift, out);
    case 2: return wide_shl_dispatch<2>(x, shift, out);
    case 3: return wide_shl_dispatch<3>(x, shift, out);
    case 4: return wide_shl_dispatch<4>(x, shift, out);
    default: break;
    }

    throw std::invalid_argument("wide_shl only supports limb counts in [1, 4]");
}

torch::Tensor wide_shr_cpu(const torch::Tensor &x,
                           int64_t              shift,
                           c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_shr_dispatch<1>(x, shift, out);
    case 2: return wide_shr_dispatch<2>(x, shift, out);
    case 3: return wide_shr_dispatch<3>(x, shift, out);
    case 4: return wide_shr_dispatch<4>(x, shift, out);
    default: break;
    }

    throw std::invalid_argument("wide_shr only supports limb counts in [1, 4]");
}

} // namespace torchmm::wide

