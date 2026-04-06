#include <stdexcept>

#include "../wide/dispatch.h"
#include "wide_cpu_common.h"

namespace torchmm::wide {

template <std::size_t N>
torch::Tensor wide_bmm_dispatch(const torch::Tensor                &x,
                                const torch::Tensor                &y,
                                const c10::optional<torch::Tensor> &out)
{
    return cpu_detail::bmm_op_impl<N>(x, y, out);
}

torch::Tensor wide_bmm_cpu(const torch::Tensor &x,
                           const torch::Tensor &y,
                           c10::optional<torch::Tensor> out)
{
    switch (x.size(-1))
    {
    case 1: return wide_bmm_dispatch<1>(x, y, out);
    case 2: return wide_bmm_dispatch<2>(x, y, out);
    case 3: return wide_bmm_dispatch<3>(x, y, out);
    case 4: return wide_bmm_dispatch<4>(x, y, out);
    default: break;
    }

    throw std::invalid_argument("wide_bmm only supports limb counts in [1, 4]");
}

} // namespace torchmm::wide

