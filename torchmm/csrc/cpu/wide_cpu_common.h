#ifndef TORCHMM_CSRC_CPU_WIDE_CPU_COMMON_H
#define TORCHMM_CSRC_CPU_WIDE_CPU_COMMON_H

#include <torch/types.h>
#include <ATen/Parallel.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include "../wide/common.h"
#include "wideint/wideint.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace torchmm::wide::cpu_detail {

template <std::size_t N>
using wide_scalar = wideint::sint<N>;

using Strides4 = std::array<int64_t, 4>;

inline Strides4 get_strides4(const torch::Tensor &tensor)
{
    return {tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)};
}

inline int64_t storage_offset(const Strides4 &strides,
                              int64_t         i,
                              int64_t         j,
                              int64_t         k)
{
    return i * strides[0] + j * strides[1] + k * strides[2];
}

inline void configure_omp_threads()
{
#ifdef _OPENMP
    omp_set_num_threads(at::get_num_threads());
#endif
}

template <std::size_t N, class Scalar = wide_scalar<N>>
Scalar load_scalar(const int64_t *base,
                   const Strides4 &strides,
                   int64_t         i,
                   int64_t         j,
                   int64_t         k)
{
    Scalar value {};
    const int64_t offset = storage_offset(strides, i, j, k);
    for (std::size_t limb = 0; limb < N; ++limb)
    {
        value.limbs[limb] = static_cast<std::uint64_t>(
            base[offset + static_cast<int64_t>(limb) * strides[3]]);
    }
    return value;
}

template <std::size_t N, class Scalar = wide_scalar<N>>
void store_scalar(int64_t        *base,
                  const Strides4 &strides,
                  int64_t         i,
                  int64_t         j,
                  int64_t         k,
                  const Scalar   &value)
{
    const int64_t offset = storage_offset(strides, i, j, k);
    for (std::size_t limb = 0; limb < N; ++limb)
    {
        base[offset + static_cast<int64_t>(limb) * strides[3]] =
            static_cast<int64_t>(value.limbs[limb]);
    }
}

template <std::size_t N, class BinaryOp>
torch::Tensor binary_op_impl(const torch::Tensor                &x,
                             const torch::Tensor                &y,
                             const c10::optional<torch::Tensor> &out,
                             BinaryOp                           &&op)
{
    const WideShape shape = compute_broadcast_shape(x, y);
    auto            result = make_output_tensor(shape, x, out);

    const auto *x_ptr = x.data_ptr<int64_t>();
    const auto *y_ptr = y.data_ptr<int64_t>();
    auto       *out_ptr = result.data_ptr<int64_t>();
    const auto  x_strides = get_strides4(x);
    const auto  y_strides = get_strides4(y);
    const auto  out_strides = get_strides4(result);
    const int64_t total = shape[0] * shape[1] * shape[2];
    const int64_t plane = shape[1] * shape[2];

    configure_omp_threads();
#ifdef _OPENMP
#pragma omp parallel for if(total > 1)
#endif
    for (int64_t linear = 0; linear < total; ++linear)
    {
        const int64_t i = linear / plane;
        const int64_t rem = linear % plane;
        const int64_t j = rem / shape[2];
        const int64_t k = rem % shape[2];

        const int64_t xi = x.size(0) == 1 ? 0 : i;
        const int64_t xj = x.size(1) == 1 ? 0 : j;
        const int64_t xk = x.size(2) == 1 ? 0 : k;
        const int64_t yi = y.size(0) == 1 ? 0 : i;
        const int64_t yj = y.size(1) == 1 ? 0 : j;
        const int64_t yk = y.size(2) == 1 ? 0 : k;

        const auto lhs = load_scalar<N>(x_ptr, x_strides, xi, xj, xk);
        const auto rhs = load_scalar<N>(y_ptr, y_strides, yi, yj, yk);
        const auto value = op(lhs, rhs);
        store_scalar<N>(out_ptr, out_strides, i, j, k, value);
    }

    return result;
}

template <std::size_t N, class ShiftOp>
torch::Tensor shift_op_impl(const torch::Tensor                &x,
                            int64_t                             shift,
                            const c10::optional<torch::Tensor> &out,
                            ShiftOp                            &&op)
{
    const WideShape shape = get_shape4(x);
    auto            result = make_output_tensor(shape, x, out);

    const auto *x_ptr = x.data_ptr<int64_t>();
    auto       *out_ptr = result.data_ptr<int64_t>();
    const auto  x_strides = get_strides4(x);
    const auto  out_strides = get_strides4(result);
    const int64_t total = shape[0] * shape[1] * shape[2];
    const int64_t plane = shape[1] * shape[2];

    configure_omp_threads();
#ifdef _OPENMP
#pragma omp parallel for if(total > 1)
#endif
    for (int64_t linear = 0; linear < total; ++linear)
    {
        const int64_t i = linear / plane;
        const int64_t rem = linear % plane;
        const int64_t j = rem / shape[2];
        const int64_t k = rem % shape[2];

        const auto src = load_scalar<N>(x_ptr, x_strides, i, j, k);
        const auto value = op(src, static_cast<unsigned int>(shift));
        store_scalar<N>(out_ptr, out_strides, i, j, k, value);
    }

    return result;
}

template <std::size_t N, class UnaryOp>
torch::Tensor unary_op_impl(const torch::Tensor                &x,
                            const c10::optional<torch::Tensor> &out,
                            UnaryOp                            &&op)
{
    const WideShape shape = get_shape4(x);
    auto            result = make_output_tensor(shape, x, out);

    const auto *x_ptr = x.data_ptr<int64_t>();
    auto       *out_ptr = result.data_ptr<int64_t>();
    const auto  x_strides = get_strides4(x);
    const auto  out_strides = get_strides4(result);
    const int64_t total = shape[0] * shape[1] * shape[2];
    const int64_t plane = shape[1] * shape[2];

    configure_omp_threads();
#ifdef _OPENMP
#pragma omp parallel for if(total > 1)
#endif
    for (int64_t linear = 0; linear < total; ++linear)
    {
        const int64_t i = linear / plane;
        const int64_t rem = linear % plane;
        const int64_t j = rem / shape[2];
        const int64_t k = rem % shape[2];

        const auto src = load_scalar<N>(x_ptr, x_strides, i, j, k);
        const auto value = op(src);
        store_scalar<N>(out_ptr, out_strides, i, j, k, value);
    }

    return result;
}

template <std::size_t N, class CompareOp>
torch::Tensor compare_op_impl(const torch::Tensor &x, const torch::Tensor &y, CompareOp &&op)
{
    const WideShape shape = compute_broadcast_shape(x, y);
    auto            result =
        torch::empty({shape[0], shape[1], shape[2]}, x.options().dtype(torch::kBool));

    const auto *x_ptr = x.data_ptr<int64_t>();
    const auto *y_ptr = y.data_ptr<int64_t>();
    auto       *out_ptr = result.data_ptr<bool>();
    const auto  x_strides = get_strides4(x);
    const auto  y_strides = get_strides4(y);
    const int64_t total = shape[0] * shape[1] * shape[2];
    const int64_t plane = shape[1] * shape[2];

    configure_omp_threads();
#ifdef _OPENMP
#pragma omp parallel for if(total > 1)
#endif
    for (int64_t linear = 0; linear < total; ++linear)
    {
        const int64_t i = linear / plane;
        const int64_t rem = linear % plane;
        const int64_t j = rem / shape[2];
        const int64_t k = rem % shape[2];

        const int64_t xi = x.size(0) == 1 ? 0 : i;
        const int64_t xj = x.size(1) == 1 ? 0 : j;
        const int64_t xk = x.size(2) == 1 ? 0 : k;
        const int64_t yi = y.size(0) == 1 ? 0 : i;
        const int64_t yj = y.size(1) == 1 ? 0 : j;
        const int64_t yk = y.size(2) == 1 ? 0 : k;

        const auto lhs = load_scalar<N>(x_ptr, x_strides, xi, xj, xk);
        const auto rhs = load_scalar<N>(y_ptr, y_strides, yi, yj, yk);
        out_ptr[linear] = op(lhs, rhs);
    }

    return result;
}

template <std::size_t N>
torch::Tensor bmm_op_impl(const torch::Tensor                &x,
                          const torch::Tensor                &y,
                          const c10::optional<torch::Tensor> &out)
{
    const WideShape shape = compute_bmm_output_shape(x, y);
    auto            result = make_output_tensor(shape, x, out);

    const auto *x_ptr = x.data_ptr<int64_t>();
    const auto *y_ptr = y.data_ptr<int64_t>();
    auto       *out_ptr = result.data_ptr<int64_t>();
    const auto  x_strides = get_strides4(x);
    const auto  y_strides = get_strides4(y);
    const auto  out_strides = get_strides4(result);
    const int64_t total = shape[0] * shape[1] * shape[2];
    const int64_t plane = shape[1] * shape[2];
    const int64_t reduction = x.size(2);

    configure_omp_threads();
#ifdef _OPENMP
#pragma omp parallel for if(total > 1)
#endif
    for (int64_t linear = 0; linear < total; ++linear)
    {
        const int64_t batch = linear / plane;
        const int64_t rem = linear % plane;
        const int64_t row = rem / shape[2];
        const int64_t col = rem % shape[2];
        const int64_t xb = x.size(0) == 1 ? 0 : batch;
        const int64_t yb = y.size(0) == 1 ? 0 : batch;

        wide_scalar<N> acc = 0;
        for (int64_t kk = 0; kk < reduction; ++kk)
        {
            acc += load_scalar<N>(x_ptr, x_strides, xb, row, kk)
                 * load_scalar<N>(y_ptr, y_strides, yb, kk, col);
        }

        store_scalar<N>(out_ptr, out_strides, batch, row, col, acc);
    }

    return result;
}

} // namespace torchmm::wide::cpu_detail

#endif



