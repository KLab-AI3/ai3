// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ai3.hpp>

#define CONV2D_PARAMS                                                          \
    Tensor, const Tensor &, const std::optional<const Tensor> &, const uint,   \
        const uint, const uint, const uint, const uint, const uint,            \
        const PaddingMode, uint

namespace conv2d {
template <typename dtype> Tensor direct(CONV2D_PARAMS);
template <typename dtype> Tensor smm(CONV2D_PARAMS);
template <typename dtype> Tensor winograd(CONV2D_PARAMS);
template <typename dtype> Tensor implicit_gemm(CONV2D_PARAMS);
template <typename dtype> Tensor implicit_precomp_gemm(CONV2D_PARAMS);
template <typename dtype> Tensor gemm(CONV2D_PARAMS);
template <typename dtype> Tensor guess(CONV2D_PARAMS);
template <typename dtype> Tensor metal(CONV2D_PARAMS);
template <typename dtype> Tensor mps(CONV2D_PARAMS);
} // namespace conv2d

#define LINEAR_PARAMS                                                          \
    Tensor, const Tensor &, const std::optional<const Tensor> &

namespace linear {
template <typename dtype> Tensor gemm(LINEAR_PARAMS);
} // namespace linear

#define AVGPOOL2D_PARAMS                                                       \
    Tensor, const uint, const uint, const uint, const uint, const uint,        \
        const uint, const bool, const bool, const std::optional<int>

namespace avgpool2d {
template <typename dtype> Tensor direct(AVGPOOL2D_PARAMS);
} // namespace avgpool2d

#define ADAPTIVEAVGPOOL2D_PARAMS                                               \
    Tensor, const std::optional<uint>, const std::optional<uint>

namespace adaptiveavgpool2d {
template <typename dtype> Tensor direct(ADAPTIVEAVGPOOL2D_PARAMS);
} // namespace adaptiveavgpool2d

#define MAXPOOL2D_PARAMS                                                       \
    Tensor, const uint, const uint, const uint, const uint, const uint,        \
        const uint, const uint, const uint, const bool

namespace maxpool2d {
template <typename dtype> Tensor direct(MAXPOOL2D_PARAMS);
} // namespace maxpool2d

#define RELU_PARAMS Tensor

namespace relu {
template <typename dtype> Tensor direct(RELU_PARAMS);
}

#define FLATTEN_PARAMS Tensor, const uint, int

namespace flatten {
template <typename dtype> Tensor direct(FLATTEN_PARAMS);
}

#define MHA_PARAMS                                                             \
    Tensor, Tensor, Tensor, const mha::MemFormat, const Tensor &,              \
        const Tensor &, const Tensor &, const std::optional<const Tensor> &,   \
        const std::optional<const Tensor> &,                                   \
        const std::optional<const Tensor> &,                                   \
        const std::optional<const Tensor> &,                                   \
        const std::optional<const Tensor> &, const Tensor &,                   \
        const std::optional<const Tensor> &, const bool, const uint,           \
        const float, const std::optional<const Tensor> &,                      \
        const std::optional<const Tensor> &, const bool, const bool,           \
        const bool, const bool

namespace mha {
enum class MemFormat { NSE, SNE, NSHD, SNHD };

template <typename dtype> Tensor standard(MHA_PARAMS);

} // namespace mha

#if defined USE_CUBLAS
const bool USING_CUBLAS = true;
#else
const bool USING_CUBLAS = false;
#endif

#if defined USE_CUDNN
const bool USING_CUDNN = true;
#else
const bool USING_CUDNN = false;
#endif

#if defined USE_MPS_METAL
const bool USING_MPS_METAL = true;
#else
const bool USING_MPS_METAL = false;
#endif

#if defined USE_SYCL
const bool USING_SYCL = true;
#else
const bool USING_SYCL = false;
#endif

#if defined DEBUG_MODE
const bool DEBUG_BUILD = true;
#else
const bool DEBUG_BUILD = false;
#endif
