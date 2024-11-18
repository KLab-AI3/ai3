// SPDX-License-Identifier: Apache-2.0

#include "exec_cudnn.hpp"
#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor conv2d::gemm(Tensor input, const Tensor &kernel,
                    const std::optional<const Tensor> &bias,
                    const uint padding_h, const uint padding_w,
                    const uint stride_h, const uint stride_w,
                    const uint dilation_h, const uint dilation_w,
                    const PaddingMode padding_mode, const uint groups) {
    return conv_bias_forward_with_algo<dtype>(
        std::move(input), kernel, bias, padding_h, padding_w, stride_h,
        stride_w, dilation_h, dilation_w, padding_mode, groups,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
}

template Tensor conv2d::gemm<float>(CONV2D_PARAMS);
template Tensor conv2d::gemm<double>(CONV2D_PARAMS);
