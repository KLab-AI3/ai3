#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{Conv2D}
 */
const bool CUSTOM_DEFAULT_CONV2D = false;

/**
 * @CUSTOM_OP{Conv2D,conv2d}
 */
template <typename dtype>
Tensor conv2d_custom(Tensor input, const Tensor &kernel,
                     const std::optional<const Tensor> &bias,
                     const uint padding_h, const uint padding_w,
                     const uint stride_h, const uint stride_w,
                     const uint dilation_h, const uint dilation_w,
                     const PaddingMode padding_mode, uint groups) {
    errs::no_user_def("conv2d");
}
