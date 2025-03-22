#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{AvgPool2D}
 */
const bool CUSTOM_DEFAULT_AVGPOOL2D = false;

/**
 * @CUSTOM_OP{AvgPool2D,avgpool2d}
 */
template <typename dtype>
Tensor avgpool2d_custom(Tensor input, const uint kernel_h, const uint kernel_w,
                        const uint padding_h, const uint padding_w,
                        const uint stride_h, const uint stride_w,
                        const bool ceil_mode, const bool count_include_pad,
                        const std::optional<int> divisor_override) {
    errs::no_user_def("avgpool2d");
}
