#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{AdaptiveAvgPool2D}
 */
const bool CUSTOM_DEFAULT_ADAPTIVEAVGPOOL2D = false;

/**
 * @CUSTOM_OP{AdaptiveAvgPool2D,adaptiveavgpool2d}
 */
template <typename dtype>
Tensor adaptiveavgpool2d_custom(Tensor input,
                                const std::optional<uint> output_h,
                                const std::optional<uint> output_w) {
    errs::no_user_def("adaptiveavgpool2d");
}
