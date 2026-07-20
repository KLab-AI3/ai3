#pragma once

#include <ai3.hpp>

/**
 * @DEFAULT_BOOL{Flatten}
 */
const bool CUSTOM_DEFAULT_FLATTEN = false;

/**
 * @CUSTOM_OP{Flatten,flatten}
 */
template <typename dtype>
Tensor flatten_custom(Tensor input, const uint start_dim, int end_dim_orig) {
    errs::no_user_def("flatten");
}
