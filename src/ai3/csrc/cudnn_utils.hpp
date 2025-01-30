// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudnn.h>
#include <iostream>
#include <sstream>

template <typename T> inline cudnnDataType_t cudnn_data_type();

template <> inline cudnnDataType_t cudnn_data_type<float>() {
    return CUDNN_DATA_FLOAT;
}

template <> inline cudnnDataType_t cudnn_data_type<double>() {
    return CUDNN_DATA_DOUBLE;
}

template <typename T> inline int64_t cudnn_byte_alignment();
template <> inline int64_t cudnn_byte_alignment<float>() { return 4; }

template <> inline int64_t cudnn_byte_alignment<double>() { return 8; }

#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudnnGetErrorString(status) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }
