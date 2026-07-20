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

#if defined DEBUG_MODE
#define CUDNN_CHECK(status)                                                    \
    {                                                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudnnGetErrorString(status) << std::endl;     \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    }
#else
#define CUDNN_CHECK(status) (void)(status)
#endif
