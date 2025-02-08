// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <cuda_utils.cuh>

template <typename T> inline cudaDataType cuda_data_type();

template <> inline cudaDataType cuda_data_type<float>() { return CUDA_R_32F; }

template <> inline cudaDataType cuda_data_type<double>() { return CUDA_R_64F; }

#ifdef DEBUG_MODE
#define CUDA_CHECK(status)                                                     \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "   \
                  << cudaGetErrorString(status) << std::endl;                  \
        std::exit(EXIT_FAILURE);                                               \
    }
#else
#define CUDA_CHECK(status) (void)(status)
#endif
