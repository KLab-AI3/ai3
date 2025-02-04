#pragma once

#include <cublas_v2.h>

#define CUBLAS_CHECK(status)                                                   \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBlas error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cublasGetStatusString(status) << std::endl;               \
        std::exit(EXIT_FAILURE);                                               \
    }
