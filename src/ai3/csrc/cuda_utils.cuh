// SPDX-License-Identifier: Apache-2.0
//
#include <cuda_runtime.h>

#pragma once

template <typename dtype>
void fill_identity_call(dtype *dev_ptr, const int rows, const int columns,
                        cudaStream_t stream);
template <typename dtype>
void transpose_call(dtype *output, dtype *input, const int in_rows,
                    const int in_columns, cudaStream_t stream);
