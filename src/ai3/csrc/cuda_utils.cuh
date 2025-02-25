// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#pragma once

template <typename dtype>
void fill_identity_call(dtype *dev_ptr, const int rows, const int columns,
                        cudaStream_t stream);
template <typename dtype>
void transpose_call(dtype *output, dtype *input, const int rows,
                    const int columns, cudaStream_t stream);

class StreamSwapper {
  public:
    StreamSwapper();
    ~StreamSwapper();

    void sync();
    cudaStream_t operator()();

  private:
    cudaStream_t streams[2];
    int current;
};
