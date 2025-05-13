// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cuda_utils.hpp>

StreamSwapper::StreamSwapper() : current(0) {
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));
}

StreamSwapper::~StreamSwapper() {
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
}

void StreamSwapper::sync() {
    CUDA_CHECK(cudaStreamSynchronize(streams[0]));
    CUDA_CHECK(cudaStreamSynchronize(streams[1]));
}

cudaStream_t StreamSwapper::operator()() {
    current = 1 - current;
    return streams[current];
}
