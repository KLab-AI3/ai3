#include <cuda_runtime.h>
#include <cuda_utils.cuh>
#include <cuda_utils.hpp>

template <typename dtype>
__global__ void fill_identity(dtype *data, const int rows, const int columns) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < rows * columns) {
        int row = x / columns;
        int col = x % columns;
        data[x] = row == col ? 1 : 0;
    }
}

template <typename dtype>
void fill_identity_call(dtype *data, const int rows, const int columns,
                        cudaStream_t stream) {
    int threads = 256;
    int blocks = (rows * columns + threads - 1) / threads;
    fill_identity<dtype><<<blocks, threads, 0, stream>>>(data, rows, columns);
}

template <typename dtype>
__global__ void transpose(dtype *output, dtype *input, const int in_rows,
                          const int in_columns) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_rows * in_columns) {
        const int d = idx / in_columns;
        const int e = idx % in_columns;

        output[e * in_rows + d] = input[d * in_columns + e];
    }
}

template <typename dtype>
void transpose_call(dtype *output, dtype *input, const int in_rows,
                    const int in_columns, cudaStream_t stream) {
    const int total_elements = in_rows * in_columns;
    const int block = 256;
    const int grid = (total_elements + block - 1) / block;

    transpose<dtype>
        <<<grid, block, 0, stream>>>(output, input, in_rows, in_columns);
}

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

template void fill_identity_call<float>(float *, const int, const int,
                                        cudaStream_t);
template void fill_identity_call<double>(double *, const int, const int,
                                         cudaStream_t);
template void transpose_call<float>(float *, float *, const int, const int,
                                    cudaStream_t);
template void transpose_call<double>(double *, double *, const int, const int,
                                     cudaStream_t);
