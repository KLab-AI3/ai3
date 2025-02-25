#include <cuda_runtime.h>
#include <cuda_utils.cuh>
#include <cuda_utils.hpp>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

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
    int threads = TILE_DIM * BLOCK_ROWS;
    int blocks = (rows * columns + threads - 1) / threads;
    fill_identity<dtype><<<blocks, threads, 0, stream>>>(data, rows, columns);
}

template <typename dtype>
__global__ void transpose(dtype *output, dtype *input, const int rows,
                          const int columns) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < columns && (y + j) < rows) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * columns + x];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < rows && (y + j) < columns) {
            output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// TODO change the int param names based on if it is for input or output
template <typename dtype>
void transpose_call(dtype *output, dtype *input, const int rows,
                    const int columns, cudaStream_t stream) {
    int tiles_y = (rows + TILE_DIM - 1) / TILE_DIM;
    int tiles_x = (columns + TILE_DIM - 1) / TILE_DIM;
    int total_blocks = tiles_x * tiles_y;

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(total_blocks);

    transpose<dtype><<<grid, block, 0, stream>>>(output, input, rows, columns);
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
