#include <cuda_runtime.h>

__global__ void chw_to_hwc(float *src, float *dst, int heads, int proj,
                           int embed) {
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= heads * proj || dy >= embed) {
        return;
    }
    int e = dy;
    int h = dx / proj;
    int p = dx % proj;
    dst[p * embed * heads + e * heads + h] =
        src[h * proj * embed + p * embed + e];
}

void chw_to_hwc_call(float *src, float *dst, int heads, int proj, int embed) {
    dim3 blockSize(32, 32);
    dim3 gridSize((heads * proj + blockSize.x - 1) / blockSize.x,
                  (embed + blockSize.y - 1) / blockSize.y);
    chw_to_hwc<<<gridSize, blockSize>>>(src, dst, heads, proj, embed);
}
