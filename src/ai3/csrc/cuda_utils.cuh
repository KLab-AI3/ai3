#include <cuda_runtime.h> // TODO remove the include and see if works

void chw_to_hwc_call(float *src, float *dst, int heads, int proj, int embed);
