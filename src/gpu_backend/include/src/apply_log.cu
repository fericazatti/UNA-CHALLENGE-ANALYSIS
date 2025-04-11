#include <cuda_runtime.h>

__global__
void apply_log_magnitude(float* __restrict__ mag_data,
                         const int total_size,
                         const float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        mag_data[idx] = logf(mag_data[idx] + epsilon);
    }
}
