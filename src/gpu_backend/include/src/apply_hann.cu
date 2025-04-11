#include <cuda_runtime.h>

__global__
void apply_hann_and_normalize(const float* __restrict__ signal,
                               float* __restrict__ windowed_out,
                               const float* __restrict__ hann,
                               const float window_norm_factor,
                               const int epoch_length) {
    extern __shared__ float shared_window[];

    int tid = threadIdx.x;
    int epoch_id = blockIdx.x;
    int global_idx = epoch_id * epoch_length + tid;
    int local_idx  = epoch_id * epoch_length + tid;

    if (tid < epoch_length)
        shared_window[tid] = hann[tid];
    __syncthreads();

    if (tid < epoch_length) {
        float val = signal[global_idx];
        float w = shared_window[tid];
        windowed_out[local_idx] = (val * w) / window_norm_factor;
    }
}
