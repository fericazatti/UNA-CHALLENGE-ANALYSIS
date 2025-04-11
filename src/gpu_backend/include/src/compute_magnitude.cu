#include <cuda_runtime.h>
#include <cufft.h>

__global__
void compute_magnitude(const cufftComplex* __restrict__ fft_output,
                       float* __restrict__ magnitude_out,
                       const int fft_output_len) {
    int epoch_id = blockIdx.x;
    int freq_id = threadIdx.x;

    if (freq_id < fft_output_len) {
        int idx = epoch_id * fft_output_len + freq_id;
        cufftComplex z = fft_output[idx];
        float mag = sqrtf(z.x * z.x + z.y * z.y);
        magnitude_out[idx] = mag;
    }
}
