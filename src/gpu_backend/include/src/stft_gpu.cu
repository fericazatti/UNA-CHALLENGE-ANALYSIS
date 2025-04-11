#include "../include/stft_gpu.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <vector>

extern void apply_hann_and_normalize(const float*, float*, const float*, float, int);
extern void compute_magnitude(const cufftComplex*, float*, int);
extern void apply_log_magnitude(float*, int, float);

pybind11::array_t<float> compute_stft_gpu(pybind11::array_t<float> signal,
                                          int epoch_len,
                                          int n_fft,
                                          float epsilon) {
    auto buf = signal.request();
    float* signal_ptr = static_cast<float*>(buf.ptr);
    int total_len = buf.size;
    int n_epochs = total_len / epoch_len;
    int fft_out_len = n_fft / 2 + 1;

    // Allocate memory
    float* d_signal; float* d_windowed; float* d_hann; cufftComplex* d_fft;
    float* d_mag;
    cudaMalloc(&d_signal, total_len * sizeof(float));
    cudaMalloc(&d_windowed, total_len * sizeof(float));
    cudaMalloc(&d_hann, n_fft * sizeof(float));
    cudaMalloc(&d_fft, n_epochs * fft_out_len * sizeof(cufftComplex));
    cudaMalloc(&d_mag, n_epochs * fft_out_len * sizeof(float));

    cudaMemcpy(d_signal, signal_ptr, total_len * sizeof(float), cudaMemcpyHostToDevice);

    // Generate Hann window + norm
    std::vector<float> hann(n_fft);
    float norm = 0.0f;
    for (int i = 0; i < n_fft; ++i) {
        hann[i] = 0.5 * (1 - cosf(2 * M_PI * i / (n_fft - 1)));
        norm += hann[i] * hann[i];
    }
    norm = sqrtf(norm);
    cudaMemcpy(d_hann, hann.data(), n_fft * sizeof(float), cudaMemcpyHostToDevice);

    // Apply window
    apply_hann_and_normalize<<<n_epochs, epoch_len, n_fft * sizeof(float)>>>(d_signal, d_windowed, d_hann, norm, epoch_len);

    // FFT
    cufftHandle plan;
    cufftPlan1d(&plan, n_fft, CUFFT_R2C, n_epochs);
    cufftExecR2C(plan, d_windowed, d_fft);
    cufftDestroy(plan);

    // Magnitude
    compute_magnitude<<<n_epochs, fft_out_len>>>(d_fft, d_mag, fft_out_len);

    // Log
    int total = n_epochs * fft_out_len;
    apply_log_magnitude<<<(total + 255) / 256, 256>>>(d_mag, total, epsilon);

    // Copy to host
    std::vector<float> h_output(total);
    cudaMemcpy(h_output.data(), d_mag, total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_signal); cudaFree(d_windowed); cudaFree(d_hann);
    cudaFree(d_fft); cudaFree(d_mag);

    return pybind11::array_t<float>({n_epochs, fft_out_len}, h_output.data());
}
