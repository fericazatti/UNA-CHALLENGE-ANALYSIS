#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/stft_gpu.hpp"

PYBIND11_MODULE(cuda_stft, m) {
    m.def("compute_stft_gpu", &compute_stft_gpu, "Compute STFT using CUDA");
}
