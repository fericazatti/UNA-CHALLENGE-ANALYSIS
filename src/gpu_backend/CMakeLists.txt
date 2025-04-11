cmake_minimum_required(VERSION 3.18)
project(cuda_stft LANGUAGES C CXX CUDA)

# Activar compilación con pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
enable_language(CUDA)

# Incluir headers
include_directories(include)

# Crear módulo Python
pybind11_add_module(cuda_stft
    bindings/pybind_stft.cpp
    src/apply_hann.cu
    src/compute_magnitude.cu
    src/apply_log.cu
    src/stft_gpu.cpp
)

# Linkear con cuFFT
target_link_libraries(cuda_stft PRIVATE cufft)
set_target_properties(cuda_stft PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
