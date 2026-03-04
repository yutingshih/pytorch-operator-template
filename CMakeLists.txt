cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path, end='')"
    OUTPUT_VARIABLE _torch_cmake_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(Torch CONFIG REQUIRED PATHS "${_torch_cmake_prefix}")
find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS "${_torch_cmake_prefix}/../../lib"
    REQUIRED
)

file(GLOB KERNEL_CUDA_SOURCES
    CONFIGURE_DEPENDS
    src/kernels/*.cu
)
pybind11_add_module(
    kernels
    src/kernels/binding.cpp
    ${KERNEL_CUDA_SOURCES}
)
target_compile_features(kernels PRIVATE cxx_std_17)
set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES native)
target_link_libraries(kernels PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
install(TARGETS kernels LIBRARY DESTINATION .)
