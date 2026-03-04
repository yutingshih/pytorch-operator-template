# PyTorch Custom CUDA Operator Template

This repository is a **minimal template** for building a **PyTorch custom operator with a CUDA kernel**. It shows how to write your own CUDA computation, bind it to PyTorch, package it, and test it.

## Project Structure

- [`src/kernels/*.cu`](src/kernels): Example CUDA kernel (vector addition). Replace or extend this with your own kernels.
- [`src/kernels/binding.cpp`](src/kernels/binding.cpp): C++ / pybind11 bindings that expose your CUDA kernels as PyTorch-callable functions.
- [`CMakeLists.txt`](CMakeLists.txt): CMake configuration for building the C++/CUDA extension.
- [`pyproject.toml`](pyproject.toml): Python packaging configuration (name, version, dependencies, build system).
- [`tests/*.py`](tests): Example test script to verify that the custom operator works correctly with PyTorch.

## Prerequisites

Make sure you have:

- Python (3.10+ recommended)
- PyTorch with matching CUDA support
- CUDA Toolkit compatible with your PyTorch install
- CMake (>= 3.18)
- A C/C++ compiler and NVIDIA GPU drivers (e.g. `gcc`/`g++`)

## Quick Start

1. Use this repository as a template

   ```bash
   git clone <this-repo-url> your-operator-name
   cd your-operator-name
   ```

2. Implement your CUDA kernel

   - Add or modify `.cu` files under `src/kernels/`
   - Use `src/kernels/vadd.cu` as a reference for your own kernel implementations

3. Register the operator in C++

   - Edit `src/kernels/binding.cpp`
   - Declare your CUDA functions and bind them using `PYBIND11_MODULE`
   - Ensure the tensor shapes, dtypes, and devices are checked properly

4. Build and install the Python package

   Install in editable mode during development:

   ```bash
   pip install -e .
   ```

   This will trigger CMake/C++/CUDA compilation and produce a loadable PyTorch extension.

5. Run the tests

   ```bash
   python tests/main.py
   ```

   Adjust or extend the tests to cover your own operators and edge cases.

## Customizing the Operator

- Add new CUDA kernels
  - Create additional `.cu` files under `src/kernels/`
  - Implement your GPU logic (matrix ops, custom activations, convolution variants, etc.)

- Update bindings
  - Expose new functions in `binding.cpp`
  - Map them to clear, documented Python-facing APIs

- Adjust build configuration (if needed)
  - Update `CMakeLists.txt` to include new source files or compiler flags
  - Add extra include/library paths as required
