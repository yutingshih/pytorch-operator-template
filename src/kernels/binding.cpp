#include <pybind11/pybind11.h>
#include <torch/extension.h>

void vadd(torch::Tensor out, torch::Tensor in1, torch::Tensor in2);

PYBIND11_MODULE(kernels, m) {
    m.def("vadd", &vadd);
}
