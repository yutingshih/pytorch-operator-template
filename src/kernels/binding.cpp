#include <pybind11/pybind11.h>
#include <torch/extension.h>

torch::Tensor vadd(torch::Tensor in1, torch::Tensor in2);

PYBIND11_MODULE(kernels, m) {
    m.def("vadd", &vadd);
}
