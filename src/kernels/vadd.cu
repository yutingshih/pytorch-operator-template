#include <cuda.h>
#include <torch/extension.h>

__global__ void vadd_(float* out, const float* in1, const float* in2, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

torch::Tensor vadd(torch::Tensor in1, torch::Tensor in2) {
    TORCH_CHECK(in1.is_cuda(), "in1 must be a CUDA tensor");
    TORCH_CHECK(in2.is_cuda(), "in2 must be a CUDA tensor");
    TORCH_CHECK(in1.sizes() == in2.sizes(), "input tensors must have the same shape");

    auto out = torch::empty_like(in1);

    int len = out.numel();
    int nthreads = 32;
    int nblocks = 1 + (len - 1) / nthreads;

    const float* in1_ = in1.data_ptr<float>();
    const float* in2_ = in2.data_ptr<float>();
    float* out_ = out.data_ptr<float>();

    vadd_<<<nblocks, nthreads>>>(out_, in1_, in2_, len);
    cudaDeviceSynchronize();

    return out;
}
