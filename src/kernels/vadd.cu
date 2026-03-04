#include <cuda.h>
#include <torch/extension.h>

__global__ void vadd_(float* out, float* in1, float* in2, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

void vadd(torch::Tensor out, torch::Tensor in1, torch::Tensor in2) {
    TORCH_CHECK(in1.is_cuda(), "in1 must be a CUDA tensor");
    TORCH_CHECK(in2.is_cuda(), "in2 must be a CUDA tensor");
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.sizes() == in1.sizes() && out.sizes() == in2.sizes(), "input tensors must have the same shape");

    int len = out.numel();
    int nthreads = 32;
    int nblocks = 1 + (len - 1) / nthreads;

    float* in1_ = in1.data_ptr<float>();
    float* in2_ = in2.data_ptr<float>();
    float* out_ = out.data_ptr<float>();

    vadd_<<<nblocks, nthreads>>>(out_, in1_, in2_, len);
    cudaDeviceSynchronize();
}
