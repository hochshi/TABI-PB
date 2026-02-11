#include <cuda_runtime.h>

#include "precondition_cuda.h"

__global__ void precondition_diag_kernel(double* z, const double* r,
                                         std::size_t num,
                                         double coeff1, double coeff2)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = num * 2;
    if (idx >= total) return;
    double denom = (idx < num) ? coeff1 : coeff2;
    z[idx] = r[idx] / denom;
}

extern "C" void precondition_diag_cuda(double* z, const double* r,
                                       std::size_t num,
                                       double coeff1, double coeff2,
                                       void* stream)
{
    if (num == 0) return;
    std::size_t total = num * 2;
    const int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    precondition_diag_kernel<<<grid, block, 0, cuda_stream>>>(z, r, num, coeff1, coeff2);
}
