#include "gmres_cuda.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

__global__ void dscal_kernel(double* x, double alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= alpha;
}

__global__ void daxpy_kernel(double* y, const double* x, double alpha, std::size_t n)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] += alpha * x[idx];
}

__global__ void dot_kernel(const double* x, const double* y, std::size_t n, double* partials)
{
    extern __shared__ double sdata[];
    const unsigned int tid = threadIdx.x;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    double sum = 0.0;
    while (idx < n) {
        sum += x[idx] * y[idx];
        idx += static_cast<std::size_t>(blockDim.x) * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

__global__ void nrm2_kernel(const double* x, std::size_t n, double* partials)
{
    extern __shared__ double sdata[];
    const unsigned int tid = threadIdx.x;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    double sum = 0.0;
    while (idx < n) {
        double v = x[idx];
        sum += v * v;
        idx += static_cast<std::size_t>(blockDim.x) * gridDim.x;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = sdata[0];
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

namespace {
inline cublasHandle_t get_cublas_handle(cudaStream_t stream)
{
    static cublasHandle_t handle = nullptr;
    static bool initialized = false;
    if (!initialized) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            return nullptr;
        }
        initialized = true;
    }
    cublasSetStream(handle, stream);
    return handle;
}
} // namespace

extern "C" void gmres_cuda_copy(double* dst, const double* src, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(dst, src, n * sizeof(double), cudaMemcpyDeviceToDevice, cuda_stream);
}

extern "C" void gmres_cuda_dscal(double* x, double alpha, std::size_t n, void* stream)
{
    constexpr int kBlock = 256;
    const int grid = grid_for(n, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    dscal_kernel<<<grid, kBlock, 0, cuda_stream>>>(x, alpha, n);
}

extern "C" void gmres_cuda_daxpy(double* y, const double* x, double alpha, std::size_t n, void* stream)
{
    constexpr int kBlock = 256;
    const int grid = grid_for(n, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    daxpy_kernel<<<grid, kBlock, 0, cuda_stream>>>(y, x, alpha, n);
}

extern "C" double gmres_cuda_ddot(const double* x, const double* y, std::size_t n,
                                  double* partials, std::size_t partials_len, void* stream)
{
    if (n == 0) return 0.0;
    constexpr int kBlock = 256;
    int grid = grid_for(n, kBlock);
    if (grid > static_cast<int>(partials_len)) {
        grid = static_cast<int>(partials_len);
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    dot_kernel<<<grid, kBlock, kBlock * sizeof(double), cuda_stream>>>(x, y, n, partials);
    std::vector<double> host_partials(grid, 0.0);
    cudaMemcpyAsync(host_partials.data(), partials, grid * sizeof(double), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    double sum = 0.0;
    for (int i = 0; i < grid; ++i) sum += host_partials[static_cast<std::size_t>(i)];
    return sum;
}

extern "C" double gmres_cuda_dnrm2(const double* x, std::size_t n,
                                   double* partials, std::size_t partials_len, void* stream)
{
    if (n == 0) return 0.0;
    constexpr int kBlock = 256;
    int grid = grid_for(n, kBlock);
    if (grid > static_cast<int>(partials_len)) {
        grid = static_cast<int>(partials_len);
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    nrm2_kernel<<<grid, kBlock, kBlock * sizeof(double), cuda_stream>>>(x, n, partials);
    std::vector<double> host_partials(grid, 0.0);
    cudaMemcpyAsync(host_partials.data(), partials, grid * sizeof(double), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    double sum = 0.0;
    for (int i = 0; i < grid; ++i) sum += host_partials[static_cast<std::size_t>(i)];
    return std::sqrt(sum);
}

extern "C" void gmres_cuda_dtrsv_upper(const double* a, std::size_t lda,
                                       double* x, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return;
    cublasDtrsv(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                static_cast<int>(n), a, static_cast<int>(lda), x, 1);
}

extern "C" void gmres_cuda_dgemv(const double* a, std::size_t lda,
                                 const double* x, double* y,
                                 std::size_t m, std::size_t n, void* stream)
{
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = get_cublas_handle(cuda_stream);
    if (!handle) return;
    const double alpha = 1.0;
    const double beta = 1.0;
    cublasDgemv(handle, CUBLAS_OP_N,
                static_cast<int>(m), static_cast<int>(n),
                &alpha, a, static_cast<int>(lda),
                x, 1, &beta, y, 1);
}
