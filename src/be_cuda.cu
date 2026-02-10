#include "be_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void clear4_kernel(double* a, double* b, double* c, double* d, std::size_t n)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = 0.0;
        b[idx] = 0.0;
        c[idx] = 0.0;
        d[idx] = 0.0;
    }
}

__global__ void potential_copy_zero_kernel(const double* in, double* temp, double* out, std::size_t n)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = in[idx];
        out[idx] = 0.0;
    }
}

__global__ void potential_combine_kernel(const double* old_ptr,
                                         const double* temp_ptr,
                                         double* new_ptr,
                                         std::size_t n,
                                         double alpha,
                                         double beta,
                                         double coeff_1,
                                         double coeff_2)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        double coeff = (idx < (n / 2)) ? coeff_1 : coeff_2;
        new_ptr[idx] = beta * temp_ptr[idx] + alpha * (coeff * old_ptr[idx] - new_ptr[idx]);
    }
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

extern "C" void be_clear_cluster_charges_cuda(
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    std::size_t num_charges,
    void* stream)
{
    if (num_charges == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num_charges, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    clear4_kernel<<<grid, kBlock, 0, cuda_stream>>>(clusters_q, clusters_q_dx, clusters_q_dy, clusters_q_dz, num_charges);
}

extern "C" void be_clear_cluster_potentials_cuda(
    double* clusters_p,
    double* clusters_p_dx,
    double* clusters_p_dy,
    double* clusters_p_dz,
    std::size_t num_potentials,
    void* stream)
{
    if (num_potentials == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num_potentials, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    clear4_kernel<<<grid, kBlock, 0, cuda_stream>>>(clusters_p, clusters_p_dx, clusters_p_dy, clusters_p_dz, num_potentials);
}

extern "C" void be_potential_copy_zero_cuda(
    const double* potential_new,
    double* potential_temp,
    double* potential_new_out,
    std::size_t count,
    void* stream)
{
    if (count == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(count, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    potential_copy_zero_kernel<<<grid, kBlock, 0, cuda_stream>>>(potential_new, potential_temp, potential_new_out, count);
}

extern "C" void be_potential_combine_cuda(
    const double* potential_old,
    const double* potential_temp,
    double* potential_new,
    std::size_t count,
    double alpha,
    double beta,
    double coeff_1,
    double coeff_2,
    void* stream)
{
    if (count == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(count, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    potential_combine_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        potential_old, potential_temp, potential_new, count, alpha, beta, coeff_1, coeff_2);
}
