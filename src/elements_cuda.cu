#include "elements_cuda.h"

#include <cuda_runtime.h>

#include "constants.h"

namespace {

__global__ void compute_charges_kernel(const double* nx,
                                       const double* ny,
                                       const double* nz,
                                       const double* area,
                                       const double* potential,
                                       double* target_q,
                                       double* target_q_dx,
                                       double* target_q_dy,
                                       double* target_q_dz,
                                       double* source_q,
                                       double* source_q_dx,
                                       double* source_q_dy,
                                       double* source_q_dz,
                                       std::size_t num)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num) {
        const double one_over_4pi = constants::ONE_OVER_4PI;
        target_q[idx] = one_over_4pi;
        target_q_dx[idx] = one_over_4pi * nx[idx];
        target_q_dy[idx] = one_over_4pi * ny[idx];
        target_q_dz[idx] = one_over_4pi * nz[idx];

        const double area_val = area[idx];
        source_q[idx] = area_val * potential[num + idx];
        source_q_dx[idx] = nx[idx] * area_val * potential[idx];
        source_q_dy[idx] = ny[idx] * area_val * potential[idx];
        source_q_dz[idx] = nz[idx] * area_val * potential[idx];
    }
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

extern "C" void elements_compute_charges_cuda(
    const double* nx,
    const double* ny,
    const double* nz,
    const double* area,
    const double* potential,
    double* target_q,
    double* target_q_dx,
    double* target_q_dy,
    double* target_q_dz,
    double* source_q,
    double* source_q_dx,
    double* source_q_dy,
    double* source_q_dz,
    std::size_t num,
    void* stream)
{
    if (num == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    compute_charges_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        nx, ny, nz, area, potential,
        target_q, target_q_dx, target_q_dy, target_q_dz,
        source_q, source_q_dx, source_q_dy, source_q_dz,
        num);
}
