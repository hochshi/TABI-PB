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

__global__ void compute_source_term_kernel(const double* elements_x,
                                           const double* elements_y,
                                           const double* elements_z,
                                           const double* elements_nx,
                                           const double* elements_ny,
                                           const double* elements_nz,
                                           const double* molecule_x,
                                           const double* molecule_y,
                                           const double* molecule_z,
                                           const double* molecule_q,
                                           double* elements_source_term,
                                           std::size_t num_elements,
                                           std::size_t num_atoms,
                                           double eps_solute)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        const double ex = elements_x[idx];
        const double ey = elements_y[idx];
        const double ez = elements_z[idx];
        const double enx = elements_nx[idx];
        const double eny = elements_ny[idx];
        const double enz = elements_nz[idx];

        double source_term_1 = 0.0;
        double source_term_2 = 0.0;
        const double inv_eps = 1.0 / eps_solute;
        const double one_over_4pi = constants::ONE_OVER_4PI;

        for (std::size_t j = 0; j < num_atoms; ++j) {
            const double dx = molecule_x[j] - ex;
            const double dy = molecule_y[j] - ey;
            const double dz = molecule_z[j] - ez;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz);
            const double inv_dist = 1.0 / dist;

            const double cos_theta = (enx * dx + eny * dy + enz * dz) * inv_dist;
            const double g0 = one_over_4pi * inv_dist;
            const double g1 = cos_theta * g0 * inv_dist;

            const double q = molecule_q[j];
            source_term_1 += q * g0 * inv_eps;
            source_term_2 += q * g1 * inv_eps;
        }

        elements_source_term[idx] += source_term_1;
        elements_source_term[num_elements + idx] += source_term_2;
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

extern "C" void elements_compute_source_term_cuda(
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* molecule_x,
    const double* molecule_y,
    const double* molecule_z,
    const double* molecule_q,
    double* elements_source_term,
    std::size_t num_elements,
    std::size_t num_atoms,
    double eps_solute,
    void* stream)
{
    if (num_elements == 0 || num_atoms == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num_elements, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    compute_source_term_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        elements_x, elements_y, elements_z,
        elements_nx, elements_ny, elements_nz,
        molecule_x, molecule_y, molecule_z, molecule_q,
        elements_source_term, num_elements, num_atoms, eps_solute);
}
