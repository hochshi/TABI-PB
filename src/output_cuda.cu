#include "output_cuda.h"

#include <cuda_runtime.h>

#include "constants.h"

namespace {

__global__ void coulombic_energy_kernel(const double* mol_x,
                                        const double* mol_y,
                                        const double* mol_z,
                                        const double* mol_q,
                                        std::size_t num_atoms,
                                        double eps_solute,
                                        double* out_energy)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_atoms) return;

    const double xi = mol_x[i];
    const double yi = mol_y[i];
    const double zi = mol_z[i];
    const double qi = mol_q[i];

    double acc = 0.0;
    for (std::size_t j = i + 1; j < num_atoms; ++j) {
        const double dx = xi - mol_x[j];
        const double dy = yi - mol_y[j];
        const double dz = zi - mol_z[j];
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        acc += qi * mol_q[j] / eps_solute / r;
    }

    atomicAdd(out_energy, acc);
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

extern "C" void output_coulombic_cuda(const double* mol_x,
                                      const double* mol_y,
                                      const double* mol_z,
                                      const double* mol_q,
                                      std::size_t num_atoms,
                                      double eps_solute,
                                      double* out_energy,
                                      void* stream)
{
    if (num_atoms == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num_atoms, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    coulombic_energy_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z, mol_q, num_atoms, eps_solute, out_energy);
}

namespace {

__global__ void solvation_energy_kernel(const double* elem_x,
                                        const double* elem_y,
                                        const double* elem_z,
                                        const double* elem_nx,
                                        const double* elem_ny,
                                        const double* elem_nz,
                                        const double* elem_area,
                                        const double* mol_x,
                                        const double* mol_y,
                                        const double* mol_z,
                                        const double* mol_q,
                                        const double* potential,
                                        std::size_t potential_offset,
                                        std::size_t num_elems,
                                        std::size_t num_atoms,
                                        double eps,
                                        double kappa,
                                        double* out_energy)
{
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_elems) return;

    const double ex = elem_x[i];
    const double ey = elem_y[i];
    const double ez = elem_z[i];
    const double enx = elem_nx[i];
    const double eny = elem_ny[i];
    const double enz = elem_nz[i];
    const double area = elem_area[i];

    const double pot = potential[i];
    const double pot_n = potential[potential_offset + i];

    double acc = 0.0;
    for (std::size_t j = 0; j < num_atoms; ++j) {
        const double dx = ex - mol_x[j];
        const double dy = ey - mol_y[j];
        const double dz = ez - mol_z[j];
        const double r = sqrt(dx * dx + dy * dy + dz * dz);
        const double inv_r = 1.0 / r;

        const double cos_theta = (enx * dx + eny * dy + enz * dz) * inv_r;
        const double kappa_r = kappa * r;
        const double exp_kappa_r = exp(-kappa_r);

        const double g0 = constants::ONE_OVER_4PI * inv_r;
        const double gk = exp_kappa_r * g0;
        const double g1 = cos_theta * g0 * inv_r;
        const double g2 = g1 * (1.0 + kappa_r) * exp_kappa_r;

        const double l1 = g1 - eps * g2;
        const double l2 = g0 - gk;

        acc += mol_q[j] * area * (l1 * pot + l2 * pot_n);
    }

    atomicAdd(out_energy, acc);
}

} // namespace

extern "C" void output_solvation_cuda(const double* elem_x,
                                      const double* elem_y,
                                      const double* elem_z,
                                      const double* elem_nx,
                                      const double* elem_ny,
                                      const double* elem_nz,
                                      const double* elem_area,
                                      const double* mol_x,
                                      const double* mol_y,
                                      const double* mol_z,
                                      const double* mol_q,
                                      const double* potential,
                                      std::size_t potential_offset,
                                      std::size_t num_elems,
                                      std::size_t num_atoms,
                                      double eps,
                                      double kappa,
                                      double* out_energy,
                                      void* stream)
{
    if (num_elems == 0 || num_atoms == 0) return;
    constexpr int kBlock = 256;
    const int grid = grid_for(num_elems, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    solvation_energy_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_nx, elem_ny, elem_nz,
        elem_area,
        mol_x, mol_y, mol_z, mol_q,
        potential, potential_offset,
        num_elems, num_atoms,
        eps, kappa, out_energy);
}
