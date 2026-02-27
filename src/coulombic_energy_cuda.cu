#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>

#include "coulombic_energy_cuda.h"

namespace {

__device__ inline double atomicAdd_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* addr_as_ull =
        reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

__global__ void coulombic_cc_kernel(
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    double* __restrict mol_clusters_p,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    int num_mol_interp_potentials_per_node,
    double inv_eps_solute)
{
    std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(num_mol_interp_potentials_per_node)) return;

    std::size_t target_interp_pts_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t target_potentials_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_potentials_per_node);
    std::size_t source_interp_pts_start =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t source_charges_start =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    int p = num_mol_interp_pts_per_node;
    std::size_t j1 = t / static_cast<std::size_t>(p * p);
    std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
    std::size_t j3 = t % static_cast<std::size_t>(p);

    std::size_t jj = target_potentials_start + t;

    double target_x = mol_clusters_x[target_interp_pts_start + j1];
    double target_y = mol_clusters_y[target_interp_pts_start + j2];
    double target_z = mol_clusters_z[target_interp_pts_start + j3];

    double pot_temp = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        double sx = mol_clusters_x[source_interp_pts_start + static_cast<std::size_t>(k1)];
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            double sy = mol_clusters_y[source_interp_pts_start + static_cast<std::size_t>(k2)];
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                double sz = mol_clusters_z[source_interp_pts_start + static_cast<std::size_t>(k3)];
                std::size_t kk = source_charges_start
                               + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                               + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                double dx = target_x - sx;
                double dy = target_y - sy;
                double dz = target_z - sz;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                pot_temp += mol_clusters_q[kk] * inv_eps_solute / r;
            }
        }
    }

    mol_clusters_p[jj] += pot_temp;
}

__global__ void coulombic_cp_kernel(
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    double* __restrict mol_clusters_p,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t target_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double inv_eps_solute)
{
    std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(num_mol_interp_potentials_per_node)) return;

    std::size_t target_interp_pts_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t target_potentials_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_potentials_per_node);

    int p = num_mol_interp_pts_per_node;
    std::size_t j1 = t / static_cast<std::size_t>(p * p);
    std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
    std::size_t j3 = t % static_cast<std::size_t>(p);

    std::size_t jj = target_potentials_start + t;

    double target_x = mol_clusters_x[target_interp_pts_start + j1];
    double target_y = mol_clusters_y[target_interp_pts_start + j2];
    double target_z = mol_clusters_z[target_interp_pts_start + j3];

    double pot_temp = 0.0;

    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = target_x - mol_x[k];
        double dy = target_y - mol_y[k];
        double dz = target_z - mol_z[k];
        double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        pot_temp += mol_q[k] * inv_eps_solute / r;
    }

    mol_clusters_p[jj] += pot_temp;
}

__global__ void coulombic_pc_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double inv_eps_solute,
    double* __restrict coul_eng)
{
    std::size_t j = target_begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= target_end) return;

    std::size_t interp_pts_start =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t charges_start =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    double target_x = mol_x[j];
    double target_y = mol_y[j];
    double target_z = mol_z[j];
    double target_q = mol_q[j];

    double pot_temp = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        double sx = mol_clusters_x[interp_pts_start + static_cast<std::size_t>(k1)];
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            double sy = mol_clusters_y[interp_pts_start + static_cast<std::size_t>(k2)];
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                double sz = mol_clusters_z[interp_pts_start + static_cast<std::size_t>(k3)];
                std::size_t kk = charges_start
                               + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                               + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                double dx = target_x - sx;
                double dy = target_y - sy;
                double dz = target_z - sz;
                double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                pot_temp += mol_clusters_q[kk] * inv_eps_solute / r;
            }
        }
    }

    pot_temp *= target_q;
    atomicAdd_double(coul_eng, pot_temp);
}

__global__ void coulombic_pp_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double inv_eps_solute,
    double* __restrict coul_eng)
{
    std::size_t j = target_begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= target_end) return;

    double target_x = mol_x[j];
    double target_y = mol_y[j];
    double target_z = mol_z[j];
    double target_q = mol_q[j];

    double pot_temp = 0.0;
    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = target_x - mol_x[k];
        double dy = target_y - mol_y[k];
        double dz = target_z - mol_z[k];
        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0) {
            pot_temp += target_q * mol_q[k] * inv_eps_solute / std::sqrt(r2);
        }
    }

    atomicAdd_double(coul_eng, pot_temp);
}

__global__ void coulombic_pp_batched_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    double* __restrict coul_eng,
    const std::uint32_t* __restrict target_node_begin,
    const std::uint32_t* __restrict target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict source_node_begin,
    const std::uint32_t* __restrict source_node_end,
    const std::uint32_t* __restrict pp_offsets,
    const std::uint32_t* __restrict pp_sources,
    double inv_eps_solute)
{
    const std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    const std::uint32_t target_begin = target_node_begin[target_node_idx];
    const std::uint32_t target_end = target_node_end[target_node_idx];
    const std::uint32_t list_begin = pp_offsets[target_node_idx];
    const std::uint32_t list_end = pp_offsets[target_node_idx + 1];

    for (std::uint32_t j = target_begin + static_cast<std::uint32_t>(threadIdx.x);
         j < target_end; j += static_cast<std::uint32_t>(blockDim.x)) {
        const double target_x = mol_x[j];
        const double target_y = mol_y[j];
        const double target_z = mol_z[j];
        const double target_q = mol_q[j];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            const std::uint32_t source_node_idx = pp_sources[list_idx];
            const std::uint32_t source_begin = source_node_begin[source_node_idx];
            const std::uint32_t source_end = source_node_end[source_node_idx];

            double pot_temp = 0.0;
            for (std::uint32_t k = source_begin; k < source_end; ++k) {
                const double dx = target_x - mol_x[k];
                const double dy = target_y - mol_y[k];
                const double dz = target_z - mol_z[k];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 > 0.0) {
                    pot_temp += target_q * mol_q[k] * inv_eps_solute / std::sqrt(r2);
                }
            }
            atomicAdd_double(coul_eng, pot_temp);
        }
    }
}

__global__ void coulombic_pc_batched_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    double* __restrict coul_eng,
    const std::uint32_t* __restrict target_node_begin,
    const std::uint32_t* __restrict target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict pc_offsets,
    const std::uint32_t* __restrict pc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double inv_eps_solute)
{
    const std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    const std::uint32_t target_begin = target_node_begin[target_node_idx];
    const std::uint32_t target_end = target_node_end[target_node_idx];
    const std::uint32_t list_begin = pc_offsets[target_node_idx];
    const std::uint32_t list_end = pc_offsets[target_node_idx + 1];

    for (std::uint32_t j = target_begin + static_cast<std::uint32_t>(threadIdx.x);
         j < target_end; j += static_cast<std::uint32_t>(blockDim.x)) {
        const double target_x = mol_x[j];
        const double target_y = mol_y[j];
        const double target_z = mol_z[j];
        const double target_q = mol_q[j];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            const std::size_t source_node_idx = static_cast<std::size_t>(pc_sources[list_idx]);
            const std::size_t interp_pts_start =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
            const std::size_t charges_start =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

            double pot_temp = 0.0;
            for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
                const double sx = mol_clusters_x[interp_pts_start + static_cast<std::size_t>(k1)];
                for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
                    const double sy = mol_clusters_y[interp_pts_start + static_cast<std::size_t>(k2)];
                    for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                        const double sz = mol_clusters_z[interp_pts_start + static_cast<std::size_t>(k3)];
                        const std::size_t kk = charges_start
                                             + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                                             + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                        const double dx = target_x - sx;
                        const double dy = target_y - sy;
                        const double dz = target_z - sz;
                        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        pot_temp += mol_clusters_q[kk] * inv_eps_solute / r;
                    }
                }
            }

            atomicAdd_double(coul_eng, pot_temp * target_q);
        }
    }
}

__global__ void coulombic_cp_batched_kernel(
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    double* __restrict mol_clusters_p,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict source_node_begin,
    const std::uint32_t* __restrict source_node_end,
    const std::uint32_t* __restrict cp_offsets,
    const std::uint32_t* __restrict cp_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_potentials_per_node,
    double inv_eps_solute)
{
    const std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    const std::uint32_t list_begin = cp_offsets[target_node_idx];
    const std::uint32_t list_end = cp_offsets[target_node_idx + 1];
    const std::size_t target_interp_pts_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    const std::size_t target_potentials_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_potentials_per_node);
    const std::size_t total = static_cast<std::size_t>(num_mol_interp_potentials_per_node);

    for (std::size_t t = static_cast<std::size_t>(threadIdx.x);
         t < total; t += static_cast<std::size_t>(blockDim.x)) {
        const int p = num_mol_interp_pts_per_node;
        const std::size_t j1 = t / static_cast<std::size_t>(p * p);
        const std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
        const std::size_t j3 = t % static_cast<std::size_t>(p);
        const std::size_t jj = target_potentials_start + t;

        const double target_x = mol_clusters_x[target_interp_pts_start + j1];
        const double target_y = mol_clusters_y[target_interp_pts_start + j2];
        const double target_z = mol_clusters_z[target_interp_pts_start + j3];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            const std::uint32_t source_node_idx = cp_sources[list_idx];
            const std::uint32_t source_begin = source_node_begin[source_node_idx];
            const std::uint32_t source_end = source_node_end[source_node_idx];

            double pot_temp = 0.0;
            for (std::uint32_t k = source_begin; k < source_end; ++k) {
                const double dx = target_x - mol_x[k];
                const double dy = target_y - mol_y[k];
                const double dz = target_z - mol_z[k];
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                pot_temp += mol_q[k] * inv_eps_solute / r;
            }
            mol_clusters_p[jj] += pot_temp;
        }
    }
}

__global__ void coulombic_cc_batched_kernel(
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    double* __restrict mol_clusters_p,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict cc_offsets,
    const std::uint32_t* __restrict cc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    int num_mol_interp_potentials_per_node,
    double inv_eps_solute)
{
    const std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    const std::uint32_t list_begin = cc_offsets[target_node_idx];
    const std::uint32_t list_end = cc_offsets[target_node_idx + 1];
    const std::size_t target_interp_pts_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    const std::size_t target_potentials_start =
        target_node_idx * static_cast<std::size_t>(num_mol_interp_potentials_per_node);
    const std::size_t total = static_cast<std::size_t>(num_mol_interp_potentials_per_node);

    for (std::size_t t = static_cast<std::size_t>(threadIdx.x);
         t < total; t += static_cast<std::size_t>(blockDim.x)) {
        const int p = num_mol_interp_pts_per_node;
        const std::size_t j1 = t / static_cast<std::size_t>(p * p);
        const std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
        const std::size_t j3 = t % static_cast<std::size_t>(p);
        const std::size_t jj = target_potentials_start + t;

        const double target_x = mol_clusters_x[target_interp_pts_start + j1];
        const double target_y = mol_clusters_y[target_interp_pts_start + j2];
        const double target_z = mol_clusters_z[target_interp_pts_start + j3];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            const std::size_t source_node_idx = static_cast<std::size_t>(cc_sources[list_idx]);
            const std::size_t source_interp_pts_start =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
            const std::size_t source_charges_start =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

            double pot_temp = 0.0;
            for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
                const double sx = mol_clusters_x[source_interp_pts_start + static_cast<std::size_t>(k1)];
                for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
                    const double sy = mol_clusters_y[source_interp_pts_start + static_cast<std::size_t>(k2)];
                    for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                        const double sz = mol_clusters_z[source_interp_pts_start + static_cast<std::size_t>(k3)];
                        const std::size_t kk = source_charges_start
                                             + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                                             + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                        const double dx = target_x - sx;
                        const double dy = target_y - sy;
                        const double dz = target_z - sz;
                        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                        pot_temp += mol_clusters_q[kk] * inv_eps_solute / r;
                    }
                }
            }
            mol_clusters_p[jj] += pot_temp;
        }
    }
}

__global__ void coulombic_up_denom_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict weights,
    std::size_t node_interp_pts_start,
    std::size_t particle_start,
    std::size_t num_particles,
    int num_mol_interp_pts_per_node,
    int* __restrict exact_idx_x,
    int* __restrict exact_idx_y,
    int* __restrict exact_idx_z,
    double* __restrict denominator)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    double denominator_x = 0.0;
    double denominator_y = 0.0;
    double denominator_z = 0.0;
    int ex = -1;
    int ey = -1;
    int ez = -1;

    double xx = mol_x[particle_start + i];
    double yy = mol_y[particle_start + i];
    double zz = mol_z[particle_start + i];

    for (int j = 0; j < num_mol_interp_pts_per_node; ++j) {
        double dist_x = xx - mol_clusters_x[node_interp_pts_start + static_cast<std::size_t>(j)];
        double dist_y = yy - mol_clusters_y[node_interp_pts_start + static_cast<std::size_t>(j)];
        double dist_z = zz - mol_clusters_z[node_interp_pts_start + static_cast<std::size_t>(j)];

        denominator_x += weights[j] / dist_x;
        denominator_y += weights[j] / dist_y;
        denominator_z += weights[j] / dist_z;

        const int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
        const int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
        const int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;

        ex = (ex > cx) ? ex : cx;
        ey = (ey > cy) ? ey : cy;
        ez = (ez > cz) ? ez : cz;
    }

    exact_idx_x[i] = ex;
    exact_idx_y[i] = ey;
    exact_idx_z[i] = ez;

    double denom = 1.0;
    if (ex == -1) denom /= denominator_x;
    if (ey == -1) denom /= denominator_y;
    if (ez == -1) denom /= denominator_z;
    denominator[i] = denom;
}

__global__ void coulombic_up_charges_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    double* __restrict mol_clusters_q,
    const double* __restrict weights,
    const int* __restrict exact_idx_x,
    const int* __restrict exact_idx_y,
    const int* __restrict exact_idx_z,
    const double* __restrict denominator,
    std::size_t node_interp_pts_start,
    std::size_t node_charges_start,
    std::size_t particle_start,
    std::size_t num_particles,
    int num_mol_interp_pts_per_node)
{
    std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t num_charges = static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node);
    if (t >= num_charges) return;

    int p = num_mol_interp_pts_per_node;
    std::size_t k1 = t / static_cast<std::size_t>(p * p);
    std::size_t k2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
    std::size_t k3 = t % static_cast<std::size_t>(p);

    std::size_t kk = node_charges_start + t;

    double cx = mol_clusters_x[node_interp_pts_start + k1];
    double w1 = weights[static_cast<int>(k1)];
    double cy = mol_clusters_y[node_interp_pts_start + k2];
    double w2 = weights[static_cast<int>(k2)];
    double cz = mol_clusters_z[node_interp_pts_start + k3];
    double w3 = weights[static_cast<int>(k3)];

    double q_temp = 0.0;

    for (std::size_t i = 0; i < num_particles; ++i) {
        double dist_x = mol_x[particle_start + i] - cx;
        double dist_y = mol_y[particle_start + i] - cy;
        double dist_z = mol_z[particle_start + i] - cz;

        double numerator = 1.0;

        if (exact_idx_x[i] == -1) {
            numerator *= w1 / dist_x;
        } else {
            if (exact_idx_x[i] != static_cast<int>(k1)) numerator = 0.0;
        }

        if (exact_idx_y[i] == -1) {
            numerator *= w2 / dist_y;
        } else {
            if (exact_idx_y[i] != static_cast<int>(k2)) numerator = 0.0;
        }

        if (exact_idx_z[i] == -1) {
            numerator *= w3 / dist_z;
        } else {
            if (exact_idx_z[i] != static_cast<int>(k3)) numerator = 0.0;
        }

        q_temp += mol_q[particle_start + i] * numerator * denominator[i];
    }

    mol_clusters_q[kk] += q_temp;
}

} // namespace

extern "C" void coulombic_cc_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    double* mol_clusters_p,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    int num_mol_interp_potentials_per_node,
    double eps_solute,
    void* stream)
{
    if (num_mol_interp_potentials_per_node <= 0) return;
    if (num_mol_interp_pts_per_node <= 0) return;

    double inv_eps_solute = 1.0 / eps_solute;
    int threads = 128;
    int blocks = (num_mol_interp_potentials_per_node + threads - 1) / threads;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_cc_kernel<<<blocks, threads, 0, cuda_stream>>>(
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        mol_clusters_q,
        mol_clusters_p,
        target_node_idx,
        source_node_idx,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        num_mol_interp_potentials_per_node,
        inv_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "coulombic_cc_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void coulombic_cp_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    double* mol_clusters_p,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double eps_solute,
    void* stream)
{
    if (num_mol_interp_potentials_per_node <= 0) return;
    if (source_end <= source_begin) return;

    double inv_eps_solute = 1.0 / eps_solute;
    int threads = 128;
    int blocks = (num_mol_interp_potentials_per_node + threads - 1) / threads;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_cp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        mol_clusters_p,
        mol_x, mol_y, mol_z, mol_q,
        target_node_idx,
        num_mol_interp_pts_per_node,
        num_mol_interp_potentials_per_node,
        source_begin, source_end,
        inv_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "coulombic_cp_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void coulombic_pc_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double eps_solute,
    double* coul_eng,
    void* stream)
{
    if (target_end <= target_begin) return;

    double inv_eps_solute = 1.0 / eps_solute;
    std::size_t count = target_end - target_begin;
    int threads = 128;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_pc_kernel<<<blocks, threads, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z, mol_q,
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        source_node_idx,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        target_begin,
        target_end,
        inv_eps_solute,
        coul_eng);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "coulombic_pc_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void coulombic_pp_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double eps_solute,
    double* coul_eng,
    void* stream)
{
    if (target_end <= target_begin) return;
    if (source_end <= source_begin) return;

    double inv_eps_solute = 1.0 / eps_solute;
    std::size_t count = target_end - target_begin;
    int threads = 128;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_pp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z, mol_q,
        target_begin,
        target_end,
        source_begin,
        source_end,
        inv_eps_solute,
        coul_eng);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "coulombic_pp_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void coulombic_up_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    double* mol_clusters_q,
    const double* weights,
    int* exact_idx_x,
    int* exact_idx_y,
    int* exact_idx_z,
    double* denominator,
    std::size_t node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    void* stream)
{
    if (num_particles == 0) return;
    if (num_mol_interp_pts_per_node <= 0) return;

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    int threads = 128;
    int blocks_particles = static_cast<int>((num_particles + threads - 1) / threads);

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t node_charges_start =
        node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    coulombic_up_denom_kernel<<<blocks_particles, threads, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z,
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        weights,
        node_interp_pts_start,
        particle_start,
        num_particles,
        num_mol_interp_pts_per_node,
        exact_idx_x, exact_idx_y, exact_idx_z,
        denominator);

    std::size_t num_charges = static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node);
    int blocks_charges = static_cast<int>((num_charges + threads - 1) / threads);

    coulombic_up_charges_kernel<<<blocks_charges, threads, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z,
        mol_q,
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        mol_clusters_q,
        weights,
        exact_idx_x, exact_idx_y, exact_idx_z,
        denominator,
        node_interp_pts_start,
        node_charges_start,
        particle_start,
        num_particles,
        num_mol_interp_pts_per_node);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "coulombic_up_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

extern "C" void coulombic_pp_batched_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    double* coul_eng,
    const std::uint32_t* target_node_begin,
    const std::uint32_t* target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* source_node_begin,
    const std::uint32_t* source_node_end,
    const std::uint32_t* pp_offsets,
    const std::uint32_t* pp_sources,
    double eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (!coul_eng) return;

    const double inv_eps_solute = 1.0 / eps_solute;
    constexpr int kBlockSize = 128;
    const int grid = static_cast<int>(num_target_nodes);
    const cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_pp_batched_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z, mol_q, coul_eng,
        target_node_begin, target_node_end, num_target_nodes,
        source_node_begin, source_node_end,
        pp_offsets, pp_sources,
        inv_eps_solute);
}

extern "C" void coulombic_pc_batched_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    double* coul_eng,
    const std::uint32_t* target_node_begin,
    const std::uint32_t* target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (num_mol_interp_pts_per_node <= 0) return;
    if (!coul_eng) return;

    const double inv_eps_solute = 1.0 / eps_solute;
    constexpr int kBlockSize = 128;
    const int grid = static_cast<int>(num_target_nodes);
    const cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_pc_batched_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        mol_x, mol_y, mol_z, mol_q,
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        coul_eng,
        target_node_begin, target_node_end, num_target_nodes,
        pc_offsets, pc_sources,
        num_mol_interp_pts_per_node, num_mol_interp_charges_per_node,
        inv_eps_solute);
}

extern "C" void coulombic_cp_batched_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    double* mol_clusters_p,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t num_target_nodes,
    const std::uint32_t* source_node_begin,
    const std::uint32_t* source_node_end,
    const std::uint32_t* cp_offsets,
    const std::uint32_t* cp_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_potentials_per_node,
    double eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (num_mol_interp_potentials_per_node <= 0) return;

    const double inv_eps_solute = 1.0 / eps_solute;
    constexpr int kBlockSize = 128;
    const int grid = static_cast<int>(num_target_nodes);
    const cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_cp_batched_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_p,
        mol_x, mol_y, mol_z, mol_q,
        num_target_nodes,
        source_node_begin, source_node_end,
        cp_offsets, cp_sources,
        num_mol_interp_pts_per_node,
        num_mol_interp_potentials_per_node,
        inv_eps_solute);
}

extern "C" void coulombic_cc_batched_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    double* mol_clusters_p,
    std::size_t num_target_nodes,
    const std::uint32_t* cc_offsets,
    const std::uint32_t* cc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    int num_mol_interp_potentials_per_node,
    double eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (num_mol_interp_potentials_per_node <= 0) return;

    const double inv_eps_solute = 1.0 / eps_solute;
    constexpr int kBlockSize = 128;
    const int grid = static_cast<int>(num_target_nodes);
    const cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    coulombic_cc_batched_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q, mol_clusters_p,
        num_target_nodes,
        cc_offsets, cc_sources,
        num_mol_interp_pts_per_node, num_mol_interp_charges_per_node,
        num_mol_interp_potentials_per_node,
        inv_eps_solute);
}
