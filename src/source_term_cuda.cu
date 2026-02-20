#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cfloat>

#include <cuda_runtime.h>

#include "source_term_cuda.h"

namespace {

__global__ void source_term_pp_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_nx,
    const double* __restrict elem_ny,
    const double* __restrict elem_nz,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double one_over_4pi_eps_solute,
    double* __restrict source_term,
    std::size_t source_term_offset)
{
    std::size_t j = target_begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= target_end) return;

    double tx = elem_x[j];
    double ty = elem_y[j];
    double tz = elem_z[j];

    double tnx = elem_nx[j];
    double tny = elem_ny[j];
    double tnz = elem_nz[j];

    double pot_temp_1  = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = mol_x[k] - tx;
        double dy = mol_y[k] - ty;
        double dz = mol_z[k] - tz;

        double r2 = dx * dx + dy * dy + dz * dz;
        double rinv = 1.0 / std::sqrt(r2);
        double G0 = one_over_4pi_eps_solute * rinv;
        double Gn = G0 * rinv * rinv;

        double q = mol_q[k];
        pot_temp_1  += G0 * q;
        pot_temp_dx += Gn * q * dx;
        pot_temp_dy += Gn * q * dy;
        pot_temp_dz += Gn * q * dz;
    }

    source_term[j] += pot_temp_1;
    source_term[j + source_term_offset] += tnx * pot_temp_dx
                                        +  tny * pot_temp_dy
                                        +  tnz * pot_temp_dz;
}

__global__ void source_term_pc_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_nx,
    const double* __restrict elem_ny,
    const double* __restrict elem_nz,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double one_over_4pi_eps_solute,
    double* __restrict source_term,
    std::size_t source_term_offset)
{
    std::size_t j = target_begin + static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= target_end) return;

    double tx = elem_x[j];
    double ty = elem_y[j];
    double tz = elem_z[j];

    double tnx = elem_nx[j];
    double tny = elem_ny[j];
    double tnz = elem_nz[j];

    std::size_t interp_pts_begin = source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t charges_begin = source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    double pot_temp_1  = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        double sx = mol_clusters_x[interp_pts_begin + static_cast<std::size_t>(k1)];
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            double sy = mol_clusters_y[interp_pts_begin + static_cast<std::size_t>(k2)];
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                double sz = mol_clusters_z[interp_pts_begin + static_cast<std::size_t>(k3)];
                std::size_t kk = charges_begin
                               + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                               + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                double dx = sx - tx;
                double dy = sy - ty;
                double dz = sz - tz;

                double r2 = dx * dx + dy * dy + dz * dz;
                double rinv = 1.0 / std::sqrt(r2);
                double G0 = one_over_4pi_eps_solute * rinv;
                double Gn = G0 * rinv * rinv;

                double q = mol_clusters_q[kk];
                pot_temp_1  += G0 * q;
                pot_temp_dx += Gn * q * dx;
                pot_temp_dy += Gn * q * dy;
                pot_temp_dz += Gn * q * dz;
            }
        }
    }

    source_term[j] += pot_temp_1;
    source_term[j + source_term_offset] += tnx * pot_temp_dx
                                        +  tny * pot_temp_dy
                                        +  tnz * pot_temp_dz;
}

__global__ void source_term_cp_kernel(
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t target_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double one_over_4pi_eps_solute)
{
    std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(num_elem_interp_potentials_per_node)) return;

    std::size_t target_cluster_interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t target_cluster_interp_potentials_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);

    int p = num_elem_interp_pts_per_node;
    std::size_t j1 = t / static_cast<std::size_t>(p * p);
    std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
    std::size_t j3 = t % static_cast<std::size_t>(p);

    std::size_t jj = target_cluster_interp_potentials_begin + t;

    double target_x = elem_clusters_x[target_cluster_interp_pts_begin + j1];
    double target_y = elem_clusters_y[target_cluster_interp_pts_begin + j2];
    double target_z = elem_clusters_z[target_cluster_interp_pts_begin + j3];

    double pot_temp_1  = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = mol_x[k] - target_x;
        double dy = mol_y[k] - target_y;
        double dz = mol_z[k] - target_z;

        double r2 = dx * dx + dy * dy + dz * dz;
        double rinv = 1.0 / std::sqrt(r2);
        double G0 = one_over_4pi_eps_solute * rinv;
        double Gn = G0 * rinv * rinv;

        double q = mol_q[k];
        pot_temp_1  += G0 * q;
        pot_temp_dx += Gn * q * dx;
        pot_temp_dy += Gn * q * dy;
        pot_temp_dz += Gn * q * dz;
    }

    elem_clusters_p[jj] += pot_temp_1;
    elem_clusters_p_dx[jj] += pot_temp_dx;
    elem_clusters_p_dy[jj] += pot_temp_dy;
    elem_clusters_p_dz[jj] += pot_temp_dz;
}

__global__ void source_term_cc_kernel(
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute)
{
    std::size_t t = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<std::size_t>(num_elem_interp_potentials_per_node)) return;

    std::size_t target_cluster_interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t target_cluster_interp_potentials_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);

    int p = num_elem_interp_pts_per_node;
    std::size_t j1 = t / static_cast<std::size_t>(p * p);
    std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
    std::size_t j3 = t % static_cast<std::size_t>(p);

    std::size_t jj = target_cluster_interp_potentials_begin + t;

    double target_x = elem_clusters_x[target_cluster_interp_pts_begin + j1];
    double target_y = elem_clusters_y[target_cluster_interp_pts_begin + j2];
    double target_z = elem_clusters_z[target_cluster_interp_pts_begin + j3];

    std::size_t source_cluster_interp_pts_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t source_cluster_interp_charges_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    double pot_temp_1  = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        double sx = mol_clusters_x[source_cluster_interp_pts_begin + static_cast<std::size_t>(k1)];
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            double sy = mol_clusters_y[source_cluster_interp_pts_begin + static_cast<std::size_t>(k2)];
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                double sz = mol_clusters_z[source_cluster_interp_pts_begin + static_cast<std::size_t>(k3)];
                std::size_t kk = source_cluster_interp_charges_begin
                               + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node)
                               + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                double dx = sx - target_x;
                double dy = sy - target_y;
                double dz = sz - target_z;

                double r2 = dx * dx + dy * dy + dz * dz;
                double rinv = 1.0 / std::sqrt(r2);
                double G0 = one_over_4pi_eps_solute * rinv;
                double Gn = G0 * rinv * rinv;

                double q = mol_clusters_q[kk];
                pot_temp_1  += G0 * q;
                pot_temp_dx += Gn * q * dx;
                pot_temp_dy += Gn * q * dy;
                pot_temp_dz += Gn * q * dz;
            }
        }
    }

    elem_clusters_p[jj] += pot_temp_1;
    elem_clusters_p_dx[jj] += pot_temp_dx;
    elem_clusters_p_dy[jj] += pot_temp_dy;
    elem_clusters_p_dz[jj] += pot_temp_dz;
}

__global__ void source_term_down_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_q_dx,
    const double* __restrict elem_q_dy,
    const double* __restrict elem_q_dz,
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    const double* __restrict elem_clusters_p,
    const double* __restrict elem_clusters_p_dx,
    const double* __restrict elem_clusters_p_dy,
    const double* __restrict elem_clusters_p_dz,
    const double* __restrict weights,
    std::size_t node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    double* __restrict source_term,
    std::size_t source_term_offset)
{
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t node_potentials_start =
        node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);

    std::size_t elem_idx = particle_start + i;

    double xx = elem_x[elem_idx];
    double yy = elem_y[elem_idx];
    double zz = elem_z[elem_idx];

    double denominator_x = 0.0;
    double denominator_y = 0.0;
    double denominator_z = 0.0;

    int exact_idx_x = -1;
    int exact_idx_y = -1;
    int exact_idx_z = -1;

    for (int j = 0; j < num_elem_interp_pts_per_node; ++j) {
        double dist_x = xx - elem_clusters_x[node_interp_pts_start + static_cast<std::size_t>(j)];
        double dist_y = yy - elem_clusters_y[node_interp_pts_start + static_cast<std::size_t>(j)];
        double dist_z = zz - elem_clusters_z[node_interp_pts_start + static_cast<std::size_t>(j)];

        denominator_x += weights[j] / dist_x;
        denominator_y += weights[j] / dist_y;
        denominator_z += weights[j] / dist_z;

        if (fabs(dist_x) < DBL_MIN) exact_idx_x = j;
        if (fabs(dist_y) < DBL_MIN) exact_idx_y = j;
        if (fabs(dist_z) < DBL_MIN) exact_idx_z = j;
    }

    double denominator = 1.0;
    if (exact_idx_x == -1) denominator /= denominator_x;
    if (exact_idx_y == -1) denominator /= denominator_y;
    if (exact_idx_z == -1) denominator /= denominator_z;

    double pot_temp_1  = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_elem_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_elem_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_elem_interp_pts_per_node; ++k3) {
                std::size_t kk = node_potentials_start
                               + static_cast<std::size_t>(k1 * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node)
                               + static_cast<std::size_t>(k2 * num_elem_interp_pts_per_node + k3);

                double dist_x = xx - elem_clusters_x[node_interp_pts_start + static_cast<std::size_t>(k1)];
                double dist_y = yy - elem_clusters_y[node_interp_pts_start + static_cast<std::size_t>(k2)];
                double dist_z = zz - elem_clusters_z[node_interp_pts_start + static_cast<std::size_t>(k3)];

                double numerator = 1.0;
                if (exact_idx_x == -1) {
                    numerator *= weights[k1] / dist_x;
                } else if (exact_idx_x != k1) {
                    numerator = 0.0;
                }

                if (exact_idx_y == -1) {
                    numerator *= weights[k2] / dist_y;
                } else if (exact_idx_y != k2) {
                    numerator = 0.0;
                }

                if (exact_idx_z == -1) {
                    numerator *= weights[k3] / dist_z;
                } else if (exact_idx_z != k3) {
                    numerator = 0.0;
                }

                pot_temp_1  += numerator * denominator * elem_clusters_p[kk];
                pot_temp_dx += numerator * denominator * elem_clusters_p_dx[kk];
                pot_temp_dy += numerator * denominator * elem_clusters_p_dy[kk];
                pot_temp_dz += numerator * denominator * elem_clusters_p_dz[kk];
            }
        }
    }

    double pot_temp_2 = elem_q_dx[elem_idx] * pot_temp_dx
                      + elem_q_dy[elem_idx] * pot_temp_dy
                      + elem_q_dz[elem_idx] * pot_temp_dz;

    source_term[elem_idx] += pot_temp_1;
    source_term[elem_idx + source_term_offset] += pot_temp_2;
}

__global__ void source_term_up_denom_kernel(
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

__global__ void source_term_up_charges_kernel(
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

__global__ void source_term_pp_batched_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_nx,
    const double* __restrict elem_ny,
    const double* __restrict elem_nz,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const std::uint32_t* __restrict target_node_begin,
    const std::uint32_t* __restrict target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict source_node_begin,
    const std::uint32_t* __restrict source_node_end,
    const std::uint32_t* __restrict pp_offsets,
    const std::uint32_t* __restrict pp_sources,
    double one_over_4pi_eps_solute,
    double* __restrict source_term,
    std::size_t source_term_offset)
{
    std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    std::uint32_t target_begin = target_node_begin[target_node_idx];
    std::uint32_t target_end = target_node_end[target_node_idx];
    std::uint32_t list_begin = pp_offsets[target_node_idx];
    std::uint32_t list_end = pp_offsets[target_node_idx + 1];

    for (std::uint32_t j = target_begin + static_cast<std::uint32_t>(threadIdx.x);
         j < target_end; j += static_cast<std::uint32_t>(blockDim.x)) {
        double tx = elem_x[j];
        double ty = elem_y[j];
        double tz = elem_z[j];
        double tnx = elem_nx[j];
        double tny = elem_ny[j];
        double tnz = elem_nz[j];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            double pot_temp_1 = 0.0;
            double pot_temp_dx = 0.0;
            double pot_temp_dy = 0.0;
            double pot_temp_dz = 0.0;

            std::uint32_t source_node_idx = pp_sources[list_idx];
            std::uint32_t source_begin = source_node_begin[source_node_idx];
            std::uint32_t source_end = source_node_end[source_node_idx];

            for (std::uint32_t k = source_begin; k < source_end; ++k) {
                double dx = mol_x[k] - tx;
                double dy = mol_y[k] - ty;
                double dz = mol_z[k] - tz;
                double r2 = dx * dx + dy * dy + dz * dz;
                double rinv = 1.0 / std::sqrt(r2);
                double G0 = one_over_4pi_eps_solute * rinv;
                double Gn = G0 * rinv * rinv;
                double q = mol_q[k];
                pot_temp_1 += G0 * q;
                pot_temp_dx += Gn * q * dx;
                pot_temp_dy += Gn * q * dy;
                pot_temp_dz += Gn * q * dz;
            }

            source_term[j] += pot_temp_1;
            source_term[j + source_term_offset] +=
                tnx * pot_temp_dx + tny * pot_temp_dy + tnz * pot_temp_dz;
        }
    }
}

__global__ void source_term_pc_batched_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_nx,
    const double* __restrict elem_ny,
    const double* __restrict elem_nz,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    const std::uint32_t* __restrict target_node_begin,
    const std::uint32_t* __restrict target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict pc_offsets,
    const std::uint32_t* __restrict pc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute,
    double* __restrict source_term,
    std::size_t source_term_offset)
{
    std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    std::uint32_t target_begin = target_node_begin[target_node_idx];
    std::uint32_t target_end = target_node_end[target_node_idx];
    std::uint32_t list_begin = pc_offsets[target_node_idx];
    std::uint32_t list_end = pc_offsets[target_node_idx + 1];

    for (std::uint32_t j = target_begin + static_cast<std::uint32_t>(threadIdx.x);
         j < target_end; j += static_cast<std::uint32_t>(blockDim.x)) {
        double tx = elem_x[j];
        double ty = elem_y[j];
        double tz = elem_z[j];
        double tnx = elem_nx[j];
        double tny = elem_ny[j];
        double tnz = elem_nz[j];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            double pot_temp_1 = 0.0;
            double pot_temp_dx = 0.0;
            double pot_temp_dy = 0.0;
            double pot_temp_dz = 0.0;

            std::size_t source_node_idx = static_cast<std::size_t>(pc_sources[list_idx]);
            std::size_t interp_pts_begin =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
            std::size_t charges_begin =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

            for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
                double sx = mol_clusters_x[interp_pts_begin + static_cast<std::size_t>(k1)];
                for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
                    double sy = mol_clusters_y[interp_pts_begin + static_cast<std::size_t>(k2)];
                    for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                        double sz = mol_clusters_z[interp_pts_begin + static_cast<std::size_t>(k3)];
                        std::size_t kk = charges_begin
                                       + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node *
                                                                  num_mol_interp_pts_per_node)
                                       + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);
                        double dx = sx - tx;
                        double dy = sy - ty;
                        double dz = sz - tz;
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double rinv = 1.0 / std::sqrt(r2);
                        double G0 = one_over_4pi_eps_solute * rinv;
                        double Gn = G0 * rinv * rinv;
                        double q = mol_clusters_q[kk];
                        pot_temp_1 += G0 * q;
                        pot_temp_dx += Gn * q * dx;
                        pot_temp_dy += Gn * q * dy;
                        pot_temp_dz += Gn * q * dz;
                    }
                }
            }

            source_term[j] += pot_temp_1;
            source_term[j + source_term_offset] +=
                tnx * pot_temp_dx + tny * pot_temp_dy + tnz * pot_temp_dz;
        }
    }
}

__global__ void source_term_cp_batched_kernel(
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict source_node_begin,
    const std::uint32_t* __restrict source_node_end,
    const std::uint32_t* __restrict cp_offsets,
    const std::uint32_t* __restrict cp_sources,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    double one_over_4pi_eps_solute)
{
    std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    std::uint32_t list_begin = cp_offsets[target_node_idx];
    std::uint32_t list_end = cp_offsets[target_node_idx + 1];
    std::size_t interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t charges_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);
    int p = num_elem_interp_pts_per_node;

    for (std::size_t t = static_cast<std::size_t>(threadIdx.x);
         t < static_cast<std::size_t>(num_elem_interp_potentials_per_node);
         t += static_cast<std::size_t>(blockDim.x)) {
        std::size_t j1 = t / static_cast<std::size_t>(p * p);
        std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
        std::size_t j3 = t % static_cast<std::size_t>(p);
        std::size_t jj = charges_begin + t;

        double tx = elem_clusters_x[interp_pts_begin + j1];
        double ty = elem_clusters_y[interp_pts_begin + j2];
        double tz = elem_clusters_z[interp_pts_begin + j3];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            double pot_temp_1 = 0.0;
            double pot_temp_dx = 0.0;
            double pot_temp_dy = 0.0;
            double pot_temp_dz = 0.0;

            std::uint32_t source_node_idx = cp_sources[list_idx];
            std::uint32_t source_begin = source_node_begin[source_node_idx];
            std::uint32_t source_end = source_node_end[source_node_idx];
            for (std::uint32_t k = source_begin; k < source_end; ++k) {
                double dx = mol_x[k] - tx;
                double dy = mol_y[k] - ty;
                double dz = mol_z[k] - tz;
                double r2 = dx * dx + dy * dy + dz * dz;
                double rinv = 1.0 / std::sqrt(r2);
                double G0 = one_over_4pi_eps_solute * rinv;
                double Gn = G0 * rinv * rinv;
                double q = mol_q[k];
                pot_temp_1 += G0 * q;
                pot_temp_dx += Gn * q * dx;
                pot_temp_dy += Gn * q * dy;
                pot_temp_dz += Gn * q * dz;
            }

            elem_clusters_p[jj] += pot_temp_1;
            elem_clusters_p_dx[jj] += pot_temp_dx;
            elem_clusters_p_dy[jj] += pot_temp_dy;
            elem_clusters_p_dz[jj] += pot_temp_dz;
        }
    }
}

__global__ void source_term_cc_batched_kernel(
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    std::size_t num_target_nodes,
    const std::uint32_t* __restrict cc_offsets,
    const std::uint32_t* __restrict cc_sources,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute)
{
    std::size_t target_node_idx = static_cast<std::size_t>(blockIdx.x);
    if (target_node_idx >= num_target_nodes) return;

    std::uint32_t list_begin = cc_offsets[target_node_idx];
    std::uint32_t list_end = cc_offsets[target_node_idx + 1];
    std::size_t target_interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t target_charges_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);
    int p = num_elem_interp_pts_per_node;

    for (std::size_t t = static_cast<std::size_t>(threadIdx.x);
         t < static_cast<std::size_t>(num_elem_interp_potentials_per_node);
         t += static_cast<std::size_t>(blockDim.x)) {
        std::size_t j1 = t / static_cast<std::size_t>(p * p);
        std::size_t j2 = (t / static_cast<std::size_t>(p)) % static_cast<std::size_t>(p);
        std::size_t j3 = t % static_cast<std::size_t>(p);
        std::size_t jj = target_charges_begin + t;

        double tx = elem_clusters_x[target_interp_pts_begin + j1];
        double ty = elem_clusters_y[target_interp_pts_begin + j2];
        double tz = elem_clusters_z[target_interp_pts_begin + j3];

        for (std::uint32_t list_idx = list_begin; list_idx < list_end; ++list_idx) {
            double pot_temp_1 = 0.0;
            double pot_temp_dx = 0.0;
            double pot_temp_dy = 0.0;
            double pot_temp_dz = 0.0;

            std::size_t source_node_idx = static_cast<std::size_t>(cc_sources[list_idx]);
            std::size_t source_interp_pts_begin =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
            std::size_t source_charges_begin =
                source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

            for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
                double sx = mol_clusters_x[source_interp_pts_begin + static_cast<std::size_t>(k1)];
                for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
                    double sy = mol_clusters_y[source_interp_pts_begin + static_cast<std::size_t>(k2)];
                    for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                        double sz = mol_clusters_z[source_interp_pts_begin + static_cast<std::size_t>(k3)];
                        std::size_t kk = source_charges_begin
                                       + static_cast<std::size_t>(k1 * num_mol_interp_pts_per_node *
                                                                  num_mol_interp_pts_per_node)
                                       + static_cast<std::size_t>(k2 * num_mol_interp_pts_per_node + k3);

                        double dx = sx - tx;
                        double dy = sy - ty;
                        double dz = sz - tz;
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double rinv = 1.0 / std::sqrt(r2);
                        double G0 = one_over_4pi_eps_solute * rinv;
                        double Gn = G0 * rinv * rinv;
                        double q = mol_clusters_q[kk];
                        pot_temp_1 += G0 * q;
                        pot_temp_dx += Gn * q * dx;
                        pot_temp_dy += Gn * q * dy;
                        pot_temp_dz += Gn * q * dz;
                    }
                }
            }

            elem_clusters_p[jj] += pot_temp_1;
            elem_clusters_p_dx[jj] += pot_temp_dx;
            elem_clusters_p_dy[jj] += pot_temp_dy;
            elem_clusters_p_dz[jj] += pot_temp_dz;
        }
    }
}

} // namespace

extern "C" void source_term_pp_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_nx,
    const double* elem_ny,
    const double* elem_nz,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double one_over_4pi_eps_solute,
    double* source_term,
    std::size_t source_term_offset,
    void* stream)
{
    if (target_end <= target_begin) return;

    std::size_t count = target_end - target_begin;
    int threads = 128;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_pp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_nx, elem_ny, elem_nz,
        mol_x, mol_y, mol_z, mol_q,
        target_begin, target_end,
        source_begin, source_end,
        one_over_4pi_eps_solute,
        source_term, source_term_offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "source_term_pp_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after the interaction loop.
}

extern "C" void source_term_pc_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_nx,
    const double* elem_ny,
    const double* elem_nz,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double one_over_4pi_eps_solute,
    double* source_term,
    std::size_t source_term_offset,
    void* stream)
{
    if (target_end <= target_begin) return;

    std::size_t count = target_end - target_begin;
    int threads = 128;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_pc_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_nx, elem_ny, elem_nz,
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        mol_clusters_q,
        source_node_idx,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        target_begin, target_end,
        one_over_4pi_eps_solute,
        source_term, source_term_offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "source_term_pc_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after the interaction loop.
}

extern "C" void source_term_cp_cuda(
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double one_over_4pi_eps_solute,
    void* stream)
{
    if (num_elem_interp_potentials_per_node <= 0) return;
    if (source_end <= source_begin) return;

    int threads = 128;
    int blocks = (num_elem_interp_potentials_per_node + threads - 1) / threads;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_cp_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        mol_x, mol_y, mol_z, mol_q,
        target_node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        source_begin, source_end,
        one_over_4pi_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "source_term_cp_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after the interaction loop.
}

extern "C" void source_term_cc_cuda(
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute,
    void* stream)
{
    if (num_elem_interp_potentials_per_node <= 0) return;
    if (num_mol_interp_pts_per_node <= 0) return;

    int threads = 128;
    int blocks = (num_elem_interp_potentials_per_node + threads - 1) / threads;
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_cc_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        target_node_idx, source_node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        one_over_4pi_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "source_term_cc_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after the interaction loop.
}

extern "C" void source_term_pp_batched_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_nx,
    const double* elem_ny,
    const double* elem_nz,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const std::uint32_t* target_node_begin,
    const std::uint32_t* target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* source_node_begin,
    const std::uint32_t* source_node_end,
    const std::uint32_t* pp_offsets,
    const std::uint32_t* pp_sources,
    double one_over_4pi_eps_solute,
    double* source_term,
    std::size_t source_term_offset,
    void* stream)
{
    if (num_target_nodes == 0) return;

    int threads = 128;
    int blocks = static_cast<int>(num_target_nodes);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_pp_batched_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_nx, elem_ny, elem_nz,
        mol_x, mol_y, mol_z, mol_q,
        target_node_begin, target_node_end, num_target_nodes,
        source_node_begin, source_node_end,
        pp_offsets, pp_sources,
        one_over_4pi_eps_solute,
        source_term, source_term_offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "source_term_pp_batched_cuda kernel launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}

extern "C" void source_term_pc_batched_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_nx,
    const double* elem_ny,
    const double* elem_nz,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    const std::uint32_t* target_node_begin,
    const std::uint32_t* target_node_end,
    std::size_t num_target_nodes,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute,
    double* source_term,
    std::size_t source_term_offset,
    void* stream)
{
    if (num_target_nodes == 0) return;

    int threads = 128;
    int blocks = static_cast<int>(num_target_nodes);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_pc_batched_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_nx, elem_ny, elem_nz,
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        target_node_begin, target_node_end, num_target_nodes,
        pc_offsets, pc_sources,
        num_mol_interp_pts_per_node, num_mol_interp_charges_per_node,
        one_over_4pi_eps_solute,
        source_term, source_term_offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "source_term_pc_batched_cuda kernel launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}

extern "C" void source_term_cp_batched_cuda(
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t num_target_nodes,
    const std::uint32_t* source_node_begin,
    const std::uint32_t* source_node_end,
    const std::uint32_t* cp_offsets,
    const std::uint32_t* cp_sources,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    double one_over_4pi_eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (num_elem_interp_potentials_per_node <= 0) return;

    int threads = 128;
    int blocks = static_cast<int>(num_target_nodes);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_cp_batched_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        mol_x, mol_y, mol_z, mol_q,
        num_target_nodes,
        source_node_begin, source_node_end,
        cp_offsets, cp_sources,
        num_elem_interp_pts_per_node, num_elem_interp_potentials_per_node,
        one_over_4pi_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "source_term_cp_batched_cuda kernel launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}

extern "C" void source_term_cc_batched_cuda(
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    std::size_t num_target_nodes,
    const std::uint32_t* cc_offsets,
    const std::uint32_t* cc_sources,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double one_over_4pi_eps_solute,
    void* stream)
{
    if (num_target_nodes == 0) return;
    if (num_elem_interp_potentials_per_node <= 0) return;

    int threads = 128;
    int blocks = static_cast<int>(num_target_nodes);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_cc_batched_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        num_target_nodes,
        cc_offsets, cc_sources,
        num_elem_interp_pts_per_node, num_elem_interp_potentials_per_node,
        num_mol_interp_pts_per_node, num_mol_interp_charges_per_node,
        one_over_4pi_eps_solute);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "source_term_cc_batched_cuda kernel launch failed: %s\n",
                     cudaGetErrorString(err));
    }
}

extern "C" void source_term_down_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_q_dx,
    const double* elem_q_dy,
    const double* elem_q_dz,
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    const double* elem_clusters_p,
    const double* elem_clusters_p_dx,
    const double* elem_clusters_p_dy,
    const double* elem_clusters_p_dz,
    const double* weights,
    std::size_t node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    double* source_term,
    std::size_t source_term_offset,
    void* stream)
{
    if (num_particles == 0) return;

    int threads = 128;
    int blocks = static_cast<int>((num_particles + threads - 1) / threads);
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    source_term_down_kernel<<<blocks, threads, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_q_dx, elem_q_dy, elem_q_dz,
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        weights,
        node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        particle_start,
        num_particles,
        source_term,
        source_term_offset);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "source_term_down_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after downward_pass finishes.
}

extern "C" void source_term_up_cuda(
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

    source_term_up_denom_kernel<<<blocks_particles, threads, 0, cuda_stream>>>(
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

    source_term_up_charges_kernel<<<blocks_charges, threads, 0, cuda_stream>>>(
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
        std::fprintf(stderr, "source_term_up_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // Stream is synchronized by OpenACC wait after upward_pass finishes.
}
