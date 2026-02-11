#include <cuda_runtime.h>
#include <cfloat>

#include "constants.h"
#include "solvation_energy_cuda.h"

namespace {

__global__ void solvation_pp_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_q_dx,
    const double* __restrict elem_q_dy,
    const double* __restrict elem_q_dz,
    const double* __restrict elem_area,
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const double* __restrict potential,
    std::size_t potential_offset,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double eps,
    double kappa,
    double* __restrict solv_eng)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t j = target_begin + idx;
    if (j >= target_end) return;

    double target_x = elem_x[j];
    double target_y = elem_y[j];
    double target_z = elem_z[j];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = target_x - mol_x[k];
        double dy = target_y - mol_y[k];
        double dz = target_z - mol_z[k];

        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double rinv = 1.0 / r;
        double G0 = constants::ONE_OVER_4PI * rinv;
        double expkr = exp(-kappa * r);

        double L2 = G0 * (1.0 - expkr);
        double L1 = G0 * rinv * rinv * (1.0 - eps * expkr * (1.0 + kappa * r));

        double q = mol_q[k];
        pot_temp_dd += L2 * q;
        pot_temp_dx += L1 * q * dx;
        pot_temp_dy += L1 * q * dy;
        pot_temp_dz += L1 * q * dz;
    }

    double area = elem_area[j];
    double pot_temp_1 = potential[j + potential_offset] * area * pot_temp_dd;
    double pot_temp_2 = potential[j] * area *
                        (elem_q_dx[j] * pot_temp_dx +
                         elem_q_dy[j] * pot_temp_dy +
                         elem_q_dz[j] * pot_temp_dz);

    atomicAdd(solv_eng, pot_temp_1 + pot_temp_2);
}

__global__ void solvation_pc_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_q_dx,
    const double* __restrict elem_q_dy,
    const double* __restrict elem_q_dz,
    const double* __restrict elem_area,
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    const double* __restrict potential,
    std::size_t potential_offset,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double eps,
    double kappa,
    double* __restrict solv_eng)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t j = target_begin + idx;
    if (j >= target_end) return;

    std::size_t source_cluster_interp_pts_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t source_cluster_interp_charges_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    double target_x = elem_x[j];
    double target_y = elem_y[j];
    double target_z = elem_z[j];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                std::size_t kk = source_cluster_interp_charges_begin
                               + static_cast<std::size_t>(k1) * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                               + static_cast<std::size_t>(k2) * num_mol_interp_pts_per_node
                               + static_cast<std::size_t>(k3);

                double dx = target_x - mol_clusters_x[source_cluster_interp_pts_begin + k1];
                double dy = target_y - mol_clusters_y[source_cluster_interp_pts_begin + k2];
                double dz = target_z - mol_clusters_z[source_cluster_interp_pts_begin + k3];

                double r = sqrt(dx * dx + dy * dy + dz * dz);
                double rinv = 1.0 / r;
                double G0 = constants::ONE_OVER_4PI * rinv;
                double expkr = exp(-kappa * r);

                double L2 = G0 * (1.0 - expkr);
                double L1 = G0 * rinv * rinv * (1.0 - eps * expkr * (1.0 + kappa * r));

                double q = mol_clusters_q[kk];
                pot_temp_dd += L2 * q;
                pot_temp_dx += L1 * q * dx;
                pot_temp_dy += L1 * q * dy;
                pot_temp_dz += L1 * q * dz;
            }
        }
    }

    double area = elem_area[j];
    double pot_temp_1 = potential[j + potential_offset] * area * pot_temp_dd;
    double pot_temp_2 = potential[j] * area *
                        (elem_q_dx[j] * pot_temp_dx +
                         elem_q_dy[j] * pot_temp_dy +
                         elem_q_dz[j] * pot_temp_dz);

    atomicAdd(solv_eng, pot_temp_1 + pot_temp_2);
}

__global__ void solvation_cp_kernel(
    const double* __restrict mol_x,
    const double* __restrict mol_y,
    const double* __restrict mol_z,
    const double* __restrict mol_q,
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    std::size_t target_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double eps,
    double kappa)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node);
    if (idx >= total) return;

    int j1 = static_cast<int>(idx / (num_elem_interp_pts_per_node * num_elem_interp_pts_per_node));
    int rem = static_cast<int>(idx % (num_elem_interp_pts_per_node * num_elem_interp_pts_per_node));
    int j2 = rem / num_elem_interp_pts_per_node;
    int j3 = rem % num_elem_interp_pts_per_node;

    std::size_t target_cluster_interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t target_cluster_interp_potentials_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);

    std::size_t jj = target_cluster_interp_potentials_begin
                   + static_cast<std::size_t>(j1) * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                   + static_cast<std::size_t>(j2) * num_elem_interp_pts_per_node
                   + static_cast<std::size_t>(j3);

    double target_x = elem_clusters_x[target_cluster_interp_pts_begin + j1];
    double target_y = elem_clusters_y[target_cluster_interp_pts_begin + j2];
    double target_z = elem_clusters_z[target_cluster_interp_pts_begin + j3];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (std::size_t k = source_begin; k < source_end; ++k) {
        double dx = target_x - mol_x[k];
        double dy = target_y - mol_y[k];
        double dz = target_z - mol_z[k];

        double r = sqrt(dx * dx + dy * dy + dz * dz);
        double rinv = 1.0 / r;
        double G0 = constants::ONE_OVER_4PI * rinv;
        double expkr = exp(-kappa * r);

        double L2 = G0 * (1.0 - expkr);
        double L1 = G0 * rinv * rinv * (1.0 - eps * expkr * (1.0 + kappa * r));

        double q = mol_q[k];
        pot_temp_dd += L2 * q;
        pot_temp_dx += L1 * q * dx;
        pot_temp_dy += L1 * q * dy;
        pot_temp_dz += L1 * q * dz;
    }

    atomicAdd(&elem_clusters_p[jj], pot_temp_dd);
    atomicAdd(&elem_clusters_p_dx[jj], pot_temp_dx);
    atomicAdd(&elem_clusters_p_dy[jj], pot_temp_dy);
    atomicAdd(&elem_clusters_p_dz[jj], pot_temp_dz);
}

__global__ void solvation_cc_kernel(
    const double* __restrict mol_clusters_x,
    const double* __restrict mol_clusters_y,
    const double* __restrict mol_clusters_z,
    const double* __restrict mol_clusters_q,
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    double* __restrict elem_clusters_p,
    double* __restrict elem_clusters_p_dx,
    double* __restrict elem_clusters_p_dy,
    double* __restrict elem_clusters_p_dz,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double eps,
    double kappa)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node);
    if (idx >= total) return;

    int j1 = static_cast<int>(idx / (num_elem_interp_pts_per_node * num_elem_interp_pts_per_node));
    int rem = static_cast<int>(idx % (num_elem_interp_pts_per_node * num_elem_interp_pts_per_node));
    int j2 = rem / num_elem_interp_pts_per_node;
    int j3 = rem % num_elem_interp_pts_per_node;

    std::size_t target_cluster_interp_pts_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t target_cluster_interp_potentials_begin =
        target_node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);
    std::size_t source_cluster_interp_pts_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t source_cluster_interp_charges_begin =
        source_node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    std::size_t jj = target_cluster_interp_potentials_begin
                   + static_cast<std::size_t>(j1) * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                   + static_cast<std::size_t>(j2) * num_elem_interp_pts_per_node
                   + static_cast<std::size_t>(j3);

    double target_x = elem_clusters_x[target_cluster_interp_pts_begin + j1];
    double target_y = elem_clusters_y[target_cluster_interp_pts_begin + j2];
    double target_z = elem_clusters_z[target_cluster_interp_pts_begin + j3];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                std::size_t kk = source_cluster_interp_charges_begin
                               + static_cast<std::size_t>(k1) * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                               + static_cast<std::size_t>(k2) * num_mol_interp_pts_per_node
                               + static_cast<std::size_t>(k3);

                double dx = target_x - mol_clusters_x[source_cluster_interp_pts_begin + k1];
                double dy = target_y - mol_clusters_y[source_cluster_interp_pts_begin + k2];
                double dz = target_z - mol_clusters_z[source_cluster_interp_pts_begin + k3];

                double r = sqrt(dx * dx + dy * dy + dz * dz);
                double rinv = 1.0 / r;
                double G0 = constants::ONE_OVER_4PI * rinv;
                double expkr = exp(-kappa * r);

                double L2 = G0 * (1.0 - expkr);
                double L1 = G0 * rinv * rinv * (1.0 - eps * expkr * (1.0 + kappa * r));

                double q = mol_clusters_q[kk];
                pot_temp_dd += L2 * q;
                pot_temp_dx += L1 * q * dx;
                pot_temp_dy += L1 * q * dy;
                pot_temp_dz += L1 * q * dz;
            }
        }
    }

    atomicAdd(&elem_clusters_p[jj], pot_temp_dd);
    atomicAdd(&elem_clusters_p_dx[jj], pot_temp_dx);
    atomicAdd(&elem_clusters_p_dy[jj], pot_temp_dy);
    atomicAdd(&elem_clusters_p_dz[jj], pot_temp_dz);
}

__global__ void solvation_up_denom_kernel(
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
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    exact_idx_x[idx] = -1;
    exact_idx_y[idx] = -1;
    exact_idx_z[idx] = -1;

    double denominator_x = 0.0;
    double denominator_y = 0.0;
    double denominator_z = 0.0;
    int ex = -1;
    int ey = -1;
    int ez = -1;

    double xx = mol_x[particle_start + idx];
    double yy = mol_y[particle_start + idx];
    double zz = mol_z[particle_start + idx];

    for (int j = 0; j < num_mol_interp_pts_per_node; ++j) {
        double dist_x = xx - mol_clusters_x[node_interp_pts_start + j];
        double dist_y = yy - mol_clusters_y[node_interp_pts_start + j];
        double dist_z = zz - mol_clusters_z[node_interp_pts_start + j];

        denominator_x += weights[j] / dist_x;
        denominator_y += weights[j] / dist_y;
        denominator_z += weights[j] / dist_z;

        int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
        int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
        int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;

        ex = (ex > cx) ? ex : cx;
        ey = (ey > cy) ? ey : cy;
        ez = (ez > cz) ? ez : cz;
    }

    exact_idx_x[idx] = ex;
    exact_idx_y[idx] = ey;
    exact_idx_z[idx] = ez;

    double denom = 1.0;
    if (ex == -1) denom /= denominator_x;
    if (ey == -1) denom /= denominator_y;
    if (ez == -1) denom /= denominator_z;
    denominator[idx] = denom;
}

__global__ void solvation_up_charges_kernel(
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
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t num_charges = static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node) *
                              static_cast<std::size_t>(num_mol_interp_pts_per_node);
    if (idx >= num_charges) return;

    int k1 = static_cast<int>(idx / (num_mol_interp_pts_per_node * num_mol_interp_pts_per_node));
    int rem = static_cast<int>(idx % (num_mol_interp_pts_per_node * num_mol_interp_pts_per_node));
    int k2 = rem / num_mol_interp_pts_per_node;
    int k3 = rem % num_mol_interp_pts_per_node;

    std::size_t kk = node_charges_start
                   + static_cast<std::size_t>(k1) * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                   + static_cast<std::size_t>(k2) * num_mol_interp_pts_per_node
                   + static_cast<std::size_t>(k3);

    double cx = mol_clusters_x[node_interp_pts_start + k1];
    double cy = mol_clusters_y[node_interp_pts_start + k2];
    double cz = mol_clusters_z[node_interp_pts_start + k3];

    double w1 = weights[k1];
    double w2 = weights[k2];
    double w3 = weights[k3];

    double q_temp = 0.0;
    for (std::size_t i = 0; i < num_particles; ++i) {
        double dist_x = mol_x[particle_start + i] - cx;
        double dist_y = mol_y[particle_start + i] - cy;
        double dist_z = mol_z[particle_start + i] - cz;

        double numerator = 1.0;
        if (exact_idx_x[i] == -1) {
            numerator *= w1 / dist_x;
        } else if (exact_idx_x[i] != k1) {
            numerator = 0.0;
        }

        if (exact_idx_y[i] == -1) {
            numerator *= w2 / dist_y;
        } else if (exact_idx_y[i] != k2) {
            numerator = 0.0;
        }

        if (exact_idx_z[i] == -1) {
            numerator *= w3 / dist_z;
        } else if (exact_idx_z[i] != k3) {
            numerator = 0.0;
        }

        q_temp += mol_q[particle_start + i] * numerator * denominator[i];
    }

    atomicAdd(&mol_clusters_q[kk], q_temp);
}

__global__ void solvation_down_kernel(
    const double* __restrict elem_x,
    const double* __restrict elem_y,
    const double* __restrict elem_z,
    const double* __restrict elem_q_dx,
    const double* __restrict elem_q_dy,
    const double* __restrict elem_q_dz,
    const double* __restrict elem_area,
    const double* __restrict elem_clusters_x,
    const double* __restrict elem_clusters_y,
    const double* __restrict elem_clusters_z,
    const double* __restrict elem_clusters_p,
    const double* __restrict elem_clusters_p_dx,
    const double* __restrict elem_clusters_p_dy,
    const double* __restrict elem_clusters_p_dz,
    const double* __restrict potential,
    std::size_t potential_offset,
    const double* __restrict weights,
    std::size_t node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    double* __restrict solv_eng)
{
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    std::size_t node_interp_pts_start = node_idx * static_cast<std::size_t>(num_elem_interp_pts_per_node);
    std::size_t node_potentials_start = node_idx * static_cast<std::size_t>(num_elem_interp_potentials_per_node);

    std::size_t i = particle_start + idx;
    double xx = elem_x[i];
    double yy = elem_y[i];
    double zz = elem_z[i];

    double denominator_x = 0.0;
    double denominator_y = 0.0;
    double denominator_z = 0.0;
    int exact_idx_x = -1;
    int exact_idx_y = -1;
    int exact_idx_z = -1;

    for (int j = 0; j < num_elem_interp_pts_per_node; ++j) {
        double dist_x = xx - elem_clusters_x[node_interp_pts_start + j];
        double dist_y = yy - elem_clusters_y[node_interp_pts_start + j];
        double dist_z = zz - elem_clusters_z[node_interp_pts_start + j];

        denominator_x += weights[j] / dist_x;
        denominator_y += weights[j] / dist_y;
        denominator_z += weights[j] / dist_z;

        int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
        int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
        int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;

        exact_idx_x = (exact_idx_x > cx) ? exact_idx_x : cx;
        exact_idx_y = (exact_idx_y > cy) ? exact_idx_y : cy;
        exact_idx_z = (exact_idx_z > cz) ? exact_idx_z : cz;
    }

    double denominator = 1.0;
    if (exact_idx_x == -1) denominator /= denominator_x;
    if (exact_idx_y == -1) denominator /= denominator_y;
    if (exact_idx_z == -1) denominator /= denominator_z;

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < num_elem_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_elem_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_elem_interp_pts_per_node; ++k3) {
                std::size_t kk = node_potentials_start
                               + static_cast<std::size_t>(k1) * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                               + static_cast<std::size_t>(k2) * num_elem_interp_pts_per_node
                               + static_cast<std::size_t>(k3);

                double dist_x = xx - elem_clusters_x[node_interp_pts_start + k1];
                double dist_y = yy - elem_clusters_y[node_interp_pts_start + k2];
                double dist_z = zz - elem_clusters_z[node_interp_pts_start + k3];

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

                double factor = numerator * denominator;
                pot_temp_dd += factor * elem_clusters_p[kk];
                pot_temp_dx += factor * elem_clusters_p_dx[kk];
                pot_temp_dy += factor * elem_clusters_p_dy[kk];
                pot_temp_dz += factor * elem_clusters_p_dz[kk];
            }
        }
    }

    double area = elem_area[i];
    double pot_temp_1 = potential[i + potential_offset] * area * pot_temp_dd;
    double pot_temp_2 = potential[i] * area *
                        (elem_q_dx[i] * pot_temp_dx + elem_q_dy[i] * pot_temp_dy + elem_q_dz[i] * pot_temp_dz);

    atomicAdd(solv_eng, pot_temp_1 + pot_temp_2);
}

} // namespace

extern "C" void solvation_pp_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_q_dx,
    const double* elem_q_dy,
    const double* elem_q_dz,
    const double* elem_area,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* potential,
    std::size_t potential_offset,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double eps,
    double kappa,
    double* solv_eng,
    void* stream)
{
    if (target_begin >= target_end || source_begin >= source_end) {
        return;
    }

    constexpr int kBlockSize = 256;
    std::size_t n_targets = target_end - target_begin;
    int grid = static_cast<int>((n_targets + kBlockSize - 1) / kBlockSize);
    solvation_pp_kernel<<<grid, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        elem_x, elem_y, elem_z,
        elem_q_dx, elem_q_dy, elem_q_dz,
        elem_area,
        mol_x, mol_y, mol_z, mol_q,
        potential, potential_offset,
        target_begin, target_end,
        source_begin, source_end,
        eps, kappa,
        solv_eng);
}

extern "C" void solvation_pc_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_q_dx,
    const double* elem_q_dy,
    const double* elem_q_dz,
    const double* elem_area,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    const double* potential,
    std::size_t potential_offset,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double eps,
    double kappa,
    double* solv_eng,
    void* stream)
{
    if (target_begin >= target_end) {
        return;
    }

    constexpr int kBlockSize = 256;
    std::size_t n_targets = target_end - target_begin;
    int grid = static_cast<int>((n_targets + kBlockSize - 1) / kBlockSize);
    solvation_pc_kernel<<<grid, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        elem_x, elem_y, elem_z,
        elem_q_dx, elem_q_dy, elem_q_dz,
        elem_area,
        mol_clusters_x, mol_clusters_y, mol_clusters_z,
        mol_clusters_q,
        potential, potential_offset,
        source_node_idx,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        target_begin, target_end,
        eps, kappa,
        solv_eng);
}

extern "C" void solvation_cp_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    std::size_t target_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double eps,
    double kappa,
    void* stream)
{
    if (source_begin >= source_end) {
        return;
    }

    std::size_t total = static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node);
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    solvation_cp_kernel<<<grid, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        mol_x, mol_y, mol_z, mol_q,
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        target_node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        source_begin, source_end,
        eps, kappa);
}

extern "C" void solvation_cc_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    double* elem_clusters_p,
    double* elem_clusters_p_dx,
    double* elem_clusters_p_dy,
    double* elem_clusters_p_dz,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    double eps,
    double kappa,
    void* stream)
{
    std::size_t total = static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node) *
                        static_cast<std::size_t>(num_elem_interp_pts_per_node);
    if (total == 0) {
        return;
    }
    constexpr int kBlockSize = 256;
    int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    solvation_cc_kernel<<<grid, kBlockSize, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        mol_clusters_x, mol_clusters_y, mol_clusters_z, mol_clusters_q,
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        target_node_idx, source_node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        num_mol_interp_pts_per_node,
        num_mol_interp_charges_per_node,
        eps, kappa);
}

extern "C" void solvation_up_cuda(
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
    if (num_particles == 0 || num_mol_interp_pts_per_node <= 0) {
        return;
    }
    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;

    int threads = 128;
    int blocks_particles = static_cast<int>((num_particles + threads - 1) / threads);

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_mol_interp_pts_per_node);
    std::size_t node_charges_start =
        node_idx * static_cast<std::size_t>(num_mol_interp_charges_per_node);

    solvation_up_denom_kernel<<<blocks_particles, threads, 0, cuda_stream>>>(
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

    solvation_up_charges_kernel<<<blocks_charges, threads, 0, cuda_stream>>>(
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
}

extern "C" void solvation_down_cuda(
    const double* elem_x,
    const double* elem_y,
    const double* elem_z,
    const double* elem_q_dx,
    const double* elem_q_dy,
    const double* elem_q_dz,
    const double* elem_area,
    const double* elem_clusters_x,
    const double* elem_clusters_y,
    const double* elem_clusters_z,
    const double* elem_clusters_p,
    const double* elem_clusters_p_dx,
    const double* elem_clusters_p_dy,
    const double* elem_clusters_p_dz,
    const double* potential,
    std::size_t potential_offset,
    const double* weights,
    std::size_t node_idx,
    int num_elem_interp_pts_per_node,
    int num_elem_interp_potentials_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    double* solv_eng,
    void* stream)
{
    if (num_particles == 0) {
        return;
    }

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    constexpr int kBlockSize = 128;
    int grid = static_cast<int>((num_particles + kBlockSize - 1) / kBlockSize);
    solvation_down_kernel<<<grid, kBlockSize, 0, cuda_stream>>>(
        elem_x, elem_y, elem_z,
        elem_q_dx, elem_q_dy, elem_q_dz,
        elem_area,
        elem_clusters_x, elem_clusters_y, elem_clusters_z,
        elem_clusters_p, elem_clusters_p_dx, elem_clusters_p_dy, elem_clusters_p_dz,
        potential, potential_offset,
        weights,
        node_idx,
        num_elem_interp_pts_per_node,
        num_elem_interp_potentials_per_node,
        particle_start,
        num_particles,
        solv_eng);
}
