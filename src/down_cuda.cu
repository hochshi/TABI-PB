#include "down_cuda.h"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace {

__global__ void downward_kernel(
    int interp_n,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_p,
    const double* clusters_p_dx,
    const double* clusters_p_dy,
    const double* clusters_p_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    const double* weights,
    double* potential,
    std::size_t potential_offset,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes)
{
    const std::size_t level_idx = static_cast<std::size_t>(blockIdx.x);
    if (level_idx >= num_level_nodes) return;

    const std::size_t node_idx = level_nodes[level_idx];
    const std::size_t node_interp_pts_start = node_idx * static_cast<std::size_t>(interp_n);
    const std::size_t node_potentials_start = node_idx * static_cast<std::size_t>(num_charges_per_node);

    const std::size_t particle_start = static_cast<std::size_t>(node_begin[node_idx]);
    const std::size_t particle_end   = static_cast<std::size_t>(node_end[node_idx]);
    const std::size_t num_particles  = (particle_end > particle_start) ? (particle_end - particle_start) : 0;

    if (num_particles == 0) return;

    __shared__ double sx[16];
    __shared__ double sy[16];
    __shared__ double sz[16];
    __shared__ double sw[16];

    if (threadIdx.x < interp_n) {
        const std::size_t idx = node_interp_pts_start + static_cast<std::size_t>(threadIdx.x);
        sx[threadIdx.x] = clusters_x[idx];
        sy[threadIdx.x] = clusters_y[idx];
        sz[threadIdx.x] = clusters_z[idx];
        sw[threadIdx.x] = weights[threadIdx.x];
    }
    __syncthreads();

    const std::size_t interp_n2 = static_cast<std::size_t>(interp_n) * static_cast<std::size_t>(interp_n);

    for (std::size_t local_i = static_cast<std::size_t>(threadIdx.x);
         local_i < num_particles;
         local_i += static_cast<std::size_t>(blockDim.x)) {
        const std::size_t i = particle_start + local_i;

        const double xx = elements_x[i];
        const double yy = elements_y[i];
        const double zz = elements_z[i];

        double x_term[16];
        double y_term[16];
        double z_term[16];

        double denominator_x = 0.0;
        double denominator_y = 0.0;
        double denominator_z = 0.0;
        int exact_idx_x = -1;
        int exact_idx_y = -1;
        int exact_idx_z = -1;

        for (int j = 0; j < interp_n; ++j) {
            const double dist_x = xx - sx[j];
            const double dist_y = yy - sy[j];
            const double dist_z = zz - sz[j];

            const double inv_x = sw[j] / dist_x;
            const double inv_y = sw[j] / dist_y;
            const double inv_z = sw[j] / dist_z;

            x_term[j] = inv_x;
            y_term[j] = inv_y;
            z_term[j] = inv_z;

            denominator_x += inv_x;
            denominator_y += inv_y;
            denominator_z += inv_z;

            const int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
            const int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
            const int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;

            exact_idx_x = (exact_idx_x > cx) ? exact_idx_x : cx;
            exact_idx_y = (exact_idx_y > cy) ? exact_idx_y : cy;
            exact_idx_z = (exact_idx_z > cz) ? exact_idx_z : cz;
        }

        if (exact_idx_x != -1) {
            for (int j = 0; j < interp_n; ++j) x_term[j] = (j == exact_idx_x) ? 1.0 : 0.0;
        }
        if (exact_idx_y != -1) {
            for (int j = 0; j < interp_n; ++j) y_term[j] = (j == exact_idx_y) ? 1.0 : 0.0;
        }
        if (exact_idx_z != -1) {
            for (int j = 0; j < interp_n; ++j) z_term[j] = (j == exact_idx_z) ? 1.0 : 0.0;
        }

        double denominator = 1.0;
        if (exact_idx_x == -1) denominator /= denominator_x;
        if (exact_idx_y == -1) denominator /= denominator_y;
        if (exact_idx_z == -1) denominator /= denominator_z;

        double pot_comp = 0.0;
        double pot_comp_dx = 0.0;
        double pot_comp_dy = 0.0;
        double pot_comp_dz = 0.0;

        for (int k1 = 0; k1 < interp_n; ++k1) {
            const std::size_t base_k1 = static_cast<std::size_t>(k1) * interp_n2;
            const double xw = x_term[k1];
            for (int k2 = 0; k2 < interp_n; ++k2) {
                const std::size_t base_k2 = base_k1 + static_cast<std::size_t>(k2) * static_cast<std::size_t>(interp_n);
                const double xy = xw * y_term[k2];
                for (int k3 = 0; k3 < interp_n; ++k3) {
                    const std::size_t kk = node_potentials_start + base_k2 + static_cast<std::size_t>(k3);
                    const double numer = xy * z_term[k3] * denominator;
                    pot_comp   += numer * clusters_p[kk];
                    pot_comp_dx += numer * clusters_p_dx[kk];
                    pot_comp_dy += numer * clusters_p_dy[kk];
                    pot_comp_dz += numer * clusters_p_dz[kk];
                }
            }
        }

        const double tq = targets_q[i];
        const double tq_dx = targets_q_dx[i];
        const double tq_dy = targets_q_dy[i];
        const double tq_dz = targets_q_dz[i];

        potential[i] = potential[i] + tq * pot_comp;
        const std::size_t norm_idx = i + potential_offset;
        potential[norm_idx] = potential[norm_idx]
                            + tq_dx * pot_comp_dx
                            + tq_dy * pot_comp_dy
                            + tq_dz * pot_comp_dz;
    }
}

inline int grid_for(std::size_t n)
{
    return static_cast<int>(n);
}

} // namespace

extern "C" void downward_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_p,
    const double* clusters_p_dx,
    const double* clusters_p_dy,
    const double* clusters_p_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    const double* weights,
    double* potential,
    std::size_t potential_offset,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    void* stream)
{
    if (num_level_nodes == 0) return;
    constexpr int kBlock = 256;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int grid = grid_for(num_level_nodes);
    downward_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        num_interp_pts_per_node,
        num_charges_per_node,
        clusters_x,
        clusters_y,
        clusters_z,
        clusters_p,
        clusters_p_dx,
        clusters_p_dy,
        clusters_p_dz,
        elements_x,
        elements_y,
        elements_z,
        targets_q,
        targets_q_dx,
        targets_q_dy,
        targets_q_dz,
        weights,
        potential,
        potential_offset,
        node_begin,
        node_end,
        level_nodes,
        num_level_nodes);
}
