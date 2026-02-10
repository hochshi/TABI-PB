#include "pppc_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

__global__ void pppc_kernel(
    int interp_n,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_q,
    const double* clusters_q_dx,
    const double* clusters_q_dy,
    const double* clusters_q_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* elements_area,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    const double* potential_old,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::uint32_t* pp_offsets,
    const std::uint32_t* pp_sources,
    std::size_t pp_offsets_count,
    std::size_t pp_sources_count,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    std::size_t pc_offsets_count,
    std::size_t pc_sources_count)
{
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= num_elements) return;

    const std::uint32_t target_node = element_node_idx[j];
    if (target_node + 1 >= pp_offsets_count || target_node + 1 >= pc_offsets_count) return;
    if (target_node >= num_nodes) return;

    std::size_t pp_start = static_cast<std::size_t>(pp_offsets[target_node]);
    std::size_t pp_end   = static_cast<std::size_t>(pp_offsets[target_node + 1]);
    if (pp_start >= pp_sources_count) pp_start = pp_sources_count;
    if (pp_end > pp_sources_count) pp_end = pp_sources_count;

    std::size_t pc_start = static_cast<std::size_t>(pc_offsets[target_node]);
    std::size_t pc_end   = static_cast<std::size_t>(pc_offsets[target_node + 1]);
    if (pc_start >= pc_sources_count) pc_start = pc_sources_count;
    if (pc_end > pc_sources_count) pc_end = pc_sources_count;

    const double target_x = elements_x[j];
    const double target_y = elements_y[j];
    const double target_z = elements_z[j];
    const double target_nx = elements_nx[j];
    const double target_ny = elements_ny[j];
    const double target_nz = elements_nz[j];

    double pot_pp_1 = 0.0;
    double pot_pp_2 = 0.0;

    for (std::size_t s = pp_start; s < pp_end; ++s) {
        const std::uint32_t source_node = pp_sources[s];
        if (source_node >= num_nodes) continue;
        const std::size_t source_begin = static_cast<std::size_t>(node_begin[source_node]);
        const std::size_t source_end   = static_cast<std::size_t>(node_end[source_node]);

        for (std::size_t k = source_begin; k < source_end; ++k) {
            const double source_x = elements_x[k];
            const double source_y = elements_y[k];
            const double source_z = elements_z[k];

            const double source_nx = elements_nx[k];
            const double source_ny = elements_ny[k];
            const double source_nz = elements_nz[k];
            const double source_area = elements_area[k];

            const double potential_old_0 = potential_old[k];
            const double potential_old_1 = potential_old[k + num_elements];

            const double dist_x = source_x - target_x;
            const double dist_y = source_y - target_y;
            const double dist_z = source_z - target_z;
            const double r = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

            if (r > 0.0) {
                const double one_over_r = 1.0 / r;
                const double G0 = 0.07957747154594767 * one_over_r;
                const double kappa_r = kappa * r;
                const double exp_kappa_r = std::exp(-kappa_r);
                const double Gk = exp_kappa_r * G0;

                const double source_cos = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                const double target_cos = (target_nx * dist_x + target_ny * dist_y + target_nz * dist_z) * one_over_r;

                const double tp1 = G0 * one_over_r;
                const double tp2 = (1.0 + kappa_r) * exp_kappa_r;

                const double dot_tqsq = source_nx * target_nx + source_ny * target_ny + source_nz * target_nz;
                const double G3 = (dot_tqsq - 3.0 * target_cos * source_cos) * one_over_r * tp1;
                const double G4 = tp2 * G3 - kappa2 * target_cos * source_cos * Gk;

                const double L1 = source_cos  * tp1 * (1.0 - tp2 * eps);
                const double L2 = G0 - Gk;
                const double L3 = G4 - G3;
                const double L4 = target_cos * tp1 * (1.0 - tp2 / eps);

                pot_pp_1 += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                pot_pp_2 += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
            }
        }
    }

    double pot_comp = 0.0;
    double pot_comp_dx = 0.0;
    double pot_comp_dy = 0.0;
    double pot_comp_dz = 0.0;

    const int n = interp_n;
    const int n2 = n * n;

    for (std::size_t s = pc_start; s < pc_end; ++s) {
        const std::uint32_t source_node = pc_sources[s];
        const std::size_t interp_begin = static_cast<std::size_t>(source_node) * static_cast<std::size_t>(n);
        const std::size_t charges_begin = static_cast<std::size_t>(source_node) * static_cast<std::size_t>(num_charges_per_node);

        for (int k1 = 0; k1 < n; ++k1) {
            const double sx = clusters_x[interp_begin + static_cast<std::size_t>(k1)];
            const double dx = target_x - sx;
            const double dx2 = dx * dx;
            for (int k2 = 0; k2 < n; ++k2) {
                const double sy = clusters_y[interp_begin + static_cast<std::size_t>(k2)];
                const double dy = target_y - sy;
                const double dy2 = dy * dy;
                const std::size_t base_k2 = charges_begin + static_cast<std::size_t>(k1) * static_cast<std::size_t>(n2)
                                            + static_cast<std::size_t>(k2) * static_cast<std::size_t>(n);
                for (int k3 = 0; k3 < n; ++k3) {
                    const double sz = clusters_z[interp_begin + static_cast<std::size_t>(k3)];
                    const double dz = target_z - sz;

                    const std::size_t kk = base_k2 + static_cast<std::size_t>(k3);

                    const double r2 = dx2 + dy2 + dz * dz;
                    const double r = std::sqrt(r2);
                    const double rinv = 1.0 / r;
                    const double r3inv = rinv * rinv * rinv;
                    const double r5inv = r3inv * rinv * rinv;

                    const double kappa_r = kappa * r;
                    const double expkr = std::exp(-kappa_r);
                    const double d1term = r3inv * expkr * (1.0 + kappa_r);
                    const double d1term1 = -r3inv + d1term * eps;
                    const double d1term2 = -r3inv + d1term / eps;
                    const double d2term = r5inv * (-3.0 + expkr * (3.0 + (3.0 * kappa_r) + (kappa2 * r2)));
                    const double d3term = r3inv * (1.0 - expkr * (1.0 + kappa_r));

                    const double source_q = clusters_q[kk];
                    const double source_q_dx = clusters_q_dx[kk];
                    const double source_q_dy = clusters_q_dy[kk];
                    const double source_q_dz = clusters_q_dz[kk];

                    pot_comp += (rinv * (1.0 - expkr) * source_q
                                 + d1term1 * (source_q_dx * dx + source_q_dy * dy + source_q_dz * dz));

                    pot_comp_dx += (source_q * (d1term2 * dx)
                                    - (source_q_dx * (dx * dx * d2term + d3term)
                                       + source_q_dy * (dx * dy * d2term)
                                       + source_q_dz * (dx * dz * d2term)));

                    pot_comp_dy += (source_q * (d1term2 * dy)
                                    - (source_q_dx * (dx * dy * d2term)
                                       + source_q_dy * (dy * dy * d2term + d3term)
                                       + source_q_dz * (dy * dz * d2term)));

                    pot_comp_dz += (source_q * (d1term2 * dz)
                                    - (source_q_dx * (dx * dz * d2term)
                                       + source_q_dy * (dy * dz * d2term)
                                       + source_q_dz * (dz * dz * d2term + d3term)));
                }
            }
        }
    }

    const double tq = targets_q[j];
    const double tq_dx = targets_q_dx[j];
    const double tq_dy = targets_q_dy[j];
    const double tq_dz = targets_q_dz[j];

    potential[j] = potential[j] + pot_pp_1 + tq * pot_comp;
    const std::size_t norm_idx = j + num_elements;
    potential[norm_idx] = potential[norm_idx]
                        + pot_pp_2
                        + tq_dx * pot_comp_dx
                        + tq_dy * pot_comp_dy
                        + tq_dz * pot_comp_dz;
}

} // namespace

extern "C" void pppc_interact_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_q,
    const double* clusters_q_dx,
    const double* clusters_q_dy,
    const double* clusters_q_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* elements_area,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    const double* potential_old,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::uint32_t* pp_offsets,
    const std::uint32_t* pp_sources,
    std::size_t pp_offsets_count,
    std::size_t pp_sources_count,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    std::size_t pc_offsets_count,
    std::size_t pc_sources_count,
    void* stream)
{
    if (num_elements == 0) return;
    constexpr int kBlock = 128;
    const int grid = static_cast<int>((num_elements + kBlock - 1) / kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    pppc_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        num_interp_pts_per_node,
        num_charges_per_node,
        eps,
        kappa,
        kappa2,
        clusters_x,
        clusters_y,
        clusters_z,
        clusters_q,
        clusters_q_dx,
        clusters_q_dy,
        clusters_q_dz,
        elements_x,
        elements_y,
        elements_z,
        elements_nx,
        elements_ny,
        elements_nz,
        elements_area,
        targets_q,
        targets_q_dx,
        targets_q_dy,
        targets_q_dz,
        potential_old,
        potential,
        num_elements,
        element_node_idx,
        num_nodes,
        node_begin,
        node_end,
        pp_offsets,
        pp_sources,
        pp_offsets_count,
        pp_sources_count,
        pc_offsets,
        pc_sources,
        pc_offsets_count,
        pc_sources_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "pppc_interact_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
