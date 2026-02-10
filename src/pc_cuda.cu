#include "pc_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

__global__ void pc_kernel(
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
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    std::size_t pc_offsets_count,
    std::size_t pc_sources_count)
{
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= num_elements) return;

    const std::uint32_t target_node = element_node_idx[j];
    if (target_node + 1 >= pc_offsets_count || target_node >= num_nodes) return;

    std::size_t pc_start = static_cast<std::size_t>(pc_offsets[target_node]);
    std::size_t pc_end   = static_cast<std::size_t>(pc_offsets[target_node + 1]);
    if (pc_start >= pc_sources_count) return;
    if (pc_end > pc_sources_count) pc_end = pc_sources_count;

    const double target_x = elements_x[j];
    const double target_y = elements_y[j];
    const double target_z = elements_z[j];

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

    potential[j] = potential[j] + tq * pot_comp;
    const std::size_t norm_idx = j + num_elements;
    potential[norm_idx] = potential[norm_idx]
                        + tq_dx * pot_comp_dx
                        + tq_dy * pot_comp_dy
                        + tq_dz * pot_comp_dz;
}

} // namespace

extern "C" void pc_interact_cuda(
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
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
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
    pc_kernel<<<grid, kBlock, 0, cuda_stream>>>(
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
        targets_q,
        targets_q_dx,
        targets_q_dy,
        targets_q_dz,
        potential,
        num_elements,
        element_node_idx,
        num_nodes,
        pc_offsets,
        pc_sources,
        pc_offsets_count,
        pc_sources_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "pc_interact_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
