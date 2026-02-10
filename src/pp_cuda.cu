#include "pp_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

__global__ void pp_kernel(
    double eps,
    double kappa,
    double kappa2,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* elements_area,
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
    std::size_t pp_sources_count)
{
    const std::size_t j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j >= num_elements) return;

    const std::uint32_t target_node = element_node_idx[j];
    if (target_node + 1 >= pp_offsets_count || target_node >= num_nodes) return;

    std::size_t pp_start = static_cast<std::size_t>(pp_offsets[target_node]);
    std::size_t pp_end   = static_cast<std::size_t>(pp_offsets[target_node + 1]);
    if (pp_start >= pp_sources_count) return;
    if (pp_end > pp_sources_count) pp_end = pp_sources_count;

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

    potential[j] += pot_pp_1;
    potential[j + num_elements] += pot_pp_2;
}

} // namespace

extern "C" void pp_interact_cuda(
    double eps,
    double kappa,
    double kappa2,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* elements_area,
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
    void* stream)
{
    if (num_elements == 0) return;
    constexpr int kBlock = 128;
    const int grid = static_cast<int>((num_elements + kBlock - 1) / kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    pp_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        eps,
        kappa,
        kappa2,
        elements_x,
        elements_y,
        elements_z,
        elements_nx,
        elements_ny,
        elements_nz,
        elements_area,
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
        pp_sources_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "pp_interact_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
