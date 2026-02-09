#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#include "cc_cuda.h"

namespace {

__global__ void cp_interact_kernel(
    int n,
    int n2,
    int n3,
    int num_interp_pts_per_node,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* __restrict clusters_x,
    const double* __restrict clusters_y,
    const double* __restrict clusters_z,
    double* __restrict clusters_p,
    double* __restrict clusters_p_dx,
    double* __restrict clusters_p_dy,
    double* __restrict clusters_p_dz,
    const std::uint32_t* __restrict node_begin,
    const std::uint32_t* __restrict node_end,
    const std::uint32_t* __restrict cp_offsets,
    const std::uint32_t* __restrict cp_sources,
    const double* __restrict elements_x,
    const double* __restrict elements_y,
    const double* __restrict elements_z,
    const double* __restrict sources_q,
    const double* __restrict sources_q_dx,
    const double* __restrict sources_q_dy,
    const double* __restrict sources_q_dz,
    int num_nodes,
    std::size_t num_elements,
    std::size_t cp_offsets_count,
    std::size_t cp_sources_count)
{
    int target_node_idx = static_cast<int>(blockIdx.x);
    if (target_node_idx >= num_nodes) return;

    int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (j >= n3) return;

    int j1 = j / n2;
    int j2 = (j / n) % n;
    int j3 = j % n;

    std::size_t target_cluster_interp_pts_begin =
        static_cast<std::size_t>(target_node_idx) * static_cast<std::size_t>(num_interp_pts_per_node);
    std::size_t target_cluster_potentials_begin =
        static_cast<std::size_t>(target_node_idx) * static_cast<std::size_t>(num_charges_per_node);

    double target_x = clusters_x[target_cluster_interp_pts_begin + j1];
    double target_y = clusters_y[target_cluster_interp_pts_begin + j2];
    double target_z = clusters_z[target_cluster_interp_pts_begin + j3];

    double pot_comp_   = 0.0;
    double pot_comp_dx = 0.0;
    double pot_comp_dy = 0.0;
    double pot_comp_dz = 0.0;

    std::uint32_t cp_start = 0;
    std::uint32_t cp_end   = 0;
    if (static_cast<std::size_t>(target_node_idx + 1) < cp_offsets_count) {
        cp_start = cp_offsets[target_node_idx];
        cp_end   = cp_offsets[target_node_idx + 1];
        if (cp_start > cp_end) cp_start = cp_end;
        if (cp_start > cp_sources_count) cp_start = static_cast<std::uint32_t>(cp_sources_count);
        if (cp_end > cp_sources_count) cp_end = static_cast<std::uint32_t>(cp_sources_count);
    }

    for (std::uint32_t s = cp_start; s < cp_end; ++s) {
        std::uint32_t source_node_idx = cp_sources[s];
        if (source_node_idx >= static_cast<std::uint32_t>(num_nodes)) continue;
        std::uint32_t source_begin = node_begin[source_node_idx];
        std::uint32_t source_end   = node_end[source_node_idx];
        if (source_begin >= num_elements) continue;
        if (source_end > num_elements) source_end = static_cast<std::uint32_t>(num_elements);

        for (std::uint32_t k = source_begin; k < source_end; ++k) {
            double dx = target_x - elements_x[k];
            double dy = target_y - elements_y[k];
            double dz = target_z - elements_z[k];

            double r2    = dx*dx + dy*dy + dz*dz;
            double r     = sqrt(r2);
            double rinv  = 1.0 / r;
            double r3inv = rinv  * rinv * rinv;
            double r5inv = r3inv * rinv * rinv;

            double kappa_r = kappa * r;
            double expkr   = exp(-kappa_r);
            double d1term  = r3inv * expkr * (1. + kappa_r);
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  = r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                   + (kappa2 * r2)));
            double d3term  = r3inv * ( 1. - expkr * (1. + kappa_r));

            pot_comp_    += (rinv * (1. - expkr) * (sources_q   [k])
                                      + d1term1 * (sources_q_dx[k] * dx
                                                 + sources_q_dy[k] * dy
                                                 + sources_q_dz[k] * dz));

            pot_comp_dx  += (sources_q   [k]  * (d1term2 * dx)
                          - (sources_q_dx[k]  * (dx * dx * d2term + d3term)
                          +  sources_q_dy[k]  * (dx * dy * d2term)
                          +  sources_q_dz[k]  * (dx * dz * d2term)));

            pot_comp_dy  += (sources_q   [k]  *  d1term2 * dy
                          - (sources_q_dx[k]  * (dx * dy * d2term)
                          +  sources_q_dy[k]  * (dy * dy * d2term + d3term)
                          +  sources_q_dz[k]  * (dy * dz * d2term)));

            pot_comp_dz  += (sources_q   [k]  *  d1term2 * dz
                          - (sources_q_dx[k]  * (dx * dz * d2term)
                          +  sources_q_dy[k]  * (dy * dz * d2term)
                          +  sources_q_dz[k]  * (dz * dz * d2term + d3term)));
        }
    }

    std::size_t out_idx = target_cluster_potentials_begin
                        + static_cast<std::size_t>(j1 * n2 + j2 * n + j3);

    clusters_p[out_idx]    += pot_comp_;
    clusters_p_dx[out_idx] += pot_comp_dx;
    clusters_p_dy[out_idx] += pot_comp_dy;
    clusters_p_dz[out_idx] += pot_comp_dz;
}

__global__ void cc_interact_kernel(
    int n,
    int n2,
    int n3,
    int num_interp_pts_per_node,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* __restrict clusters_x,
    const double* __restrict clusters_y,
    const double* __restrict clusters_z,
    const double* __restrict clusters_q,
    const double* __restrict clusters_q_dx,
    const double* __restrict clusters_q_dy,
    const double* __restrict clusters_q_dz,
    double* __restrict clusters_p,
    double* __restrict clusters_p_dx,
    double* __restrict clusters_p_dy,
    double* __restrict clusters_p_dz,
    const std::uint32_t* __restrict node_begin,
    const std::uint32_t* __restrict node_end,
    const std::uint32_t* __restrict cp_offsets,
    const std::uint32_t* __restrict cp_sources,
    const std::uint32_t* __restrict cc_offsets,
    const std::uint32_t* __restrict cc_sources,
    const double* __restrict elements_x,
    const double* __restrict elements_y,
    const double* __restrict elements_z,
    const double* __restrict sources_q,
    const double* __restrict sources_q_dx,
    const double* __restrict sources_q_dy,
    const double* __restrict sources_q_dz,
    int num_nodes,
    std::size_t num_elements,
    std::size_t cp_offsets_count,
    std::size_t cp_sources_count,
    std::size_t cc_offsets_count,
    std::size_t cc_sources_count,
    int enable_cp)
{
    int target_node_idx = static_cast<int>(blockIdx.x);
    if (target_node_idx >= num_nodes) return;

    int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    bool active = (j < n3);

    int j1 = 0;
    int j2 = 0;
    int j3 = 0;
    if (active) {
        j1 = j / n2;
        j2 = (j / n) % n;
        j3 = j % n;
    }

    std::size_t target_cluster_interp_pts_begin =
        static_cast<std::size_t>(target_node_idx) * static_cast<std::size_t>(num_interp_pts_per_node);
    std::size_t target_cluster_potentials_begin =
        static_cast<std::size_t>(target_node_idx) * static_cast<std::size_t>(num_charges_per_node);

    double target_x = 0.0;
    double target_y = 0.0;
    double target_z = 0.0;
    if (active) {
        target_x = clusters_x[target_cluster_interp_pts_begin + j1];
        target_y = clusters_y[target_cluster_interp_pts_begin + j2];
        target_z = clusters_z[target_cluster_interp_pts_begin + j3];
    }

    double pot_comp_   = 0.0;
    double pot_comp_dx = 0.0;
    double pot_comp_dy = 0.0;
    double pot_comp_dz = 0.0;

    if (active && enable_cp) {
    std::uint32_t cp_start = 0;
    std::uint32_t cp_end   = 0;
    if (static_cast<std::size_t>(target_node_idx + 1) < cp_offsets_count) {
        cp_start = cp_offsets[target_node_idx];
        cp_end   = cp_offsets[target_node_idx + 1];
        if (cp_start > cp_end) cp_start = cp_end;
        if (cp_start > cp_sources_count) cp_start = static_cast<std::uint32_t>(cp_sources_count);
        if (cp_end > cp_sources_count) cp_end = static_cast<std::uint32_t>(cp_sources_count);
    }

        for (std::uint32_t s = cp_start; s < cp_end; ++s) {
            std::uint32_t source_node_idx = cp_sources[s];
            if (source_node_idx >= static_cast<std::uint32_t>(num_nodes)) continue;
            std::uint32_t source_begin = node_begin[source_node_idx];
            std::uint32_t source_end   = node_end[source_node_idx];
            if (source_begin >= num_elements) continue;
            if (source_end > num_elements) source_end = static_cast<std::uint32_t>(num_elements);

            for (std::uint32_t k = source_begin; k < source_end; ++k) {
                double dx = target_x - elements_x[k];
                double dy = target_y - elements_y[k];
                double dz = target_z - elements_z[k];

                double r2    = dx*dx + dy*dy + dz*dz;
                double r     = sqrt(r2);
                double rinv  = 1.0 / r;
                double r3inv = rinv  * rinv * rinv;
                double r5inv = r3inv * rinv * rinv;

                double kappa_r = kappa * r;
                double expkr   = exp(-kappa_r);
                double d1term  = r3inv * expkr * (1. + kappa_r);
                double d1term1 = -r3inv + d1term * eps;
                double d1term2 = -r3inv + d1term / eps;
                double d2term  = r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                       + (kappa2 * r2)));
                double d3term  = r3inv * ( 1. - expkr * (1. + kappa_r));

                pot_comp_    += (rinv * (1. - expkr) * (sources_q   [k])
                                          + d1term1 * (sources_q_dx[k] * dx
                                                     + sources_q_dy[k] * dy
                                                     + sources_q_dz[k] * dz));

                pot_comp_dx  += (sources_q   [k]  * (d1term2 * dx)
                              - (sources_q_dx[k]  * (dx * dx * d2term + d3term)
                              +  sources_q_dy[k]  * (dx * dy * d2term)
                              +  sources_q_dz[k]  * (dx * dz * d2term)));

                pot_comp_dy  += (sources_q   [k]  *  d1term2 * dy
                              - (sources_q_dx[k]  * (dx * dy * d2term)
                              +  sources_q_dy[k]  * (dy * dy * d2term + d3term)
                              +  sources_q_dz[k]  * (dy * dz * d2term)));

                pot_comp_dz  += (sources_q   [k]  *  d1term2 * dz
                              - (sources_q_dx[k]  * (dx * dz * d2term)
                              +  sources_q_dy[k]  * (dy * dz * d2term)
                              +  sources_q_dz[k]  * (dz * dz * d2term + d3term)));
            }
        }
    }

    extern __shared__ double shmem[];
    double* src_x    = shmem;
    double* src_y    = src_x    + n;
    double* src_z    = src_y    + n;
    double* src_q    = src_z    + n;
    double* src_q_dx = src_q    + n3;
    double* src_q_dy = src_q_dx + n3;
    double* src_q_dz = src_q_dy + n3;

    std::uint32_t cc_start = 0;
    std::uint32_t cc_end   = 0;
    if (static_cast<std::size_t>(target_node_idx + 1) < cc_offsets_count) {
        cc_start = cc_offsets[target_node_idx];
        cc_end   = cc_offsets[target_node_idx + 1];
        if (cc_start > cc_end) cc_start = cc_end;
        if (cc_start > cc_sources_count) cc_start = static_cast<std::uint32_t>(cc_sources_count);
        if (cc_end > cc_sources_count) cc_end = static_cast<std::uint32_t>(cc_sources_count);
    }

    for (std::uint32_t s = cc_start; s < cc_end; ++s) {
        std::uint32_t source_node_idx = cc_sources[s];
        if (source_node_idx >= static_cast<std::uint32_t>(num_nodes)) continue;

        std::size_t source_cluster_interp_pts_begin =
            static_cast<std::size_t>(source_node_idx) * static_cast<std::size_t>(num_interp_pts_per_node);
        std::size_t source_cluster_charges_begin =
            static_cast<std::size_t>(source_node_idx) * static_cast<std::size_t>(num_charges_per_node);

        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            src_x[i] = clusters_x[source_cluster_interp_pts_begin + i];
            src_y[i] = clusters_y[source_cluster_interp_pts_begin + i];
            src_z[i] = clusters_z[source_cluster_interp_pts_begin + i];
        }

        for (int i = threadIdx.x; i < n3; i += blockDim.x) {
            std::size_t kk = source_cluster_charges_begin + static_cast<std::size_t>(i);
            src_q[i]    = clusters_q   [kk];
            src_q_dx[i] = clusters_q_dx[kk];
            src_q_dy[i] = clusters_q_dy[kk];
            src_q_dz[i] = clusters_q_dz[kk];
        }

        __syncthreads();

        if (active) {
            for (int k1 = 0; k1 < n; ++k1) {
            for (int k2 = 0; k2 < n; ++k2) {
            for (int k3 = 0; k3 < n; ++k3) {
                int kk_local = k1 * n2 + k2 * n + k3;

                double dx = target_x - src_x[k1];
                double dy = target_y - src_y[k2];
                double dz = target_z - src_z[k3];

                double r2    = dx*dx + dy*dy + dz*dz;
                double r     = sqrt(r2);
                double rinv  = 1.0 / r;
                double r3inv = rinv  * rinv * rinv;
                double r5inv = r3inv * rinv * rinv;

                double kappa_r = kappa * r;
                double expkr   = exp(-kappa_r);
                double d1term  = r3inv * expkr * (1. + kappa_r);
                double d1term1 = -r3inv + d1term * eps;
                double d1term2 = -r3inv + d1term / eps;
                double d2term  = r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                       + (kappa2 * r2)));
                double d3term  = r3inv * ( 1. - expkr * (1. + kappa_r));

                pot_comp_    += (rinv * (1. - expkr) * (src_q   [kk_local])
                                          + d1term1 * (src_q_dx[kk_local] * dx
                                                     + src_q_dy[kk_local] * dy
                                                     + src_q_dz[kk_local] * dz));

                pot_comp_dx  += (src_q   [kk_local]  * (d1term2 * dx)
                              - (src_q_dx[kk_local]  * (dx * dx * d2term + d3term)
                              +  src_q_dy[kk_local]  * (dx * dy * d2term)
                              +  src_q_dz[kk_local]  * (dx * dz * d2term)));

                pot_comp_dy  += (src_q   [kk_local]  *  d1term2 * dy
                              - (src_q_dx[kk_local]  * (dx * dy * d2term)
                              +  src_q_dy[kk_local]  * (dy * dy * d2term + d3term)
                              +  src_q_dz[kk_local]  * (dy * dz * d2term)));

                pot_comp_dz  += (src_q   [kk_local]  *  d1term2 * dz
                              - (src_q_dx[kk_local]  * (dx * dz * d2term)
                              +  src_q_dy[kk_local]  * (dy * dz * d2term)
                              +  src_q_dz[kk_local]  * (dz * dz * d2term + d3term)));
            }
            }
            }
        }

        __syncthreads();
    }

    if (active) {
        std::size_t out_idx = target_cluster_potentials_begin
                            + static_cast<std::size_t>(j1 * n2 + j2 * n + j3);

        clusters_p[out_idx]    += pot_comp_;
        clusters_p_dx[out_idx] += pot_comp_dx;
        clusters_p_dy[out_idx] += pot_comp_dy;
        clusters_p_dz[out_idx] += pot_comp_dz;
    }
}

} // namespace

extern "C" void cc_interact_cuda(
    int n,
    int n2,
    int n3,
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
    double* clusters_p,
    double* clusters_p_dx,
    double* clusters_p_dy,
    double* clusters_p_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::uint32_t* cp_offsets,
    const std::uint32_t* cp_sources,
    const std::uint32_t* cc_offsets,
    const std::uint32_t* cc_sources,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    std::size_t num_elements,
    std::size_t num_nodes,
    std::size_t cp_offsets_count,
    std::size_t cp_sources_count,
    std::size_t cc_offsets_count,
    std::size_t cc_sources_count,
    int enable_cp,
    void* stream)
{
    if (n <= 0 || n3 <= 0 || num_nodes == 0) return;

    int num_nodes_i = static_cast<int>(num_nodes);
    int threads = 128;
    int blocks_y = (n3 + threads - 1) / threads;
    dim3 grid(num_nodes_i, blocks_y, 1);
    dim3 block(threads, 1, 1);

    std::size_t shmem_bytes = sizeof(double) * (static_cast<std::size_t>(3 * n) + static_cast<std::size_t>(4 * n3));

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    cc_interact_kernel<<<grid, block, shmem_bytes, cuda_stream>>>(
        n, n2, n3, num_interp_pts_per_node, num_charges_per_node, eps, kappa, kappa2,
        clusters_x, clusters_y, clusters_z,
        clusters_q, clusters_q_dx, clusters_q_dy, clusters_q_dz,
        clusters_p, clusters_p_dx, clusters_p_dy, clusters_p_dz,
        node_begin, node_end, cp_offsets, cp_sources, cc_offsets, cc_sources,
        elements_x, elements_y, elements_z,
        sources_q, sources_q_dx, sources_q_dy, sources_q_dz,
        num_nodes_i,
        num_elements,
        cp_offsets_count,
        cp_sources_count,
        cc_offsets_count,
        cc_sources_count,
        enable_cp);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cc_interact_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(cuda_stream);
}

extern "C" void cp_interact_cuda(
    int n,
    int n2,
    int n3,
    int num_interp_pts_per_node,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    double* clusters_p,
    double* clusters_p_dx,
    double* clusters_p_dy,
    double* clusters_p_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::uint32_t* cp_offsets,
    const std::uint32_t* cp_sources,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    std::size_t num_elements,
    std::size_t num_nodes,
    std::size_t cp_offsets_count,
    std::size_t cp_sources_count,
    void* stream)
{
    if (n <= 0 || n3 <= 0 || num_nodes == 0) return;

    int num_nodes_i = static_cast<int>(num_nodes);
    int threads = 128;
    int blocks_y = (n3 + threads - 1) / threads;
    dim3 grid(num_nodes_i, blocks_y, 1);
    dim3 block(threads, 1, 1);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    cp_interact_kernel<<<grid, block, 0, cuda_stream>>>(
        n, n2, n3, num_interp_pts_per_node, num_charges_per_node, eps, kappa, kappa2,
        clusters_x, clusters_y, clusters_z,
        clusters_p, clusters_p_dx, clusters_p_dy, clusters_p_dz,
        node_begin, node_end, cp_offsets, cp_sources,
        elements_x, elements_y, elements_z,
        sources_q, sources_q_dx, sources_q_dy, sources_q_dz,
        num_nodes_i,
        num_elements,
        cp_offsets_count,
        cp_sources_count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cp_interact_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(cuda_stream);
}
