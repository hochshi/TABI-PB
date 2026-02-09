#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "up_cuda.h"

namespace {

constexpr int kMaxInterpPts = 16;
constexpr int kParticleTile = 128;
constexpr int kMaxJPerThread = 16;

__global__ void upward_denom_kernel(
    int num_interp_pts_per_node,
    const double* __restrict clusters_x,
    const double* __restrict clusters_y,
    const double* __restrict clusters_z,
    const double* __restrict weights,
    const double* __restrict elements_x,
    const double* __restrict elements_y,
    const double* __restrict elements_z,
    std::size_t num_elements,
    const std::uint32_t* __restrict node_begin,
    const std::uint32_t* __restrict node_end,
    const std::size_t* __restrict level_nodes,
    std::size_t num_level_nodes,
    int* __restrict exact_idx_x,
    int* __restrict exact_idx_y,
    int* __restrict exact_idx_z,
    double* __restrict denominator)
{
    std::size_t level_idx = static_cast<std::size_t>(blockIdx.x);
    if (level_idx >= num_level_nodes) return;

    std::size_t node_idx = level_nodes[level_idx];
    std::size_t particle_start = static_cast<std::size_t>(node_begin[node_idx]);
    std::size_t particle_end = static_cast<std::size_t>(node_end[node_idx]);
    if (particle_start >= num_elements) return;
    if (particle_end > num_elements) particle_end = num_elements;

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_interp_pts_per_node);

    __shared__ double node_x[kMaxInterpPts];
    __shared__ double node_y[kMaxInterpPts];
    __shared__ double node_z[kMaxInterpPts];
    __shared__ double w_cache[kMaxInterpPts];

    if (threadIdx.x < num_interp_pts_per_node) {
        int j = threadIdx.x;
        node_x[j] = clusters_x[node_interp_pts_start + j];
        node_y[j] = clusters_y[node_interp_pts_start + j];
        node_z[j] = clusters_z[node_interp_pts_start + j];
        w_cache[j] = weights[j];
    }
    __syncthreads();

    for (std::size_t p = particle_start + threadIdx.x; p < particle_end; p += blockDim.x) {
        double denominator_x = 0.0;
        double denominator_y = 0.0;
        double denominator_z = 0.0;
        int ex = -1;
        int ey = -1;
        int ez = -1;

        double xx = elements_x[p];
        double yy = elements_y[p];
        double zz = elements_z[p];

        for (int j = 0; j < num_interp_pts_per_node; ++j) {
            double dist_x = xx - node_x[j];
            double dist_y = yy - node_y[j];
            double dist_z = zz - node_z[j];

            denominator_x += w_cache[j] / dist_x;
            denominator_y += w_cache[j] / dist_y;
            denominator_z += w_cache[j] / dist_z;

            int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
            int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
            int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;
            if (cx > ex) ex = cx;
            if (cy > ey) ey = cy;
            if (cz > ez) ez = cz;
        }

        exact_idx_x[p] = ex;
        exact_idx_y[p] = ey;
        exact_idx_z[p] = ez;

        double denom = 1.0;
        if (ex == -1) denom /= denominator_x;
        if (ey == -1) denom /= denominator_y;
        if (ez == -1) denom /= denominator_z;
        denominator[p] = denom;
    }
}

__global__ void upward_charge_kernel(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* __restrict clusters_x,
    const double* __restrict clusters_y,
    const double* __restrict clusters_z,
    const double* __restrict weights,
    const double* __restrict elements_x,
    const double* __restrict elements_y,
    const double* __restrict elements_z,
    const double* __restrict sources_q,
    const double* __restrict sources_q_dx,
    const double* __restrict sources_q_dy,
    const double* __restrict sources_q_dz,
    const std::uint32_t* __restrict node_begin,
    const std::uint32_t* __restrict node_end,
    const std::size_t* __restrict level_nodes,
    std::size_t num_level_nodes,
    const int* __restrict exact_idx_x,
    const int* __restrict exact_idx_y,
    const int* __restrict exact_idx_z,
    const double* __restrict denominator,
    double* __restrict clusters_q,
    double* __restrict clusters_q_dx,
    double* __restrict clusters_q_dy,
    double* __restrict clusters_q_dz)
{
    std::size_t level_idx = static_cast<std::size_t>(blockIdx.x);
    if (level_idx >= num_level_nodes) return;

    std::size_t node_idx = level_nodes[level_idx];
    std::size_t particle_start = static_cast<std::size_t>(node_begin[node_idx]);
    std::size_t particle_end = static_cast<std::size_t>(node_end[node_idx]);
    if (particle_end <= particle_start) return;

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_interp_pts_per_node);
    std::size_t node_charges_start =
        node_idx * static_cast<std::size_t>(num_charges_per_node);

    int n = num_interp_pts_per_node;
    int n2 = n * n;
    int n3 = n2 * n;

    __shared__ double node_x[kMaxInterpPts];
    __shared__ double node_y[kMaxInterpPts];
    __shared__ double node_z[kMaxInterpPts];
    __shared__ double w_cache[kMaxInterpPts];

    if (threadIdx.x < n) {
        int j = threadIdx.x;
        node_x[j] = clusters_x[node_interp_pts_start + j];
        node_y[j] = clusters_y[node_interp_pts_start + j];
        node_z[j] = clusters_z[node_interp_pts_start + j];
        w_cache[j] = weights[j];
    }
    __syncthreads();

    __shared__ double tile_x[kParticleTile];
    __shared__ double tile_y[kParticleTile];
    __shared__ double tile_z[kParticleTile];
    __shared__ double tile_q[kParticleTile];
    __shared__ double tile_q_dx[kParticleTile];
    __shared__ double tile_q_dy[kParticleTile];
    __shared__ double tile_q_dz[kParticleTile];
    __shared__ double tile_denom[kParticleTile];
    __shared__ int tile_ex[kParticleTile];
    __shared__ int tile_ey[kParticleTile];
    __shared__ int tile_ez[kParticleTile];

    int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    bool active = (j < n3);

    int k1 = 0;
    int k2 = 0;
    int k3 = 0;
    if (active) {
        k1 = j / n2;
        k2 = (j / n) % n;
        k3 = j % n;
    }

    double cx = 0.0;
    double cy = 0.0;
    double cz = 0.0;
    double w1 = 0.0;
    double w2 = 0.0;
    double w3 = 0.0;
    if (active) {
        cx = node_x[k1];
        cy = node_y[k2];
        cz = node_z[k3];
        w1 = w_cache[k1];
        w2 = w_cache[k2];
        w3 = w_cache[k3];
    }

    double q_temp = 0.0;
    double q_dx_temp = 0.0;
    double q_dy_temp = 0.0;
    double q_dz_temp = 0.0;

    std::size_t num_particles = particle_end - particle_start;
    for (std::size_t tile_start = 0; tile_start < num_particles; tile_start += kParticleTile) {
        int tile_count = static_cast<int>(num_particles - tile_start);
        if (tile_count > kParticleTile) tile_count = kParticleTile;

        for (int ii = threadIdx.x; ii < tile_count; ii += blockDim.x) {
            std::size_t pidx = particle_start + tile_start + static_cast<std::size_t>(ii);
            tile_x[ii] = elements_x[pidx];
            tile_y[ii] = elements_y[pidx];
            tile_z[ii] = elements_z[pidx];
            tile_q[ii] = sources_q[pidx];
            tile_q_dx[ii] = sources_q_dx[pidx];
            tile_q_dy[ii] = sources_q_dy[pidx];
            tile_q_dz[ii] = sources_q_dz[pidx];
            tile_denom[ii] = denominator[pidx];
            tile_ex[ii] = exact_idx_x[pidx];
            tile_ey[ii] = exact_idx_y[pidx];
            tile_ez[ii] = exact_idx_z[pidx];
        }
        __syncthreads();

        if (active) {
            for (int ii = 0; ii < tile_count; ++ii) {
                double dist_x = tile_x[ii] - cx;
                double dist_y = tile_y[ii] - cy;
                double dist_z = tile_z[ii] - cz;

                double numerator = 1.0;
                if (tile_ex[ii] == -1) {
                    numerator *= w1 / dist_x;
                } else {
                    if (tile_ex[ii] != k1) numerator *= 0.0;
                }

                if (tile_ey[ii] == -1) {
                    numerator *= w2 / dist_y;
                } else {
                    if (tile_ey[ii] != k2) numerator *= 0.0;
                }

                if (tile_ez[ii] == -1) {
                    numerator *= w3 / dist_z;
                } else {
                    if (tile_ez[ii] != k3) numerator *= 0.0;
                }

                double denom = tile_denom[ii];
                q_temp    += tile_q[ii]    * numerator * denom;
                q_dx_temp += tile_q_dx[ii] * numerator * denom;
                q_dy_temp += tile_q_dy[ii] * numerator * denom;
                q_dz_temp += tile_q_dz[ii] * numerator * denom;
            }
        }
        __syncthreads();
    }

    if (active) {
        std::size_t kk = node_charges_start + static_cast<std::size_t>(j);
        clusters_q   [kk] += q_temp;
        clusters_q_dx[kk] += q_dx_temp;
        clusters_q_dy[kk] += q_dy_temp;
        clusters_q_dz[kk] += q_dz_temp;
    }
}

__global__ void upward_fused_kernel(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* __restrict clusters_x,
    const double* __restrict clusters_y,
    const double* __restrict clusters_z,
    const double* __restrict weights,
    const double* __restrict elements_x,
    const double* __restrict elements_y,
    const double* __restrict elements_z,
    const double* __restrict sources_q,
    const double* __restrict sources_q_dx,
    const double* __restrict sources_q_dy,
    const double* __restrict sources_q_dz,
    const std::uint32_t* __restrict node_begin,
    const std::uint32_t* __restrict node_end,
    const std::size_t* __restrict level_nodes,
    std::size_t num_level_nodes,
    double* __restrict clusters_q,
    double* __restrict clusters_q_dx,
    double* __restrict clusters_q_dy,
    double* __restrict clusters_q_dz)
{
    std::size_t level_idx = static_cast<std::size_t>(blockIdx.x);
    if (level_idx >= num_level_nodes) return;

    std::size_t node_idx = level_nodes[level_idx];
    std::size_t particle_start = static_cast<std::size_t>(node_begin[node_idx]);
    std::size_t particle_end = static_cast<std::size_t>(node_end[node_idx]);
    if (particle_end <= particle_start) return;

    std::size_t node_interp_pts_start =
        node_idx * static_cast<std::size_t>(num_interp_pts_per_node);
    std::size_t node_charges_start =
        node_idx * static_cast<std::size_t>(num_charges_per_node);

    int n = num_interp_pts_per_node;
    int n2 = n * n;
    int n3 = n2 * n;

    __shared__ double node_x[kMaxInterpPts];
    __shared__ double node_y[kMaxInterpPts];
    __shared__ double node_z[kMaxInterpPts];
    __shared__ double w_cache[kMaxInterpPts];

    if (threadIdx.x < n) {
        int j = threadIdx.x;
        node_x[j] = clusters_x[node_interp_pts_start + j];
        node_y[j] = clusters_y[node_interp_pts_start + j];
        node_z[j] = clusters_z[node_interp_pts_start + j];
        w_cache[j] = weights[j];
    }
    __syncthreads();

    int j_list[kMaxJPerThread];
    int k1_list[kMaxJPerThread];
    int k2_list[kMaxJPerThread];
    int k3_list[kMaxJPerThread];
    double cx_list[kMaxJPerThread];
    double cy_list[kMaxJPerThread];
    double cz_list[kMaxJPerThread];
    double w1_list[kMaxJPerThread];
    double w2_list[kMaxJPerThread];
    double w3_list[kMaxJPerThread];

    int m_count = 0;
    for (int j = threadIdx.x; j < n3; j += blockDim.x) {
        if (m_count >= kMaxJPerThread) break;
        int k1 = j / n2;
        int k2 = (j / n) % n;
        int k3 = j % n;
        j_list[m_count] = j;
        k1_list[m_count] = k1;
        k2_list[m_count] = k2;
        k3_list[m_count] = k3;
        cx_list[m_count] = node_x[k1];
        cy_list[m_count] = node_y[k2];
        cz_list[m_count] = node_z[k3];
        w1_list[m_count] = w_cache[k1];
        w2_list[m_count] = w_cache[k2];
        w3_list[m_count] = w_cache[k3];
        ++m_count;
    }

    double q_temp[kMaxJPerThread];
    double q_dx_temp[kMaxJPerThread];
    double q_dy_temp[kMaxJPerThread];
    double q_dz_temp[kMaxJPerThread];
    for (int m = 0; m < m_count; ++m) {
        q_temp[m] = 0.0;
        q_dx_temp[m] = 0.0;
        q_dy_temp[m] = 0.0;
        q_dz_temp[m] = 0.0;
    }

    __shared__ double tile_x[kParticleTile];
    __shared__ double tile_y[kParticleTile];
    __shared__ double tile_z[kParticleTile];
    __shared__ double tile_q[kParticleTile];
    __shared__ double tile_q_dx[kParticleTile];
    __shared__ double tile_q_dy[kParticleTile];
    __shared__ double tile_q_dz[kParticleTile];
    __shared__ double tile_denom[kParticleTile];
    __shared__ int tile_ex[kParticleTile];
    __shared__ int tile_ey[kParticleTile];
    __shared__ int tile_ez[kParticleTile];

    std::size_t num_particles = particle_end - particle_start;
    for (std::size_t tile_start = 0; tile_start < num_particles; tile_start += kParticleTile) {
        int tile_count = static_cast<int>(num_particles - tile_start);
        if (tile_count > kParticleTile) tile_count = kParticleTile;

        for (int ii = threadIdx.x; ii < tile_count; ii += blockDim.x) {
            std::size_t pidx = particle_start + tile_start + static_cast<std::size_t>(ii);
            double px = elements_x[pidx];
            double py = elements_y[pidx];
            double pz = elements_z[pidx];

            tile_x[ii] = px;
            tile_y[ii] = py;
            tile_z[ii] = pz;
            tile_q[ii] = sources_q[pidx];
            tile_q_dx[ii] = sources_q_dx[pidx];
            tile_q_dy[ii] = sources_q_dy[pidx];
            tile_q_dz[ii] = sources_q_dz[pidx];

            double denom_x = 0.0;
            double denom_y = 0.0;
            double denom_z = 0.0;
            int ex = -1;
            int ey = -1;
            int ez = -1;

            for (int j = 0; j < n; ++j) {
                double dist_x = px - node_x[j];
                double dist_y = py - node_y[j];
                double dist_z = pz - node_z[j];

                denom_x += w_cache[j] / dist_x;
                denom_y += w_cache[j] / dist_y;
                denom_z += w_cache[j] / dist_z;

                int cx = (fabs(dist_x) < DBL_MIN) ? j : -1;
                int cy = (fabs(dist_y) < DBL_MIN) ? j : -1;
                int cz = (fabs(dist_z) < DBL_MIN) ? j : -1;
                if (cx > ex) ex = cx;
                if (cy > ey) ey = cy;
                if (cz > ez) ez = cz;
            }

            double denom = 1.0;
            if (ex == -1) denom /= denom_x;
            if (ey == -1) denom /= denom_y;
            if (ez == -1) denom /= denom_z;

            tile_ex[ii] = ex;
            tile_ey[ii] = ey;
            tile_ez[ii] = ez;
            tile_denom[ii] = denom;
        }
        __syncthreads();

        for (int m = 0; m < m_count; ++m) {
            double cx = cx_list[m];
            double cy = cy_list[m];
            double cz = cz_list[m];
            double w1 = w1_list[m];
            double w2 = w2_list[m];
            double w3 = w3_list[m];
            int k1 = k1_list[m];
            int k2 = k2_list[m];
            int k3 = k3_list[m];

            for (int ii = 0; ii < tile_count; ++ii) {
                double dist_x = tile_x[ii] - cx;
                double dist_y = tile_y[ii] - cy;
                double dist_z = tile_z[ii] - cz;

                double numerator = 1.0;
                if (tile_ex[ii] == -1) {
                    numerator *= w1 / dist_x;
                } else {
                    if (tile_ex[ii] != k1) numerator *= 0.0;
                }

                if (tile_ey[ii] == -1) {
                    numerator *= w2 / dist_y;
                } else {
                    if (tile_ey[ii] != k2) numerator *= 0.0;
                }

                if (tile_ez[ii] == -1) {
                    numerator *= w3 / dist_z;
                } else {
                    if (tile_ez[ii] != k3) numerator *= 0.0;
                }

                double denom = tile_denom[ii];
                q_temp[m]    += tile_q[ii]    * numerator * denom;
                q_dx_temp[m] += tile_q_dx[ii] * numerator * denom;
                q_dy_temp[m] += tile_q_dy[ii] * numerator * denom;
                q_dz_temp[m] += tile_q_dz[ii] * numerator * denom;
            }
        }
        __syncthreads();
    }

    for (int m = 0; m < m_count; ++m) {
        std::size_t kk = node_charges_start + static_cast<std::size_t>(j_list[m]);
        clusters_q   [kk] += q_temp[m];
        clusters_q_dx[kk] += q_dx_temp[m];
        clusters_q_dy[kk] += q_dy_temp[m];
        clusters_q_dz[kk] += q_dz_temp[m];
    }
}

} // namespace

extern "C" void upward_denom_cuda(
    int num_interp_pts_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    std::size_t num_elements,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    int* exact_idx_x,
    int* exact_idx_y,
    int* exact_idx_z,
    double* denominator,
    void* stream)
{
    if (num_interp_pts_per_node <= 0 || num_interp_pts_per_node > kMaxInterpPts) return;
    if (num_level_nodes == 0) return;

    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid(static_cast<unsigned int>(num_level_nodes), 1, 1);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    upward_denom_kernel<<<grid, block, 0, cuda_stream>>>(
        num_interp_pts_per_node,
        clusters_x, clusters_y, clusters_z,
        weights,
        elements_x, elements_y, elements_z,
        num_elements,
        node_begin, node_end,
        level_nodes,
        num_level_nodes,
        exact_idx_x, exact_idx_y, exact_idx_z,
        denominator);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "upward_denom_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(cuda_stream);
}

extern "C" void upward_charge_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    const int* exact_idx_x,
    const int* exact_idx_y,
    const int* exact_idx_z,
    const double* denominator,
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    void* stream)
{
    if (num_interp_pts_per_node <= 0 || num_interp_pts_per_node > kMaxInterpPts) return;
    if (num_level_nodes == 0) return;

    int n = num_interp_pts_per_node;
    int n3 = n * n * n;
    int threads = 256;
    int blocks_y = (n3 + threads - 1) / threads;
    dim3 block(threads, 1, 1);
    dim3 grid(static_cast<unsigned int>(num_level_nodes),
              static_cast<unsigned int>(blocks_y),
              1);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    upward_charge_kernel<<<grid, block, 0, cuda_stream>>>(
        num_interp_pts_per_node,
        num_charges_per_node,
        clusters_x, clusters_y, clusters_z,
        weights,
        elements_x, elements_y, elements_z,
        sources_q, sources_q_dx, sources_q_dy, sources_q_dz,
        node_begin, node_end,
        level_nodes,
        num_level_nodes,
        exact_idx_x, exact_idx_y, exact_idx_z,
        denominator,
        clusters_q, clusters_q_dx, clusters_q_dy, clusters_q_dz);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "upward_charge_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(cuda_stream);
}

extern "C" void upward_fused_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    void* stream)
{
    if (num_interp_pts_per_node <= 0 || num_interp_pts_per_node > kMaxInterpPts) return;
    if (num_level_nodes == 0) return;

    int threads = 256;
    dim3 block(threads, 1, 1);
    dim3 grid(static_cast<unsigned int>(num_level_nodes), 1, 1);

    cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream) : 0;
    upward_fused_kernel<<<grid, block, 0, cuda_stream>>>(
        num_interp_pts_per_node,
        num_charges_per_node,
        clusters_x, clusters_y, clusters_z,
        weights,
        elements_x, elements_y, elements_z,
        sources_q, sources_q_dx, sources_q_dy, sources_q_dz,
        node_begin, node_end,
        level_nodes,
        num_level_nodes,
        clusters_q, clusters_q_dx, clusters_q_dy, clusters_q_dz);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "upward_fused_cuda kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaStreamSynchronize(cuda_stream);
}
