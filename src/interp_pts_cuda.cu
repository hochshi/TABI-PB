#include "interp_pts_cuda.h"

#include <cuda_runtime.h>

#include "constants.h"

namespace {

__global__ void interp_pts_kernel(const double* node_bounds,
                                  double* interp_x,
                                  double* interp_y,
                                  double* interp_z,
                                  std::size_t num_nodes,
                                  int num_interp_pts_per_node)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_nodes * static_cast<std::size_t>(num_interp_pts_per_node);
    if (idx >= total) return;

    const std::size_t node_idx = idx / static_cast<std::size_t>(num_interp_pts_per_node);
    const int i = static_cast<int>(idx - node_idx * static_cast<std::size_t>(num_interp_pts_per_node));

    const int degree = num_interp_pts_per_node - 1;
    const double tt = (degree == 0) ? 1.0 : cos(static_cast<double>(i) * constants::PI / static_cast<double>(degree));

    const double* bounds = node_bounds + node_idx * 6;
    const double x_min = bounds[0];
    const double x_max = bounds[1];
    const double y_min = bounds[2];
    const double y_max = bounds[3];
    const double z_min = bounds[4];
    const double z_max = bounds[5];

    const std::size_t out_idx = node_idx * static_cast<std::size_t>(num_interp_pts_per_node) + static_cast<std::size_t>(i);
    interp_x[out_idx] = x_min + (tt + 1.0) * 0.5 * (x_max - x_min);
    interp_y[out_idx] = y_min + (tt + 1.0) * 0.5 * (y_max - y_min);
    interp_z[out_idx] = z_min + (tt + 1.0) * 0.5 * (z_max - z_min);
}

inline int grid_for(std::size_t n, int block)
{
    return static_cast<int>((n + static_cast<std::size_t>(block) - 1) / static_cast<std::size_t>(block));
}

} // namespace

extern "C" void interp_pts_cuda(const double* node_bounds,
                                double* interp_x,
                                double* interp_y,
                                double* interp_z,
                                std::size_t num_nodes,
                                int num_interp_pts_per_node,
                                void* stream)
{
    if (num_nodes == 0 || num_interp_pts_per_node <= 0) return;
    constexpr int kBlock = 256;
    const std::size_t total = num_nodes * static_cast<std::size_t>(num_interp_pts_per_node);
    const int grid = grid_for(total, kBlock);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    interp_pts_kernel<<<grid, kBlock, 0, cuda_stream>>>(
        node_bounds, interp_x, interp_y, interp_z, num_nodes, num_interp_pts_per_node);
}
