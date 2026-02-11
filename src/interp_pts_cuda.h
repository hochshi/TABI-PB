#pragma once

#include <cstddef>

extern "C" void interp_pts_cuda(const double* node_bounds,
                                double* interp_x,
                                double* interp_y,
                                double* interp_z,
                                std::size_t num_nodes,
                                int num_interp_pts_per_node,
                                void* stream);
