#ifndef H_TABIPB_INTERP_PTS_BACKEND_CUDA_H
#define H_TABIPB_INTERP_PTS_BACKEND_CUDA_H

#include <cstddef>

#include "interp_pts.h"

bool interp_pts_try_compute_cuda(const double* node_bounds,
                                 std::size_t num_nodes,
                                 int num_interp_pts_per_node,
                                 const InterpolationPoints::View& interp_view,
                                 void* stream);

#endif
