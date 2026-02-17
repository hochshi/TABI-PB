#ifndef H_TABIPB_INTERP_PTS_BACKEND_CPU_H
#define H_TABIPB_INTERP_PTS_BACKEND_CPU_H

#include <cstddef>

#include "interp_pts.h"

void interp_pts_compute_cpu(const double* node_bounds,
                            std::size_t num_nodes,
                            int num_interp_pts_per_node,
                            const InterpolationPoints::View& interp_view);

#endif
