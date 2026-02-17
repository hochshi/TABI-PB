#include "interp_pts_backend_cpu.h"

#include <cmath>

#include "constants.h"

void interp_pts_compute_cpu(const double* node_bounds,
                            std::size_t num_nodes,
                            int num_interp_pts_per_node,
                            const InterpolationPoints::View& interp_view) {
    if (!node_bounds || num_nodes == 0 || num_interp_pts_per_node <= 0 ||
        !interp_view.interp_x || !interp_view.interp_y || !interp_view.interp_z) {
        return;
    }

    const int degree = num_interp_pts_per_node - 1;
    for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
        const std::size_t node_start = node_idx * static_cast<std::size_t>(num_interp_pts_per_node);
        const std::size_t base = node_idx * 6;

        const double x_min = node_bounds[base + 0];
        const double x_max = node_bounds[base + 1];
        const double y_min = node_bounds[base + 2];
        const double y_max = node_bounds[base + 3];
        const double z_min = node_bounds[base + 4];
        const double z_max = node_bounds[base + 5];

        for (int i = 0; i < num_interp_pts_per_node; ++i) {
            const double tt = (degree == 0) ? 1.0 :
                              std::cos(static_cast<double>(i) * constants::PI /
                                       static_cast<double>(degree));
            const std::size_t out = node_start + static_cast<std::size_t>(i);
            interp_view.interp_x[out] = x_min + (tt + 1.0) * 0.5 * (x_max - x_min);
            interp_view.interp_y[out] = y_min + (tt + 1.0) * 0.5 * (y_max - y_min);
            interp_view.interp_z[out] = z_min + (tt + 1.0) * 0.5 * (z_max - z_min);
        }
    }
}
