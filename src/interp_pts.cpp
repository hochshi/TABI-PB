#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "interp_pts.h"
#include "interp_pts_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "interp_pts_backend_cuda.h"
#endif

namespace {
std::vector<double> build_node_bounds(const Tree& tree) {
    const std::size_t num_nodes = tree.num_nodes();
    std::vector<double> bounds(num_nodes * 6);
    for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
        const auto node_bounds = tree.node_particle_bounds(node_idx);
        const std::size_t base = node_idx * 6;
        bounds[base + 0] = node_bounds[0];
        bounds[base + 1] = node_bounds[1];
        bounds[base + 2] = node_bounds[2];
        bounds[base + 3] = node_bounds[3];
        bounds[base + 4] = node_bounds[4];
        bounds[base + 5] = node_bounds[5];
    }
    return bounds;
}
}

InterpolationPoints::InterpolationPoints(const class Tree& tree, int degree)
    : tree_(tree)
{
    //timers_.ctor.start();

    num_interp_pts_per_node_ = degree + 1;
    num_interp_pts_ = tree_.num_nodes() * num_interp_pts_per_node_;

    interp_x_.resize(num_interp_pts_);
    interp_y_.resize(num_interp_pts_);
    interp_z_.resize(num_interp_pts_);

    //timers_.ctor.stop();
}


void InterpolationPoints::compute_all_interp_pts()
{
    //timers_.compute_all_interp_pts.start();
    const std::size_t num_nodes = tree_.num_nodes();
    const int num_interp_pts_per_node = num_interp_pts_per_node_;
    const auto bounds = build_node_bounds(tree_);

#ifdef USE_CUDA_CC
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = !(require_all_env && std::strcmp(require_all_env, "0") == 0);
    if (validate_device_buffers_compute_interp_pts_()) {
        const auto interp_view = device_view();
        if (interp_pts_try_compute_cuda(bounds.data(), num_nodes, num_interp_pts_per_node,
                                        interp_view, nullptr)) {
            return;
        }
    }
    if (require_all) {
        std::fprintf(stderr,
                     "[CUDA_INTERP] TABIPB_CUDA_REQUIRE_ALL=1 but interp "
                     "device buffers are not mapped.\n");
        std::abort();
    }
#endif

    const auto interp_view = host_view();
    interp_pts_compute_cpu(bounds.data(), num_nodes, num_interp_pts_per_node, interp_view);

    //timers_.compute_all_interp_pts.stop();
}




void InterpolationPoints::copyin_to_device() const
{
//    timers_.copyin_to_device.start();

#ifdef USE_CUDA_CC
    copyin_to_device_cuda_();
    const std::size_t num_interp_pts = num_interp_pts_;
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = !(require_all_env && std::strcmp(require_all_env, "0") == 0);
    if (require_all && num_interp_pts > 0) {
        if (!validate_device_buffers_compute_interp_pts_()) {
            std::fprintf(stderr,
                         "[CUDA_INTERP] missing device buffers under "
                         "TABIPB_CUDA_REQUIRE_ALL=1\n");
            std::abort();
        }
    }
#endif

//    timers_.copyin_to_device.stop();
}


void InterpolationPoints::delete_from_device() const
{
//    timers_.delete_from_device.start();

#ifdef USE_CUDA_CC
    delete_from_device_cuda_();
#endif

//    timers_.delete_from_device.stop();
}
