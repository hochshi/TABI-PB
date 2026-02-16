#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "interp_pts.h"
#include "tree.h"
#include "constants.h"

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#include "interp_pts_cuda.h"
#endif

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

    double* __restrict clusters_x_ptr   = interp_x_.data();
    double* __restrict clusters_y_ptr   = interp_y_.data();
    double* __restrict clusters_z_ptr   = interp_z_.data();

    bool require_all = false;
#ifdef USE_CUDA_CC
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
#endif
    
    int num_interp_pts_per_node = num_interp_pts_per_node_;
    int degree = num_interp_pts_per_node - 1;

#ifdef USE_CUDA_CC
    const bool use_cuda = (device_state_ == CudaDeviceState::DeviceMapped);
    if (require_all && !use_cuda) {
        std::fprintf(stderr,
                     "[CUDA_INTERP] TABIPB_CUDA_REQUIRE_ALL=1 but interp "
                     "device buffers are not mapped.\n");
        std::abort();
    }

    if (use_cuda) {
        const std::size_t num_nodes = tree_.num_nodes();
        std::vector<double> bounds(num_nodes * 6);
        for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
            auto node_bounds = tree_.node_particle_bounds(node_idx);
            const std::size_t base = node_idx * 6;
            bounds[base + 0] = node_bounds[0];
            bounds[base + 1] = node_bounds[1];
            bounds[base + 2] = node_bounds[2];
            bounds[base + 3] = node_bounds[3];
            bounds[base + 4] = node_bounds[4];
            bounds[base + 5] = node_bounds[5];
        }

        double* bounds_dev = nullptr;
        CUDA_MALLOC_OR_DIE(&bounds_dev, bounds.size() * sizeof(double));

        cudaStream_t stream = nullptr;

        CUDA_MEMCPY_ASYNC(bounds_dev, bounds.data(),
                          bounds.size() * sizeof(double),
                          cudaMemcpyHostToDevice, stream);

        auto &buf = device_buffers_;
        interp_pts_cuda(bounds_dev, buf.interp_x_dev, buf.interp_y_dev,
                        buf.interp_z_dev, num_nodes, num_interp_pts_per_node,
                        stream);
        CUDA_CHECK_LAST_KERNEL();
        CUDA_SYNC_AND_CHECK();
        CUDA_FREE_AND_NULL(bounds_dev);
        return;
    }
#endif

    for (std::size_t node_idx = 0; node_idx < tree_.num_nodes(); ++node_idx) {
        std::size_t node_start = node_idx * num_interp_pts_per_node_;
        auto node_bounds = tree_.node_particle_bounds(node_idx);

        for (int i = 0; i < num_interp_pts_per_node; ++i) {
            double tt = std::cos(i * constants::PI / degree);
            clusters_x_ptr[node_start + i] = node_bounds[0] + (tt + 1.) / 2. * (node_bounds[1] - node_bounds[0]);
            clusters_y_ptr[node_start + i] = node_bounds[2] + (tt + 1.) / 2. * (node_bounds[3] - node_bounds[2]);
            clusters_z_ptr[node_start + i] = node_bounds[4] + (tt + 1.) / 2. * (node_bounds[5] - node_bounds[4]);
        }
    }

    //timers_.compute_all_interp_pts.stop();
}




void InterpolationPoints::copyin_to_device() const
{
//    timers_.copyin_to_device.start();

#ifdef USE_CUDA_CC
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);

    const std::size_t num_interp_pts = num_interp_pts_;
    auto &buf = device_buffers_;
    if (buf.num_interp_pts != 0 && buf.num_interp_pts != num_interp_pts) {
        CUDA_FREE_AND_NULL(buf.interp_x_dev);
        CUDA_FREE_AND_NULL(buf.interp_y_dev);
        CUDA_FREE_AND_NULL(buf.interp_z_dev);
        buf.num_interp_pts = 0;
        buf.ready = false;
    }

    if (buf.num_interp_pts == 0 && num_interp_pts > 0) {
        CUDA_MALLOC_OR_DIE(&buf.interp_x_dev,
                           num_interp_pts * sizeof(double));
        CUDA_MALLOC_OR_DIE(&buf.interp_y_dev,
                           num_interp_pts * sizeof(double));
        CUDA_MALLOC_OR_DIE(&buf.interp_z_dev,
                           num_interp_pts * sizeof(double));
        buf.num_interp_pts = num_interp_pts;
    }

    CUDA_SYNC_AND_CHECK();
    buf.ready = true;
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all && num_interp_pts > 0) {
        if (!buf.interp_x_dev || !buf.interp_y_dev || !buf.interp_z_dev) {
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
    auto &buf = device_buffers_;
    CUDA_FREE_AND_NULL(buf.interp_x_dev);
    CUDA_FREE_AND_NULL(buf.interp_y_dev);
    CUDA_FREE_AND_NULL(buf.interp_z_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
#endif

//    timers_.delete_from_device.stop();
}
