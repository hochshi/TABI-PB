#include "interp_pts.h"

#ifdef USE_CUDA_CC
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuda_helpers.h"
#include "interp_pts_cuda.h"

bool InterpolationPoints::validate_device_buffers_compute_interp_pts_() const {
    const auto& buf = device_buffers_;
    return cuda_device_ready() &&
           buf.interp_x_dev &&
           buf.interp_y_dev &&
           buf.interp_z_dev &&
           buf.num_interp_pts == num_interp_pts_;
}

bool InterpolationPoints::try_compute_all_interp_pts_cuda_(const double* node_bounds,
                                                           std::size_t num_nodes,
                                                           int num_interp_pts_per_node) const {
    if (!validate_device_buffers_compute_interp_pts_()) {
        return false;
    }
    if (!node_bounds || num_nodes == 0 || num_interp_pts_per_node <= 0) {
        return true;
    }

    double* bounds_dev = nullptr;
    CUDA_MALLOC_OR_DIE(&bounds_dev, num_nodes * 6 * sizeof(double));

    cudaStream_t stream = nullptr;
    CUDA_MEMCPY_ASYNC(bounds_dev, node_bounds, num_nodes * 6 * sizeof(double),
                      cudaMemcpyHostToDevice, stream);

    const auto& buf = device_buffers_;
    interp_pts_cuda(bounds_dev, buf.interp_x_dev, buf.interp_y_dev, buf.interp_z_dev,
                    num_nodes, num_interp_pts_per_node, stream);
    CUDA_CHECK_LAST_KERNEL();
    CUDA_SYNC_AND_CHECK();
    CUDA_FREE_AND_NULL(bounds_dev);
    return true;
}

void InterpolationPoints::copyin_to_device_cuda_() const {
    const std::size_t num_interp_pts = num_interp_pts_;
    auto& buf = device_buffers_;
    if (buf.num_interp_pts != 0 && buf.num_interp_pts != num_interp_pts) {
        CUDA_FREE_AND_NULL(buf.interp_x_dev);
        CUDA_FREE_AND_NULL(buf.interp_y_dev);
        CUDA_FREE_AND_NULL(buf.interp_z_dev);
        buf.num_interp_pts = 0;
        buf.ready = false;
    }

    if (buf.num_interp_pts == 0 && num_interp_pts > 0) {
        CUDA_MALLOC_OR_DIE(&buf.interp_x_dev, num_interp_pts * sizeof(double));
        CUDA_MALLOC_OR_DIE(&buf.interp_y_dev, num_interp_pts * sizeof(double));
        CUDA_MALLOC_OR_DIE(&buf.interp_z_dev, num_interp_pts * sizeof(double));
        buf.num_interp_pts = num_interp_pts;
    }

    CUDA_SYNC_AND_CHECK();
    buf.ready = true;
    device_state_ = CudaDeviceState::DeviceMapped;
}

void InterpolationPoints::delete_from_device_cuda_() const {
    auto& buf = device_buffers_;
    CUDA_FREE_AND_NULL(buf.interp_x_dev);
    CUDA_FREE_AND_NULL(buf.interp_y_dev);
    CUDA_FREE_AND_NULL(buf.interp_z_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
