#ifndef H_TABIPB_INTERP_PTS_STRUCT_H
#define H_TABIPB_INTERP_PTS_STRUCT_H

#include <cstddef>

#include "tree.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

class InterpolationPoints
{
private:
    friend class BoundaryElement;

    const class Tree& tree_;

    int num_interp_pts_per_node_;
    std::size_t num_interp_pts_;

    std::vector<double> interp_x_;
    std::vector<double> interp_y_;
    std::vector<double> interp_z_;

#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class InterpolationPoints;
        friend class BoundaryElement;
    private:
        bool ready = false;
        double* interp_x_dev = nullptr;
        double* interp_y_dev = nullptr;
        double* interp_z_dev = nullptr;
        std::size_t num_interp_pts = 0;
    };

    mutable DeviceBuffers device_buffers_;
    mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
    bool validate_device_buffers_compute_interp_pts_() const;
    bool try_compute_all_interp_pts_cuda_(const double* node_bounds,
                                          std::size_t num_nodes,
                                          int num_interp_pts_per_node) const;
    void copyin_to_device_cuda_() const;
    void delete_from_device_cuda_() const;
#endif
    
    
public:
    InterpolationPoints(const class Tree&, int degree);
    ~InterpolationPoints() = default;
    
    std::size_t num_interp_pts_per_node() const { return num_interp_pts_per_node_; };
    
    const std::array<std::size_t, 2> cluster_interp_pts_idxs(std::size_t node_idx) const {
        return std::array<std::size_t, 2> {num_interp_pts_per_node_ *  node_idx,
                                           num_interp_pts_per_node_ * (node_idx + 1)};
    };
    
    const double* interp_x_ptr() const { return interp_x_.data(); };
    const double* interp_y_ptr() const { return interp_y_.data(); };
    const double* interp_z_ptr() const { return interp_z_.data(); };

    struct View {
        bool ready = false;
        double* interp_x = nullptr;
        double* interp_y = nullptr;
        double* interp_z = nullptr;
        std::size_t num_interp_pts = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };
    using DeviceView = View;

    View device_view() const {
        View view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.interp_x = device_buffers_.interp_x_dev;
        view.interp_y = device_buffers_.interp_y_dev;
        view.interp_z = device_buffers_.interp_z_dev;
        view.num_interp_pts = device_buffers_.num_interp_pts;
        view.state = device_state_;
#endif
        return view;
    }

    View host_view() {
        View view;
        view.ready = true;
        view.interp_x = interp_x_.data();
        view.interp_y = interp_y_.data();
        view.interp_z = interp_z_.data();
        view.num_interp_pts = num_interp_pts_;
#ifdef USE_CUDA_CC
        view.state = CudaDeviceState::HostOnly;
#endif
        return view;
    }

#ifdef USE_CUDA_CC
    bool cuda_device_ready() const {
        return device_state_ == CudaDeviceState::DeviceMapped &&
               device_buffers_.ready;
    }
#else
    bool cuda_device_ready() const { return false; }
#endif
    
    void compute_all_interp_pts();
    void copyin_to_device() const;
    void delete_from_device() const;
    
};

#endif /* H_TABIPB_INTERP_PTS_STRUCT_H */
