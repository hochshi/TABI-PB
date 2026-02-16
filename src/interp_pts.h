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
