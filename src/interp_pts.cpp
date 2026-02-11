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
#include <cuda_runtime.h>
#include "interp_pts_cuda.h"
#endif

#ifdef OPENACC_ENABLED
#include <openacc.h>
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
#if defined(USE_CUDA_CC) && defined(OPENACC_ENABLED)
    if (require_all) {
        const std::size_t num_nodes = tree_.num_nodes();
        const std::size_t num_interp_pts = num_interp_pts_;
        const bool present_ok =
            acc_is_present((void*)clusters_x_ptr, num_interp_pts * sizeof(double)) &&
            acc_is_present((void*)clusters_y_ptr, num_interp_pts * sizeof(double)) &&
            acc_is_present((void*)clusters_z_ptr, num_interp_pts * sizeof(double));
        if (!present_ok) {
            std::cerr << "[CUDA_INTERP] require_all set but interp buffers not present. "
                      << "Aborting to avoid OpenACC fallback.\n";
            std::exit(1);
        }

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

        auto check = [](cudaError_t err, const char* what) {
            if (err != cudaSuccess) {
                std::cerr << "[CUDA_INTERP] " << what << " failed: "
                          << cudaGetErrorString(err) << "\n";
                std::exit(1);
            }
        };

        double* bounds_dev = nullptr;
        check(cudaMalloc(&bounds_dev, bounds.size() * sizeof(double)), "cudaMalloc bounds");
        check(cudaMemcpy(bounds_dev, bounds.data(), bounds.size() * sizeof(double),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy bounds");

        acc_wait(acc_async_sync);
        void* stream = acc_get_cuda_stream(acc_async_sync);
        #pragma acc host_data use_device(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr)
        {
            interp_pts_cuda(bounds_dev, clusters_x_ptr, clusters_y_ptr, clusters_z_ptr,
                            num_nodes, num_interp_pts_per_node, stream);
        }
        check(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
              "cudaStreamSynchronize interp_pts");
        check(cudaFree(bounds_dev), "cudaFree bounds");
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

#ifdef OPENACC_ENABLED
    {
        const std::size_t num_interp_pts = num_interp_pts_;
        #pragma acc update device(clusters_x_ptr[0:num_interp_pts], \
                                  clusters_y_ptr[0:num_interp_pts], \
                                  clusters_z_ptr[0:num_interp_pts])
    }
#endif

    //timers_.compute_all_interp_pts.stop();
}




void InterpolationPoints::copyin_to_device() const
{
//    timers_.copyin_to_device.start();

#ifdef OPENACC_ENABLED
    const double* x_ptr = interp_x_.data();
    const double* y_ptr = interp_y_.data();
    const double* z_ptr = interp_z_.data();
    
    std::size_t x_num = interp_x_.size();
    std::size_t y_num = interp_y_.size();
    std::size_t z_num = interp_z_.size();
    
    #pragma acc enter data create(x_ptr[0:x_num], y_ptr[0:y_num], z_ptr[0:z_num])
#endif

//    timers_.copyin_to_device.stop();
}


void InterpolationPoints::delete_from_device() const
{
//    timers_.delete_from_device.start();

#ifdef OPENACC_ENABLED
    const double* x_ptr = interp_x_.data();
    const double* y_ptr = interp_y_.data();
    const double* z_ptr = interp_z_.data();
    
    std::size_t x_num = interp_x_.size();
    std::size_t y_num = interp_y_.size();
    std::size_t z_num = interp_z_.size();
    
    #pragma acc exit data delete(x_ptr[0:x_num], y_ptr[0:y_num], z_ptr[0:z_num])
#endif

//    timers_.delete_from_device.stop();
}
