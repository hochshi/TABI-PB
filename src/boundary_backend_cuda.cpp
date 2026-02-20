#include "boundary_element.h"

#ifdef USE_CUDA_CC
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_helpers.h"
#include "cc_cuda.h"
#include "up_cuda.h"
#include "be_cuda.h"
#include "elements_cuda.h"
#include "down_cuda.h"
#include "pc_cuda.h"
#include "pp_cuda.h"
#include "pppc_cuda.h"

void BoundaryElement::reset_device_buffers_() const
{
    device_buffers_ = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
}

void BoundaryElement::matrix_vector_cuda(double alpha, const double* potential_old_dev,
                                         double beta, double* potential_new_dev,
                                         void* stream)
{
    timers_.matrix_vector.start();

    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        if (require_all) {
            std::cerr << "[CUDA_BE] require_all set but CUDA pointers not cached. "
                      << "Aborting to avoid OpenACC fallback.\n";
            std::exit(1);
        }
        timers_.matrix_vector.stop();
        return;
    }

    const DeviceBuffers& cp = device_buffers_;
    const std::size_t potential_num = potential_.size();
    const double potential_coeff_1 = 0.5 * (1. + params_.phys_eps_);
    const double potential_coeff_2 = 0.5 * (1. + 1. / params_.phys_eps_);

    const std::size_t num_nodes = cp.num_nodes;
    const std::size_t num_elements = elements_.num();
    const std::size_t num_charges = static_cast<std::size_t>(num_charges_per_node_) * num_nodes;
    const int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    constexpr int kMaxInterpPts = 16;

    if (require_all && num_interp_pts_per_node > kMaxInterpPts) {
        std::cerr << "[CUDA_BE] require_all set but num_interp_pts_per_node="
                  << num_interp_pts_per_node
                  << " exceeds CUDA upward/downward limit " << kMaxInterpPts
                  << ". Aborting to avoid OpenACC fallback.\n";
        std::exit(1);
    }

    be_potential_copy_zero_cuda(potential_new_dev, cp.potential_temp, potential_new_dev,
                                potential_num, stream);

    cuda_timer_queue_.begin(timers_.clear_cluster_charges, stream);
    be_clear_cluster_charges_cuda(cp.clusters_q, cp.clusters_q_dx, cp.clusters_q_dy, cp.clusters_q_dz,
                                  num_charges, stream);
    cuda_timer_queue_.end(stream);

    cuda_timer_queue_.begin(timers_.clear_cluster_potentials, stream);
    be_clear_cluster_potentials_cuda(cp.clusters_p, cp.clusters_p_dx, cp.clusters_p_dy, cp.clusters_p_dz,
                                     num_charges, stream);
    cuda_timer_queue_.end(stream);

    cuda_timer_queue_.begin(elements_.compute_charges_timer(), stream);
    elements_compute_charges_cuda(cp.elements_nx, cp.elements_ny, cp.elements_nz, cp.elements_area,
                                  potential_old_dev,
                                  cp.targets_q, cp.targets_q_dx, cp.targets_q_dy, cp.targets_q_dz,
                                  cp.sources_q, cp.sources_q_dx, cp.sources_q_dy, cp.sources_q_dz,
                                  num_elements, stream);
    cuda_timer_queue_.end(stream);

    if (num_interp_pts_per_node <= kMaxInterpPts && cp.level_nodes_num > 0) {
        cuda_timer_queue_.begin(timers_.upward_pass, stream);
        upward_fused_cuda(num_interp_pts_per_node, num_charges_per_node_,
                          cp.clusters_x, cp.clusters_y, cp.clusters_z,
                          cp.weights,
                          cp.elements_x, cp.elements_y, cp.elements_z,
                          cp.sources_q, cp.sources_q_dx, cp.sources_q_dy, cp.sources_q_dz,
                          cp.node_begin, cp.node_end,
                          cp.level_nodes, cp.level_nodes_num,
                          cp.clusters_q, cp.clusters_q_dx, cp.clusters_q_dy, cp.clusters_q_dz,
                          stream);
        cuda_timer_queue_.end(stream);
    }

    const double eps = params_.phys_eps_;
    const double kappa = params_.phys_kappa_;
    const double kappa2 = params_.phys_kappa2_;

    bool use_fused_pppc = false;
    const char* fused_env = std::getenv("TABIPB_CUDA_PPPC_FUSED");
    const char* require_fused_env = std::getenv("TABIPB_CUDA_REQUIRE_PPPC");
    const bool require_fused = require_all || (require_fused_env && std::strcmp(require_fused_env, "0") != 0);
    use_fused_pppc = require_fused || (fused_env && std::strcmp(fused_env, "0") != 0);

    if (use_fused_pppc) {
        cuda_timer_queue_.begin(timers_.particle_cluster_interact, stream);
        pppc_interact_cuda(num_interp_pts_per_node, num_charges_per_node_,
                           eps, kappa, kappa2,
                           cp.clusters_x, cp.clusters_y, cp.clusters_z,
                           cp.clusters_q, cp.clusters_q_dx, cp.clusters_q_dy, cp.clusters_q_dz,
                           cp.elements_x, cp.elements_y, cp.elements_z,
                           cp.elements_nx, cp.elements_ny, cp.elements_nz,
                           cp.elements_area,
                           cp.targets_q, cp.targets_q_dx, cp.targets_q_dy, cp.targets_q_dz,
                           potential_old_dev, potential_new_dev,
                           num_elements,
                           cp.element_node_idx, num_nodes,
                           cp.node_begin, cp.node_end,
                           cp.pp_offsets, cp.pp_sources,
                           pp_offsets_u32_.size(), pp_sources_u32_.size(),
                           cp.pc_offsets, cp.pc_sources,
                           pc_offsets_u32_.size(), pc_sources_u32_.size(),
                           stream);
        cuda_timer_queue_.end(stream);
    } else {
        cuda_timer_queue_.begin(timers_.particle_particle_interact, stream);
        pp_interact_cuda(eps, kappa, kappa2,
                         cp.elements_x, cp.elements_y, cp.elements_z,
                         cp.elements_nx, cp.elements_ny, cp.elements_nz,
                         cp.elements_area,
                         potential_old_dev, potential_new_dev,
                         num_elements,
                         cp.element_node_idx,
                         num_nodes,
                         cp.node_begin, cp.node_end,
                         cp.pp_offsets, cp.pp_sources,
                         pp_offsets_u32_.size(), pp_sources_u32_.size(),
                         stream);
        cuda_timer_queue_.end(stream);

        cuda_timer_queue_.begin(timers_.particle_cluster_interact, stream);
        pc_interact_cuda(num_interp_pts_per_node, num_charges_per_node_,
                         eps, kappa, kappa2,
                         cp.clusters_x, cp.clusters_y, cp.clusters_z,
                         cp.clusters_q, cp.clusters_q_dx, cp.clusters_q_dy, cp.clusters_q_dz,
                         cp.elements_x, cp.elements_y, cp.elements_z,
                         cp.targets_q, cp.targets_q_dx, cp.targets_q_dy, cp.targets_q_dz,
                         potential_new_dev,
                         num_elements,
                         cp.element_node_idx,
                         num_nodes,
                         cp.pc_offsets, cp.pc_sources,
                         pc_offsets_u32_.size(), pc_sources_u32_.size(),
                         stream);
        cuda_timer_queue_.end(stream);
    }

    const char* disable_cp_env = std::getenv("TABIPB_CUDA_CC_DISABLE_CP");
    const bool disable_cp = (disable_cp_env && std::strcmp(disable_cp_env, "0") != 0);
    if (!disable_cp) {
        int n = num_interp_pts_per_node;
        int n2 = n * n;
        int n3 = n2 * n;
        cuda_timer_queue_.begin(timers_.cluster_particle_interact, stream);
        cp_interact_cuda(n, n2, n3, num_interp_pts_per_node, num_charges_per_node_,
                         eps, kappa, kappa2,
                         cp.clusters_x, cp.clusters_y, cp.clusters_z,
                         cp.clusters_p, cp.clusters_p_dx, cp.clusters_p_dy, cp.clusters_p_dz,
                         cp.node_begin, cp.node_end,
                         cp.cp_offsets, cp.cp_sources,
                         cp.elements_x, cp.elements_y, cp.elements_z,
                         cp.sources_q, cp.sources_q_dx, cp.sources_q_dy, cp.sources_q_dz,
                         num_elements,
                         num_nodes,
                         cp_offsets_u32_.size(), cp_sources_u32_.size(),
                         stream);
        cuda_timer_queue_.end(stream);
    }

    {
        int n = num_interp_pts_per_node;
        int n2 = n * n;
        int n3 = n2 * n;
        cuda_timer_queue_.begin(timers_.cluster_cluster_interact, stream);
        cc_interact_cuda(n, n2, n3, num_interp_pts_per_node, num_charges_per_node_,
                         eps, kappa, kappa2,
                         cp.clusters_x, cp.clusters_y, cp.clusters_z,
                         cp.clusters_q, cp.clusters_q_dx, cp.clusters_q_dy, cp.clusters_q_dz,
                         cp.clusters_p, cp.clusters_p_dx, cp.clusters_p_dy, cp.clusters_p_dz,
                         cp.node_begin, cp.node_end,
                         cp.cp_offsets, cp.cp_sources,
                         cp.cc_offsets, cp.cc_sources,
                         cp.elements_x, cp.elements_y, cp.elements_z,
                         cp.sources_q, cp.sources_q_dx, cp.sources_q_dy, cp.sources_q_dz,
                         num_elements,
                         num_nodes,
                         cp_offsets_u32_.size(), cp_sources_u32_.size(),
                         cc_offsets_u32_.size(), cc_sources_u32_.size(),
                         0,
                         stream);
        cuda_timer_queue_.end(stream);
    }

    const std::size_t* level_offsets_ptr = level_offsets_.data();
    const std::size_t level_count = level_offsets_.empty() ? 0 : (level_offsets_.size() - 1);
    const std::size_t potential_offset = elements_.num();
    for (std::size_t level = 0; level < level_count; ++level) {
        std::size_t level_begin = level_offsets_ptr[level];
        std::size_t level_end = level_offsets_ptr[level + 1];
        std::size_t num_level_nodes = level_end - level_begin;
        if (num_level_nodes == 0) continue;
        const std::size_t* level_nodes_dev = cp.level_nodes + level_begin;
        cuda_timer_queue_.begin(timers_.downward_pass, stream);
        downward_cuda(num_interp_pts_per_node, num_charges_per_node_,
                      cp.clusters_x, cp.clusters_y, cp.clusters_z,
                      cp.clusters_p, cp.clusters_p_dx, cp.clusters_p_dy, cp.clusters_p_dz,
                      cp.elements_x, cp.elements_y, cp.elements_z,
                      cp.targets_q, cp.targets_q_dx, cp.targets_q_dy, cp.targets_q_dz,
                      cp.weights,
                      potential_new_dev, potential_offset,
                      cp.node_begin, cp.node_end,
                      level_nodes_dev, num_level_nodes,
                      stream);
        cuda_timer_queue_.end(stream);
    }

    be_potential_combine_cuda(potential_old_dev, cp.potential_temp, potential_new_dev,
                              potential_num, alpha, beta,
                              potential_coeff_1, potential_coeff_2,
                              stream);

    timers_.matrix_vector.stop();
}

bool BoundaryElement::validate_device_buffers_matrix_vector_(
    const double* potential_old, const double* potential_new) const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!device_buffers_.potential_temp) {
        return false;
    }
    const std::size_t potential_num = potential_.size();
    if (potential_num > 0) {
        if (!cuda_pointer_is_device_accessible(potential_new)) {
            return false;
        }
        if (potential_old &&
            !cuda_pointer_is_device_accessible(potential_old)) {
            return false;
        }
    }
    return true;
}

bool BoundaryElement::validate_device_buffers_particle_particle_() const
{
    const auto& b = device_buffers_;
    return device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
           elements_.cuda_device_ready() && interp_pts_.cuda_device_ready() &&
           b.elements_x && b.elements_y && b.elements_z &&
           b.elements_nx && b.elements_ny && b.elements_nz && b.elements_area &&
           b.node_begin && b.node_end && b.element_node_idx &&
           b.pp_offsets && b.pp_sources;
}

bool BoundaryElement::validate_device_buffers_particle_cluster_(bool include_pp) const
{
    const auto& b = device_buffers_;
    const bool base_ok =
        device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
        elements_.cuda_device_ready() && interp_pts_.cuda_device_ready() &&
        b.clusters_x && b.clusters_y && b.clusters_z &&
        b.clusters_q && b.clusters_q_dx && b.clusters_q_dy && b.clusters_q_dz &&
        b.elements_x && b.elements_y && b.elements_z &&
        b.targets_q && b.targets_q_dx && b.targets_q_dy && b.targets_q_dz &&
        b.element_node_idx && b.pc_offsets && b.pc_sources;
    if (!base_ok) {
        return false;
    }
    if (include_pp) {
        return b.elements_nx && b.elements_ny && b.elements_nz && b.elements_area &&
               b.node_begin && b.node_end &&
               b.pp_offsets && b.pp_sources;
    }
    return true;
}

bool BoundaryElement::validate_device_buffers_cluster_mixed_() const
{
    const auto& b = device_buffers_;
    return device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
           elements_.cuda_device_ready() && interp_pts_.cuda_device_ready() &&
           b.clusters_x && b.clusters_y && b.clusters_z &&
           b.clusters_q && b.clusters_q_dx && b.clusters_q_dy && b.clusters_q_dz &&
           b.clusters_p && b.clusters_p_dx && b.clusters_p_dy && b.clusters_p_dz &&
           b.elements_x && b.elements_y && b.elements_z &&
           b.sources_q && b.sources_q_dx && b.sources_q_dy && b.sources_q_dz &&
           b.node_begin && b.node_end &&
           b.cp_offsets && b.cp_sources && b.cc_offsets && b.cc_sources;
}

bool BoundaryElement::validate_device_buffers_upward_(bool use_split) const
{
    const auto& b = device_buffers_;
    bool ok = device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
              elements_.cuda_device_ready() && interp_pts_.cuda_device_ready() &&
              b.clusters_x && b.clusters_y && b.clusters_z &&
              b.clusters_q && b.clusters_q_dx && b.clusters_q_dy && b.clusters_q_dz &&
              b.weights && b.elements_x && b.elements_y && b.elements_z &&
              b.sources_q && b.sources_q_dx && b.sources_q_dy && b.sources_q_dz &&
              b.node_begin && b.node_end && b.level_nodes;
    if (!ok) {
        return false;
    }
    if (use_split) {
        return b.exact_idx_x && b.exact_idx_y && b.exact_idx_z && b.denominator;
    }
    return true;
}

bool BoundaryElement::validate_device_buffers_downward_(const double* potential) const
{
    const auto& b = device_buffers_;
    if (!(device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
          elements_.cuda_device_ready() && interp_pts_.cuda_device_ready() &&
          b.clusters_x && b.clusters_y && b.clusters_z &&
          b.clusters_p && b.clusters_p_dx && b.clusters_p_dy && b.clusters_p_dz &&
          b.elements_x && b.elements_y && b.elements_z &&
          b.targets_q && b.targets_q_dx && b.targets_q_dy && b.targets_q_dz &&
          b.weights && b.node_begin && b.node_end && b.level_nodes)) {
        return false;
    }
    const std::size_t num = potential_.size();
    if (num > 0 && !cuda_pointer_is_device_accessible(potential)) {
        return false;
    }
    return true;
}

bool BoundaryElement::validate_device_buffers_clear_cluster_charges_() const
{
    const auto& b = device_buffers_;
    return device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
           b.clusters_q && b.clusters_q_dx && b.clusters_q_dy && b.clusters_q_dz;
}

bool BoundaryElement::validate_device_buffers_clear_cluster_potentials_() const
{
    const auto& b = device_buffers_;
    return device_state_ == CudaDeviceState::DeviceMapped && b.ready &&
           b.clusters_p && b.clusters_p_dx && b.clusters_p_dy && b.clusters_p_dz;
}

void BoundaryElement::cache_device_buffers_() const
{
    // Interop-free phase: copyin_clusters_to_device() is the only owner of
    // populating device_buffers_. Keep this as a no-op compatibility hook.
    if (device_state_ != CudaDeviceState::DeviceMapped) {
        reset_device_buffers_();
    }
}

void BoundaryElement::CudaTimerQueue::begin(Timer& timer, void* stream)
{
    CudaTimerSection section;
    section.timer = &timer;
    if (cudaEventCreate(&section.start) != cudaSuccess) return;
    if (cudaEventCreate(&section.stop) != cudaSuccess) {
        cudaEventDestroy(section.start);
        return;
    }
    cudaEventRecord(section.start, reinterpret_cast<cudaStream_t>(stream));
    sections.push_back(section);
}

void BoundaryElement::CudaTimerQueue::end(void* stream)
{
    if (sections.empty()) return;
    cudaEventRecord(sections.back().stop, reinterpret_cast<cudaStream_t>(stream));
}

void BoundaryElement::CudaTimerQueue::flush()
{
    if (sections.empty()) return;
    cudaEventSynchronize(sections.back().stop);
    for (auto& section : sections) {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, section.start, section.stop) == cudaSuccess) {
            section.timer->add_seconds(static_cast<double>(ms) / 1000.0);
        }
        cudaEventDestroy(section.start);
        cudaEventDestroy(section.stop);
    }
    sections.clear();
}

void BoundaryElement::flush_cuda_timers_()
{
    cuda_timer_queue_.flush();
}
#endif
