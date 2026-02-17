#include "boundary_element.h"

#ifdef USE_CUDA_CC
#include <cstdlib>

#include "cuda_helpers.h"

void BoundaryElement::reset_device_buffers_() const
{
    device_buffers_ = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
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
