#include "source_term_backend_cuda.h"

#ifdef USE_CUDA_CC
#include "elements.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_helpers.h"
#include "source_term_cuda.h"

bool SourceTermCompute::validate_device_buffers_common_() const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!elements_.cuda_device_ready() || !molecule_.cuda_device_ready() ||
        !elem_interp_pts_.cuda_device_ready() || !mol_interp_pts_.cuda_device_ready()) {
        return false;
    }

    const auto& buf = device_buffers_;
    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = elem_interp_potential_.size();
    const std::size_t p_dx_num = elem_interp_potential_dx_.size();
    const std::size_t p_dy_num = elem_interp_potential_dy_.size();
    const std::size_t p_dz_num = elem_interp_potential_dz_.size();
    const std::size_t mol_weights_num = mol_weights_.size();
    const std::size_t elem_weights_num = elem_weights_.size();
    const std::size_t scratch_num = exact_idx_x_.size();
    const std::size_t target_nodes_num = target_node_begin_u32_.size();
    const std::size_t source_nodes_num = source_node_begin_u32_.size();
    const std::size_t pp_offsets_num = pp_offsets_u32_.size();
    const std::size_t pp_sources_num = pp_sources_u32_.size();
    const std::size_t pc_offsets_num = pc_offsets_u32_.size();
    const std::size_t pc_sources_num = pc_sources_u32_.size();
    const std::size_t cp_offsets_num = cp_offsets_u32_.size();
    const std::size_t cp_sources_num = cp_sources_u32_.size();
    const std::size_t cc_offsets_num = cc_offsets_u32_.size();
    const std::size_t cc_sources_num = cc_sources_u32_.size();
    const std::size_t expected_offsets_num = target_nodes_num + 1;

    if (pp_offsets_num != expected_offsets_num ||
        pc_offsets_num != expected_offsets_num ||
        cp_offsets_num != expected_offsets_num ||
        cc_offsets_num != expected_offsets_num) {
        return false;
    }

    if (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
        buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
        buf.mol_weights_num != mol_weights_num ||
        buf.elem_weights_num != elem_weights_num ||
        buf.scratch_num != scratch_num ||
        buf.target_nodes_num != target_nodes_num ||
        buf.source_nodes_num != source_nodes_num ||
        buf.pp_offsets_num != pp_offsets_num ||
        buf.pp_sources_num != pp_sources_num ||
        buf.pc_offsets_num != pc_offsets_num ||
        buf.pc_sources_num != pc_sources_num ||
        buf.cp_offsets_num != cp_offsets_num ||
        buf.cp_sources_num != cp_sources_num ||
        buf.cc_offsets_num != cc_offsets_num ||
        buf.cc_sources_num != cc_sources_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (p_dx_num > 0 && !buf.p_dx_dev) ||
        (p_dy_num > 0 && !buf.p_dy_dev) ||
        (p_dz_num > 0 && !buf.p_dz_dev) ||
        (mol_weights_num > 0 && !buf.mol_weights_dev) ||
        (elem_weights_num > 0 && !buf.elem_weights_dev) ||
        (scratch_num > 0 &&
         (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
          !buf.denominator_dev)) ||
        (target_nodes_num > 0 &&
         (!buf.target_node_begin_dev || !buf.target_node_end_dev)) ||
        (source_nodes_num > 0 &&
         (!buf.source_node_begin_dev || !buf.source_node_end_dev)) ||
        (pp_offsets_num > 0 && !buf.pp_offsets_dev) ||
        (pp_sources_num > 0 && !buf.pp_sources_dev) ||
        (pc_offsets_num > 0 && !buf.pc_offsets_dev) ||
        (pc_sources_num > 0 && !buf.pc_sources_dev) ||
        (cp_offsets_num > 0 && !buf.cp_offsets_dev) ||
        (cp_sources_num > 0 && !buf.cp_sources_dev) ||
        (cc_offsets_num > 0 && !buf.cc_offsets_dev) ||
        (cc_sources_num > 0 && !buf.cc_sources_dev)) {
        return false;
    }
    return true;
}

bool SourceTermCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_downward_pass_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::run_batched_interactions_cuda_()
{
    const char* use_batched_env = std::getenv("TABIPB_CUDA_SOURCE_TERM_BATCHED");
    const bool use_batched =
        !(use_batched_env && std::strcmp(use_batched_env, "0") != 0);
    if (!use_batched) {
        return false;
    }

    const char* pp_env = std::getenv("TABIPB_CUDA_SOURCE_TERM_PP");
    const char* pc_env = std::getenv("TABIPB_CUDA_SOURCE_TERM_PC");
    const char* cp_env = std::getenv("TABIPB_CUDA_SOURCE_TERM_CP");
    const char* cc_env = std::getenv("TABIPB_CUDA_SOURCE_TERM_CC");
    const bool pp_enabled = !(pp_env && std::strcmp(pp_env, "0") == 0);
    const bool pc_enabled = !(pc_env && std::strcmp(pc_env, "0") == 0);
    const bool cp_enabled = !(cp_env && std::strcmp(cp_env, "0") == 0);
    const bool cc_enabled = !(cc_env && std::strcmp(cc_env, "0") == 0);

    if (!(pp_enabled && pc_enabled && cp_enabled && cc_enabled)) {
        return false;
    }
    if (!validate_device_buffers_common_()) {
        return false;
    }

    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

    const auto elem_dev = elements_.device_view();
    const auto mol_dev = molecule_.device_view();
    const auto elem_interp_dev = elem_interp_pts_.device_view();
    const auto mol_interp_dev = mol_interp_pts_.device_view();
    const auto self_dev = device_view();

    if (!source_term_try_upward_pass_cuda(mol_dev, mol_interp_dev, self_dev,
                                          source_tree_, params, nullptr)) {
        return false;
    }

    const auto& buf = device_buffers_;
    source_term_pp_batched_cuda(
        elem_dev.x, elem_dev.y, elem_dev.z,
        elem_dev.nx, elem_dev.ny, elem_dev.nz,
        mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
        buf.target_node_begin_dev, buf.target_node_end_dev, buf.target_nodes_num,
        buf.source_node_begin_dev, buf.source_node_end_dev,
        buf.pp_offsets_dev, buf.pp_sources_dev,
        params.one_over_4pi_eps_solute,
        elem_dev.source_term, params.source_term_offset,
        nullptr);
    CUDA_CHECK_LAST_KERNEL();

    source_term_pc_batched_cuda(
        elem_dev.x, elem_dev.y, elem_dev.z,
        elem_dev.nx, elem_dev.ny, elem_dev.nz,
        mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
        self_dev.q,
        buf.target_node_begin_dev, buf.target_node_end_dev, buf.target_nodes_num,
        buf.pc_offsets_dev, buf.pc_sources_dev,
        params.num_mol_interp_pts_per_node, params.num_mol_interp_charges_per_node,
        params.one_over_4pi_eps_solute,
        elem_dev.source_term, params.source_term_offset,
        nullptr);
    CUDA_CHECK_LAST_KERNEL();

    source_term_cp_batched_cuda(
        elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
        self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
        mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
        buf.target_nodes_num,
        buf.source_node_begin_dev, buf.source_node_end_dev,
        buf.cp_offsets_dev, buf.cp_sources_dev,
        params.num_elem_interp_pts_per_node, params.num_elem_interp_potentials_per_node,
        params.one_over_4pi_eps_solute,
        nullptr);
    CUDA_CHECK_LAST_KERNEL();

    source_term_cc_batched_cuda(
        elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
        self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
        mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
        self_dev.q,
        buf.target_nodes_num,
        buf.cc_offsets_dev, buf.cc_sources_dev,
        params.num_elem_interp_pts_per_node, params.num_elem_interp_potentials_per_node,
        params.num_mol_interp_pts_per_node, params.num_mol_interp_charges_per_node,
        params.one_over_4pi_eps_solute,
        nullptr);
    CUDA_CHECK_LAST_KERNEL();

    if (!source_term_try_downward_pass_cuda(elem_dev, elem_interp_dev, self_dev,
                                            target_tree_, params, nullptr)) {
        return false;
    }
    return true;
}

bool source_term_try_particle_particle_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.source_term &&
                         mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge;
    if (!view_ok) {
        return false;
    }

    source_term_pp_cuda(
        elem_view.x, elem_view.y, elem_view.z,
        elem_view.nx, elem_view.ny, elem_view.nz,
        mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
        target_node_idxs[0], target_node_idxs[1],
        source_node_idxs[0], source_node_idxs[1],
        params.one_over_4pi_eps_solute,
        elem_view.source_term, params.source_term_offset,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool source_term_try_particle_cluster_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.source_term &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q;
    if (!view_ok) {
        return false;
    }

    source_term_pc_cuda(
        elem_view.x, elem_view.y, elem_view.z,
        elem_view.nx, elem_view.ny, elem_view.nz,
        mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
        self_view.q,
        source_node_idx,
        params.num_mol_interp_pts_per_node,
        params.num_mol_interp_charges_per_node,
        target_node_idxs[0],
        target_node_idxs[1],
        params.one_over_4pi_eps_solute,
        elem_view.source_term, params.source_term_offset,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool source_term_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& elem_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz &&
                         mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge;
    if (!view_ok) {
        return false;
    }

    source_term_cp_cuda(
        elem_interp_view.interp_x, elem_interp_view.interp_y, elem_interp_view.interp_z,
        self_view.p, self_view.p_dx,
        self_view.p_dy, self_view.p_dz,
        mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
        target_node_idx,
        params.num_elem_interp_pts_per_node,
        params.num_elem_interp_potentials_per_node,
        source_node_idxs[0],
        source_node_idxs[1],
        params.one_over_4pi_eps_solute,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool source_term_try_cluster_cluster_cuda(
    const InterpolationPoints::View& elem_interp_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz;
    if (!view_ok) {
        return false;
    }

    source_term_cc_cuda(
        elem_interp_view.interp_x, elem_interp_view.interp_y, elem_interp_view.interp_z,
        self_view.p, self_view.p_dx,
        self_view.p_dy, self_view.p_dz,
        mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
        self_view.q,
        target_node_idx,
        source_node_idx,
        params.num_elem_interp_pts_per_node,
        params.num_elem_interp_potentials_per_node,
        params.num_mol_interp_pts_per_node,
        params.num_mol_interp_charges_per_node,
        params.one_over_4pi_eps_solute,
        stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool source_term_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.mol_weights &&
                         self_view.exact_idx_x && self_view.exact_idx_y &&
                         self_view.exact_idx_z && self_view.denominator;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < source_tree.num_nodes(); ++node_idx) {
        auto particle_idxs = source_tree.node_particle_idxs(node_idx);
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        source_term_up_cuda(
            mol_view.particles_x, mol_view.particles_y, mol_view.particles_z, mol_view.charge,
            mol_interp_view.interp_x, mol_interp_view.interp_y, mol_interp_view.interp_z,
            self_view.q,
            self_view.mol_weights,
            self_view.exact_idx_x, self_view.exact_idx_y, self_view.exact_idx_z,
            self_view.denominator,
            node_idx,
            params.num_mol_interp_pts_per_node,
            params.num_mol_interp_charges_per_node,
            particle_start,
            num_particles,
            stream);
        CUDA_CHECK_LAST_KERNEL();
    }
    CUDA_SYNC_AND_CHECK();
    return true;
}

bool source_term_try_downward_pass_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& elem_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const Tree& target_tree,
    const SourceTermBackendParams& params,
    void* stream) {
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.source_term &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz &&
                         self_view.elem_weights;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < target_tree.num_nodes(); ++node_idx) {
        auto particle_idxs = target_tree.node_particle_idxs(node_idx);
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        source_term_down_cuda(
            elem_view.x, elem_view.y, elem_view.z,
            elem_view.nx, elem_view.ny, elem_view.nz,
            elem_interp_view.interp_x, elem_interp_view.interp_y, elem_interp_view.interp_z,
            self_view.p, self_view.p_dx,
            self_view.p_dy, self_view.p_dz,
            self_view.elem_weights,
            node_idx,
            params.num_elem_interp_pts_per_node,
            params.num_elem_interp_potentials_per_node,
            particle_start,
            num_particles,
            elem_view.source_term,
            params.source_term_offset,
            stream);
        CUDA_CHECK_LAST_KERNEL();
    }
    CUDA_SYNC_AND_CHECK();
    return true;
}

void SourceTermCompute::copyin_clusters_to_device_cuda_() const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);

    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();

    const double* p_ptr    = elem_interp_potential_.data();
    const double* p_dx_ptr = elem_interp_potential_dx_.data();
    const double* p_dy_ptr = elem_interp_potential_dy_.data();
    const double* p_dz_ptr = elem_interp_potential_dz_.data();

    std::size_t p_num    = elem_interp_potential_.size();
    std::size_t p_dx_num = elem_interp_potential_dx_.size();
    std::size_t p_dy_num = elem_interp_potential_dy_.size();
    std::size_t p_dz_num = elem_interp_potential_dz_.size();

    const double* mol_weights_ptr = mol_weights_.data();
    std::size_t mol_weights_num = mol_weights_.size();
    const double* elem_weights_ptr = elem_weights_.data();
    std::size_t elem_weights_num = elem_weights_.size();
    const std::uint32_t* target_node_begin_ptr = target_node_begin_u32_.data();
    const std::uint32_t* target_node_end_ptr = target_node_end_u32_.data();
    const std::uint32_t* source_node_begin_ptr = source_node_begin_u32_.data();
    const std::uint32_t* source_node_end_ptr = source_node_end_u32_.data();
    const std::uint32_t* pp_offsets_ptr = pp_offsets_u32_.data();
    const std::uint32_t* pp_sources_ptr = pp_sources_u32_.data();
    const std::uint32_t* pc_offsets_ptr = pc_offsets_u32_.data();
    const std::uint32_t* pc_sources_ptr = pc_sources_u32_.data();
    const std::uint32_t* cp_offsets_ptr = cp_offsets_u32_.data();
    const std::uint32_t* cp_sources_ptr = cp_sources_u32_.data();
    const std::uint32_t* cc_offsets_ptr = cc_offsets_u32_.data();
    const std::uint32_t* cc_sources_ptr = cc_sources_u32_.data();

    const std::size_t scratch_num = max_mol_particles_per_node_;
    const std::size_t target_nodes_num = target_node_begin_u32_.size();
    const std::size_t source_nodes_num = source_node_begin_u32_.size();
    const std::size_t pp_offsets_num = pp_offsets_u32_.size();
    const std::size_t pp_sources_num = pp_sources_u32_.size();
    const std::size_t pc_offsets_num = pc_offsets_u32_.size();
    const std::size_t pc_sources_num = pc_sources_u32_.size();
    const std::size_t cp_offsets_num = cp_offsets_u32_.size();
    const std::size_t cp_sources_num = cp_sources_u32_.size();
    const std::size_t cc_offsets_num = cc_offsets_u32_.size();
    const std::size_t cc_sources_num = cc_sources_u32_.size();

    auto &buf = device_buffers_;
    if (buf.q_num != 0 && (buf.q_num != q_num || buf.p_num != p_num ||
                           buf.p_dx_num != p_dx_num || buf.p_dy_num != p_dy_num ||
                           buf.p_dz_num != p_dz_num || buf.mol_weights_num != mol_weights_num ||
                           buf.elem_weights_num != elem_weights_num ||
                           buf.scratch_num != scratch_num ||
                           buf.target_nodes_num != target_nodes_num ||
                           buf.source_nodes_num != source_nodes_num ||
                           buf.pp_offsets_num != pp_offsets_num ||
                           buf.pp_sources_num != pp_sources_num ||
                           buf.pc_offsets_num != pc_offsets_num ||
                           buf.pc_sources_num != pc_sources_num ||
                           buf.cp_offsets_num != cp_offsets_num ||
                           buf.cp_sources_num != cp_sources_num ||
                           buf.cc_offsets_num != cc_offsets_num ||
                           buf.cc_sources_num != cc_sources_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.mol_weights_dev);
        CUDA_FREE_AND_NULL(buf.elem_weights_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        CUDA_FREE_AND_NULL(buf.target_node_begin_dev);
        CUDA_FREE_AND_NULL(buf.target_node_end_dev);
        CUDA_FREE_AND_NULL(buf.source_node_begin_dev);
        CUDA_FREE_AND_NULL(buf.source_node_end_dev);
        CUDA_FREE_AND_NULL(buf.pp_offsets_dev);
        CUDA_FREE_AND_NULL(buf.pp_sources_dev);
        CUDA_FREE_AND_NULL(buf.pc_offsets_dev);
        CUDA_FREE_AND_NULL(buf.pc_sources_dev);
        CUDA_FREE_AND_NULL(buf.cp_offsets_dev);
        CUDA_FREE_AND_NULL(buf.cp_sources_dev);
        CUDA_FREE_AND_NULL(buf.cc_offsets_dev);
        CUDA_FREE_AND_NULL(buf.cc_sources_dev);
        buf = DeviceBuffers{};
    }

    if (buf.q_num == 0 && (q_num > 0 || p_num > 0 || mol_weights_num > 0 ||
                           elem_weights_num > 0 || scratch_num > 0 ||
                           target_nodes_num > 0 || source_nodes_num > 0 ||
                           pp_offsets_num > 0 || pp_sources_num > 0 ||
                           pc_offsets_num > 0 || pc_sources_num > 0 ||
                           cp_offsets_num > 0 || cp_sources_num > 0 ||
                           cc_offsets_num > 0 || cc_sources_num > 0)) {
        if (q_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.q_dev, q_num * sizeof(double));
            buf.q_num = q_num;
        }
        if (p_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dev, p_num * sizeof(double));
            buf.p_num = p_num;
        }
        if (p_dx_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dx_dev, p_dx_num * sizeof(double));
            buf.p_dx_num = p_dx_num;
        }
        if (p_dy_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dy_dev, p_dy_num * sizeof(double));
            buf.p_dy_num = p_dy_num;
        }
        if (p_dz_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dz_dev, p_dz_num * sizeof(double));
            buf.p_dz_num = p_dz_num;
        }
        if (mol_weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.mol_weights_dev, mol_weights_num * sizeof(double));
            buf.mol_weights_num = mol_weights_num;
        }
        if (elem_weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.elem_weights_dev, elem_weights_num * sizeof(double));
            buf.elem_weights_num = elem_weights_num;
        }
        if (scratch_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_x_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_y_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_z_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.denominator_dev, scratch_num * sizeof(double));
            buf.scratch_num = scratch_num;
        }
        if (target_nodes_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.target_node_begin_dev,
                               target_nodes_num * sizeof(std::uint32_t));
            CUDA_MALLOC_OR_DIE(&buf.target_node_end_dev,
                               target_nodes_num * sizeof(std::uint32_t));
            buf.target_nodes_num = target_nodes_num;
        }
        if (source_nodes_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.source_node_begin_dev,
                               source_nodes_num * sizeof(std::uint32_t));
            CUDA_MALLOC_OR_DIE(&buf.source_node_end_dev,
                               source_nodes_num * sizeof(std::uint32_t));
            buf.source_nodes_num = source_nodes_num;
        }
        if (pp_offsets_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.pp_offsets_dev,
                               pp_offsets_num * sizeof(std::uint32_t));
            buf.pp_offsets_num = pp_offsets_num;
        }
        if (pp_sources_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.pp_sources_dev,
                               pp_sources_num * sizeof(std::uint32_t));
            buf.pp_sources_num = pp_sources_num;
        }
        if (pc_offsets_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.pc_offsets_dev,
                               pc_offsets_num * sizeof(std::uint32_t));
            buf.pc_offsets_num = pc_offsets_num;
        }
        if (pc_sources_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.pc_sources_dev,
                               pc_sources_num * sizeof(std::uint32_t));
            buf.pc_sources_num = pc_sources_num;
        }
        if (cp_offsets_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.cp_offsets_dev,
                               cp_offsets_num * sizeof(std::uint32_t));
            buf.cp_offsets_num = cp_offsets_num;
        }
        if (cp_sources_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.cp_sources_dev,
                               cp_sources_num * sizeof(std::uint32_t));
            buf.cp_sources_num = cp_sources_num;
        }
        if (cc_offsets_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.cc_offsets_dev,
                               cc_offsets_num * sizeof(std::uint32_t));
            buf.cc_offsets_num = cc_offsets_num;
        }
        if (cc_sources_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.cc_sources_dev,
                               cc_sources_num * sizeof(std::uint32_t));
            buf.cc_sources_num = cc_sources_num;
        }
    }

    cudaStream_t stream = nullptr;

    if (q_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dx_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dx_dev, p_dx_ptr, p_dx_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dy_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dy_dev, p_dy_ptr, p_dy_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dz_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dz_dev, p_dz_ptr, p_dz_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (mol_weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.mol_weights_dev, mol_weights_ptr,
                          mol_weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (elem_weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.elem_weights_dev, elem_weights_ptr,
                          elem_weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (target_nodes_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.target_node_begin_dev, target_node_begin_ptr,
                          target_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(buf.target_node_end_dev, target_node_end_ptr,
                          target_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (source_nodes_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.source_node_begin_dev, source_node_begin_ptr,
                          source_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(buf.source_node_end_dev, source_node_end_ptr,
                          source_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pp_offsets_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.pp_offsets_dev, pp_offsets_ptr,
                          pp_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pp_sources_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.pp_sources_dev, pp_sources_ptr,
                          pp_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pc_offsets_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.pc_offsets_dev, pc_offsets_ptr,
                          pc_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pc_sources_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.pc_sources_dev, pc_sources_ptr,
                          pc_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cp_offsets_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.cp_offsets_dev, cp_offsets_ptr,
                          cp_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cp_sources_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.cp_sources_dev, cp_sources_ptr,
                          cp_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cc_offsets_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.cc_offsets_dev, cc_offsets_ptr,
                          cc_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cc_sources_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.cc_sources_dev, cc_sources_ptr,
                          cc_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    buf.ready = true;
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        if ((q_num > 0 && !buf.q_dev) ||
            (p_num > 0 && !buf.p_dev) ||
            (p_dx_num > 0 && !buf.p_dx_dev) ||
            (p_dy_num > 0 && !buf.p_dy_dev) ||
            (p_dz_num > 0 && !buf.p_dz_dev) ||
            (mol_weights_num > 0 && !buf.mol_weights_dev) ||
            (elem_weights_num > 0 && !buf.elem_weights_dev) ||
            (scratch_num > 0 && (!buf.exact_idx_x_dev ||
                                 !buf.exact_idx_y_dev ||
                                 !buf.exact_idx_z_dev ||
                                 !buf.denominator_dev)) ||
            (target_nodes_num > 0 &&
             (!buf.target_node_begin_dev || !buf.target_node_end_dev)) ||
            (source_nodes_num > 0 &&
             (!buf.source_node_begin_dev || !buf.source_node_end_dev)) ||
            (pp_offsets_num > 0 && !buf.pp_offsets_dev) ||
            (pp_sources_num > 0 && !buf.pp_sources_dev) ||
            (pc_offsets_num > 0 && !buf.pc_offsets_dev) ||
            (pc_sources_num > 0 && !buf.pc_sources_dev) ||
            (cp_offsets_num > 0 && !buf.cp_offsets_dev) ||
            (cp_sources_num > 0 && !buf.cp_sources_dev) ||
            (cc_offsets_num > 0 && !buf.cc_offsets_dev) ||
            (cc_sources_num > 0 && !buf.cc_sources_dev)) {
            std::fprintf(stderr,
                         "[CUDA] SourceTermCompute missing device buffers "
                         "under TABIPB_CUDA_REQUIRE_ALL=1\n");
            std::abort();
        }
    }
}

void SourceTermCompute::delete_clusters_from_device_cuda_() const
{
    auto &buf = device_buffers_;
    CUDA_FREE_AND_NULL(buf.q_dev);
    CUDA_FREE_AND_NULL(buf.p_dev);
    CUDA_FREE_AND_NULL(buf.p_dx_dev);
    CUDA_FREE_AND_NULL(buf.p_dy_dev);
    CUDA_FREE_AND_NULL(buf.p_dz_dev);
    CUDA_FREE_AND_NULL(buf.mol_weights_dev);
    CUDA_FREE_AND_NULL(buf.elem_weights_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
    CUDA_FREE_AND_NULL(buf.denominator_dev);
    CUDA_FREE_AND_NULL(buf.target_node_begin_dev);
    CUDA_FREE_AND_NULL(buf.target_node_end_dev);
    CUDA_FREE_AND_NULL(buf.source_node_begin_dev);
    CUDA_FREE_AND_NULL(buf.source_node_end_dev);
    CUDA_FREE_AND_NULL(buf.pp_offsets_dev);
    CUDA_FREE_AND_NULL(buf.pp_sources_dev);
    CUDA_FREE_AND_NULL(buf.pc_offsets_dev);
    CUDA_FREE_AND_NULL(buf.pc_sources_dev);
    CUDA_FREE_AND_NULL(buf.cp_offsets_dev);
    CUDA_FREE_AND_NULL(buf.cp_sources_dev);
    CUDA_FREE_AND_NULL(buf.cc_offsets_dev);
    CUDA_FREE_AND_NULL(buf.cc_sources_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
