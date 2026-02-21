#include "solvation_backend_cuda.h"

#ifdef USE_CUDA_CC
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_helpers.h"
#include "solvation_energy_cuda.h"

bool SolvationEnergyCompute::validate_device_buffers_common_() const
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
    const std::size_t solv_eng_num = solv_eng_vec_.size();
    const std::size_t weights_up_num = mol_weights_.size();
    const std::size_t weights_down_num = elem_weights_.size();
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
        buf.solv_eng_num != solv_eng_num ||
        buf.weights_up_num != weights_up_num ||
        buf.weights_down_num != weights_down_num ||
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
        (solv_eng_num > 0 && !buf.solv_eng_dev) ||
        (weights_up_num > 0 && !buf.weights_up_dev) ||
        (weights_down_num > 0 && !buf.weights_down_dev) ||
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

bool SolvationEnergyCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}

bool SolvationEnergyCompute::validate_device_buffers_downward_pass_() const
{
    return validate_device_buffers_common_() && potential_device_ptr_ != nullptr;
}

bool SolvationEnergyCompute::run_batched_interactions_cuda_()
{
    const char* use_batched_env = std::getenv("TABIPB_CUDA_SOLVATION_BATCHED");
    const bool use_batched =
        !(use_batched_env && std::strcmp(use_batched_env, "0") == 0);
    if (!use_batched) {
        return false;
    }

    if (potential_device_ptr_ == nullptr) {
        return false;
    }

    const char* pp_env = std::getenv("TABIPB_CUDA_SOLVATION_PP");
    const char* pc_env = std::getenv("TABIPB_CUDA_SOLVATION_PC");
    const char* cp_env = std::getenv("TABIPB_CUDA_SOLVATION_CP");
    const char* cc_env = std::getenv("TABIPB_CUDA_SOLVATION_CC");
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

    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

    const auto elem_dev = elements_.device_view();
    const auto mol_dev = molecule_.device_view();
    const auto elem_interp_dev = elem_interp_pts_.device_view();
    const auto mol_interp_dev = mol_interp_pts_.device_view();
    const auto self_dev = device_view();

    if (!solvation_try_upward_pass_cuda(mol_dev, mol_interp_dev, self_dev,
                                        source_tree_, params, nullptr)) {
        return false;
    }

    const char* batch_pp_env = std::getenv("TABIPB_CUDA_SOLVATION_BATCHED_PP");
    const char* batch_pc_env = std::getenv("TABIPB_CUDA_SOLVATION_BATCHED_PC");
    const char* batch_cp_env = std::getenv("TABIPB_CUDA_SOLVATION_BATCHED_CP");
    const char* batch_cc_env = std::getenv("TABIPB_CUDA_SOLVATION_BATCHED_CC");
    const bool batch_pp = !(batch_pp_env && std::strcmp(batch_pp_env, "0") == 0);
    const bool batch_pc = !(batch_pc_env && std::strcmp(batch_pc_env, "0") == 0);
    const bool batch_cp = !(batch_cp_env && std::strcmp(batch_cp_env, "0") == 0);
    const bool batch_cc = !(batch_cc_env && std::strcmp(batch_cc_env, "0") == 0);

    auto run_pp_legacy = [&]() -> bool {
        const std::size_t num_target_nodes = target_tree_.num_nodes();
        for (std::size_t target_node_idx = 0; target_node_idx < num_target_nodes; ++target_node_idx) {
            const auto target_node_idxs = target_tree_.node_particle_idxs(target_node_idx);
            for (auto source_node_idx : interaction_list_.particle_particle(target_node_idx)) {
                const auto source_node_idxs = source_tree_.node_particle_idxs(source_node_idx);
                if (!solvation_try_particle_particle_cuda(
                        elem_dev, mol_dev, self_dev, target_node_idxs, source_node_idxs,
                        potential_device_ptr_, params, nullptr)) {
                    return false;
                }
            }
        }
        return true;
    };

    auto run_pc_legacy = [&]() -> bool {
        const std::size_t num_target_nodes = target_tree_.num_nodes();
        for (std::size_t target_node_idx = 0; target_node_idx < num_target_nodes; ++target_node_idx) {
            const auto target_node_idxs = target_tree_.node_particle_idxs(target_node_idx);
            for (auto source_node_idx : interaction_list_.particle_cluster(target_node_idx)) {
                if (!solvation_try_particle_cluster_cuda(
                        elem_dev, mol_interp_dev, self_dev, target_node_idxs, source_node_idx,
                        potential_device_ptr_, params, nullptr)) {
                    return false;
                }
            }
        }
        return true;
    };

    auto run_cp_legacy = [&]() -> bool {
        const std::size_t num_target_nodes = target_tree_.num_nodes();
        for (std::size_t target_node_idx = 0; target_node_idx < num_target_nodes; ++target_node_idx) {
            for (auto source_node_idx : interaction_list_.cluster_particle(target_node_idx)) {
                const auto source_node_idxs = source_tree_.node_particle_idxs(source_node_idx);
                if (!solvation_try_cluster_particle_cuda(
                        mol_dev, elem_interp_dev, self_dev, target_node_idx, source_node_idxs,
                        params, nullptr)) {
                    return false;
                }
            }
        }
        return true;
    };

    auto run_cc_legacy = [&]() -> bool {
        const std::size_t num_target_nodes = target_tree_.num_nodes();
        for (std::size_t target_node_idx = 0; target_node_idx < num_target_nodes; ++target_node_idx) {
            for (auto source_node_idx : interaction_list_.cluster_cluster(target_node_idx)) {
                if (!solvation_try_cluster_cluster_cuda(
                        mol_interp_dev, elem_interp_dev, self_dev, target_node_idx, source_node_idx,
                        params, nullptr)) {
                    return false;
                }
            }
        }
        return true;
    };

    const auto& buf = device_buffers_;
    if (batch_pp) {
        solvation_pp_batched_cuda(
            elem_dev.x, elem_dev.y, elem_dev.z,
            elem_dev.nx, elem_dev.ny, elem_dev.nz,
            elem_dev.area,
            mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
            potential_device_ptr_, params.potential_offset,
            buf.target_node_begin_dev, buf.target_node_end_dev, buf.target_nodes_num,
            buf.source_node_begin_dev, buf.source_node_end_dev,
            buf.pp_offsets_dev, buf.pp_sources_dev,
            params.eps, params.kappa,
            self_dev.solv_eng, nullptr);
        CUDA_CHECK_LAST_KERNEL();
    } else if (!run_pp_legacy()) {
        return false;
    }

    if (batch_pc) {
        solvation_pc_batched_cuda(
            elem_dev.x, elem_dev.y, elem_dev.z,
            elem_dev.nx, elem_dev.ny, elem_dev.nz,
            elem_dev.area,
            mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
            self_dev.q,
            potential_device_ptr_, params.potential_offset,
            buf.target_node_begin_dev, buf.target_node_end_dev, buf.target_nodes_num,
            buf.pc_offsets_dev, buf.pc_sources_dev,
            params.num_mol_interp_pts_per_node, params.num_mol_interp_charges_per_node,
            params.eps, params.kappa,
            self_dev.solv_eng, nullptr);
        CUDA_CHECK_LAST_KERNEL();
    } else if (!run_pc_legacy()) {
        return false;
    }

    if (batch_cp) {
        solvation_cp_batched_cuda(
            mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
            elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
            self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
            buf.target_nodes_num,
            buf.source_node_begin_dev, buf.source_node_end_dev,
            buf.cp_offsets_dev, buf.cp_sources_dev,
            params.num_elem_interp_pts_per_node, params.num_elem_interp_potentials_per_node,
            params.eps, params.kappa, nullptr);
        CUDA_CHECK_LAST_KERNEL();
    } else if (!run_cp_legacy()) {
        return false;
    }

    if (batch_cc) {
        solvation_cc_batched_cuda(
            mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
            self_dev.q,
            elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
            self_dev.p, self_dev.p_dx, self_dev.p_dy, self_dev.p_dz,
            buf.target_nodes_num,
            buf.cc_offsets_dev, buf.cc_sources_dev,
            params.num_elem_interp_pts_per_node, params.num_elem_interp_potentials_per_node,
            params.num_mol_interp_pts_per_node, params.num_mol_interp_charges_per_node,
            params.eps, params.kappa, nullptr);
        CUDA_CHECK_LAST_KERNEL();
    } else if (!run_cc_legacy()) {
        return false;
    }

    if (!solvation_try_downward_pass_cuda(elem_dev, elem_interp_dev, self_dev,
                                          target_tree_, potential_device_ptr_,
                                          params, nullptr)) {
        return false;
    }
    return true;
}

bool solvation_try_particle_particle_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         self_view.solv_eng && potential;
    if (!view_ok) {
        return false;
    }

    solvation_pp_cuda(elem_view.x, elem_view.y, elem_view.z,
                      elem_view.nx, elem_view.ny, elem_view.nz,
                      elem_view.area,
                      mol_view.particles_x, mol_view.particles_y,
                      mol_view.particles_z, mol_view.charge,
                      potential, params.potential_offset,
                      target_node_idxs[0], target_node_idxs[1],
                      source_node_idxs[0], source_node_idxs[1],
                      params.eps, params.kappa,
                      self_view.solv_eng, stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_particle_cluster_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& mol_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.solv_eng && potential;
    if (!view_ok) {
        return false;
    }

    solvation_pc_cuda(elem_view.x, elem_view.y, elem_view.z,
                      elem_view.nx, elem_view.ny, elem_view.nz,
                      elem_view.area,
                      mol_interp_view.interp_x, mol_interp_view.interp_y,
                      mol_interp_view.interp_z,
                      self_view.q,
                      potential, params.potential_offset,
                      source_node_idx,
                      params.num_mol_interp_pts_per_node,
                      params.num_mol_interp_charges_per_node,
                      target_node_idxs[0], target_node_idxs[1],
                      params.eps, params.kappa,
                      self_view.solv_eng, stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz;
    if (!view_ok) {
        return false;
    }

    solvation_cp_cuda(mol_view.particles_x, mol_view.particles_y,
                      mol_view.particles_z, mol_view.charge,
                      elem_interp_view.interp_x, elem_interp_view.interp_y,
                      elem_interp_view.interp_z,
                      self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
                      target_node_idx,
                      params.num_elem_interp_pts_per_node,
                      params.num_elem_interp_potentials_per_node,
                      source_node_idxs[0], source_node_idxs[1],
                      params.eps, params.kappa,
                      stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_cluster_cluster_cuda(
    const InterpolationPoints::View& mol_interp_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.q && self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz;
    if (!view_ok) {
        return false;
    }

    solvation_cc_cuda(mol_interp_view.interp_x, mol_interp_view.interp_y,
                      mol_interp_view.interp_z,
                      self_view.q,
                      elem_interp_view.interp_x, elem_interp_view.interp_y,
                      elem_interp_view.interp_z,
                      self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
                      target_node_idx, source_node_idx,
                      params.num_elem_interp_pts_per_node,
                      params.num_elem_interp_potentials_per_node,
                      params.num_mol_interp_pts_per_node,
                      params.num_mol_interp_charges_per_node,
                      params.eps, params.kappa,
                      stream);
    CUDA_CHECK_LAST_KERNEL();
    return true;
}

bool solvation_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = mol_view.particles_x && mol_view.particles_y &&
                         mol_view.particles_z && mol_view.charge &&
                         mol_interp_view.interp_x && mol_interp_view.interp_y &&
                         mol_interp_view.interp_z &&
                         self_view.q && self_view.weights_up &&
                         self_view.exact_idx_x && self_view.exact_idx_y &&
                         self_view.exact_idx_z && self_view.denominator;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < source_tree.num_nodes(); ++node_idx) {
        const auto particle_idxs = source_tree.node_particle_idxs(node_idx);
        const std::size_t particle_start = particle_idxs[0];
        const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        if (num_particles > self_view.scratch_num) {
            return false;
        }
        solvation_up_cuda(
            mol_view.particles_x, mol_view.particles_y, mol_view.particles_z,
            mol_view.charge,
            mol_interp_view.interp_x, mol_interp_view.interp_y,
            mol_interp_view.interp_z,
            self_view.q,
            self_view.weights_up,
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

bool solvation_try_downward_pass_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& elem_interp_view,
    const SolvationEnergyCompute::DeviceView& self_view,
    const Tree& target_tree,
    const double* potential,
    const SolvationBackendParams& params,
    void* stream)
{
    const bool view_ok = elem_view.x && elem_view.y && elem_view.z &&
                         elem_view.nx && elem_view.ny && elem_view.nz &&
                         elem_view.area &&
                         elem_interp_view.interp_x && elem_interp_view.interp_y &&
                         elem_interp_view.interp_z &&
                         self_view.p && self_view.p_dx &&
                         self_view.p_dy && self_view.p_dz &&
                         self_view.solv_eng && self_view.weights_down &&
                         potential;
    if (!view_ok) {
        return false;
    }

    for (std::size_t node_idx = 0; node_idx < target_tree.num_nodes(); ++node_idx) {
        const auto particle_idxs = target_tree.node_particle_idxs(node_idx);
        const std::size_t particle_start = particle_idxs[0];
        const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles == 0) {
            continue;
        }
        solvation_down_cuda(
            elem_view.x, elem_view.y, elem_view.z,
            elem_view.nx, elem_view.ny, elem_view.nz,
            elem_view.area,
            elem_interp_view.interp_x, elem_interp_view.interp_y,
            elem_interp_view.interp_z,
            self_view.p, self_view.p_dx, self_view.p_dy, self_view.p_dz,
            potential, params.potential_offset,
            self_view.weights_down,
            node_idx,
            params.num_elem_interp_pts_per_node,
            params.num_elem_interp_potentials_per_node,
            particle_start,
            num_particles,
            self_view.solv_eng,
            stream);
        CUDA_CHECK_LAST_KERNEL();
    }
    CUDA_SYNC_AND_CHECK();
    return true;
}

void SolvationEnergyCompute::copyin_clusters_to_device_cuda_() const
{
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        !(require_all_env && std::strcmp(require_all_env, "0") == 0);

    const double* q_ptr = mol_interp_charge_.data();
    const std::size_t q_num = mol_interp_charge_.size();

    const double* p_ptr = elem_interp_potential_.data();
    const double* p_dx_ptr = elem_interp_potential_dx_.data();
    const double* p_dy_ptr = elem_interp_potential_dy_.data();
    const double* p_dz_ptr = elem_interp_potential_dz_.data();

    const std::size_t p_num = elem_interp_potential_.size();
    const std::size_t p_dx_num = elem_interp_potential_dx_.size();
    const std::size_t p_dy_num = elem_interp_potential_dy_.size();
    const std::size_t p_dz_num = elem_interp_potential_dz_.size();

    const double* solv_eng_ptr = solv_eng_vec_.data();
    const std::size_t solv_eng_num = solv_eng_vec_.size();

    const double* weights_up_ptr = mol_weights_.data();
    const std::size_t weights_up_num = mol_weights_.size();
    const double* weights_down_ptr = elem_weights_.data();
    const std::size_t weights_down_num = elem_weights_.size();
    const std::size_t scratch_num = max_mol_particles_per_node_;
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

    auto& buf = device_buffers_;
    if (buf.ready &&
        (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
         buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
         buf.solv_eng_num != solv_eng_num ||
         buf.weights_up_num != weights_up_num ||
         buf.weights_down_num != weights_down_num ||
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
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
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

    if (!buf.ready &&
        (q_num > 0 || p_num > 0 || solv_eng_num > 0 ||
         weights_up_num > 0 || weights_down_num > 0 || scratch_num > 0 ||
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
        if (solv_eng_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.solv_eng_dev, solv_eng_num * sizeof(double));
            buf.solv_eng_num = solv_eng_num;
        }
        if (weights_up_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_up_dev, weights_up_num * sizeof(double));
            buf.weights_up_num = weights_up_num;
        }
        if (weights_down_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_down_dev, weights_down_num * sizeof(double));
            buf.weights_down_num = weights_down_num;
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
        buf.ready = true;
    }

    cudaStream_t stream = nullptr;
    if (q_num > 0 && buf.q_dev) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0 && buf.p_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dx_num > 0 && buf.p_dx_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dx_dev, p_dx_ptr, p_dx_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dy_num > 0 && buf.p_dy_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dy_dev, p_dy_ptr, p_dy_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dz_num > 0 && buf.p_dz_dev) {
        CUDA_MEMCPY_ASYNC(buf.p_dz_dev, p_dz_ptr, p_dz_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (solv_eng_num > 0 && buf.solv_eng_dev) {
        CUDA_MEMCPY_ASYNC(buf.solv_eng_dev, solv_eng_ptr,
                          solv_eng_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_up_num > 0 && buf.weights_up_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_up_dev, weights_up_ptr,
                          weights_up_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_down_num > 0 && buf.weights_down_dev) {
        CUDA_MEMCPY_ASYNC(buf.weights_down_dev, weights_down_ptr,
                          weights_down_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (target_nodes_num > 0 && buf.target_node_begin_dev && buf.target_node_end_dev) {
        CUDA_MEMCPY_ASYNC(buf.target_node_begin_dev, target_node_begin_ptr,
                          target_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(buf.target_node_end_dev, target_node_end_ptr,
                          target_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (source_nodes_num > 0 && buf.source_node_begin_dev && buf.source_node_end_dev) {
        CUDA_MEMCPY_ASYNC(buf.source_node_begin_dev, source_node_begin_ptr,
                          source_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
        CUDA_MEMCPY_ASYNC(buf.source_node_end_dev, source_node_end_ptr,
                          source_nodes_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pp_offsets_num > 0 && buf.pp_offsets_dev) {
        CUDA_MEMCPY_ASYNC(buf.pp_offsets_dev, pp_offsets_ptr,
                          pp_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pp_sources_num > 0 && buf.pp_sources_dev) {
        CUDA_MEMCPY_ASYNC(buf.pp_sources_dev, pp_sources_ptr,
                          pp_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pc_offsets_num > 0 && buf.pc_offsets_dev) {
        CUDA_MEMCPY_ASYNC(buf.pc_offsets_dev, pc_offsets_ptr,
                          pc_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (pc_sources_num > 0 && buf.pc_sources_dev) {
        CUDA_MEMCPY_ASYNC(buf.pc_sources_dev, pc_sources_ptr,
                          pc_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cp_offsets_num > 0 && buf.cp_offsets_dev) {
        CUDA_MEMCPY_ASYNC(buf.cp_offsets_dev, cp_offsets_ptr,
                          cp_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cp_sources_num > 0 && buf.cp_sources_dev) {
        CUDA_MEMCPY_ASYNC(buf.cp_sources_dev, cp_sources_ptr,
                          cp_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cc_offsets_num > 0 && buf.cc_offsets_dev) {
        CUDA_MEMCPY_ASYNC(buf.cc_offsets_dev, cc_offsets_ptr,
                          cc_offsets_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }
    if (cc_sources_num > 0 && buf.cc_sources_dev) {
        CUDA_MEMCPY_ASYNC(buf.cc_sources_dev, cc_sources_ptr,
                          cc_sources_num * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        if (!buf.ready ||
            (q_num > 0 && !buf.q_dev) ||
            (p_num > 0 && !buf.p_dev) ||
            (p_dx_num > 0 && !buf.p_dx_dev) ||
            (p_dy_num > 0 && !buf.p_dy_dev) ||
            (p_dz_num > 0 && !buf.p_dz_dev) ||
            (solv_eng_num > 0 && !buf.solv_eng_dev) ||
            (weights_up_num > 0 && !buf.weights_up_dev) ||
            (weights_down_num > 0 && !buf.weights_down_dev) ||
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
            std::cerr << "[CUDA_SOLVATION] require_all set but device buffers not ready.\n";
            std::exit(1);
        }
    }
}

void SolvationEnergyCompute::delete_clusters_from_device_cuda_() const
{
    auto& buf = device_buffers_;
    const std::size_t solv_eng_num = solv_eng_vec_.size();

    if (buf.ready) {
        cudaStream_t stream = nullptr;
        if (solv_eng_num > 0 && buf.solv_eng_dev) {
            CUDA_MEMCPY_ASYNC(solv_eng_vec_.data(), buf.solv_eng_dev,
                              solv_eng_num * sizeof(double),
                              cudaMemcpyDeviceToHost, stream);
            CUDA_STREAM_SYNC_AND_CHECK(stream);
        }
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.solv_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_up_dev);
        CUDA_FREE_AND_NULL(buf.weights_down_dev);
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
    device_state_ = CudaDeviceState::HostOnly;
}
#endif
