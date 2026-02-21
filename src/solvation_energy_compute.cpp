#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "solvation_backend_common.h"
#include "solvation_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "solvation_backend_cuda.h"
#endif
#include "solvation_energy_compute.h"

namespace {

#ifdef USE_CUDA_CC
bool cuda_require_all_enabled() {
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  return !(require_all_env && std::strcmp(require_all_env, "0") == 0);
}

void abort_require_all(const char* step) {
  std::cerr << "[CUDA_SOLVATION] require_all set but CUDA path unavailable in "
            << step << ". Aborting to avoid CPU fallback.\n";
  std::exit(1);
}
#endif

}  // namespace

SolvationEnergyCompute::SolvationEnergyCompute(std::vector<double>& potential,
                      class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps, double phys_kappa,
                      const double* potential_device_ptr)
    : TreeCompute(mol_tree, elem_tree, interaction_list),
      elements_(elements), elem_interp_pts_(elem_interp_pts),
      molecule_(molecule), mol_interp_pts_ (mol_interp_pts),
      eps_(phys_eps), kappa_(phys_kappa),
      potential_offset_(elements.num()), potential_(potential),
      potential_device_ptr_(potential_device_ptr)
{
//    timers_.ctor.start();

    /* Target clusters */

    num_elem_interp_pts_per_node_        = elem_interp_pts_.num_interp_pts_per_node();
    num_elem_interp_potentials_per_node_ = std::pow(num_elem_interp_pts_per_node_, 3);
    num_elem_potentials_                 = target_tree_.num_nodes() * num_elem_interp_potentials_per_node_;

    elem_interp_potential_   .assign(num_elem_potentials_, 0.);
    elem_interp_potential_dx_.assign(num_elem_potentials_, 0.);
    elem_interp_potential_dy_.assign(num_elem_potentials_, 0.);
    elem_interp_potential_dz_.assign(num_elem_potentials_, 0.);


    /* Source clusters */

    num_mol_interp_pts_per_node_     = mol_interp_pts_.num_interp_pts_per_node();
    num_mol_interp_charges_per_node_ = std::pow(num_mol_interp_pts_per_node_, 3);
    num_mol_charges_                 = source_tree_.num_nodes() * num_mol_interp_charges_per_node_;

    mol_interp_charge_.assign(num_mol_charges_, 0.);

    max_mol_particles_per_node_ = 0;
    for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
        auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
        std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles > max_mol_particles_per_node_) {
            max_mol_particles_per_node_ = num_particles;
        }
    }

    mol_weights_.resize(num_mol_interp_pts_per_node_);
    for (int i = 0; i < num_mol_interp_pts_per_node_; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        if (i == 0 || i == num_mol_interp_pts_per_node_ - 1) {
            w *= 0.5;
        }
        mol_weights_[i] = w;
    }

    elem_weights_.resize(num_elem_interp_pts_per_node_);
    for (int i = 0; i < num_elem_interp_pts_per_node_; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        if (i == 0 || i == num_elem_interp_pts_per_node_ - 1) {
            w *= 0.5;
        }
        elem_weights_[i] = w;
    }

    exact_idx_x_.assign(max_mol_particles_per_node_, -1);
    exact_idx_y_.assign(max_mol_particles_per_node_, -1);
    exact_idx_z_.assign(max_mol_particles_per_node_, -1);
    denominator_.assign(max_mol_particles_per_node_, 0.0);

    target_node_begin_u32_.resize(target_tree_.num_nodes());
    target_node_end_u32_.resize(target_tree_.num_nodes());
    for (std::size_t node_idx = 0; node_idx < target_tree_.num_nodes(); ++node_idx) {
        auto idxs = target_tree_.node_particle_idxs(node_idx);
        target_node_begin_u32_[node_idx] = static_cast<std::uint32_t>(idxs[0]);
        target_node_end_u32_[node_idx] = static_cast<std::uint32_t>(idxs[1]);
    }

    source_node_begin_u32_.resize(source_tree_.num_nodes());
    source_node_end_u32_.resize(source_tree_.num_nodes());
    for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
        auto idxs = source_tree_.node_particle_idxs(node_idx);
        source_node_begin_u32_[node_idx] = static_cast<std::uint32_t>(idxs[0]);
        source_node_end_u32_[node_idx] = static_cast<std::uint32_t>(idxs[1]);
    }

    const auto& pp_offsets = interaction_list_.particle_particle_offsets();
    const auto& pp_sources = interaction_list_.particle_particle_flat();
    const auto& pc_offsets = interaction_list_.particle_cluster_offsets();
    const auto& pc_sources = interaction_list_.particle_cluster_flat();
    const auto& cp_offsets = interaction_list_.cluster_particle_offsets();
    const auto& cp_sources = interaction_list_.cluster_particle_flat();
    const auto& cc_offsets = interaction_list_.cluster_cluster_offsets();
    const auto& cc_sources = interaction_list_.cluster_cluster_flat();

    pp_offsets_u32_.resize(pp_offsets.size());
    pp_sources_u32_.resize(pp_sources.size());
    pc_offsets_u32_.resize(pc_offsets.size());
    pc_sources_u32_.resize(pc_sources.size());
    cp_offsets_u32_.resize(cp_offsets.size());
    cp_sources_u32_.resize(cp_sources.size());
    cc_offsets_u32_.resize(cc_offsets.size());
    cc_sources_u32_.resize(cc_sources.size());

    for (std::size_t i = 0; i < pp_offsets.size(); ++i)
        pp_offsets_u32_[i] = static_cast<std::uint32_t>(pp_offsets[i]);
    for (std::size_t i = 0; i < pp_sources.size(); ++i)
        pp_sources_u32_[i] = static_cast<std::uint32_t>(pp_sources[i]);
    for (std::size_t i = 0; i < pc_offsets.size(); ++i)
        pc_offsets_u32_[i] = static_cast<std::uint32_t>(pc_offsets[i]);
    for (std::size_t i = 0; i < pc_sources.size(); ++i)
        pc_sources_u32_[i] = static_cast<std::uint32_t>(pc_sources[i]);
    for (std::size_t i = 0; i < cp_offsets.size(); ++i)
        cp_offsets_u32_[i] = static_cast<std::uint32_t>(cp_offsets[i]);
    for (std::size_t i = 0; i < cp_sources.size(); ++i)
        cp_sources_u32_[i] = static_cast<std::uint32_t>(cp_sources[i]);
    for (std::size_t i = 0; i < cc_offsets.size(); ++i)
        cc_offsets_u32_[i] = static_cast<std::uint32_t>(cc_offsets[i]);
    for (std::size_t i = 0; i < cc_sources.size(); ++i)
        cc_sources_u32_[i] = static_cast<std::uint32_t>(cc_sources[i]);

    /* Solvation energy */

    solv_eng_vec_.resize(1);
    solv_eng_vec_[0] = 0.;

    solvation_energy_ = 0.;

//    timers_.ctor.stop();
}




double SolvationEnergyCompute::compute()
{
    SolvationEnergyCompute::copyin_clusters_to_device();
#ifdef USE_CUDA_CC
    const bool require_all_dbg = cuda_require_all_enabled();
    if (require_all_dbg && !validate_device_buffers_common_()) {
        std::cerr << "[CUDA_SOLVATION] require_all set but device buffers not ready. "
                  << "Aborting to avoid CPU fallback.\n";
        std::exit(1);
    }
    if (run_batched_interactions_cuda_()) {
        SolvationEnergyCompute::delete_clusters_from_device();
        solvation_energy_ = solv_eng_vec_[0];
        return solvation_energy_;
    }
#endif
    SolvationEnergyCompute::run();
    SolvationEnergyCompute::delete_clusters_from_device();

    solvation_energy_ = solv_eng_vec_[0];

    return solvation_energy_;
}


void SolvationEnergyCompute::particle_particle_interact(std::array<std::size_t, 2> target_node_idxs,
                                                        std::array<std::size_t, 2> source_node_idxs)
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_PP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_particle_particle_()) {
        const auto elem_dev = elements_.device_view();
        const auto mol_dev = molecule_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_particle_particle_cuda(elem_dev, mol_dev, self_dev,
                                                 target_node_idxs, source_node_idxs,
                                                 potential_device_ptr_, params,
                                                 nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("particle_particle_interact");
    }
#endif

    const auto elem_host = elements_.host_view();
    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    solvation_particle_particle_cpu(elem_host, mol_host, self_host,
                                    target_node_idxs, source_node_idxs,
                                    potential_.data(), params);
}


void SolvationEnergyCompute::particle_cluster_interact(std::array<std::size_t, 2> target_node_idxs,
                                                       std::size_t source_node_idx)
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_PC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_particle_cluster_()) {
        const auto elem_dev = elements_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_particle_cluster_cuda(elem_dev, mol_interp_dev, self_dev,
                                                target_node_idxs, source_node_idx,
                                                potential_device_ptr_, params,
                                                nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("particle_cluster_interact");
    }
#endif

    const auto elem_host = elements_.host_view();
    auto self_host = host_view();
    solvation_particle_cluster_cpu(elem_host,
                                   mol_interp_pts_.interp_x_ptr(),
                                   mol_interp_pts_.interp_y_ptr(),
                                   mol_interp_pts_.interp_z_ptr(),
                                   self_host,
                                   target_node_idxs,
                                   source_node_idx,
                                   potential_.data(),
                                   params);
}


void SolvationEnergyCompute::cluster_particle_interact(std::size_t target_node_idx,
                                                       std::array<std::size_t, 2> source_node_idxs)
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_CP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_cluster_particle_()) {
        const auto mol_dev = molecule_.device_view();
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_cluster_particle_cuda(mol_dev, elem_interp_dev, self_dev,
                                                target_node_idx, source_node_idxs,
                                                params, nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("cluster_particle_interact");
    }
#endif

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    solvation_cluster_particle_cpu(mol_host,
                                   elem_interp_pts_.interp_x_ptr(),
                                   elem_interp_pts_.interp_y_ptr(),
                                   elem_interp_pts_.interp_z_ptr(),
                                   self_host,
                                   target_node_idx,
                                   source_node_idxs,
                                   params);
}


void SolvationEnergyCompute::cluster_cluster_interact(std::size_t target_node_idx,
                                                      std::size_t source_node_idx)
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_CC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_cluster_cluster_()) {
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_cluster_cluster_cuda(mol_interp_dev, elem_interp_dev, self_dev,
                                               target_node_idx, source_node_idx,
                                               params, nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("cluster_cluster_interact");
    }
#endif

    auto self_host = host_view();
    solvation_cluster_cluster_cpu(elem_interp_pts_.interp_x_ptr(),
                                  elem_interp_pts_.interp_y_ptr(),
                                  elem_interp_pts_.interp_z_ptr(),
                                  mol_interp_pts_.interp_x_ptr(),
                                  mol_interp_pts_.interp_y_ptr(),
                                  mol_interp_pts_.interp_z_ptr(),
                                  self_host,
                                  target_node_idx,
                                  source_node_idx,
                                  params);
}


void SolvationEnergyCompute::upward_pass()
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_UP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_upward_pass_()) {
        const auto mol_dev = molecule_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_upward_pass_cuda(mol_dev, mol_interp_dev, self_dev,
                                           source_tree_, params, nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("upward_pass");
    }
#endif

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    solvation_upward_pass_cpu(mol_host,
                              mol_interp_pts_.interp_x_ptr(),
                              mol_interp_pts_.interp_y_ptr(),
                              mol_interp_pts_.interp_z_ptr(),
                              self_host,
                              source_tree_,
                              params);
}


void SolvationEnergyCompute::downward_pass()
{
    const SolvationBackendParams params{
        eps_,
        kappa_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        potential_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOLVATION_DOWN");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_downward_pass_()) {
        const auto elem_dev = elements_.device_view();
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (solvation_try_downward_pass_cuda(elem_dev, elem_interp_dev, self_dev,
                                             target_tree_, potential_device_ptr_,
                                             params, nullptr)) {
            return;
        }
    }
    if (use_cuda && cuda_require_all_enabled()) {
        abort_require_all("downward_pass");
    }
#endif

    const auto elem_host = elements_.host_view();
    auto self_host = host_view();
    solvation_downward_pass_cpu(elem_host,
                                elem_interp_pts_.interp_x_ptr(),
                                elem_interp_pts_.interp_y_ptr(),
                                elem_interp_pts_.interp_z_ptr(),
                                potential_.data(),
                                self_host,
                                target_tree_,
                                params);
}

void SolvationEnergyCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    copyin_clusters_to_device_cuda_();
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void SolvationEnergyCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    delete_clusters_from_device_cuda_();
#endif

//    timers_.delete_clusters_from_device.stop();
}
