#include <cmath>
// #include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#include "elements.h"
#include "constants.h"
#include "source_term_backend_common.h"
#include "source_term_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "source_term_backend_cuda.h"
#endif
#include "source_term_compute.h"

SourceTermCompute::SourceTermCompute(class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps_solute)
    : TreeCompute(mol_tree, elem_tree, interaction_list),
      elements_(elements),  elem_interp_pts_(elem_interp_pts),
      molecule_(molecule),  mol_interp_pts_ (mol_interp_pts),
      one_over_4pi_eps_solute_(constants::ONE_OVER_4PI / phys_eps_solute),
      source_term_offset_(elements_.num())
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
    
//    timers_.ctor.stop();
}




void SourceTermCompute::compute()
{
    SourceTermCompute::copyin_clusters_to_device();
#if defined(USE_CUDA_CC)
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);
    if (require_all && !validate_device_buffers_common_()) {
        std::cerr << "[CUDA_SOURCE_TERM] require_all set but device buffers not ready. "
                  << "Aborting to avoid CPU/OpenACC fallback.\n";
        std::exit(1);
    }
#endif
    SourceTermCompute::run();
    SourceTermCompute::delete_clusters_from_device();
}


void SourceTermCompute::particle_particle_interact(std::array<std::size_t, 2> target_node_idxs,
                                                   std::array<std::size_t, 2> source_node_idxs)
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_PP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_particle_particle_()) {
        const auto elem_dev = elements_.device_view();
        const auto mol_dev = molecule_.device_view();
        if (source_term_try_particle_particle_cuda(elem_dev, mol_dev,
                                                   target_node_idxs, source_node_idxs,
                                                   params, nullptr)) {
            return;
        }
    }
#endif

    const auto elem_host = elements_.host_view();
    const auto mol_host = molecule_.host_view();
    source_term_particle_particle_cpu(elem_host, mol_host,
                                      target_node_idxs, source_node_idxs,
                                      params);
}


void SourceTermCompute::particle_cluster_interact(std::array<std::size_t, 2> target_node_idxs,
                                         std::size_t source_node_idx)
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_PC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_particle_cluster_()) {
        const auto elem_dev = elements_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (source_term_try_particle_cluster_cuda(elem_dev, mol_interp_dev, self_dev,
                                                  target_node_idxs, source_node_idx,
                                                  params, nullptr)) {
            return;
        }
    }
#endif

    const auto elem_host = elements_.host_view();
    auto self_host = host_view();
    source_term_particle_cluster_cpu(elem_host,
                                     mol_interp_pts_.interp_x_ptr(),
                                     mol_interp_pts_.interp_y_ptr(),
                                     mol_interp_pts_.interp_z_ptr(),
                                     self_host,
                                     target_node_idxs,
                                     source_node_idx,
                                     params);
}


void SourceTermCompute::cluster_particle_interact(std::size_t target_node_idx,
                                                  std::array<std::size_t, 2> source_node_idxs)
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_CP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_cluster_particle_()) {
        const auto mol_dev = molecule_.device_view();
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (source_term_try_cluster_particle_cuda(mol_dev, elem_interp_dev, self_dev,
                                                  target_node_idx, source_node_idxs,
                                                  params, nullptr)) {
            return;
        }
    }
#endif

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    source_term_cluster_particle_cpu(mol_host,
                                     elem_interp_pts_.interp_x_ptr(),
                                     elem_interp_pts_.interp_y_ptr(),
                                     elem_interp_pts_.interp_z_ptr(),
                                     self_host,
                                     target_node_idx,
                                     source_node_idxs,
                                     params);
}


void SourceTermCompute::cluster_cluster_interact(std::size_t target_node_idx,
                                                 std::size_t source_node_idx)
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_CC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_cluster_cluster_()) {
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (source_term_try_cluster_cluster_cuda(elem_interp_dev, mol_interp_dev, self_dev,
                                                 target_node_idx, source_node_idx,
                                                 params, nullptr)) {
            return;
        }
    }
#endif

    auto self_host = host_view();
    source_term_cluster_cluster_cpu(elem_interp_pts_.interp_x_ptr(),
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


void SourceTermCompute::upward_pass()
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_UP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_upward_pass_()) {
        const auto mol_dev = molecule_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (source_term_try_upward_pass_cuda(mol_dev, mol_interp_dev, self_dev,
                                             source_tree_, params, nullptr)) {
            return;
        }
    }
#endif

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    source_term_upward_pass_cpu(mol_host,
                                mol_interp_pts_.interp_x_ptr(),
                                mol_interp_pts_.interp_y_ptr(),
                                mol_interp_pts_.interp_z_ptr(),
                                self_host,
                                source_tree_,
                                params);
}


void SourceTermCompute::downward_pass()
{
    const SourceTermBackendParams params{
        one_over_4pi_eps_solute_,
        num_elem_interp_pts_per_node_,
        num_elem_interp_potentials_per_node_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        source_term_offset_};

#ifdef USE_CUDA_CC
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_DOWN");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda && validate_device_buffers_downward_pass_()) {
        const auto elem_dev = elements_.device_view();
        const auto elem_interp_dev = elem_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (source_term_try_downward_pass_cuda(elem_dev, elem_interp_dev, self_dev,
                                               target_tree_, params, nullptr)) {
            return;
        }
    }
#endif

    const auto elem_host = elements_.host_view();
    const auto self_host = host_view();
    source_term_downward_pass_cpu(elem_host,
                                  elem_interp_pts_.interp_x_ptr(),
                                  elem_interp_pts_.interp_y_ptr(),
                                  elem_interp_pts_.interp_z_ptr(),
                                  self_host,
                                  target_tree_,
                                  params);
}

void SourceTermCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    copyin_clusters_to_device_cuda_();
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void SourceTermCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    delete_clusters_from_device_cuda_();
#endif

//    timers_.delete_clusters_from_device.stop();
}
