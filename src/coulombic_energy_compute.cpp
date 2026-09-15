#include <cmath>
// #include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

// #include "constants.h"
#include "coulombic_backend_common.h"
#include "coulombic_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "coulombic_backend_cuda.h"
#endif
#include "coulombic_energy_compute.h"

CoulombicEnergyCompute::CoulombicEnergyCompute(const class Molecule& molecule,
                      const class InterpolationPoints& mol_interp_pts, const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps_solute)
    : TreeCompute(mol_tree, interaction_list),
      molecule_(molecule), mol_interp_pts_(mol_interp_pts), eps_solute_(phys_eps_solute)
{
//    timers_.ctor.start();
    
    /* Target and Source clusters */
    
    num_mol_interp_pts_per_node_        = mol_interp_pts_.num_interp_pts_per_node();
    num_mol_interp_charges_per_node_    = std::pow(num_mol_interp_pts_per_node_, 3);
    num_mol_charges_                    = source_tree_.num_nodes() * num_mol_interp_charges_per_node_;
    
    num_mol_interp_potentials_per_node_ = num_mol_interp_charges_per_node_;
    num_mol_potentials_                 = num_mol_charges_;
    
#ifndef USE_CUDA_CC
    mol_interp_charge_.assign(num_mol_charges_, 0.);
#endif
#ifndef USE_CUDA_CC
    mol_interp_potential_.assign(num_mol_potentials_, 0.);
#endif

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
    pp_offsets_u32_.resize(pp_offsets.size());
    pp_sources_u32_.resize(pp_sources.size());
    pc_offsets_u32_.resize(pc_offsets.size());
    pc_sources_u32_.resize(pc_sources.size());

    for (std::size_t i = 0; i < pp_offsets.size(); ++i)
        pp_offsets_u32_[i] = static_cast<std::uint32_t>(pp_offsets[i]);
    for (std::size_t i = 0; i < pp_sources.size(); ++i)
        pp_sources_u32_[i] = static_cast<std::uint32_t>(pp_sources[i]);
    for (std::size_t i = 0; i < pc_offsets.size(); ++i)
        pc_offsets_u32_[i] = static_cast<std::uint32_t>(pc_offsets[i]);
    for (std::size_t i = 0; i < pc_sources.size(); ++i)
        pc_sources_u32_[i] = static_cast<std::uint32_t>(pc_sources[i]);
    /* Coulombic energy */

    coul_eng_vec_.resize(1);
    coul_eng_vec_[0] = 0.;

    coulombic_energy_ = 0.;

//    timers_.ctor.stop();
}


double CoulombicEnergyCompute::compute()
{
    CoulombicEnergyCompute::copyin_clusters_to_device();
#ifdef USE_CUDA_CC
    if (!validate_device_buffers_common_()) {
        std::cerr << "[CUDA_COULOMBIC] device buffers not ready.\n";
        std::exit(1);
    }
#ifdef USE_CUDA_BATCHED_INTERACTIONS
    if (!run_batched_interactions_cuda_()) {
        std::cerr << "[CUDA_COULOMBIC] batched CUDA path unavailable.\n";
        std::exit(1);
    }
#else
    CoulombicEnergyCompute::run();
#endif
#else
    CoulombicEnergyCompute::run();
#endif
    CoulombicEnergyCompute::delete_clusters_from_device();
    
    coulombic_energy_ = coul_eng_vec_[0] / 2;
    
    return coulombic_energy_;
}


void CoulombicEnergyCompute::particle_particle_interact(std::array<std::size_t, 2> target_node_idxs,
                                                        std::array<std::size_t, 2> source_node_idxs)
{
    const CoulombicBackendParams params{
        eps_solute_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        num_mol_interp_potentials_per_node_};

#ifdef USE_CUDA_CC
    if (validate_device_buffers_particle_particle_()) {
        const auto mol_dev = molecule_.device_view();
        const auto self_dev = device_view();
        if (coulombic_try_particle_particle_cuda(mol_dev, self_dev,
                                                 target_node_idxs, source_node_idxs,
                                                 params, nullptr)) {
            return;
        }
    }
    std::cerr << "[CUDA_COULOMBIC] particle-particle CUDA path unavailable.\n";
    std::exit(1);
#else

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    coulombic_particle_particle_cpu(mol_host, self_host,
                                    target_node_idxs, source_node_idxs,
                                    params);
#endif
}


void CoulombicEnergyCompute::particle_cluster_interact(std::array<std::size_t, 2> target_node_idxs,
                                                       std::size_t source_node_idx)
{
    const CoulombicBackendParams params{
        eps_solute_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        num_mol_interp_potentials_per_node_};

#ifdef USE_CUDA_CC
    if (validate_device_buffers_particle_cluster_()) {
        const auto mol_dev = molecule_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (coulombic_try_particle_cluster_cuda(mol_dev, mol_interp_dev, self_dev,
                                                target_node_idxs, source_node_idx,
                                                params, nullptr)) {
            return;
        }
    }
    std::cerr << "[CUDA_COULOMBIC] particle-cluster CUDA path unavailable.\n";
    std::exit(1);
#else

    const auto mol_host = molecule_.host_view();
    const auto self_host = host_view();
    coulombic_particle_cluster_cpu(mol_host,
                                   mol_interp_pts_.interp_x_ptr(),
                                   mol_interp_pts_.interp_y_ptr(),
                                   mol_interp_pts_.interp_z_ptr(),
                                   self_host,
                                   target_node_idxs,
                                   source_node_idx,
                                   params);
#endif
}


void CoulombicEnergyCompute::cluster_particle_interact(std::size_t target_node_idx,
                                                       std::array<std::size_t, 2> source_node_idxs)
{
#ifdef USE_CUDA_CC
    // Match the CPU reference: CP only updates the unused interpolation potential.
    (void)target_node_idx;
    (void)source_node_idxs;
#else
    const CoulombicBackendParams params{
        eps_solute_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        num_mol_interp_potentials_per_node_};

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    coulombic_cluster_particle_cpu(mol_host,
                                   mol_interp_pts_.interp_x_ptr(),
                                   mol_interp_pts_.interp_y_ptr(),
                                   mol_interp_pts_.interp_z_ptr(),
                                   self_host,
                                   target_node_idx,
                                   source_node_idxs,
                                   params);
#endif
}


void CoulombicEnergyCompute::cluster_cluster_interact(std::size_t target_node_idx,
                                                      std::size_t source_node_idx)
{
#ifdef USE_CUDA_CC
    // Match the CPU reference: CC only updates the unused interpolation potential.
    (void)target_node_idx;
    (void)source_node_idx;
#else
    const CoulombicBackendParams params{
        eps_solute_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        num_mol_interp_potentials_per_node_};

    auto self_host = host_view();
    coulombic_cluster_cluster_cpu(mol_interp_pts_.interp_x_ptr(),
                                  mol_interp_pts_.interp_y_ptr(),
                                  mol_interp_pts_.interp_z_ptr(),
                                  self_host,
                                  target_node_idx,
                                  source_node_idx,
                                  params);
#endif
}


void CoulombicEnergyCompute::upward_pass()
{
    const CoulombicBackendParams params{
        eps_solute_,
        num_mol_interp_pts_per_node_,
        num_mol_interp_charges_per_node_,
        num_mol_interp_potentials_per_node_};

#ifdef USE_CUDA_CC
    if (mol_interp_pts_.charge_cache_ready(num_mol_charges_)) {
        return;
    }
    if (validate_device_buffers_upward_pass_()) {
        const auto mol_dev = molecule_.device_view();
        const auto mol_interp_dev = mol_interp_pts_.device_view();
        const auto self_dev = device_view();
        if (coulombic_try_upward_pass_cuda(mol_dev, mol_interp_dev, self_dev,
                                           source_tree_, params, nullptr)) {
            mol_interp_pts_.mark_charge_cache_ready(num_mol_charges_);
            return;
        }
    }
    std::cerr << "[CUDA_COULOMBIC] upward CUDA path unavailable.\n";
    std::exit(1);
#else

    const auto mol_host = molecule_.host_view();
    auto self_host = host_view();
    coulombic_upward_pass_cpu(mol_host,
                              mol_interp_pts_.interp_x_ptr(),
                              mol_interp_pts_.interp_y_ptr(),
                              mol_interp_pts_.interp_z_ptr(),
                              self_host,
                              source_tree_,
                              params);
#endif
}


void CoulombicEnergyCompute::downward_pass()
{
}

void CoulombicEnergyCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    copyin_clusters_to_device_cuda_();
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void CoulombicEnergyCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    delete_clusters_from_device_cuda_();
#endif

//    timers_.delete_clusters_from_device.stop();
}
