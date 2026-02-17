#ifndef H_TABIPB_COULOMBIC_BACKEND_CPU_H
#define H_TABIPB_COULOMBIC_BACKEND_CPU_H

#include <array>
#include <cstddef>

#include "coulombic_backend_common.h"
#include "coulombic_energy_compute.h"
#include "molecule.h"
#include "tree.h"

void coulombic_particle_particle_cpu(
    const Molecule::View& mol_view,
    CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params);

void coulombic_particle_cluster_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params);

void coulombic_cluster_particle_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params);

void coulombic_cluster_cluster_cpu(
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params);

void coulombic_upward_pass_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const CoulombicBackendParams& params);

#endif
