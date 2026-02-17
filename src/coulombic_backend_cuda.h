#ifndef H_TABIPB_COULOMBIC_BACKEND_CUDA_H
#define H_TABIPB_COULOMBIC_BACKEND_CUDA_H

#include <array>
#include <cstddef>

#include "coulombic_backend_common.h"
#include "coulombic_energy_compute.h"
#include "interp_pts.h"
#include "molecule.h"
#include "tree.h"

bool coulombic_try_particle_particle_cuda(
    const Molecule::View& mol_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params,
    void* stream);

bool coulombic_try_particle_cluster_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params,
    void* stream);

bool coulombic_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params,
    void* stream);

bool coulombic_try_cluster_cluster_cuda(
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params,
    void* stream);

bool coulombic_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const CoulombicBackendParams& params,
    void* stream);

#endif
