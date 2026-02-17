#ifndef H_TABIPB_SOURCE_TERM_BACKEND_CPU_H
#define H_TABIPB_SOURCE_TERM_BACKEND_CPU_H

#include <array>
#include <cstddef>

#include "elements.h"
#include "molecule.h"
#include "source_term_backend_common.h"
#include "source_term_compute.h"
#include "tree.h"

void source_term_particle_particle_cpu(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params);

void source_term_particle_cluster_cpu(
    const Elements::View& elem_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    const SourceTermCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params);

void source_term_cluster_particle_cpu(
    const Molecule::View& mol_view,
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params);

void source_term_cluster_cluster_cpu(
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params);

void source_term_upward_pass_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    SourceTermCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SourceTermBackendParams& params);

void source_term_downward_pass_cpu(
    const Elements::View& elem_view,
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    const SourceTermCompute::DeviceView& self_view,
    const Tree& target_tree,
    const SourceTermBackendParams& params);

#endif
