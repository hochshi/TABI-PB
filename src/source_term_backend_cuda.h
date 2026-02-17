#ifndef H_TABIPB_SOURCE_TERM_BACKEND_CUDA_H
#define H_TABIPB_SOURCE_TERM_BACKEND_CUDA_H

#include <array>
#include <cstddef>

#include "elements.h"
#include "interp_pts.h"
#include "molecule.h"
#include "source_term_backend_common.h"
#include "source_term_compute.h"
#include "tree.h"

bool source_term_try_particle_particle_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params,
    void* stream);

bool source_term_try_particle_cluster_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params,
    void* stream);

bool source_term_try_cluster_particle_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& elem_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SourceTermBackendParams& params,
    void* stream);

bool source_term_try_cluster_cluster_cuda(
    const InterpolationPoints::View& elem_interp_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SourceTermBackendParams& params,
    void* stream);

bool source_term_try_upward_pass_cuda(
    const Molecule::View& mol_view,
    const InterpolationPoints::View& mol_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SourceTermBackendParams& params,
    void* stream);

bool source_term_try_downward_pass_cuda(
    const Elements::View& elem_view,
    const InterpolationPoints::View& elem_interp_view,
    const SourceTermCompute::DeviceView& self_view,
    const Tree& target_tree,
    const SourceTermBackendParams& params,
    void* stream);

#endif
