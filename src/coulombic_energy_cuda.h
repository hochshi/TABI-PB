#pragma once

#include <cstddef>

extern "C" {

void coulombic_cc_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    double* mol_clusters_p,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    int num_mol_interp_potentials_per_node,
    double eps_solute,
    void* stream);

void coulombic_cp_cuda(
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    double* mol_clusters_p,
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_potentials_per_node,
    std::size_t source_begin,
    std::size_t source_end,
    double eps_solute,
    void* stream);

void coulombic_pc_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    const double* mol_clusters_q,
    std::size_t source_node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t target_begin,
    std::size_t target_end,
    double eps_solute,
    double* coul_eng,
    void* stream);

void coulombic_pp_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    std::size_t target_begin,
    std::size_t target_end,
    std::size_t source_begin,
    std::size_t source_end,
    double eps_solute,
    double* coul_eng,
    void* stream);

void coulombic_up_cuda(
    const double* mol_x,
    const double* mol_y,
    const double* mol_z,
    const double* mol_q,
    const double* mol_clusters_x,
    const double* mol_clusters_y,
    const double* mol_clusters_z,
    double* mol_clusters_q,
    const double* weights,
    int* exact_idx_x,
    int* exact_idx_y,
    int* exact_idx_z,
    double* denominator,
    std::size_t node_idx,
    int num_mol_interp_pts_per_node,
    int num_mol_interp_charges_per_node,
    std::size_t particle_start,
    std::size_t num_particles,
    void* stream);

} // extern "C"
