#ifndef H_TABIPB_PC_CUDA_H
#define H_TABIPB_PC_CUDA_H

#include <cstddef>
#include <cstdint>

extern "C" void pc_interact_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    double eps,
    double kappa,
    double kappa2,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_q,
    const double* clusters_q_dx,
    const double* clusters_q_dy,
    const double* clusters_q_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
    const std::uint32_t* pc_offsets,
    const std::uint32_t* pc_sources,
    std::size_t pc_offsets_count,
    std::size_t pc_sources_count,
    void* stream);

#endif
