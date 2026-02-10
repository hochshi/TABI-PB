#ifndef H_TABIPB_DOWN_CUDA_H
#define H_TABIPB_DOWN_CUDA_H

#include <cstddef>
#include <cstdint>

extern "C" void downward_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* clusters_p,
    const double* clusters_p_dx,
    const double* clusters_p_dy,
    const double* clusters_p_dz,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* targets_q,
    const double* targets_q_dx,
    const double* targets_q_dy,
    const double* targets_q_dz,
    const double* weights,
    double* potential,
    std::size_t potential_offset,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    void* stream);

#endif
