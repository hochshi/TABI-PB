#ifndef H_TABIPB_UP_CUDA_H
#define H_TABIPB_UP_CUDA_H

#include <cstddef>
#include <cstdint>

extern "C" void upward_denom_cuda(
    int num_interp_pts_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    std::size_t num_elements,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    int* exact_idx_x,
    int* exact_idx_y,
    int* exact_idx_z,
    double* denominator,
    void* stream);

extern "C" void upward_charge_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    const int* exact_idx_x,
    const int* exact_idx_y,
    const int* exact_idx_z,
    const double* denominator,
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    void* stream);

extern "C" void upward_fused_cuda(
    int num_interp_pts_per_node,
    int num_charges_per_node,
    const double* clusters_x,
    const double* clusters_y,
    const double* clusters_z,
    const double* weights,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* sources_q,
    const double* sources_q_dx,
    const double* sources_q_dy,
    const double* sources_q_dz,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::size_t* level_nodes,
    std::size_t num_level_nodes,
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    void* stream);

#endif
