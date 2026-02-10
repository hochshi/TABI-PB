#ifndef H_TABIPB_PP_CUDA_H
#define H_TABIPB_PP_CUDA_H

#include <cstddef>
#include <cstdint>

extern "C" void pp_interact_cuda(
    double eps,
    double kappa,
    double kappa2,
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* elements_area,
    const double* potential_old,
    double* potential,
    std::size_t num_elements,
    const std::uint32_t* element_node_idx,
    std::size_t num_nodes,
    const std::uint32_t* node_begin,
    const std::uint32_t* node_end,
    const std::uint32_t* pp_offsets,
    const std::uint32_t* pp_sources,
    std::size_t pp_offsets_count,
    std::size_t pp_sources_count,
    void* stream);

#endif
