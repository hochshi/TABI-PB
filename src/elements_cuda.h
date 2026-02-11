#ifndef H_TABIPB_ELEMENTS_CUDA_H
#define H_TABIPB_ELEMENTS_CUDA_H

#include <cstddef>

extern "C" void elements_compute_charges_cuda(
    const double* nx,
    const double* ny,
    const double* nz,
    const double* area,
    const double* potential,
    double* target_q,
    double* target_q_dx,
    double* target_q_dy,
    double* target_q_dz,
    double* source_q,
    double* source_q_dx,
    double* source_q_dy,
    double* source_q_dz,
    std::size_t num,
    void* stream);

extern "C" void elements_compute_source_term_cuda(
    const double* elements_x,
    const double* elements_y,
    const double* elements_z,
    const double* elements_nx,
    const double* elements_ny,
    const double* elements_nz,
    const double* molecule_x,
    const double* molecule_y,
    const double* molecule_z,
    const double* molecule_q,
    double* elements_source_term,
    std::size_t num_elements,
    std::size_t num_atoms,
    double eps_solute,
    void* stream);

#endif
