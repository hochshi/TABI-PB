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

#endif
