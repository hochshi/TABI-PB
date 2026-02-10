#ifndef H_TABIPB_BE_CUDA_H
#define H_TABIPB_BE_CUDA_H

#include <cstddef>

extern "C" void be_clear_cluster_charges_cuda(
    double* clusters_q,
    double* clusters_q_dx,
    double* clusters_q_dy,
    double* clusters_q_dz,
    std::size_t num_charges,
    void* stream);

extern "C" void be_clear_cluster_potentials_cuda(
    double* clusters_p,
    double* clusters_p_dx,
    double* clusters_p_dy,
    double* clusters_p_dz,
    std::size_t num_potentials,
    void* stream);

extern "C" void be_potential_copy_zero_cuda(
    const double* potential_new,
    double* potential_temp,
    double* potential_new_out,
    std::size_t count,
    void* stream);

extern "C" void be_potential_combine_cuda(
    const double* potential_old,
    const double* potential_temp,
    double* potential_new,
    std::size_t count,
    double alpha,
    double beta,
    double coeff_1,
    double coeff_2,
    void* stream);

#endif
