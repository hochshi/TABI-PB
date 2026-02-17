#ifndef H_TABIPB_ELEMENTS_BACKEND_CUDA_H
#define H_TABIPB_ELEMENTS_BACKEND_CUDA_H

#include <cstddef>

#include "elements.h"
#include "molecule.h"

bool elements_try_compute_source_term_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    std::size_t num_elements,
    std::size_t num_atoms,
    double eps_solute,
    void* stream);

bool elements_try_compute_charges_cuda(
    const Elements::View& elem_view,
    const double* potential_dev,
    std::size_t num,
    void* stream);

#endif
