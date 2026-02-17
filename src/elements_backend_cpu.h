#ifndef H_TABIPB_ELEMENTS_BACKEND_CPU_H
#define H_TABIPB_ELEMENTS_BACKEND_CPU_H

#include "elements.h"
#include "molecule.h"

void elements_compute_source_term_cpu(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    double eps_solute);

void elements_compute_charges_cpu(
    const Elements::View& elem_view,
    const double* potential,
    std::size_t num);

#endif
