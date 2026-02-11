#pragma once

#include <cstddef>

extern "C" void output_coulombic_cuda(const double* mol_x,
                                      const double* mol_y,
                                      const double* mol_z,
                                      const double* mol_q,
                                      std::size_t num_atoms,
                                      double eps_solute,
                                      double* out_energy,
                                      void* stream);

extern "C" void output_solvation_cuda(const double* elem_x,
                                      const double* elem_y,
                                      const double* elem_z,
                                      const double* elem_nx,
                                      const double* elem_ny,
                                      const double* elem_nz,
                                      const double* elem_area,
                                      const double* mol_x,
                                      const double* mol_y,
                                      const double* mol_z,
                                      const double* mol_q,
                                      const double* potential,
                                      std::size_t potential_offset,
                                      std::size_t num_elems,
                                      std::size_t num_atoms,
                                      double eps,
                                      double kappa,
                                      double* out_energy,
                                      void* stream);
