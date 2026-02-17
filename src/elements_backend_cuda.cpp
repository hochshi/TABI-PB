#include "elements_backend_cuda.h"

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#include "elements_cuda.h"
#endif

bool elements_try_compute_source_term_cuda(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    std::size_t num_elements,
    std::size_t num_atoms,
    double eps_solute,
    void* stream) {
#ifdef USE_CUDA_CC
  const bool view_ok = (elem_view.num == num_elements) &&
                       (mol_view.num_particles == num_atoms) && elem_view.x &&
                       elem_view.y && elem_view.z && elem_view.nx &&
                       elem_view.ny && elem_view.nz && elem_view.source_term &&
                       mol_view.particles_x && mol_view.particles_y &&
                       mol_view.particles_z && mol_view.charge;
  if (!view_ok) {
    return false;
  }

  elements_compute_source_term_cuda(
      elem_view.x, elem_view.y, elem_view.z, elem_view.nx, elem_view.ny,
      elem_view.nz, mol_view.particles_x, mol_view.particles_y,
      mol_view.particles_z, mol_view.charge, elem_view.source_term, num_elements,
      num_atoms, eps_solute, stream);
  return true;
#else
  (void)elem_view;
  (void)mol_view;
  (void)num_elements;
  (void)num_atoms;
  (void)eps_solute;
  (void)stream;
  return false;
#endif
}

bool elements_try_compute_charges_cuda(
    const Elements::View& elem_view,
    const double* potential_dev,
    std::size_t num,
    void* stream) {
#ifdef USE_CUDA_CC
  if (!cuda_pointer_is_device_accessible(potential_dev)) {
    return false;
  }

  const bool view_ok = (elem_view.num == num) && elem_view.nx && elem_view.ny &&
                       elem_view.nz && elem_view.area && elem_view.target_q &&
                       elem_view.target_q_dx && elem_view.target_q_dy &&
                       elem_view.target_q_dz && elem_view.source_q &&
                       elem_view.source_q_dx && elem_view.source_q_dy &&
                       elem_view.source_q_dz;
  if (!view_ok) {
    return false;
  }

  elements_compute_charges_cuda(
      elem_view.nx, elem_view.ny, elem_view.nz, elem_view.area, potential_dev,
      elem_view.target_q, elem_view.target_q_dx, elem_view.target_q_dy,
      elem_view.target_q_dz, elem_view.source_q, elem_view.source_q_dx,
      elem_view.source_q_dy, elem_view.source_q_dz, num, stream);
  return true;
#else
  (void)elem_view;
  (void)potential_dev;
  (void)num;
  (void)stream;
  return false;
#endif
}
