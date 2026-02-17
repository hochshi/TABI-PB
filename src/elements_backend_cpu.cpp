#include "elements_backend_cpu.h"

#include <cmath>

#include "constants.h"

void elements_compute_source_term_cpu(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    double eps_solute) {
  const std::size_t num_elements = elem_view.num;
  const std::size_t num_atoms = mol_view.num_particles;

#ifdef OPENMP_ENABLED
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < num_elements; ++i) {
    double source_term_1 = 0.;
    double source_term_2 = 0.;

    for (std::size_t j = 0; j < num_atoms; ++j) {
      const double x_dist = mol_view.particles_x[j] - elem_view.x[i];
      const double y_dist = mol_view.particles_y[j] - elem_view.y[i];
      const double z_dist = mol_view.particles_z[j] - elem_view.z[i];
      const double dist =
          std::sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);
      const double cos_theta =
          (elem_view.nx[i] * x_dist + elem_view.ny[i] * y_dist +
           elem_view.nz[i] * z_dist) /
          dist;
      const double g0 = constants::ONE_OVER_4PI / dist;
      const double g1 = cos_theta * g0 / dist;
      source_term_1 += mol_view.charge[j] * g0 / eps_solute;
      source_term_2 += mol_view.charge[j] * g1 / eps_solute;
    }

    elem_view.source_term[i] += source_term_1;
    elem_view.source_term[num_elements + i] += source_term_2;
  }
}

void elements_compute_charges_cpu(
    const Elements::View& elem_view,
    const double* potential,
    std::size_t num) {
#ifdef OPENMP_ENABLED
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < num; ++i) {
    elem_view.target_q[i] = constants::ONE_OVER_4PI;
    elem_view.target_q_dx[i] = constants::ONE_OVER_4PI * elem_view.nx[i];
    elem_view.target_q_dy[i] = constants::ONE_OVER_4PI * elem_view.ny[i];
    elem_view.target_q_dz[i] = constants::ONE_OVER_4PI * elem_view.nz[i];

    elem_view.source_q[i] = elem_view.area[i] * potential[num + i];
    elem_view.source_q_dx[i] = elem_view.nx[i] * elem_view.area[i] * potential[i];
    elem_view.source_q_dy[i] = elem_view.ny[i] * elem_view.area[i] * potential[i];
    elem_view.source_q_dz[i] = elem_view.nz[i] * elem_view.area[i] * potential[i];
  }
}
