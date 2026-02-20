#include "solvation_backend_cpu.h"

#include <cmath>
#include <limits>

#include "constants.h"

void solvation_particle_particle_cpu(
    const Elements::View& elem_view,
    const Molecule::View& mol_view,
    SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const double* potential,
    const SolvationBackendParams& params) {
  const std::size_t target_node_begin = target_node_idxs[0];
  const std::size_t target_node_end = target_node_idxs[1];
  const std::size_t source_node_begin = source_node_idxs[0];
  const std::size_t source_node_end = source_node_idxs[1];

  for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
    const double target_x = elem_view.x[j];
    const double target_y = elem_view.y[j];
    const double target_z = elem_view.z[j];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (std::size_t k = source_node_begin; k < source_node_end; ++k) {
      const double dx = target_x - mol_view.particles_x[k];
      const double dy = target_y - mol_view.particles_y[k];
      const double dz = target_z - mol_view.particles_z[k];

      const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double rinv = 1.0 / r;
      const double g0 = constants::ONE_OVER_4PI * rinv;
      const double expkr = std::exp(-params.kappa * r);
      const double l2 = g0 * (1.0 - expkr);
      const double l1 =
          g0 * rinv * rinv *
          (1.0 - params.eps * expkr * (1.0 + params.kappa * r));

      pot_temp_dd += l2 * mol_view.charge[k];
      pot_temp_dx += l1 * mol_view.charge[k] * dx;
      pot_temp_dy += l1 * mol_view.charge[k] * dy;
      pot_temp_dz += l1 * mol_view.charge[k] * dz;
    }

    const double pot_temp_1 =
        potential[j + params.potential_offset] * elem_view.area[j] * pot_temp_dd;
    const double pot_temp_2 =
        potential[j] * elem_view.area[j] *
        (elem_view.nx[j] * pot_temp_dx + elem_view.ny[j] * pot_temp_dy +
         elem_view.nz[j] * pot_temp_dz);

#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
    self_view.solv_eng[0] += pot_temp_1 + pot_temp_2;
  }
}

void solvation_particle_cluster_cpu(
    const Elements::View& elem_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    SolvationEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const double* potential,
    const SolvationBackendParams& params) {
  const std::size_t target_node_begin = target_node_idxs[0];
  const std::size_t target_node_end = target_node_idxs[1];

  const int n_interp = params.num_mol_interp_pts_per_node;
  const int n_charge = params.num_mol_interp_charges_per_node;
  const std::size_t source_cluster_interp_pts_begin = source_node_idx * n_interp;
  const std::size_t source_cluster_interp_charges_begin = source_node_idx * n_charge;

  for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
    const double target_x = elem_view.x[j];
    const double target_y = elem_view.y[j];
    const double target_z = elem_view.z[j];

    double pot_temp_dd = 0.0;
    double pot_temp_dx = 0.0;
    double pot_temp_dy = 0.0;
    double pot_temp_dz = 0.0;

    for (int k1 = 0; k1 < n_interp; ++k1) {
      for (int k2 = 0; k2 < n_interp; ++k2) {
        for (int k3 = 0; k3 < n_interp; ++k3) {
          const std::size_t kk =
              source_cluster_interp_charges_begin +
              static_cast<std::size_t>(k1 * n_interp * n_interp + k2 * n_interp + k3);

          const double dx = target_x - mol_interp_x[source_cluster_interp_pts_begin + k1];
          const double dy = target_y - mol_interp_y[source_cluster_interp_pts_begin + k2];
          const double dz = target_z - mol_interp_z[source_cluster_interp_pts_begin + k3];

          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          const double rinv = 1.0 / r;
          const double g0 = constants::ONE_OVER_4PI * rinv;
          const double expkr = std::exp(-params.kappa * r);
          const double l2 = g0 * (1.0 - expkr);
          const double l1 =
              g0 * rinv * rinv *
              (1.0 - params.eps * expkr * (1.0 + params.kappa * r));

          pot_temp_dd += l2 * self_view.q[kk];
          pot_temp_dx += l1 * self_view.q[kk] * dx;
          pot_temp_dy += l1 * self_view.q[kk] * dy;
          pot_temp_dz += l1 * self_view.q[kk] * dz;
        }
      }
    }

    const double pot_temp_1 =
        potential[j + params.potential_offset] * elem_view.area[j] * pot_temp_dd;
    const double pot_temp_2 =
        potential[j] * elem_view.area[j] *
        (elem_view.nx[j] * pot_temp_dx + elem_view.ny[j] * pot_temp_dy +
         elem_view.nz[j] * pot_temp_dz);

#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
    self_view.solv_eng[0] += pot_temp_1 + pot_temp_2;
  }
}

void solvation_cluster_particle_cpu(
    const Molecule::View& mol_view,
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const SolvationBackendParams& params) {
  const int n_interp = params.num_elem_interp_pts_per_node;
  const int n_pot = params.num_elem_interp_potentials_per_node;

  const std::size_t target_cluster_interp_pts_begin = target_node_idx * n_interp;
  const std::size_t target_cluster_interp_potentials_begin = target_node_idx * n_pot;

  const std::size_t source_node_begin = source_node_idxs[0];
  const std::size_t source_node_end = source_node_idxs[1];

  for (int j1 = 0; j1 < n_interp; ++j1) {
    for (int j2 = 0; j2 < n_interp; ++j2) {
      for (int j3 = 0; j3 < n_interp; ++j3) {
        const std::size_t jj =
            target_cluster_interp_potentials_begin +
            static_cast<std::size_t>(j1 * n_interp * n_interp + j2 * n_interp + j3);

        const double target_x = elem_interp_x[target_cluster_interp_pts_begin + j1];
        const double target_y = elem_interp_y[target_cluster_interp_pts_begin + j2];
        const double target_z = elem_interp_z[target_cluster_interp_pts_begin + j3];

        double pot_temp_dd = 0.0;
        double pot_temp_dx = 0.0;
        double pot_temp_dy = 0.0;
        double pot_temp_dz = 0.0;

        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {
          const double dx = target_x - mol_view.particles_x[k];
          const double dy = target_y - mol_view.particles_y[k];
          const double dz = target_z - mol_view.particles_z[k];

          const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
          const double rinv = 1.0 / r;
          const double g0 = constants::ONE_OVER_4PI * rinv;
          const double expkr = std::exp(-params.kappa * r);
          const double l2 = g0 * (1.0 - expkr);
          const double l1 =
              g0 * rinv * rinv *
              (1.0 - params.eps * expkr * (1.0 + params.kappa * r));

          pot_temp_dd += l2 * mol_view.charge[k];
          pot_temp_dx += l1 * mol_view.charge[k] * dx;
          pot_temp_dy += l1 * mol_view.charge[k] * dy;
          pot_temp_dz += l1 * mol_view.charge[k] * dz;
        }

#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p[jj] += pot_temp_dd;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dx[jj] += pot_temp_dx;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dy[jj] += pot_temp_dy;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dz[jj] += pot_temp_dz;
      }
    }
  }
}

void solvation_cluster_cluster_cpu(
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    SolvationEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const SolvationBackendParams& params) {
  const int n_elem_interp = params.num_elem_interp_pts_per_node;
  const int n_elem_pot = params.num_elem_interp_potentials_per_node;
  const int n_mol_interp = params.num_mol_interp_pts_per_node;
  const int n_mol_charge = params.num_mol_interp_charges_per_node;

  const std::size_t target_cluster_interp_pts_begin = target_node_idx * n_elem_interp;
  const std::size_t target_cluster_interp_potentials_begin = target_node_idx * n_elem_pot;

  const std::size_t source_cluster_interp_pts_begin = source_node_idx * n_mol_interp;
  const std::size_t source_cluster_interp_charges_begin = source_node_idx * n_mol_charge;

  for (int j1 = 0; j1 < n_elem_interp; ++j1) {
    for (int j2 = 0; j2 < n_elem_interp; ++j2) {
      for (int j3 = 0; j3 < n_elem_interp; ++j3) {
        const std::size_t jj =
            target_cluster_interp_potentials_begin +
            static_cast<std::size_t>(j1 * n_elem_interp * n_elem_interp +
                                     j2 * n_elem_interp + j3);

        const double target_x = elem_interp_x[target_cluster_interp_pts_begin + j1];
        const double target_y = elem_interp_y[target_cluster_interp_pts_begin + j2];
        const double target_z = elem_interp_z[target_cluster_interp_pts_begin + j3];

        double pot_temp_dd = 0.0;
        double pot_temp_dx = 0.0;
        double pot_temp_dy = 0.0;
        double pot_temp_dz = 0.0;

        for (int k1 = 0; k1 < n_mol_interp; ++k1) {
          for (int k2 = 0; k2 < n_mol_interp; ++k2) {
            for (int k3 = 0; k3 < n_mol_interp; ++k3) {
              const std::size_t kk =
                  source_cluster_interp_charges_begin +
                  static_cast<std::size_t>(k1 * n_mol_interp * n_mol_interp +
                                           k2 * n_mol_interp + k3);

              const double dx =
                  target_x - mol_interp_x[source_cluster_interp_pts_begin + k1];
              const double dy =
                  target_y - mol_interp_y[source_cluster_interp_pts_begin + k2];
              const double dz =
                  target_z - mol_interp_z[source_cluster_interp_pts_begin + k3];

              const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
              const double rinv = 1.0 / r;
              const double g0 = constants::ONE_OVER_4PI * rinv;
              const double expkr = std::exp(-params.kappa * r);
              const double l2 = g0 * (1.0 - expkr);
              const double l1 =
                  g0 * rinv * rinv *
                  (1.0 - params.eps * expkr * (1.0 + params.kappa * r));

              pot_temp_dd += l2 * self_view.q[kk];
              pot_temp_dx += l1 * self_view.q[kk] * dx;
              pot_temp_dy += l1 * self_view.q[kk] * dy;
              pot_temp_dz += l1 * self_view.q[kk] * dz;
            }
          }
        }

#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p[jj] += pot_temp_dd;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dx[jj] += pot_temp_dx;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dy[jj] += pot_temp_dy;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p_dz[jj] += pot_temp_dz;
      }
    }
  }
}

void solvation_upward_pass_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    SolvationEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const SolvationBackendParams& params) {
  const int n_interp = params.num_mol_interp_pts_per_node;
  const int n_charge = params.num_mol_interp_charges_per_node;

  for (std::size_t node_idx = 0; node_idx < source_tree.num_nodes(); ++node_idx) {
    const auto particle_idxs = source_tree.node_particle_idxs(node_idx);

    const std::size_t node_interp_pts_start = node_idx * n_interp;
    const std::size_t node_charges_start = node_idx * n_charge;

    const std::size_t particle_start = particle_idxs[0];
    const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];

    for (std::size_t i = 0; i < num_particles; ++i) {
      self_view.exact_idx_x[i] = -1;
      self_view.exact_idx_y[i] = -1;
      self_view.exact_idx_z[i] = -1;
    }

    for (std::size_t i = 0; i < num_particles; ++i) {
      double denominator_x = 0.0;
      double denominator_y = 0.0;
      double denominator_z = 0.0;
      int ex = -1;
      int ey = -1;
      int ez = -1;

      const double xx = mol_view.particles_x[particle_start + i];
      const double yy = mol_view.particles_y[particle_start + i];
      const double zz = mol_view.particles_z[particle_start + i];

      for (int j = 0; j < n_interp; ++j) {
        const double dist_x = xx - mol_interp_x[node_interp_pts_start + j];
        const double dist_y = yy - mol_interp_y[node_interp_pts_start + j];
        const double dist_z = zz - mol_interp_z[node_interp_pts_start + j];

        denominator_x += self_view.weights_up[j] / dist_x;
        denominator_y += self_view.weights_up[j] / dist_y;
        denominator_z += self_view.weights_up[j] / dist_z;

        const int cx = (std::abs(dist_x) < std::numeric_limits<double>::min()) ? j : -1;
        const int cy = (std::abs(dist_y) < std::numeric_limits<double>::min()) ? j : -1;
        const int cz = (std::abs(dist_z) < std::numeric_limits<double>::min()) ? j : -1;

        ex = (ex > cx) ? ex : cx;
        ey = (ey > cy) ? ey : cy;
        ez = (ez > cz) ? ez : cz;
      }

      self_view.exact_idx_x[i] = ex;
      self_view.exact_idx_y[i] = ey;
      self_view.exact_idx_z[i] = ez;

      self_view.denominator[i] = 1.0;
      if (self_view.exact_idx_x[i] == -1) self_view.denominator[i] /= denominator_x;
      if (self_view.exact_idx_y[i] == -1) self_view.denominator[i] /= denominator_y;
      if (self_view.exact_idx_z[i] == -1) self_view.denominator[i] /= denominator_z;
    }

    for (int k1 = 0; k1 < n_interp; ++k1) {
      for (int k2 = 0; k2 < n_interp; ++k2) {
        for (int k3 = 0; k3 < n_interp; ++k3) {
          const std::size_t kk =
              node_charges_start +
              static_cast<std::size_t>(k1 * n_interp * n_interp + k2 * n_interp + k3);

          const double cx = mol_interp_x[node_interp_pts_start + k1];
          const double w1 = self_view.weights_up[k1];
          const double cy = mol_interp_y[node_interp_pts_start + k2];
          const double w2 = self_view.weights_up[k2];
          const double cz = mol_interp_z[node_interp_pts_start + k3];
          const double w3 = self_view.weights_up[k3];

          double q_temp = 0.0;
          for (std::size_t i = 0; i < num_particles; ++i) {
            const double dist_x = mol_view.particles_x[particle_start + i] - cx;
            const double dist_y = mol_view.particles_y[particle_start + i] - cy;
            const double dist_z = mol_view.particles_z[particle_start + i] - cz;

            double numerator = 1.0;
            if (self_view.exact_idx_x[i] == -1) {
              numerator *= w1 / dist_x;
            } else if (self_view.exact_idx_x[i] != k1) {
              numerator *= 0.0;
            }

            if (self_view.exact_idx_y[i] == -1) {
              numerator *= w2 / dist_y;
            } else if (self_view.exact_idx_y[i] != k2) {
              numerator *= 0.0;
            }

            if (self_view.exact_idx_z[i] == -1) {
              numerator *= w3 / dist_z;
            } else if (self_view.exact_idx_z[i] != k3) {
              numerator *= 0.0;
            }

            q_temp += mol_view.charge[particle_start + i] * numerator *
                      self_view.denominator[i];
          }
          self_view.q[kk] += q_temp;
        }
      }
    }
  }
}

void solvation_downward_pass_cpu(
    const Elements::View& elem_view,
    const double* elem_interp_x,
    const double* elem_interp_y,
    const double* elem_interp_z,
    const double* potential,
    SolvationEnergyCompute::DeviceView& self_view,
    const Tree& target_tree,
    const SolvationBackendParams& params) {
  const int n_interp = params.num_elem_interp_pts_per_node;
  const int n_pot = params.num_elem_interp_potentials_per_node;

  for (std::size_t node_idx = 0; node_idx < target_tree.num_nodes(); ++node_idx) {
    const auto particle_idxs = target_tree.node_particle_idxs(node_idx);
    const std::size_t node_interp_pts_start = node_idx * n_interp;
    const std::size_t node_potentials_start = node_idx * n_pot;

    const std::size_t particle_start = particle_idxs[0];
    const std::size_t num_particles = particle_idxs[1] - particle_idxs[0];

    for (std::size_t i = 0; i < num_particles; ++i) {
      double denominator_x = 0.0;
      double denominator_y = 0.0;
      double denominator_z = 0.0;

      int exact_idx_x = -1;
      int exact_idx_y = -1;
      int exact_idx_z = -1;

      const double xx = elem_view.x[particle_start + i];
      const double yy = elem_view.y[particle_start + i];
      const double zz = elem_view.z[particle_start + i];

      for (int j = 0; j < n_interp; ++j) {
        const double dist_x = xx - elem_interp_x[node_interp_pts_start + j];
        const double dist_y = yy - elem_interp_y[node_interp_pts_start + j];
        const double dist_z = zz - elem_interp_z[node_interp_pts_start + j];

        denominator_x += self_view.weights_down[j] / dist_x;
        denominator_y += self_view.weights_down[j] / dist_y;
        denominator_z += self_view.weights_down[j] / dist_z;

        const int cx = (std::abs(dist_x) < std::numeric_limits<double>::min()) ? j : -1;
        const int cy = (std::abs(dist_y) < std::numeric_limits<double>::min()) ? j : -1;
        const int cz = (std::abs(dist_z) < std::numeric_limits<double>::min()) ? j : -1;

        exact_idx_x = (exact_idx_x > cx) ? exact_idx_x : cx;
        exact_idx_y = (exact_idx_y > cy) ? exact_idx_y : cy;
        exact_idx_z = (exact_idx_z > cz) ? exact_idx_z : cz;
      }

      double denominator = 1.0;
      if (exact_idx_x == -1) denominator /= denominator_x;
      if (exact_idx_y == -1) denominator /= denominator_y;
      if (exact_idx_z == -1) denominator /= denominator_z;

      double pot_temp_dd = 0.0;
      double pot_temp_dx = 0.0;
      double pot_temp_dy = 0.0;
      double pot_temp_dz = 0.0;

      for (int k1 = 0; k1 < n_interp; ++k1) {
        for (int k2 = 0; k2 < n_interp; ++k2) {
          for (int k3 = 0; k3 < n_interp; ++k3) {
            const std::size_t kk =
                node_potentials_start +
                static_cast<std::size_t>(k1 * n_interp * n_interp + k2 * n_interp + k3);

            const double dist_x = xx - elem_interp_x[node_interp_pts_start + k1];
            const double dist_y = yy - elem_interp_y[node_interp_pts_start + k2];
            const double dist_z = zz - elem_interp_z[node_interp_pts_start + k3];

            double numerator = 1.0;
            if (exact_idx_x == -1) {
              numerator *= self_view.weights_down[k1] / dist_x;
            } else if (exact_idx_x != k1) {
              numerator *= 0.0;
            }

            if (exact_idx_y == -1) {
              numerator *= self_view.weights_down[k2] / dist_y;
            } else if (exact_idx_y != k2) {
              numerator *= 0.0;
            }

            if (exact_idx_z == -1) {
              numerator *= self_view.weights_down[k3] / dist_z;
            } else if (exact_idx_z != k3) {
              numerator *= 0.0;
            }

            pot_temp_dd += numerator * denominator * self_view.p[kk];
            pot_temp_dx += numerator * denominator * self_view.p_dx[kk];
            pot_temp_dy += numerator * denominator * self_view.p_dy[kk];
            pot_temp_dz += numerator * denominator * self_view.p_dz[kk];
          }
        }
      }

      const double pot_temp_1 =
          potential[particle_start + i + params.potential_offset] *
          elem_view.area[particle_start + i] * pot_temp_dd;
      const double pot_temp_2 =
          potential[particle_start + i] * elem_view.area[particle_start + i] *
          (elem_view.nx[particle_start + i] * pot_temp_dx +
           elem_view.ny[particle_start + i] * pot_temp_dy +
           elem_view.nz[particle_start + i] * pot_temp_dz);
      self_view.solv_eng[0] += pot_temp_1 + pot_temp_2;
    }
  }
}
