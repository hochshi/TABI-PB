#include "coulombic_backend_cpu.h"

#include <cmath>
#include <limits>

void coulombic_particle_particle_cpu(
    const Molecule::View& mol_view,
    CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params) {
  const std::size_t target_node_begin = target_node_idxs[0];
  const std::size_t target_node_end = target_node_idxs[1];
  const std::size_t source_node_begin = source_node_idxs[0];
  const std::size_t source_node_end = source_node_idxs[1];

  for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
    const double target_x = mol_view.particles_x[j];
    const double target_y = mol_view.particles_y[j];
    const double target_z = mol_view.particles_z[j];
    const double target_q = mol_view.charge[j];

    double pot_temp = 0.0;
    for (std::size_t k = source_node_begin; k < source_node_end; ++k) {
      const double dx = target_x - mol_view.particles_x[k];
      const double dy = target_y - mol_view.particles_y[k];
      const double dz = target_z - mol_view.particles_z[k];
      const double r = dx * dx + dy * dy + dz * dz;
      if (r > 0.0) {
        pot_temp += target_q * mol_view.charge[k] / params.eps_solute / std::sqrt(r);
      }
    }
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
    self_view.coul_eng[0] += pot_temp;
  }
}

void coulombic_particle_cluster_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    const CoulombicEnergyCompute::DeviceView& self_view,
    const std::array<std::size_t, 2>& target_node_idxs,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params) {
  const std::size_t target_node_begin = target_node_idxs[0];
  const std::size_t target_node_end = target_node_idxs[1];

  const int n_interp = params.num_mol_interp_pts_per_node;
  const int n_charge = params.num_mol_interp_charges_per_node;
  const std::size_t source_cluster_interp_pts_begin = source_node_idx * n_interp;
  const std::size_t source_cluster_interp_charges_begin = source_node_idx * n_charge;

  for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
    const double target_x = mol_view.particles_x[j];
    const double target_y = mol_view.particles_y[j];
    const double target_z = mol_view.particles_z[j];
    const double target_q = mol_view.charge[j];

    double pot_temp = 0.0;
    for (int k1 = 0; k1 < n_interp; ++k1) {
      for (int k2 = 0; k2 < n_interp; ++k2) {
        for (int k3 = 0; k3 < n_interp; ++k3) {
          const std::size_t kk = source_cluster_interp_charges_begin +
                                 static_cast<std::size_t>(k1 * n_interp * n_interp +
                                                          k2 * n_interp + k3);

          const double dx = target_x - mol_interp_x[source_cluster_interp_pts_begin + k1];
          const double dy = target_y - mol_interp_y[source_cluster_interp_pts_begin + k2];
          const double dz = target_z - mol_interp_z[source_cluster_interp_pts_begin + k3];

          pot_temp += self_view.q[kk] / params.eps_solute / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
      }
    }
    pot_temp *= target_q;
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
    self_view.coul_eng[0] += pot_temp;
  }
}

void coulombic_cluster_particle_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    const std::array<std::size_t, 2>& source_node_idxs,
    const CoulombicBackendParams& params) {
  const int n_interp = params.num_mol_interp_pts_per_node;
  const int n_pot = params.num_mol_interp_potentials_per_node;

  const std::size_t target_cluster_interp_pts_begin = target_node_idx * n_interp;
  const std::size_t target_cluster_interp_potentials_begin = target_node_idx * n_pot;

  const std::size_t source_node_begin = source_node_idxs[0];
  const std::size_t source_node_end = source_node_idxs[1];

  for (int j1 = 0; j1 < n_interp; ++j1) {
    for (int j2 = 0; j2 < n_interp; ++j2) {
      for (int j3 = 0; j3 < n_interp; ++j3) {
        const std::size_t jj = target_cluster_interp_potentials_begin +
                               static_cast<std::size_t>(j1 * n_interp * n_interp +
                                                        j2 * n_interp + j3);

        const double target_x = mol_interp_x[target_cluster_interp_pts_begin + j1];
        const double target_y = mol_interp_y[target_cluster_interp_pts_begin + j2];
        const double target_z = mol_interp_z[target_cluster_interp_pts_begin + j3];

        double pot_temp = 0.0;
        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {
          const double dx = target_x - mol_view.particles_x[k];
          const double dy = target_y - mol_view.particles_y[k];
          const double dz = target_z - mol_view.particles_z[k];
          pot_temp += mol_view.charge[k] / params.eps_solute / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p[jj] += pot_temp;
      }
    }
  }
}

void coulombic_cluster_cluster_cpu(
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    std::size_t target_node_idx,
    std::size_t source_node_idx,
    const CoulombicBackendParams& params) {
  const int n_interp = params.num_mol_interp_pts_per_node;
  const int n_charge = params.num_mol_interp_charges_per_node;
  const int n_pot = params.num_mol_interp_potentials_per_node;

  const std::size_t target_cluster_interp_pts_begin = target_node_idx * n_interp;
  const std::size_t target_cluster_interp_potentials_begin = target_node_idx * n_pot;

  const std::size_t source_cluster_interp_pts_begin = source_node_idx * n_interp;
  const std::size_t source_cluster_interp_charges_begin = source_node_idx * n_charge;

  for (int j1 = 0; j1 < n_interp; ++j1) {
    for (int j2 = 0; j2 < n_interp; ++j2) {
      for (int j3 = 0; j3 < n_interp; ++j3) {
        const std::size_t jj = target_cluster_interp_potentials_begin +
                               static_cast<std::size_t>(j1 * n_interp * n_interp +
                                                        j2 * n_interp + j3);

        const double target_x = mol_interp_x[target_cluster_interp_pts_begin + j1];
        const double target_y = mol_interp_y[target_cluster_interp_pts_begin + j2];
        const double target_z = mol_interp_z[target_cluster_interp_pts_begin + j3];

        double pot_temp = 0.0;
        for (int k1 = 0; k1 < n_interp; ++k1) {
          for (int k2 = 0; k2 < n_interp; ++k2) {
            for (int k3 = 0; k3 < n_interp; ++k3) {
              const std::size_t kk = source_cluster_interp_charges_begin +
                                     static_cast<std::size_t>(k1 * n_interp * n_interp +
                                                              k2 * n_interp + k3);

              const double dx = target_x - mol_interp_x[source_cluster_interp_pts_begin + k1];
              const double dy = target_y - mol_interp_y[source_cluster_interp_pts_begin + k2];
              const double dz = target_z - mol_interp_z[source_cluster_interp_pts_begin + k3];

              pot_temp += self_view.q[kk] / params.eps_solute / std::sqrt(dx * dx + dy * dy + dz * dz);
            }
          }
        }
#ifdef OPENMP_ENABLED
#pragma omp atomic update
#endif
        self_view.p[jj] += pot_temp;
      }
    }
  }
}

void coulombic_upward_pass_cpu(
    const Molecule::View& mol_view,
    const double* mol_interp_x,
    const double* mol_interp_y,
    const double* mol_interp_z,
    CoulombicEnergyCompute::DeviceView& self_view,
    const Tree& source_tree,
    const CoulombicBackendParams& params) {
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

        denominator_x += self_view.weights[j] / dist_x;
        denominator_y += self_view.weights[j] / dist_y;
        denominator_z += self_view.weights[j] / dist_z;

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
          const double w1 = self_view.weights[k1];
          const double cy = mol_interp_y[node_interp_pts_start + k2];
          const double w2 = self_view.weights[k2];
          const double cz = mol_interp_z[node_interp_pts_start + k3];
          const double w3 = self_view.weights[k3];

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

            q_temp += mol_view.charge[particle_start + i] * numerator * self_view.denominator[i];
          }

          self_view.q[kk] += q_temp;
        }
      }
    }
  }
}
