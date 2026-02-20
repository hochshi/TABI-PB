#ifndef H_TABIPB_SOLVATION_BACKEND_COMMON_H
#define H_TABIPB_SOLVATION_BACKEND_COMMON_H

#include <cstddef>

struct SolvationBackendParams {
  double eps = 0.0;
  double kappa = 0.0;
  int num_elem_interp_pts_per_node = 0;
  int num_elem_interp_potentials_per_node = 0;
  int num_mol_interp_pts_per_node = 0;
  int num_mol_interp_charges_per_node = 0;
  std::size_t potential_offset = 0;

  SolvationBackendParams() = default;
  SolvationBackendParams(double eps_,
                         double kappa_,
                         int num_elem_interp_pts_per_node_,
                         int num_elem_interp_potentials_per_node_,
                         int num_mol_interp_pts_per_node_,
                         int num_mol_interp_charges_per_node_,
                         std::size_t potential_offset_)
      : eps(eps_),
        kappa(kappa_),
        num_elem_interp_pts_per_node(num_elem_interp_pts_per_node_),
        num_elem_interp_potentials_per_node(
            num_elem_interp_potentials_per_node_),
        num_mol_interp_pts_per_node(num_mol_interp_pts_per_node_),
        num_mol_interp_charges_per_node(num_mol_interp_charges_per_node_),
        potential_offset(potential_offset_) {}
};

#endif
