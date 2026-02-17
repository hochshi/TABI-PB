#ifndef H_TABIPB_COULOMBIC_BACKEND_COMMON_H
#define H_TABIPB_COULOMBIC_BACKEND_COMMON_H

#include <cstddef>

struct CoulombicBackendParams {
  double eps_solute = 0.0;
  int num_mol_interp_pts_per_node = 0;
  int num_mol_interp_charges_per_node = 0;
  int num_mol_interp_potentials_per_node = 0;

  CoulombicBackendParams() = default;
  CoulombicBackendParams(double eps_solute_,
                         int num_mol_interp_pts_per_node_,
                         int num_mol_interp_charges_per_node_,
                         int num_mol_interp_potentials_per_node_)
      : eps_solute(eps_solute_),
        num_mol_interp_pts_per_node(num_mol_interp_pts_per_node_),
        num_mol_interp_charges_per_node(num_mol_interp_charges_per_node_),
        num_mol_interp_potentials_per_node(num_mol_interp_potentials_per_node_) {}
};

#endif
