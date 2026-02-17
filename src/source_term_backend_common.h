#ifndef H_TABIPB_SOURCE_TERM_BACKEND_COMMON_H
#define H_TABIPB_SOURCE_TERM_BACKEND_COMMON_H

#include <cstddef>

struct SourceTermBackendParams {
  double one_over_4pi_eps_solute = 0.0;
  int num_elem_interp_pts_per_node = 0;
  int num_elem_interp_potentials_per_node = 0;
  int num_mol_interp_pts_per_node = 0;
  int num_mol_interp_charges_per_node = 0;
  std::size_t source_term_offset = 0;

  SourceTermBackendParams() = default;
  SourceTermBackendParams(double one_over_4pi_eps_solute_,
                          int num_elem_interp_pts_per_node_,
                          int num_elem_interp_potentials_per_node_,
                          int num_mol_interp_pts_per_node_,
                          int num_mol_interp_charges_per_node_,
                          std::size_t source_term_offset_)
      : one_over_4pi_eps_solute(one_over_4pi_eps_solute_),
        num_elem_interp_pts_per_node(num_elem_interp_pts_per_node_),
        num_elem_interp_potentials_per_node(
            num_elem_interp_potentials_per_node_),
        num_mol_interp_pts_per_node(num_mol_interp_pts_per_node_),
        num_mol_interp_charges_per_node(num_mol_interp_charges_per_node_),
        source_term_offset(source_term_offset_) {}
};

#endif
