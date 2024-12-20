#ifndef H_TABIPB_PARAMS_STRUCT_H
#define H_TABIPB_PARAMS_STRUCT_H

#include <fstream>
#include <string>
#include <unordered_map>

#ifdef TABIPB_APBS
#include "tabipb_wrap/TABIPBStruct.h"
#endif

struct Params {
  enum Mesh { SES, SKIN };
  enum MeshFormat { MSMS, PLY };

  std::unordered_map<std::string, enum Mesh> const mesh_table_ = {
      {"ses", Mesh::SES}, {"skin", Mesh::SKIN}};

  std::unordered_map<std::string, enum MeshFormat> const mesh_format_table_ = {
      {"msms", MeshFormat::MSMS}, {"ply", MeshFormat::PLY}};

  /* pqr file location */
  std::ifstream pqr_file_;

  /* mesh settings */
  enum Mesh mesh_;
  enum MeshFormat mesh_format_;
  double mesh_density_;
  double mesh_probe_radius_;

  /* physical parameters */
  double phys_temp_;
  double phys_eps_solute_;
  double phys_eps_solvent_;
  double phys_bulk_strength_;

  /* set and used locally */
  double phys_eps_;
  double phys_kappa_;
  double phys_kappa2_;

  /* boundary_element parameters */
  int tree_degree_;
  int tree_max_per_leaf_;
  double tree_theta_;

  /* preconditioning */
  bool precondition_;

  /* GMRES */
  long int gmres_restart_ = 10;
  double gmres_residual_   = 1e-4;
  long int gmres_num_iter_ = 1000;

  /* nonpolar energy */
  int nonpolar_;

  /* output of potential data */
  bool output_vtk_;
  bool output_ply_;
  bool output_csv_;
  bool output_csv_headers_;
  bool output_timers_;

  std::string output_prefix_;
  std::string input_mesh_prefix_;

  Params(char *paramfile);
  ~Params() = default;

#ifdef TABIPB_APBS
  Params(TABIPBInput tabipbIn);
#endif
};

#endif /* H_TABIPB_PARAMS_STRUCT_H */
