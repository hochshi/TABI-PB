#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "constants.h"
#include "params.h"

Params::Params(char *infile) {
  std::ifstream paramfile(infile, std::ifstream::in);
  if (!paramfile.good()) {
    std::cout << "param file is not readable. exiting. " << std::endl;
    std::exit(1);
  }

  output_vtk_ = false;
  output_ply_ = false;
  output_csv_ = false;
  output_csv_headers_ = false;
  output_timers_ = false;
  precondition_ = false;
  output_prefix_ = "output";
  input_mesh_prefix_ = "";

  mesh_ = Params::Mesh::SES;
  mesh_format_ = Params::MeshFormat::MSMS;

  std::string line;

  while (std::getline(paramfile, line)) {

    std::istringstream iss(line);
    std::vector<std::string> tokenized_line{
        std::istream_iterator<std::string>{iss},
        std::istream_iterator<std::string>{}};

    std::string param_token = tokenized_line[0];
    std::string param_value = tokenized_line[1];

    std::transform(param_token.begin(), param_token.end(), param_token.begin(),
                   [=](unsigned char c) { return std::tolower(c); });

    std::transform(param_value.begin(), param_value.end(), param_value.begin(),
                   [=](unsigned char c) { return std::tolower(c); });

    if (param_token == "mol" || param_token == "pqr") {
      pqr_file_.open(tokenized_line[1], std::ifstream::in);
      if (!pqr_file_.good()) {
        std::cout << "pqr file is not readable. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "pdie") {
      phys_eps_solute_ = std::stod(param_value);

    } else if (param_token == "sdie") {
      phys_eps_solvent_ = std::stod(param_value);

    } else if (param_token == "bulk") {
      phys_bulk_strength_ = std::stod(param_value);

    } else if (param_token == "temp") {
      phys_temp_ = std::stod(param_value);

    } else if (param_token == "tree_degree") {
      tree_degree_ = std::stoi(param_value);
      if (tree_degree_ <= 0) {
        std::cout << "invalid tree_degree value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "tree_theta") {
      tree_theta_ = std::stod(param_value);
      if (tree_theta_ < 0. || tree_theta_ > 1.) {
        std::cout << "invalid tree_theta value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "tree_max_per_leaf") {
      tree_max_per_leaf_ = std::stoi(param_value);
      if (tree_max_per_leaf_ <= 0) {
        std::cout << "invalid tree_max_per_leaf value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "gmres_restart") {
      gmres_restart_ = std::stoi(param_value);
      if (gmres_restart_ <= 0) {
        std::cout << "invalid GMRES restart value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "gmres_residual") {
      gmres_residual_ = std::stod(param_value);
      if (gmres_residual_ < 0. || gmres_residual_ > 1.) {
        std::cout << "invalid gmres_residual value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "gmres_num_iter") {
      gmres_num_iter_ = std::stoi(param_value);
      if (gmres_num_iter_ <= 0) {
        std::cout << "invalid gmres_num_iter value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "mesh") {
      auto it = mesh_table_.find(param_value);
      if (it == mesh_table_.end()) {
        std::cout << "invalid mesh value. exiting. " << std::endl;
        std::exit(1);
      }
      mesh_ = it->second;

    } else if (param_token == "mesh_format") {
      auto it = mesh_format_table_.find(param_value);
      if (it == mesh_format_table_.end()) {
        std::cout << "invalid mesh value. exiting. " << std::endl;
        std::exit(1);
      }
      mesh_format_ = it->second;

    } else if (param_token == "sdens") {
      mesh_density_ = std::stod(param_value);
      if (mesh_density_ < 0) {
        std::cout << "invalid density value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "srad") {
      mesh_probe_radius_ = std::stod(param_value);
      if (mesh_probe_radius_ < 0) {
        std::cout << "invalid probe radius value. exiting. " << std::endl;
        std::exit(1);
      }

    } else if (param_token == "precondition") {
      if (param_value == "true" || param_value == "on")
        precondition_ = true;

    } else if (param_token == "nonpolar") {
      if (param_value == "true")
        nonpolar_ = true;

    } else if (param_token == "outdata") {
      if (param_value == "vtk")
        output_vtk_ = true;
      if (param_value == "ply")
        output_ply_ = true;
      if (param_value == "csv")
        output_csv_ = true;
      if (param_value == "csv_headers")
        output_csv_headers_ = true;
      if (param_value == "timers")
        output_timers_ = true;

    } else if (param_token == "output_prefix") {
      if (!param_value.empty())
        output_prefix_ = param_value;

    } else if (param_token == "input_mesh_prefix") {
      if (!param_value.empty())
        input_mesh_prefix_ = param_value;

    } else {
      std::cout << "Skipping undefined token: " << param_token << std::endl;
    }
  }

  phys_eps_ = phys_eps_solvent_ / phys_eps_solute_;
  phys_kappa2_ = constants::BULK_COEFF * phys_bulk_strength_ /
                 phys_eps_solvent_ / phys_temp_;
  phys_kappa_ = std::sqrt(phys_kappa2_);
}
