#include "params.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#ifdef PLY_ENABLED
#include <tinyply.h>
#endif // PLY_ENABLED

#include "constants.h"
#include "elements.h"
#include "source_term_compute.h"
#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#include "elements_cuda.h"
#endif

#ifdef OPENACC_ENABLED
#include <openacc.h>
#endif

static double triangle_area(std::array<std::array<double, 3>, 3> v);

Elements::Elements(const class Molecule &mol, const struct Params &params,
                   struct Timers_Elements &timers)
    : Particles(params), molecule_(mol), timers_(timers) {
  timers_.ctor.start();

  Elements::generate_elements(params_.mesh_, params_.mesh_format_,
                              params_.mesh_density_, params_.mesh_probe_radius_,
                              params_.input_mesh_prefix_);

  source_charge_.assign(num_, 0.);
  source_charge_dx_.assign(num_, 0.);
  source_charge_dy_.assign(num_, 0.);
  source_charge_dz_.assign(num_, 0.);

  target_charge_.assign(num_, 0.);
  target_charge_dx_.assign(num_, 0.);
  target_charge_dy_.assign(num_, 0.);
  target_charge_dz_.assign(num_, 0.);

  source_term_.assign(num_ * 2, 0.);

  order_.resize(num_);
  std::iota(order_.begin(), order_.end(), 0);

  timers_.ctor.stop();
}

bool Elements::file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

bool Elements::read_ply_file(const std::string &filepath) {
#ifdef PLY_ENABLED
  struct double3 {
    double x, y, z;
  };
  struct uint3 {
    uint32_t x, y, z;
  };
  std::unique_ptr<std::istream> file_stream;
  std::vector<uint8_t> byte_buffer;

  try {
    file_stream.reset(new std::ifstream(filepath, std::ios::binary));

    if (!file_stream || file_stream->fail())
      throw std::runtime_error("file_stream failed to open " + filepath);

    file_stream->seekg(0, std::ios::end);
    const float size_mb = file_stream->tellg() * float(1e-6);
    file_stream->seekg(0, std::ios::beg);

    tinyply::PlyFile file;
    file.parse_header(*file_stream);

    std::cout << "\t[ply_header] Type: "
              << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto &c : file.get_comments())
      std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto &c : file.get_info())
      std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : file.get_elements()) {
      std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")"
                << std::endl;
      for (const auto &p : e.properties) {
        std::cout << "\t[ply_header] \tproperty: " << p.name
                  << " (type=" << tinyply::PropertyTable[p.propertyType].str
                  << ")";
        if (p.isList)
          std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str
                    << ")";
        std::cout << std::endl;
      }
    }

    // Because most people have their own mesh types, tinyply treats parsed data
    // as structured/typed byte buffers. See examples below on how to marry your
    // own application-specific data structures with this one.
    std::shared_ptr<tinyply::PlyData> vertices, normals, faces;

    // The header information can be used to programmatically extract properties
    // on elements known to exist in the header prior to reading the data. For
    // brevity of this sample, properties like vertex position are hard-coded:
    try {
      vertices =
          file.request_properties_from_element("vertex", {"x", "y", "z"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    try {
      normals =
          file.request_properties_from_element("vertex", {"nx", "ny", "nz"});
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    // Providing a list size hint (the last argument) is a 2x performance
    // improvement. If you have arbitrary ply files, it is best to leave this 0.
    try {
      faces =
          file.request_properties_from_element("face", {"vertex_indices"}, 0);
    } catch (const std::exception &e) {
      std::cerr << "tinyply exception: " << e.what() << std::endl;
    }

    file.read(*file_stream);

    if (vertices)
      std::cout << "\tRead " << vertices->count << " total vertices "
                << std::endl;
    if (normals)
      std::cout << "\tRead " << normals->count << " total vertex normals "
                << std::endl;
    if (faces)
      std::cout << "\tRead " << faces->count << " total faces (triangles) "
                << std::endl;

    if (vertices) {
      num_ = vertices->count;

      x_.reserve(num_);
      y_.reserve(num_);
      z_.reserve(num_);
      const size_t numVerticesBytes = vertices->buffer.size_bytes();
      std::vector<double3> verts(vertices->count);
      std::memcpy(verts.data(), vertices->buffer.get(), numVerticesBytes);
      for (auto &triplet : verts) {
        x_.push_back(triplet.x);
        y_.push_back(triplet.y);
        z_.push_back(triplet.z);
      }
    }
    if (normals) {
      nx_.reserve(num_);
      ny_.reserve(num_);
      nz_.reserve(num_);
      const size_t numNormalsBytes = normals->buffer.size_bytes();
      std::vector<double3> normalsVec(normals->count);
      std::memcpy(normalsVec.data(), normals->buffer.get(), numNormalsBytes);
      for (auto &triplet : normalsVec) {
        nx_.push_back(triplet.x);
        ny_.push_back(triplet.y);
        nz_.push_back(triplet.z);
      }
    }

    if (faces) {
      num_faces_ = faces->count;

      face_x_.reserve(num_faces_);
      face_y_.reserve(num_faces_);
      face_z_.reserve(num_faces_);
      const size_t numFacesBytes = faces->buffer.size_bytes();
      std::vector<uint3> facesVec(faces->count);
      std::memcpy(facesVec.data(), faces->buffer.get(), numFacesBytes);
      for (uint3 &triplet : facesVec) {
        face_x_.push_back(static_cast<uint32_t>(triplet.x));
        face_y_.push_back(static_cast<uint32_t>(triplet.y));
        face_z_.push_back(static_cast<uint32_t>(triplet.z));
      }
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
  }
#endif
  return false;
}

bool Elements::read_msms_file(const std::string &input_mesh_prefix) {
  std::string line;

  // Read in the vert file
  std::cout << "Reading " << input_mesh_prefix << ".vert" << std::endl;
  std::ifstream vert_file(input_mesh_prefix + ".vert");
  // Throw away first two lines
  std::getline(vert_file, line);
  std::getline(vert_file, line);

  std::getline(vert_file, line);
  try {
    num_ = std::stoul(line);
    std::cout << "Reading " << num_ <<" vertices" << std::endl;

    x_.reserve(num_);
    y_.reserve(num_);
    z_.reserve(num_);
    nx_.reserve(num_);
    ny_.reserve(num_);
    nz_.reserve(num_);

    while (std::getline(vert_file, line)) {

      std::istringstream iss(line);
      std::vector<std::string> tokenized_line{
          std::istream_iterator<std::string>{iss},
          std::istream_iterator<std::string>{}};

      x_.push_back(std::stod(tokenized_line[0]));
      y_.push_back(std::stod(tokenized_line[1]));
      z_.push_back(std::stod(tokenized_line[2]));
      nx_.push_back(std::stod(tokenized_line[3]));
      ny_.push_back(std::stod(tokenized_line[4]));
      nz_.push_back(std::stod(tokenized_line[5]));
  }
  } catch (std::invalid_argument const& ex) {
     std::cout << "Vert: line: |" << line << "| caused invalid argument exception" << std::endl;
     throw;
  }

  vert_file.close();

  // Read in the face file
  std::cout << "Reading " << input_mesh_prefix << ".face" << std::endl;
  std::ifstream face_file(input_mesh_prefix + ".face");
  // Throw away first two lines
  std::getline(face_file, line);
  std::getline(face_file, line);

  std::getline(face_file, line);
  try{
    num_faces_ = std::stoul(line);
    std::cout << "Reading " << num_faces_ <<" faces" << std::endl;

    face_x_.reserve(num_faces_);
    face_y_.reserve(num_faces_);
    face_z_.reserve(num_faces_);

    while (std::getline(face_file, line)) {

      std::istringstream iss(line);
      std::vector<std::string> tokenized_line{
          std::istream_iterator<std::string>{iss},
          std::istream_iterator<std::string>{}};

      face_x_.push_back(std::stoul(tokenized_line[0]));
      face_y_.push_back(std::stoul(tokenized_line[1]));
      face_z_.push_back(std::stoul(tokenized_line[2]));
  }
  } catch (std::invalid_argument const& ex) {
     std::cout << "Face: line: |" << line << "| caused invalid argument exception" << std::endl;
     throw;
  }

  face_file.close();

  return true;
}

void Elements::write_nanaoshaper_config(Params::Mesh mesh,
                                        Params::MeshFormat mesh_format,
                                        double mesh_density,
                                        double probe_radius) {
  std::ofstream NS_param_file("surfaceConfiguration.prm");

  NS_param_file << "Grid_scale = " << mesh_density << std::endl;
  NS_param_file << "Grid_perfil = " << 90.0 << std::endl;
  NS_param_file << "XYZR_FileName = " << "molecule.xyzr" << std::endl;
  NS_param_file << "Build_epsilon_maps = " << "false" << std::endl;
  NS_param_file << "Build_status_map = " << "false" << std::endl;

  if (Params::MeshFormat::PLY == mesh_format) {
    NS_param_file << "Save_Mesh_PLY_Format = " << "true" << std::endl;
  } else {
    NS_param_file << "Save_Mesh_MSMS_Format = " << "true" << std::endl;
  }

  NS_param_file << "Compute_Vertex_Normals = " << "true" << std::endl;

  if (mesh == Params::Mesh::SES)
    NS_param_file << "Surface = ses" << std::endl;
  if (mesh == Params::Mesh::SKIN)
    NS_param_file << "Surface = skin" << std::endl;

  NS_param_file << "Smooth_Mesh = " << "true" << std::endl;
  NS_param_file << "Skin_Surface_Parameter = " << 0.45 << std::endl;
  NS_param_file << "Cavity_Detection_Filling = " << "false" << std::endl;
  NS_param_file << "Conditional_Volume_Filling_Value = " << 11.4 << std::endl;
  NS_param_file << "Keep_Water_Shaped_Cavities = " << "false" << std::endl;
  NS_param_file << "Probe_Radius = " << probe_radius << std::endl;
  NS_param_file << "Accurate_Triangulation = " << "true" << std::endl;
  NS_param_file << "Triangulation = " << "true" << std::endl;
  NS_param_file << "Check_duplicated_vertices = " << "true" << std::endl;
  NS_param_file << "Save_Status_map = " << "false" << std::endl;
  NS_param_file << "Save_PovRay = " << "false" << std::endl;
  NS_param_file << "Max_ses_patches_per_auxiliary_grid_2d_cell = " << 1600
                << std::endl;
  NS_param_file << "Max_ses_patches_auxiliary_grid_2d_size = " << 50
                << std::endl;

  NS_param_file.close();
}

void Elements::generate_elements(Params::Mesh mesh,
                                 Params::MeshFormat mesh_format,
                                 double mesh_density, double probe_radius,
                                 const std::string &input_mesh_prefix) {
  std::string input_mesh_file_name = "";
  if (input_mesh_prefix.empty()) {
    // Gotta write the files and run NanoShaper
    input_mesh_file_name = "triangulatedSurf";
    write_nanaoshaper_config(mesh, mesh_format, mesh_density, probe_radius);
#ifdef _WIN32
    std::system("NanoShaper.exe");
#else
    std::system("NanoShaper");
#endif

    std::remove("stderror.txt");
    std::remove("surfaceConfiguration.prm");
    std::remove("triangleAreas.txt");
    std::remove("exposed.xyz");
    std::remove("exposedIndices.txt");
  } else {
    input_mesh_file_name = input_mesh_prefix;
  }

  if (Params::MeshFormat::PLY == mesh_format) {
    read_ply_file(input_mesh_file_name + ".ply");
  } else {
    read_msms_file(input_mesh_file_name);
  }

  if (input_mesh_prefix.empty()) {
    if (Params::MeshFormat::PLY == mesh_format) {
      std::remove("triangulatedSurf.ply");
    } else {
      std::remove("triangulatedSurf.vert");
      std::remove("triangulatedSurf.face");
    }
    std::remove("molecule.xyzr");
  }

  area_.assign(num_, 0.);

  int face_vertex_index_shift = 1;
  if (Params::MeshFormat::PLY == mesh_format) {
    face_vertex_index_shift = 0;
  } else {
    face_vertex_index_shift = 1;
  }

  for (std::size_t i = 0; i < num_faces_; ++i) {
    std::array<uint32_t, 3> iface{face_x_[i], face_y_[i], face_z_[i]};
    std::array<std::array<double, 3>, 3> r;

    for (int ii = 0; ii < 3; ++ii) {
      r[0][ii] = x_[iface[ii] - face_vertex_index_shift];
      r[1][ii] = y_[iface[ii] - face_vertex_index_shift];
      r[2][ii] = z_[iface[ii] - face_vertex_index_shift];
    }

    for (int j = 0; j < 3; ++j) {
      area_[iface[j] - face_vertex_index_shift] += triangle_area(r);
    }
  }

  std::transform(area_.begin(), area_.end(), area_.begin(),
                 [=](double x) { return x / 3.; });
  surface_area_ = std::accumulate(area_.begin(), area_.end(),
                                  decltype(area_)::value_type(0));
  std::cout << "Surface area of triangulated mesh is " << surface_area_ << ". "
            << std::endl
            << std::endl;
}

void Elements::compute_source_term() {
  /* this computes the source term where
   * S1=sum(qk*G0)/e1 S2=sim(qk*G0')/e1 */
  timers_.compute_source_term.start();

  double eps_solute = params_.phys_eps_solute_;

  std::size_t num_atoms = molecule_.num();
  std::size_t num = num_;

  const double *__restrict elements_x_ptr = x_.data();
  const double *__restrict elements_y_ptr = y_.data();
  const double *__restrict elements_z_ptr = z_.data();

  const double *__restrict elements_nx_ptr = nx_.data();
  const double *__restrict elements_ny_ptr = ny_.data();
  const double *__restrict elements_nz_ptr = nz_.data();

  const double *__restrict molecule_x_ptr = molecule_.x_ptr();
  const double *__restrict molecule_y_ptr = molecule_.y_ptr();
  const double *__restrict molecule_z_ptr = molecule_.z_ptr();
  const double *__restrict molecule_charge_ptr = molecule_.charge_ptr();

  double *__restrict elements_source_term_ptr = source_term_.data();

#ifdef OPENACC_ENABLED
#ifdef USE_CUDA_CC
  {
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool present_ok = validate_device_buffers_compute_source_term_();
    if (present_ok) {
      acc_wait(acc_async_sync);
      void* stream = acc_get_cuda_stream(acc_async_sync);
      #pragma acc host_data use_device(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                       elements_nx_ptr, elements_ny_ptr, elements_nz_ptr, \
                                       molecule_x_ptr, molecule_y_ptr, molecule_z_ptr, \
                                       molecule_charge_ptr, elements_source_term_ptr)
      {
        elements_compute_source_term_cuda(elements_x_ptr, elements_y_ptr, elements_z_ptr,
                                          elements_nx_ptr, elements_ny_ptr, elements_nz_ptr,
                                          molecule_x_ptr, molecule_y_ptr, molecule_z_ptr,
                                          molecule_charge_ptr, elements_source_term_ptr,
                                          num, num_atoms, eps_solute, stream);
      }
      Elements::update_source_term_on_host();
      timers_.compute_source_term.stop();
      return;
    }
    if (require_all) {
      std::cerr << "[CUDA_ELEM] require_all set but device pointers not present. "
                << "Aborting to avoid OpenACC fallback.\n";
      std::exit(1);
    }
  }
#endif
#endif
#ifdef OPENMP_ENABLED
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < num; ++i) {

    double source_term_1 = 0.;
    double source_term_2 = 0.;

    for (std::size_t j = 0; j < num_atoms; ++j) {

      /* r_s = distance of charge position to triangular */
      double x_dist = molecule_x_ptr[j] - elements_x_ptr[i];
      double y_dist = molecule_y_ptr[j] - elements_y_ptr[i];
      double z_dist = molecule_z_ptr[j] - elements_z_ptr[i];
      double dist =
          std::sqrt(x_dist * x_dist + y_dist * y_dist + z_dist * z_dist);

      /* cos_theta = <tr_q,r_s>/||r_s||_2 */
      double cos_theta =
          (elements_nx_ptr[i] * x_dist + elements_ny_ptr[i] * y_dist +
           elements_nz_ptr[i] * z_dist) /
          dist;

      /* G0 = 1/(4pi*||r_s||_2) */
      double G0 = constants::ONE_OVER_4PI / dist;

      /* G1 = cos_theta*G0/||r_s||_2 */
      double G1 = cos_theta * G0 / dist;

      /* update source term */
      source_term_1 += molecule_charge_ptr[j] * G0 / eps_solute;
      source_term_2 += molecule_charge_ptr[j] * G1 / eps_solute;
    }

    elements_source_term_ptr[i] += source_term_1;
    elements_source_term_ptr[num + i] += source_term_2;
  }

  Elements::update_source_term_on_host();

  timers_.compute_source_term.stop();
}

void Elements::compute_source_term(
    const class InterpolationPoints &elem_interp_pts,
    const class Tree &elem_tree, const class Molecule &molecule,
    const class InterpolationPoints &mol_interp_pts, const class Tree &mol_tree,
    const class InteractionList &interaction_list) {
  /* this computes the source term where
   * S1=sum(qk*G0)/e1 S2=sim(qk*G0')/e1 */
  timers_.compute_source_term.start();

  class SourceTermCompute source_term(
      source_term_, *this, elem_interp_pts, elem_tree, molecule, mol_interp_pts,
      mol_tree, interaction_list, params_.phys_eps_solute_);

  source_term.compute();
  Elements::update_source_term_on_host();

  timers_.compute_source_term.stop();
}

void Elements::reorder() {
  apply_order(order_.begin(), order_.end(), nx_.begin());
  apply_order(order_.begin(), order_.end(), ny_.begin());
  apply_order(order_.begin(), order_.end(), nz_.begin());

  apply_order(order_.begin(), order_.end(), area_.begin());
  apply_order(order_.begin(), order_.end(), source_term_.begin());
  apply_order(order_.begin(), order_.end(), source_term_.begin() + num_);
}

void Elements::unorder() {
  apply_unorder(order_.begin(), order_.end(), x_.begin());
  apply_unorder(order_.begin(), order_.end(), y_.begin());
  apply_unorder(order_.begin(), order_.end(), z_.begin());

  apply_unorder(order_.begin(), order_.end(), nx_.begin());
  apply_unorder(order_.begin(), order_.end(), ny_.begin());
  apply_unorder(order_.begin(), order_.end(), nz_.begin());

  apply_unorder(order_.begin(), order_.end(), area_.begin());
  apply_unorder(order_.begin(), order_.end(), source_term_.begin());
  apply_unorder(order_.begin(), order_.end(), source_term_.begin() + num_);
}

void Elements::unorder(std::vector<double> &potential) {
  apply_unorder(order_.begin(), order_.end(), x_.begin());
  apply_unorder(order_.begin(), order_.end(), y_.begin());
  apply_unorder(order_.begin(), order_.end(), z_.begin());

  apply_unorder(order_.begin(), order_.end(), nx_.begin());
  apply_unorder(order_.begin(), order_.end(), ny_.begin());
  apply_unorder(order_.begin(), order_.end(), nz_.begin());

  apply_unorder(order_.begin(), order_.end(), area_.begin());
  apply_unorder(order_.begin(), order_.end(), source_term_.begin());
  apply_unorder(order_.begin(), order_.end(), source_term_.begin() + num_);

  apply_unorder(order_.begin(), order_.end(), potential.begin());
  apply_unorder(order_.begin(), order_.end(), potential.begin() + num_);
}

void Elements::compute_charges(const double *__restrict potential_ptr) {
  timers_.compute_charges.start();

  std::size_t num = num_;

  const double *__restrict nx_ptr = nx_.data();
  const double *__restrict ny_ptr = ny_.data();
  const double *__restrict nz_ptr = nz_.data();
  const double *__restrict area_ptr = area_.data();

  double *__restrict target_q_ptr = target_charge_.data();
  double *__restrict target_q_dx_ptr = target_charge_dx_.data();
  double *__restrict target_q_dy_ptr = target_charge_dy_.data();
  double *__restrict target_q_dz_ptr = target_charge_dz_.data();

  double *__restrict source_q_ptr = source_charge_.data();
  double *__restrict source_q_dx_ptr = source_charge_dx_.data();
  double *__restrict source_q_dy_ptr = source_charge_dy_.data();
  double *__restrict source_q_dz_ptr = source_charge_dz_.data();

#ifdef OPENACC_ENABLED
#ifdef USE_CUDA_CC
  {
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool present_ok = validate_device_buffers_compute_charges_();
    if (present_ok) {
      acc_wait(acc_async_sync);
      void* stream = acc_get_cuda_stream(acc_async_sync);
      #pragma acc host_data use_device(nx_ptr, ny_ptr, nz_ptr, area_ptr, potential_ptr, \
                                       target_q_ptr, target_q_dx_ptr, target_q_dy_ptr, target_q_dz_ptr, \
                                       source_q_ptr, source_q_dx_ptr, source_q_dy_ptr, source_q_dz_ptr)
      {
        elements_compute_charges_cuda(nx_ptr, ny_ptr, nz_ptr, area_ptr, potential_ptr,
                                      target_q_ptr, target_q_dx_ptr, target_q_dy_ptr, target_q_dz_ptr,
                                      source_q_ptr, source_q_dx_ptr, source_q_dy_ptr, source_q_dz_ptr,
                                      num, stream);
      }
      timers_.compute_charges.stop();
      return;
    }
    if (require_all) {
      std::cerr << "[CUDA_ELEM] require_all set but device pointers not present. "
                << "Aborting to avoid OpenACC fallback.\n";
      std::exit(1);
    }
  }
#endif
#elif OPENMP_ENABLED
#pragma omp parallel for
#endif
  for (std::size_t i = 0; i < num; ++i) {
    target_q_ptr[i] = constants::ONE_OVER_4PI;
    target_q_dx_ptr[i] = constants::ONE_OVER_4PI * nx_ptr[i];
    target_q_dy_ptr[i] = constants::ONE_OVER_4PI * ny_ptr[i];
    target_q_dz_ptr[i] = constants::ONE_OVER_4PI * nz_ptr[i];

    source_q_ptr[i] = area_ptr[i] * potential_ptr[num + i];
    source_q_dx_ptr[i] = nx_ptr[i] * area_ptr[i] * potential_ptr[i];
    source_q_dy_ptr[i] = ny_ptr[i] * area_ptr[i] * potential_ptr[i];
    source_q_dz_ptr[i] = nz_ptr[i] * area_ptr[i] * potential_ptr[i];
  }

  timers_.compute_charges.stop();
}

Timer& Elements::compute_charges_timer() {
  return timers_.compute_charges;
}

#ifdef USE_CUDA_CC
bool Elements::validate_device_buffers_compute_source_term_() const {
  if (!cuda_device_ready() || !molecule_.cuda_device_ready()) {
    return false;
  }
  const auto& buf = device_buffers_;
  return buf.x && buf.y && buf.z &&
         buf.nx && buf.ny && buf.nz &&
         buf.source_term &&
         buf.num == num_;
}

bool Elements::validate_device_buffers_compute_charges_() const {
  if (!cuda_device_ready()) {
    return false;
  }
  const auto& buf = device_buffers_;
  return buf.nx && buf.ny && buf.nz &&
         buf.area && buf.target_q &&
         buf.target_q_dx && buf.target_q_dy && buf.target_q_dz &&
         buf.source_q && buf.source_q_dx &&
         buf.source_q_dy && buf.source_q_dz &&
         buf.num == num_;
}

void Elements::reset_device_buffers_() const {
  device_buffers_ = DeviceBuffers{};
  device_state_ = CudaDeviceState::HostOnly;
}
#endif

void Elements::copyin_to_device() const {
  timers_.copyin_to_device.start();

#ifdef USE_CUDA_CC
  const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
  const bool require_all =
      (require_all_env && std::strcmp(require_all_env, "0") != 0);
  const std::size_t num = num_;
  const std::size_t x_num = x_.size();
  const std::size_t y_num = y_.size();
  const std::size_t z_num = z_.size();
  const std::size_t nx_num = nx_.size();
  const std::size_t ny_num = ny_.size();
  const std::size_t nz_num = nz_.size();
  const std::size_t area_num = area_.size();
  const std::size_t source_term_num = source_term_.size();
  const std::size_t tq_num = target_charge_.size();
  const std::size_t tq_dx_num = target_charge_dx_.size();
  const std::size_t tq_dy_num = target_charge_dy_.size();
  const std::size_t tq_dz_num = target_charge_dz_.size();
  const std::size_t sq_num = source_charge_.size();
  const std::size_t sq_dx_num = source_charge_dx_.size();
  const std::size_t sq_dy_num = source_charge_dy_.size();
  const std::size_t sq_dz_num = source_charge_dz_.size();

  if (x_num != num || y_num != num || z_num != num ||
      nx_num != num || ny_num != num || nz_num != num ||
      area_num != num ||
      tq_num != num || tq_dx_num != num || tq_dy_num != num || tq_dz_num != num ||
      sq_num != num || sq_dx_num != num || sq_dy_num != num || sq_dz_num != num ||
      source_term_num != 2 * num) {
    std::fprintf(stderr,
                 "[CUDA_ELEMENTS] size mismatch: num=%zu x=%zu y=%zu z=%zu "
                 "nx=%zu ny=%zu nz=%zu area=%zu source_term=%zu "
                 "tq=%zu tq_dx=%zu tq_dy=%zu tq_dz=%zu "
                 "sq=%zu sq_dx=%zu sq_dy=%zu sq_dz=%zu\n",
                 num, x_num, y_num, z_num, nx_num, ny_num, nz_num, area_num,
                 source_term_num, tq_num, tq_dx_num, tq_dy_num, tq_dz_num,
                 sq_num, sq_dx_num, sq_dy_num, sq_dz_num);
    std::abort();
  }

  auto &ptrs = device_buffers_;
  if (ptrs.ready && ptrs.num != num) {
    CUDA_FREE_AND_NULL(ptrs.x);
    CUDA_FREE_AND_NULL(ptrs.y);
    CUDA_FREE_AND_NULL(ptrs.z);
    CUDA_FREE_AND_NULL(ptrs.nx);
    CUDA_FREE_AND_NULL(ptrs.ny);
    CUDA_FREE_AND_NULL(ptrs.nz);
    CUDA_FREE_AND_NULL(ptrs.area);
    CUDA_FREE_AND_NULL(ptrs.source_term);
    CUDA_FREE_AND_NULL(ptrs.target_q);
    CUDA_FREE_AND_NULL(ptrs.target_q_dx);
    CUDA_FREE_AND_NULL(ptrs.target_q_dy);
    CUDA_FREE_AND_NULL(ptrs.target_q_dz);
    CUDA_FREE_AND_NULL(ptrs.source_q);
    CUDA_FREE_AND_NULL(ptrs.source_q_dx);
    CUDA_FREE_AND_NULL(ptrs.source_q_dy);
    CUDA_FREE_AND_NULL(ptrs.source_q_dz);
    reset_device_buffers_();
  }

  if (!ptrs.ready && num > 0) {
    CUDA_MALLOC_OR_DIE(&ptrs.x, x_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.y, y_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.z, z_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.nx, nx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.ny, ny_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.nz, nz_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.area, area_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_term, source_term_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q, tq_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dx, tq_dx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dy, tq_dy_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.target_q_dz, tq_dz_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q, sq_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dx, sq_dx_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dy, sq_dy_num * sizeof(double));
    CUDA_MALLOC_OR_DIE(&ptrs.source_q_dz, sq_dz_num * sizeof(double));
    ptrs.num = num;
    ptrs.ready = true;
  }

  cudaStream_t stream = nullptr;
#ifdef OPENACC_ENABLED
  stream = static_cast<cudaStream_t>(acc_get_cuda_stream(acc_async_sync));
#endif

  if (num > 0) {
    CUDA_MEMCPY_ASYNC(ptrs.x, x_.data(), x_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.y, y_.data(), y_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.z, z_.data(), z_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.nx, nx_.data(), nx_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.ny, ny_.data(), ny_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.nz, nz_.data(), nz_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.area, area_.data(), area_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
    CUDA_MEMCPY_ASYNC(ptrs.source_term, source_term_.data(),
                      source_term_num * sizeof(double),
                      cudaMemcpyHostToDevice, stream);
  }

#ifdef OPENACC_ENABLED
  CUDA_ACC_UNMAP_IF_PRESENT(x_.data(), x_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(y_.data(), y_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(z_.data(), z_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(nx_.data(), nx_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(ny_.data(), ny_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(nz_.data(), nz_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(area_.data(), area_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(source_term_.data(), source_term_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(target_charge_.data(), tq_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dx_.data(), tq_dx_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dy_.data(), tq_dy_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dz_.data(), tq_dz_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(source_charge_.data(), sq_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dx_.data(), sq_dx_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dy_.data(), sq_dy_num * sizeof(double));
  CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dz_.data(), sq_dz_num * sizeof(double));

  CUDA_ACC_MAP_CONST(x_.data(), ptrs.x, x_num * sizeof(double));
  CUDA_ACC_MAP_CONST(y_.data(), ptrs.y, y_num * sizeof(double));
  CUDA_ACC_MAP_CONST(z_.data(), ptrs.z, z_num * sizeof(double));
  CUDA_ACC_MAP_CONST(nx_.data(), ptrs.nx, nx_num * sizeof(double));
  CUDA_ACC_MAP_CONST(ny_.data(), ptrs.ny, ny_num * sizeof(double));
  CUDA_ACC_MAP_CONST(nz_.data(), ptrs.nz, nz_num * sizeof(double));
  CUDA_ACC_MAP_CONST(area_.data(), ptrs.area, area_num * sizeof(double));
  CUDA_ACC_MAP_CONST(source_term_.data(), ptrs.source_term,
                     source_term_num * sizeof(double));
  CUDA_ACC_MAP_CONST(target_charge_.data(), ptrs.target_q, tq_num * sizeof(double));
  CUDA_ACC_MAP_CONST(target_charge_dx_.data(), ptrs.target_q_dx,
                     tq_dx_num * sizeof(double));
  CUDA_ACC_MAP_CONST(target_charge_dy_.data(), ptrs.target_q_dy,
                     tq_dy_num * sizeof(double));
  CUDA_ACC_MAP_CONST(target_charge_dz_.data(), ptrs.target_q_dz,
                     tq_dz_num * sizeof(double));
  CUDA_ACC_MAP_CONST(source_charge_.data(), ptrs.source_q, sq_num * sizeof(double));
  CUDA_ACC_MAP_CONST(source_charge_dx_.data(), ptrs.source_q_dx,
                     sq_dx_num * sizeof(double));
  CUDA_ACC_MAP_CONST(source_charge_dy_.data(), ptrs.source_q_dy,
                     sq_dy_num * sizeof(double));
  CUDA_ACC_MAP_CONST(source_charge_dz_.data(), ptrs.source_q_dz,
                     sq_dz_num * sizeof(double));
#endif

  CUDA_SYNC_AND_CHECK();
  device_state_ = CudaDeviceState::DeviceMapped;

  if (require_all && num > 0) {
    if (!ptrs.ready || !ptrs.x || !ptrs.y || !ptrs.z || !ptrs.nx || !ptrs.ny ||
        !ptrs.nz || !ptrs.area || !ptrs.source_term || !ptrs.target_q ||
        !ptrs.target_q_dx || !ptrs.target_q_dy || !ptrs.target_q_dz ||
        !ptrs.source_q || !ptrs.source_q_dx || !ptrs.source_q_dy ||
        !ptrs.source_q_dz) {
      std::fprintf(stderr,
                   "[CUDA_ELEMENTS] copyin missing device buffers under "
                   "TABIPB_CUDA_REQUIRE_ALL=1\n");
      std::abort();
    }
  }
#endif

#if defined(OPENACC_ENABLED) && !defined(USE_CUDA_CC)
  const double *x_ptr = x_.data();
  const double *y_ptr = y_.data();
  const double *z_ptr = z_.data();

  std::size_t x_num = x_.size();
  std::size_t y_num = y_.size();
  std::size_t z_num = z_.size();

  const double *nx_ptr = nx_.data();
  const double *ny_ptr = ny_.data();
  const double *nz_ptr = nz_.data();

  std::size_t nx_num = nx_.size();
  std::size_t ny_num = ny_.size();
  std::size_t nz_num = nz_.size();

  const double *area_ptr = area_.data();
  std::size_t area_num = area_.size();

  const double *source_term_ptr = source_term_.data();
  std::size_t source_term_num = source_term_.size();

  const double *tq_ptr = target_charge_.data();
  const double *tq_dx_ptr = target_charge_dx_.data();
  const double *tq_dy_ptr = target_charge_dy_.data();
  const double *tq_dz_ptr = target_charge_dz_.data();

  std::size_t tq_num = target_charge_.size();
  std::size_t tq_dx_num = target_charge_dx_.size();
  std::size_t tq_dy_num = target_charge_dy_.size();
  std::size_t tq_dz_num = target_charge_dz_.size();

  const double *sq_ptr = source_charge_.data();
  const double *sq_dx_ptr = source_charge_dx_.data();
  const double *sq_dy_ptr = source_charge_dy_.data();
  const double *sq_dz_ptr = source_charge_dz_.data();

  std::size_t sq_num = source_charge_.size();
  std::size_t sq_dx_num = source_charge_dx_.size();
  std::size_t sq_dy_num = source_charge_dy_.size();
  std::size_t sq_dz_num = source_charge_dz_.size();

#pragma acc enter data copyin(                                                 \
    x_ptr[0 : x_num], y_ptr[0 : y_num], z_ptr[0 : z_num], nx_ptr[0 : nx_num],  \
    ny_ptr[0 : ny_num], nz_ptr[0 : nz_num], area_ptr[0 : area_num])
#pragma acc enter data create(                                                 \
    source_term_ptr[0 : source_term_num], tq_ptr[0 : tq_num],                  \
    tq_dx_ptr[0 : tq_dx_num], tq_dy_ptr[0 : tq_dy_num],                        \
    tq_dz_ptr[0 : tq_dz_num], sq_ptr[0 : sq_num], sq_dx_ptr[0 : sq_dx_num],    \
    sq_dy_ptr[0 : sq_dy_num], sq_dz_ptr[0 : sq_dz_num])
#endif

  timers_.copyin_to_device.stop();
}

void Elements::update_source_term_on_host() {
#if defined(USE_CUDA_CC) && defined(OPENACC_ENABLED)
  const std::size_t source_term_num = source_term_.size();
  if (device_state_ != CudaDeviceState::DeviceMapped ||
      !device_buffers_.ready || source_term_num == 0 ||
      !device_buffers_.source_term) {
    return;
  }
  cudaStream_t stream = nullptr;
  stream = static_cast<cudaStream_t>(acc_get_cuda_stream(acc_async_sync));
  CUDA_MEMCPY_ASYNC(source_term_.data(), device_buffers_.source_term,
                    source_term_num * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
#elif defined(OPENACC_ENABLED)
  const double *source_term_ptr = source_term_.data();
  std::size_t source_term_num = source_term_.size();

#pragma acc update self(source_term_ptr[0 : source_term_num])
#endif
}

void Elements::delete_from_device() const {
  timers_.delete_from_device.start();

#ifdef USE_CUDA_CC
  if (device_buffers_.ready) {
#ifdef OPENACC_ENABLED
    const std::size_t x_num = x_.size();
    const std::size_t y_num = y_.size();
    const std::size_t z_num = z_.size();
    const std::size_t nx_num = nx_.size();
    const std::size_t ny_num = ny_.size();
    const std::size_t nz_num = nz_.size();
    const std::size_t area_num = area_.size();
    const std::size_t source_term_num = source_term_.size();
    const std::size_t tq_num = target_charge_.size();
    const std::size_t tq_dx_num = target_charge_dx_.size();
    const std::size_t tq_dy_num = target_charge_dy_.size();
    const std::size_t tq_dz_num = target_charge_dz_.size();
    const std::size_t sq_num = source_charge_.size();
    const std::size_t sq_dx_num = source_charge_dx_.size();
    const std::size_t sq_dy_num = source_charge_dy_.size();
    const std::size_t sq_dz_num = source_charge_dz_.size();

    CUDA_ACC_UNMAP_IF_PRESENT(x_.data(), x_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(y_.data(), y_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(z_.data(), z_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(nx_.data(), nx_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(ny_.data(), ny_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(nz_.data(), nz_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(area_.data(), area_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(source_term_.data(), source_term_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(target_charge_.data(), tq_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dx_.data(), tq_dx_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dy_.data(), tq_dy_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(target_charge_dz_.data(), tq_dz_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(source_charge_.data(), sq_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dx_.data(), sq_dx_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dy_.data(), sq_dy_num * sizeof(double));
    CUDA_ACC_UNMAP_IF_PRESENT(source_charge_dz_.data(), sq_dz_num * sizeof(double));
#endif
    CUDA_FREE_AND_NULL(device_buffers_.x);
    CUDA_FREE_AND_NULL(device_buffers_.y);
    CUDA_FREE_AND_NULL(device_buffers_.z);
    CUDA_FREE_AND_NULL(device_buffers_.nx);
    CUDA_FREE_AND_NULL(device_buffers_.ny);
    CUDA_FREE_AND_NULL(device_buffers_.nz);
    CUDA_FREE_AND_NULL(device_buffers_.area);
    CUDA_FREE_AND_NULL(device_buffers_.source_term);
    CUDA_FREE_AND_NULL(device_buffers_.target_q);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dx);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dy);
    CUDA_FREE_AND_NULL(device_buffers_.target_q_dz);
    CUDA_FREE_AND_NULL(device_buffers_.source_q);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dx);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dy);
    CUDA_FREE_AND_NULL(device_buffers_.source_q_dz);
    reset_device_buffers_();
  }
  device_state_ = CudaDeviceState::HostOnly;
#elif defined(OPENACC_ENABLED)
  const double *x_ptr = x_.data();
  const double *y_ptr = y_.data();
  const double *z_ptr = z_.data();

  std::size_t x_num = x_.size();
  std::size_t y_num = y_.size();
  std::size_t z_num = z_.size();

  const double *nx_ptr = nx_.data();
  const double *ny_ptr = ny_.data();
  const double *nz_ptr = nz_.data();

  std::size_t nx_num = nx_.size();
  std::size_t ny_num = ny_.size();
  std::size_t nz_num = nz_.size();

  const double *area_ptr = area_.data();
  std::size_t area_num = area_.size();

  const double *source_term_ptr = source_term_.data();
  std::size_t source_term_num = source_term_.size();

  const double *tq_ptr = target_charge_.data();
  const double *tq_dx_ptr = target_charge_dx_.data();
  const double *tq_dy_ptr = target_charge_dy_.data();
  const double *tq_dz_ptr = target_charge_dz_.data();

  std::size_t tq_num = target_charge_.size();
  std::size_t tq_dx_num = target_charge_dx_.size();
  std::size_t tq_dy_num = target_charge_dy_.size();
  std::size_t tq_dz_num = target_charge_dz_.size();

  const double *sq_ptr = source_charge_.data();
  const double *sq_dx_ptr = source_charge_dx_.data();
  const double *sq_dy_ptr = source_charge_dy_.data();
  const double *sq_dz_ptr = source_charge_dz_.data();

  std::size_t sq_num = source_charge_.size();
  std::size_t sq_dx_num = source_charge_dx_.size();
  std::size_t sq_dy_num = source_charge_dy_.size();
  std::size_t sq_dz_num = source_charge_dz_.size();

#pragma acc exit data delete (                                                 \
    x_ptr[0 : x_num], y_ptr[0 : y_num], z_ptr[0 : z_num], nx_ptr[0 : nx_num],  \
    ny_ptr[0 : ny_num], nz_ptr[0 : nz_num], area_ptr[0 : area_num])
#pragma acc exit data delete (                                                 \
    source_term_ptr[0 : source_term_num], tq_ptr[0 : tq_num],                  \
    tq_dx_ptr[0 : tq_dx_num], tq_dy_ptr[0 : tq_dy_num],                        \
    tq_dz_ptr[0 : tq_dz_num], sq_ptr[0 : sq_num], sq_dx_ptr[0 : sq_dx_num],    \
    sq_dy_ptr[0 : sq_dy_num], sq_dz_ptr[0 : sq_dz_num])
#endif

  timers_.delete_from_device.stop();
}

static double triangle_area(std::array<std::array<double, 3>, 3> v) {
  std::array<double, 3> a, b, c;

  for (int i = 0; i < 3; ++i) {
    a[i] = v[i][0] - v[i][1];
    b[i] = v[i][0] - v[i][2];
    c[i] = v[i][1] - v[i][2];
  }

  double aa = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  double bb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  double cc = std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);

  double ss = 0.5 * (aa + bb + cc);
  return std::sqrt(ss * (ss - aa) * (ss - bb) * (ss - cc));
}

void Timers_Elements::print() const {
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(5);
  std::cout << "|...Elements function times (s)...." << std::endl;
  std::cout << "|   |...ctor.......................: ";
  std::cout << std::setw(12) << std::right << ctor.elapsed_time() << std::endl;
  std::cout << "|   |...compute_source_term........: ";
  std::cout << std::setw(12) << std::right << compute_source_term.elapsed_time()
            << std::endl;
  std::cout << "|   |...compute_charges............: ";
  std::cout << std::setw(12) << std::right << compute_charges.elapsed_time()
            << std::endl;
#ifdef OPENACC_ENABLED
  std::cout << "|   |...copyin_to_device...........: ";
  std::cout << std::setw(12) << std::right << copyin_to_device.elapsed_time()
            << std::endl;
  std::cout << "|   |...delete_from_device.........: ";
  std::cout << std::setw(12) << std::right << copyin_to_device.elapsed_time()
            << std::endl;
#endif
  std::cout << "|" << std::endl;
}

std::string Timers_Elements::get_durations() const {
  std::string durations;
  durations.append(std::to_string(ctor.elapsed_time())).append(", ");
  durations.append(std::to_string(compute_source_term.elapsed_time()))
      .append(", ");
  durations.append(std::to_string(compute_charges.elapsed_time())).append(", ");
  durations.append(std::to_string(copyin_to_device.elapsed_time()))
      .append(", ");
  durations.append(std::to_string(delete_from_device.elapsed_time()))
      .append(", ");

  return durations;
}

std::string Timers_Elements::get_headers() const {
  std::string headers;
  headers.append("Elements ctor, ");
  headers.append("Elements compute_source_term, ");
  headers.append("Elements compute_charges, ");
  headers.append("Elements copyin_to_device, ");
  headers.append("Elements delete_from_device, ");

  return headers;
}
