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
#include "elements_backend_common.h"
#include "elements_backend_cpu.h"
#ifdef USE_CUDA_CC
#include "elements_backend_cuda.h"
#endif
#include "elements.h"
#include "source_term_compute.h"
#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
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

#ifdef USE_CUDA_CC
  const std::size_t num_atoms = molecule_.num();
  const std::size_t num = num_;
  const bool require_all = elements_cuda_require_all();
  if (validate_device_buffers_compute_source_term_()) {
    const auto elem_view = device_view();
    const auto mol_view = molecule_.device_view();
    if (elements_try_compute_source_term_cuda(elem_view, mol_view, num, num_atoms,
                                              eps_solute, nullptr)) {
      Elements::update_source_term_on_host();
      timers_.compute_source_term.stop();
      return;
    }
  }
  if (require_all) {
    std::cerr << "[CUDA_ELEM] require_all set but device pointers not present. "
              << "Aborting to avoid OpenACC fallback.\n";
    std::exit(1);
  }
#endif

  const auto elem_view = host_view();
  const auto mol_view = molecule_.host_view();
  elements_compute_source_term_cpu(elem_view, mol_view, eps_solute);

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

#ifdef USE_CUDA_CC
  if (validate_device_buffers_compute_charges_()) {
    const auto elem_view = device_view();
    if (elements_try_compute_charges_cuda(elem_view, potential_ptr, num,
                                          nullptr)) {
      timers_.compute_charges.stop();
      return;
    }
  }
  if (elements_cuda_require_all()) {
    if (!cuda_pointer_is_device_accessible(potential_ptr)) {
      std::cerr << "[CUDA_ELEM] require_all set but potential pointer is not a CUDA device pointer. "
                << "Aborting to avoid OpenACC interop fallback.\n";
    } else {
      std::cerr << "[CUDA_ELEM] require_all set but device pointers not present. "
                << "Aborting to avoid OpenACC fallback.\n";
    }
    std::exit(1);
  }
#endif

  const auto elem_view = host_view();
  elements_compute_charges_cpu(elem_view, potential_ptr, num);

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
  copyin_to_device_cuda_();
#endif

  timers_.copyin_to_device.stop();
}

void Elements::update_source_term_on_host() {
#ifdef USE_CUDA_CC
  update_source_term_on_host_cuda_();
#endif
}

void Elements::delete_from_device() const {
  timers_.delete_from_device.start();

#ifdef USE_CUDA_CC
  delete_from_device_cuda_();
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
#ifdef USE_CUDA_CC
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
