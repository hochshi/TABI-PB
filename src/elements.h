#ifndef H_TABIPB_ELEMENTS_STRUCT_H
#define H_TABIPB_ELEMENTS_STRUCT_H

#include <cstdlib>
#include <vector>

#include "params.h"
#include "timer.h"
// #include "source_term_compute.h"
#include "molecule.h"
#include "particles.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

struct Timers_Elements;

class Elements : public Particles {
private:
  const class Molecule &molecule_;
  struct Timers_Elements &timers_;

  std::size_t num_faces_;
  double surface_area_;

  std::vector<uint32_t> face_x_;
  std::vector<uint32_t> face_y_;
  std::vector<uint32_t> face_z_;

  std::vector<double> nx_;
  std::vector<double> ny_;
  std::vector<double> nz_;

  std::vector<double> area_;
  std::vector<double> source_term_;

  std::vector<double> target_charge_;
  std::vector<double> target_charge_dx_;
  std::vector<double> target_charge_dy_;
  std::vector<double> target_charge_dz_;

  std::vector<double> source_charge_;
  std::vector<double> source_charge_dx_;
  std::vector<double> source_charge_dy_;
  std::vector<double> source_charge_dz_;

  void write_nanaoshaper_config(Params::Mesh, Params::MeshFormat, double,
                                double);
  void generate_elements(Params::Mesh, Params::MeshFormat, double, double,
                         const std::string &);
  bool read_msms_file(const std::string &);
  bool read_ply_file(const std::string &filepath);
  bool file_exists(const std::string &name);
  void update_source_term_on_host();

public:
  Elements(const class Molecule &, const struct Params &,
           struct Timers_Elements &);
  ~Elements() = default;

  std::size_t num_faces() const { return num_faces_; };
  double surface_area() const { return surface_area_; };

  const uint32_t *face_x_ptr() const { return face_x_.data(); };
  const uint32_t *face_y_ptr() const { return face_y_.data(); };
  const uint32_t *face_z_ptr() const { return face_z_.data(); };

  const double *nx_ptr() const { return nx_.data(); };
  const double *ny_ptr() const { return ny_.data(); };
  const double *nz_ptr() const { return nz_.data(); };

  const double *area_ptr() const { return area_.data(); };
  const double *source_term_ptr() const { return source_term_.data(); };

  const double *target_charge_ptr() const { return target_charge_.data(); };
  const double *target_charge_dx_ptr() const {
    return target_charge_dx_.data();
  };
  const double *target_charge_dy_ptr() const {
    return target_charge_dy_.data();
  };
  const double *target_charge_dz_ptr() const {
    return target_charge_dz_.data();
  };

  const double *source_charge_ptr() const { return source_charge_.data(); };
  const double *source_charge_dx_ptr() const {
    return source_charge_dx_.data();
  };
  const double *source_charge_dy_ptr() const {
    return source_charge_dy_.data();
  };
  const double *source_charge_dz_ptr() const {
    return source_charge_dz_.data();
  };

  void reorder() override;
  void unorder() override;
  void unorder(std::vector<double> &potential);

  void compute_source_term();
  void compute_source_term(const class InterpolationPoints &elem_interp_pts,
                           const class Tree &elem_tree,
                           const class Molecule &molecule,
                           const class InterpolationPoints &mol_interp_pts,
                           const class Tree &mol_tree,
                           const class InteractionList &interaction_list);

  void compute_charges(const double *potential);
  Timer& compute_charges_timer();

  void copyin_to_device() const override;
  void delete_from_device() const override;

private:
#ifdef USE_CUDA_CC
  class DeviceBuffers {
    friend class Elements;
  private:
    bool ready = false;
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* nx = nullptr;
    double* ny = nullptr;
    double* nz = nullptr;
    double* area = nullptr;
    double* source_term = nullptr;
    double* target_q = nullptr;
    double* target_q_dx = nullptr;
    double* target_q_dy = nullptr;
    double* target_q_dz = nullptr;
    double* source_q = nullptr;
    double* source_q_dx = nullptr;
    double* source_q_dy = nullptr;
    double* source_q_dz = nullptr;
    std::size_t num = 0;
  };
  void reset_device_buffers_() const;
  bool validate_device_buffers_compute_source_term_() const;
  bool validate_device_buffers_compute_charges_() const;
  void copyin_to_device_cuda_() const;
  void delete_from_device_cuda_() const;
  void update_source_term_on_host_cuda_();
#endif

#ifdef USE_CUDA_CC
  mutable DeviceBuffers device_buffers_;
  mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
#endif

public:
#ifdef USE_CUDA_CC
  bool cuda_device_ready() const {
    return device_state_ == CudaDeviceState::DeviceMapped && device_buffers_.ready;
  }
#else
  bool cuda_device_ready() const { return false; }
#endif

  struct View {
    bool ready = false;
    const double* x = nullptr;
    const double* y = nullptr;
    const double* z = nullptr;
    const double* nx = nullptr;
    const double* ny = nullptr;
    const double* nz = nullptr;
    const double* area = nullptr;
    double* source_term = nullptr;
    double* target_q = nullptr;
    double* target_q_dx = nullptr;
    double* target_q_dy = nullptr;
    double* target_q_dz = nullptr;
    double* source_q = nullptr;
    double* source_q_dx = nullptr;
    double* source_q_dy = nullptr;
    double* source_q_dz = nullptr;
    std::size_t num = 0;
#ifdef USE_CUDA_CC
    CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
  };
  using DeviceView = View;

  View device_view() const {
    View view;
#ifdef USE_CUDA_CC
    view.ready = device_buffers_.ready;
    view.x = device_buffers_.x;
    view.y = device_buffers_.y;
    view.z = device_buffers_.z;
    view.nx = device_buffers_.nx;
    view.ny = device_buffers_.ny;
    view.nz = device_buffers_.nz;
    view.area = device_buffers_.area;
    view.source_term = device_buffers_.source_term;
    view.target_q = device_buffers_.target_q;
    view.target_q_dx = device_buffers_.target_q_dx;
    view.target_q_dy = device_buffers_.target_q_dy;
    view.target_q_dz = device_buffers_.target_q_dz;
    view.source_q = device_buffers_.source_q;
    view.source_q_dx = device_buffers_.source_q_dx;
    view.source_q_dy = device_buffers_.source_q_dy;
    view.source_q_dz = device_buffers_.source_q_dz;
    view.num = device_buffers_.num;
    view.state = device_state_;
#endif
    return view;
  }

  View host_view() {
    View view;
    view.ready = true;
    view.x = x_.data();
    view.y = y_.data();
    view.z = z_.data();
    view.nx = nx_.data();
    view.ny = ny_.data();
    view.nz = nz_.data();
    view.area = area_.data();
    view.source_term = source_term_.data();
    view.target_q = target_charge_.data();
    view.target_q_dx = target_charge_dx_.data();
    view.target_q_dy = target_charge_dy_.data();
    view.target_q_dz = target_charge_dz_.data();
    view.source_q = source_charge_.data();
    view.source_q_dx = source_charge_dx_.data();
    view.source_q_dy = source_charge_dy_.data();
    view.source_q_dz = source_charge_dz_.data();
    view.num = num_;
#ifdef USE_CUDA_CC
    view.state = CudaDeviceState::HostOnly;
#endif
    return view;
  }
};

struct Timers_Elements {
  Timer ctor;
  Timer compute_source_term;
  Timer compute_charges;
  Timer copyin_to_device;
  Timer delete_from_device;
  Timer output_VTK;

  void print() const;
  std::string get_durations() const;
  std::string get_headers() const;

  Timers_Elements() = default;
  ~Timers_Elements() = default;
};

#endif /* H_TABIPB_ELEMENTS_STRUCT_H */
