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
  public:
    bool getReady() const { return ready; }
    double* getX() const { return x; }
    double* getY() const { return y; }
    double* getZ() const { return z; }
    double* getNX() const { return nx; }
    double* getNY() const { return ny; }
    double* getNZ() const { return nz; }
    double* getArea() const { return area; }
    double* getSourceTerm() const { return source_term; }
    double* getTargetQ() const { return target_q; }
    double* getTargetQDX() const { return target_q_dx; }
    double* getTargetQDY() const { return target_q_dy; }
    double* getTargetQDZ() const { return target_q_dz; }
    double* getSourceQ() const { return source_q; }
    double* getSourceQDX() const { return source_q_dx; }
    double* getSourceQDY() const { return source_q_dy; }
    double* getSourceQDZ() const { return source_q_dz; }
    std::size_t getNum() const { return num; }
  };

  const DeviceBuffers& device_buffers() const { return device_buffers_; }
  void reset_device_buffers_() const;
#endif

private:
#ifdef USE_CUDA_CC
  mutable DeviceBuffers device_buffers_;
  mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
#endif
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
