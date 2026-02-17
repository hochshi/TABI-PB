#ifndef H_OUTPUT_H
#define H_OUTPUT_H

// #include <array>
#include <vector>

// #include "solvation_energy_compute.h"
#include "molecule.h"
#include "elements.h"
#include "params.h"
#include "timer.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

struct Timers;
struct Timers_Output;

class Output
{
private:

    class Molecule& molecule_;
    class Elements& elements_;
    const struct Params& params_;
    struct Timers_Output& timers_;
    
    /* iteration data */
    long int num_iter_;
    double residual_;
    
    /* output */
    const std::size_t potential_offset_;
    std::vector<double> potential_;
    
    double solvation_energy_;
    double free_energy_;
    double coulombic_energy_;
    
    double pot_min_;
    double pot_max_;
    double pot_normal_min_;
    double pot_normal_max_;

#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class Output;
    private:
        bool ready = false;
        double* potential_dev = nullptr;
        std::size_t potential_num = 0;
    };

    mutable DeviceBuffers device_buffers_;
    mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
    bool validate_device_buffers_compute_coulombic_energy_() const;
    bool validate_device_buffers_compute_solvation_energy_() const;
    void copyin_potential_to_device_cuda_(const double* potential_ptr,
                                          std::size_t potential_num) const;
    void cleanup_potential_device_buffer_cuda_() const;
    bool try_compute_coulombic_energy_cuda_(double& coulombic_energy) const;
    bool try_compute_solvation_energy_cuda_(double& solvation_energy) const;
#endif

public:

    Output(class Molecule&, class Elements&, const struct Params&, struct Timers_Output&);
    ~Output() = default;
    
    std::vector<double>& potential() { return potential_; };
    std::size_t potential_offset() { return potential_offset_; };
    
    void set_num_iter(long int num_iter) { num_iter_ = num_iter; }
    void set_residual(double residual) { residual_ = residual; }

    struct DeviceView {
        bool ready = false;
        double* potential = nullptr;
        std::size_t potential_num = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };

    DeviceView device_view() const {
        DeviceView view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.potential = device_buffers_.potential_dev;
        view.potential_num = device_buffers_.potential_num;
        view.state = device_state_;
#endif
        return view;
    }

#ifdef USE_CUDA_CC
    bool cuda_device_ready() const {
        return device_state_ == CudaDeviceState::DeviceMapped &&
               device_buffers_.ready;
    }
#else
    bool cuda_device_ready() const { return false; }
#endif
    
    void compute_solvation_energy();
    void compute_solvation_energy(const class InterpolationPoints& elem_interp_pts, const class Tree& elem_tree,
                                  const class InterpolationPoints& mol_interp_pts,  const class Tree& mol_tree,
                                  const class InteractionList& interaction_list);

    void compute_coulombic_energy();
    void compute_coulombic_energy(const class InterpolationPoints& mol_interp_pts,  const class Tree& mol_tree,
                                  const class InteractionList& interaction_list);
    
    void compute_free_energy();
    
    void finalize();
    void files(const struct Timers&) const;
    void output_VTK() const;
    void output_PLY() const;
};


struct Timers_Output
{
    Timer ctor;
    Timer compute_solvation_energy;
    Timer compute_coulombic_energy;
    Timer output_VTK;
    Timer finalize;
    
    void print() const;
    std::string get_durations() const;
    std::string get_headers() const;

    Timers_Output() = default;
    ~Timers_Output() = default;
};

#endif
