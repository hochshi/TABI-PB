#ifndef H_TABIPB_MOLECULE_STRUCT_H
#define H_TABIPB_MOLECULE_STRUCT_H

#include <vector>
#include <string>
// #include <fstream>
#include <cstddef>

#include "timer.h"
#include "particles.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

#ifdef TABIPB_APBS
    #include "generic/valist.h"
#endif

struct Timers_Molecule;

class Molecule : public Particles
{
private:
    struct Timers_Molecule& timers_;
    
    double coulombic_energy_;
    std::vector<double> charge_;
    std::vector<double> radius_;

#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class Molecule;
    private:
        bool ready = false;
        double* particles_x_dev = nullptr;
        double* particles_y_dev = nullptr;
        double* particles_z_dev = nullptr;
        double* charge_dev = nullptr;
        std::size_t num_particles = 0;
    };

    mutable DeviceBuffers device_buffers_;
    mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
#endif

public:
    Molecule(struct Params&, struct Timers_Molecule&);
    ~Molecule() = default;
    
#ifdef TABIPB_APBS
    Molecule(Valist*, struct Params&, struct Timers_Molecule&);
#endif
    
    void build_xyzr_file() const;
    
    const double* charge_ptr() const { return charge_.data(); };
    const double* radius_ptr() const { return radius_.data(); };

    struct View {
        bool ready = false;
        const double* particles_x = nullptr;
        const double* particles_y = nullptr;
        const double* particles_z = nullptr;
        const double* charge = nullptr;
        std::size_t num_particles = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };
    using DeviceView = View;

    View device_view() const {
        View view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.particles_x = device_buffers_.particles_x_dev;
        view.particles_y = device_buffers_.particles_y_dev;
        view.particles_z = device_buffers_.particles_z_dev;
        view.charge = device_buffers_.charge_dev;
        view.num_particles = device_buffers_.num_particles;
        view.state = device_state_;
#endif
        return view;
    }

    View host_view() const {
        View view;
        view.ready = true;
        view.particles_x = x_.data();
        view.particles_y = y_.data();
        view.particles_z = z_.data();
        view.charge = charge_.data();
        view.num_particles = num_;
#ifdef USE_CUDA_CC
        view.state = CudaDeviceState::HostOnly;
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
    
    void reorder() override;
    void unorder() override;
    
    void copyin_to_device() const override;
    void delete_from_device() const override;
};


struct Timers_Molecule
{
    Timer ctor;
    Timer build_xyzr_file;
    Timer copyin_to_device;
    Timer delete_from_device;
    
    void print() const;
    std::string get_durations() const;
    std::string get_headers() const;

    Timers_Molecule() = default;
    ~Timers_Molecule() = default;
};

#endif /* H_MOLECULE_STRUCT_H */
