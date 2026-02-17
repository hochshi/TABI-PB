#ifndef H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H
#define H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H

// #include "timer.h"
#include "elements.h"
#include "molecule.h"
#include "interp_pts.h"
#include "tree_compute.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif
//struct Timers_SolvationEnergyCompute;
//struct Timers;

class SolvationEnergyCompute : public TreeCompute
{
private:
    class Elements& elements_;
    const class InterpolationPoints& elem_interp_pts_;
    
    const class Molecule& molecule_;
    const class InterpolationPoints& mol_interp_pts_;
    
    const double eps_;
    const double kappa_;
    
    
    /* Target clusters */
    
    int num_elem_interp_pts_per_node_;
    int num_elem_interp_potentials_per_node_;
    std::size_t num_elem_potentials_;
    
    std::vector<double> elem_interp_potential_;
    std::vector<double> elem_interp_potential_dx_;
    std::vector<double> elem_interp_potential_dy_;
    std::vector<double> elem_interp_potential_dz_;
    
    const std::size_t potential_offset_;
    std::vector<double>& potential_;
    const double* potential_device_ptr_;
    
    
    /* Source clusters */
    
    int num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node_;
    std::size_t num_mol_charges_;
    
    std::vector<double> mol_interp_charge_;
    
    
    /* Solvation energy */
    
    mutable std::vector<double> solv_eng_vec_;
    double solvation_energy_;
    
#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class SolvationEnergyCompute;
    private:
        double* weights_up_dev = nullptr;
        double* weights_down_dev = nullptr;
        double* q_dev = nullptr;
        double* p_dev = nullptr;
        double* p_dx_dev = nullptr;
        double* p_dy_dev = nullptr;
        double* p_dz_dev = nullptr;
        double* solv_eng_dev = nullptr;

        std::size_t weights_up_num = 0;
        std::size_t weights_down_num = 0;
        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t p_dx_num = 0;
        std::size_t p_dy_num = 0;
        std::size_t p_dz_num = 0;
        std::size_t solv_eng_num = 0;

        bool ready = false;
    };

    mutable DeviceBuffers device_buffers_;
    mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
    bool validate_device_buffers_common_() const;
    bool validate_device_buffers_particle_particle_() const;
    bool validate_device_buffers_particle_cluster_() const;
    bool validate_device_buffers_cluster_particle_() const;
    bool validate_device_buffers_cluster_cluster_() const;
    bool validate_device_buffers_upward_pass_() const;
    bool validate_device_buffers_downward_pass_() const;
    bool try_particle_particle_interact_cuda_(std::size_t target_node_begin,
                                              std::size_t target_node_end,
                                              std::size_t source_node_begin,
                                              std::size_t source_node_end) const;
    bool try_particle_cluster_interact_cuda_(std::size_t target_node_begin,
                                             std::size_t target_node_end,
                                             std::size_t source_node_idx) const;
    bool try_cluster_particle_interact_cuda_(std::size_t target_node_idx,
                                             std::size_t source_node_begin,
                                             std::size_t source_node_end) const;
    bool try_cluster_cluster_interact_cuda_(std::size_t target_node_idx,
                                            std::size_t source_node_idx) const;
    bool try_upward_pass_cuda_() const;
    bool try_downward_pass_cuda_() const;
    void copyin_clusters_to_device_cuda_() const;
    void delete_clusters_from_device_cuda_() const;
#endif

    
    void particle_particle_interact(std::array<std::size_t, 2> target_node_particle_idxs,
                                    std::array<std::size_t, 2> source_node_particle_idxs) override;
    
    void particle_cluster_interact(std::array<std::size_t, 2> target_node_particle_idxs,
                                   std::size_t source_node_idx) override;
                                   
    void cluster_particle_interact(std::size_t target_node_idx,
                                   std::array<std::size_t, 2> source_node_particle_idxs) override;
            
    void cluster_cluster_interact(std::size_t target_node_idx, std::size_t source_node_idx) override;
            
    void upward_pass() override;
    void downward_pass() override;

    void copyin_clusters_to_device() const override;
    void delete_clusters_from_device() const override;

    
public:
    struct DeviceView {
        bool ready = false;
        double* weights_up = nullptr;
        double* weights_down = nullptr;
        double* q = nullptr;
        double* p = nullptr;
        double* p_dx = nullptr;
        double* p_dy = nullptr;
        double* p_dz = nullptr;
        double* solv_eng = nullptr;

        std::size_t weights_up_num = 0;
        std::size_t weights_down_num = 0;
        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t p_dx_num = 0;
        std::size_t p_dy_num = 0;
        std::size_t p_dz_num = 0;
        std::size_t solv_eng_num = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };

    DeviceView device_view() const {
        DeviceView view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.weights_up = device_buffers_.weights_up_dev;
        view.weights_down = device_buffers_.weights_down_dev;
        view.q = device_buffers_.q_dev;
        view.p = device_buffers_.p_dev;
        view.p_dx = device_buffers_.p_dx_dev;
        view.p_dy = device_buffers_.p_dy_dev;
        view.p_dz = device_buffers_.p_dz_dev;
        view.solv_eng = device_buffers_.solv_eng_dev;
        view.weights_up_num = device_buffers_.weights_up_num;
        view.weights_down_num = device_buffers_.weights_down_num;
        view.q_num = device_buffers_.q_num;
        view.p_num = device_buffers_.p_num;
        view.p_dx_num = device_buffers_.p_dx_num;
        view.p_dy_num = device_buffers_.p_dy_num;
        view.p_dz_num = device_buffers_.p_dz_num;
        view.solv_eng_num = device_buffers_.solv_eng_num;
        view.state = device_state_;
#endif
        return view;
    }

    SolvationEnergyCompute(std::vector<double>& potential,
                      class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps, double phys_kappa,
                      const double* potential_device_ptr);

    ~SolvationEnergyCompute() = default;
    
    double compute();
    
};

#endif /* H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H */
