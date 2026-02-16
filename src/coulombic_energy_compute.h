#ifndef H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H
#define H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H

// #include "timer.h"
#include "molecule.h"
#include "interp_pts.h"
#include "tree_compute.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

//struct Timers;

class CoulombicEnergyCompute : public TreeCompute
{
private:
    
    const class Molecule& molecule_;
    const class InterpolationPoints& mol_interp_pts_;
    
    const double eps_solute_;
    
    /* Target and Source clusters */
    
    int num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node_;
    std::size_t num_mol_charges_;
    
    int num_mol_interp_potentials_per_node_;
    std::size_t num_mol_potentials_;

    std::vector<double> mol_interp_charge_;
    std::vector<double> mol_interp_potential_;

    std::size_t max_mol_particles_per_node_;
    std::vector<double> mol_weights_;
    mutable std::vector<int> exact_idx_x_;
    mutable std::vector<int> exact_idx_y_;
    mutable std::vector<int> exact_idx_z_;
    mutable std::vector<double> denominator_;

#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class CoulombicEnergyCompute;
    private:
        double* q_dev = nullptr;
        double* p_dev = nullptr;
        double* coul_eng_dev = nullptr;
        double* weights_dev = nullptr;
        int* exact_idx_x_dev = nullptr;
        int* exact_idx_y_dev = nullptr;
        int* exact_idx_z_dev = nullptr;
        double* denominator_dev = nullptr;

        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t coul_eng_num = 0;
        std::size_t weights_num = 0;
        std::size_t scratch_num = 0;
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
#endif
    
    
    /* Coulombic energy */
   
    mutable std::vector<double> coul_eng_vec_; 
    double coulombic_energy_;
    
    
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

    CoulombicEnergyCompute(const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree, const class InteractionList& interaction_list,
                      double phys_eps_solute);

    ~CoulombicEnergyCompute() = default;
    
    double compute();
    
};

#endif /* H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H */
