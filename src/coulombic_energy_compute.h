#ifndef H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H
#define H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H

// #include "timer.h"
#include "molecule.h"
#include "interp_pts.h"
#include "tree_compute.h"
#include <cstdint>

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

    std::vector<std::uint32_t> target_node_begin_u32_;
    std::vector<std::uint32_t> target_node_end_u32_;
    std::vector<std::uint32_t> source_node_begin_u32_;
    std::vector<std::uint32_t> source_node_end_u32_;
    std::vector<std::uint32_t> pp_offsets_u32_;
    std::vector<std::uint32_t> pp_sources_u32_;
    std::vector<std::uint32_t> pc_offsets_u32_;
    std::vector<std::uint32_t> pc_sources_u32_;
    std::vector<std::uint32_t> cp_offsets_u32_;
    std::vector<std::uint32_t> cp_sources_u32_;
    std::vector<std::uint32_t> cc_offsets_u32_;
    std::vector<std::uint32_t> cc_sources_u32_;

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
        std::uint32_t* target_node_begin_dev = nullptr;
        std::uint32_t* target_node_end_dev = nullptr;
        std::uint32_t* source_node_begin_dev = nullptr;
        std::uint32_t* source_node_end_dev = nullptr;
        std::uint32_t* pp_offsets_dev = nullptr;
        std::uint32_t* pp_sources_dev = nullptr;
        std::uint32_t* pc_offsets_dev = nullptr;
        std::uint32_t* pc_sources_dev = nullptr;
        std::uint32_t* cp_offsets_dev = nullptr;
        std::uint32_t* cp_sources_dev = nullptr;
        std::uint32_t* cc_offsets_dev = nullptr;
        std::uint32_t* cc_sources_dev = nullptr;

        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t coul_eng_num = 0;
        std::size_t weights_num = 0;
        std::size_t scratch_num = 0;
        std::size_t target_nodes_num = 0;
        std::size_t source_nodes_num = 0;
        std::size_t pp_offsets_num = 0;
        std::size_t pp_sources_num = 0;
        std::size_t pc_offsets_num = 0;
        std::size_t pc_sources_num = 0;
        std::size_t cp_offsets_num = 0;
        std::size_t cp_sources_num = 0;
        std::size_t cc_offsets_num = 0;
        std::size_t cc_sources_num = 0;
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
    bool run_batched_interactions_cuda_();
    void copyin_clusters_to_device_cuda_() const;
    void delete_clusters_from_device_cuda_() const;
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
    struct DeviceView {
        bool ready = false;
        double* q = nullptr;
        double* p = nullptr;
        double* coul_eng = nullptr;
        double* weights = nullptr;
        int* exact_idx_x = nullptr;
        int* exact_idx_y = nullptr;
        int* exact_idx_z = nullptr;
        double* denominator = nullptr;

        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t coul_eng_num = 0;
        std::size_t weights_num = 0;
        std::size_t scratch_num = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };

    DeviceView device_view() const {
        DeviceView view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.q = device_buffers_.q_dev;
        view.p = device_buffers_.p_dev;
        view.coul_eng = device_buffers_.coul_eng_dev;
        view.weights = device_buffers_.weights_dev;
        view.exact_idx_x = device_buffers_.exact_idx_x_dev;
        view.exact_idx_y = device_buffers_.exact_idx_y_dev;
        view.exact_idx_z = device_buffers_.exact_idx_z_dev;
        view.denominator = device_buffers_.denominator_dev;
        view.q_num = device_buffers_.q_num;
        view.p_num = device_buffers_.p_num;
        view.coul_eng_num = device_buffers_.coul_eng_num;
        view.weights_num = device_buffers_.weights_num;
        view.scratch_num = device_buffers_.scratch_num;
        view.state = device_state_;
#endif
        return view;
    }

    DeviceView host_view() {
        DeviceView view;
        view.ready = true;
        view.q = mol_interp_charge_.data();
        view.p = mol_interp_potential_.data();
        view.coul_eng = coul_eng_vec_.data();
        view.weights = mol_weights_.data();
        view.exact_idx_x = exact_idx_x_.data();
        view.exact_idx_y = exact_idx_y_.data();
        view.exact_idx_z = exact_idx_z_.data();
        view.denominator = denominator_.data();
        view.q_num = mol_interp_charge_.size();
        view.p_num = mol_interp_potential_.size();
        view.coul_eng_num = coul_eng_vec_.size();
        view.weights_num = mol_weights_.size();
        view.scratch_num = exact_idx_x_.size();
#ifdef USE_CUDA_CC
        view.state = CudaDeviceState::HostOnly;
#endif
        return view;
    }


    CoulombicEnergyCompute(const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree, const class InteractionList& interaction_list,
                      double phys_eps_solute);

    ~CoulombicEnergyCompute() = default;
    
    double compute();
    
};

#endif /* H_TABIPB_COULOMBIC_ENERGY_COMPUTE_STRUCT_H */
