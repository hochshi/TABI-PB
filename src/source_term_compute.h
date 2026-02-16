#ifndef H_TABIPB_SOURCE_TERM_COMPUTE_STRUCT_H
#define H_TABIPB_SOURCE_TERM_COMPUTE_STRUCT_H

#include "molecule.h"
#include "interp_pts.h"
#include "tree_compute.h"

#ifdef USE_CUDA_CC
#include "cuda_state.h"
#endif

//struct Timers_SourceTermCompute;
//struct Timers;

class SourceTermCompute : public TreeCompute
{
private:
    class Elements& elements_;
    const class InterpolationPoints& elem_interp_pts_;
    
    const class Molecule& molecule_;
    const class InterpolationPoints& mol_interp_pts_;
    
    const double one_over_4pi_eps_solute_;
    
    
    /* Target clusters */
    
    int num_elem_interp_pts_per_node_;
    int num_elem_interp_potentials_per_node_;
    std::size_t num_elem_potentials_;
    
    std::vector<double> elem_interp_potential_;
    std::vector<double> elem_interp_potential_dx_;
    std::vector<double> elem_interp_potential_dy_;
    std::vector<double> elem_interp_potential_dz_;
    
    
    /* Source clusters */
    
    int num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node_;
    std::size_t num_mol_charges_;
    
    std::vector<double> mol_interp_charge_;

    /* Persistent weights + scratch */

    std::size_t max_mol_particles_per_node_;

    std::vector<double> mol_weights_;
    std::vector<double> elem_weights_;

    mutable std::vector<int> exact_idx_x_;
    mutable std::vector<int> exact_idx_y_;
    mutable std::vector<int> exact_idx_z_;
    mutable std::vector<double> denominator_;

#ifdef USE_CUDA_CC
    class DeviceBuffers {
        friend class SourceTermCompute;
    private:
        double* q_dev = nullptr;
        double* p_dev = nullptr;
        double* p_dx_dev = nullptr;
        double* p_dy_dev = nullptr;
        double* p_dz_dev = nullptr;
        double* mol_weights_dev = nullptr;
        double* elem_weights_dev = nullptr;
        int* exact_idx_x_dev = nullptr;
        int* exact_idx_y_dev = nullptr;
        int* exact_idx_z_dev = nullptr;
        double* denominator_dev = nullptr;

        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t p_dx_num = 0;
        std::size_t p_dy_num = 0;
        std::size_t p_dz_num = 0;
        std::size_t mol_weights_num = 0;
        std::size_t elem_weights_num = 0;
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
    bool validate_device_buffers_downward_pass_() const;
#endif

    
    
    /* Potentials */
    
    const std::size_t source_term_offset_;
    std::vector<double>& source_term_;
    
    
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
        double* p_dx = nullptr;
        double* p_dy = nullptr;
        double* p_dz = nullptr;
        double* mol_weights = nullptr;
        double* elem_weights = nullptr;
        int* exact_idx_x = nullptr;
        int* exact_idx_y = nullptr;
        int* exact_idx_z = nullptr;
        double* denominator = nullptr;

        std::size_t q_num = 0;
        std::size_t p_num = 0;
        std::size_t p_dx_num = 0;
        std::size_t p_dy_num = 0;
        std::size_t p_dz_num = 0;
        std::size_t mol_weights_num = 0;
        std::size_t elem_weights_num = 0;
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
        view.p_dx = device_buffers_.p_dx_dev;
        view.p_dy = device_buffers_.p_dy_dev;
        view.p_dz = device_buffers_.p_dz_dev;
        view.mol_weights = device_buffers_.mol_weights_dev;
        view.elem_weights = device_buffers_.elem_weights_dev;
        view.exact_idx_x = device_buffers_.exact_idx_x_dev;
        view.exact_idx_y = device_buffers_.exact_idx_y_dev;
        view.exact_idx_z = device_buffers_.exact_idx_z_dev;
        view.denominator = device_buffers_.denominator_dev;
        view.q_num = device_buffers_.q_num;
        view.p_num = device_buffers_.p_num;
        view.p_dx_num = device_buffers_.p_dx_num;
        view.p_dy_num = device_buffers_.p_dy_num;
        view.p_dz_num = device_buffers_.p_dz_num;
        view.mol_weights_num = device_buffers_.mol_weights_num;
        view.elem_weights_num = device_buffers_.elem_weights_num;
        view.scratch_num = device_buffers_.scratch_num;
        view.state = device_state_;
#endif
        return view;
    }

    SourceTermCompute(std::vector<double>& source_term,
                      class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps_solute);

    ~SourceTermCompute() = default;
    
    void compute();
    
};

#endif /* H_TABIPB_SOURCE_TERM_COMPUTE_STRUCT_H */
