#ifndef H_TABIPB_TREECODE_STRUCT_H
#define H_TABIPB_TREECODE_STRUCT_H

#include <cstdint>

#include "timer.h"
#include "output.h"
#include "elements.h"
#include "interp_pts.h"
#include "interaction_list.h"
#ifdef USE_CUDA_CC
#include <cuda_runtime.h>
#include "cuda_state.h"
#endif

struct Timers_BoundaryElement;

class BoundaryElement
{
private:
    class Elements& elements_;
    const class InterpolationPoints& interp_pts_;
    const class Tree& tree_;
    const class InteractionList& interaction_list_;
    const class Molecule& molecule_;
    const struct Params& params_;
    class Output& output_;
    struct Timers_BoundaryElement& timers_;
    
    std::vector<double> potential_;
    
    /* cluster specific data */
    int num_charges_per_node_;
    std::size_t num_charges_;
    
    std::vector<double> interp_charge_;
    std::vector<double> interp_charge_dx_;
    std::vector<double> interp_charge_dy_;
    std::vector<double> interp_charge_dz_;
    
    std::vector<double> interp_potential_;
    std::vector<double> interp_potential_dx_;
    std::vector<double> interp_potential_dy_;
    std::vector<double> interp_potential_dz_;

    std::vector<double> potential_temp_;
    std::vector<double> weights_;
    std::vector<int> exact_idx_x_;
    std::vector<int> exact_idx_y_;
    std::vector<int> exact_idx_z_;
    std::vector<double> denominator_;
    std::vector<std::size_t> level_offsets_;
    std::vector<std::size_t> level_nodes_;

    std::vector<std::size_t> node_particles_begin_;
    std::vector<std::size_t> node_particles_end_;
    std::vector<std::size_t> element_node_idx_;

    std::vector<std::uint32_t> node_particles_begin_u32_;
    std::vector<std::uint32_t> node_particles_end_u32_;
    std::vector<std::uint32_t> element_node_idx_u32_;

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
        friend class BoundaryElement;
    private:
        bool ready = false;
        bool owns_clusters_xyz = true;
        double* clusters_x = nullptr;
        double* clusters_y = nullptr;
        double* clusters_z = nullptr;
        double* clusters_q = nullptr;
        double* clusters_q_dx = nullptr;
        double* clusters_q_dy = nullptr;
        double* clusters_q_dz = nullptr;
        double* clusters_p = nullptr;
        double* clusters_p_dx = nullptr;
        double* clusters_p_dy = nullptr;
        double* clusters_p_dz = nullptr;
        const double* elements_x = nullptr;
        const double* elements_y = nullptr;
        const double* elements_z = nullptr;
        const double* elements_nx = nullptr;
        const double* elements_ny = nullptr;
        const double* elements_nz = nullptr;
        const double* elements_area = nullptr;
        double* targets_q = nullptr;
        double* targets_q_dx = nullptr;
        double* targets_q_dy = nullptr;
        double* targets_q_dz = nullptr;
        double* sources_q = nullptr;
        double* sources_q_dx = nullptr;
        double* sources_q_dy = nullptr;
        double* sources_q_dz = nullptr;
        double* weights = nullptr;
        double* potential_temp = nullptr;
        int* exact_idx_x = nullptr;
        int* exact_idx_y = nullptr;
        int* exact_idx_z = nullptr;
        double* denominator = nullptr;
        std::uint32_t* node_begin = nullptr;
        std::uint32_t* node_end = nullptr;
        std::uint32_t* element_node_idx = nullptr;
        std::uint32_t* pp_offsets = nullptr;
        std::uint32_t* pp_sources = nullptr;
        std::uint32_t* pc_offsets = nullptr;
        std::uint32_t* pc_sources = nullptr;
        std::uint32_t* cp_offsets = nullptr;
        std::uint32_t* cp_sources = nullptr;
        std::uint32_t* cc_offsets = nullptr;
        std::uint32_t* cc_sources = nullptr;
        std::size_t* level_nodes = nullptr;
        std::size_t level_nodes_num = 0;
        std::size_t num_nodes = 0;
    };
    mutable DeviceBuffers device_buffers_;
    mutable CudaDeviceState device_state_ = CudaDeviceState::HostOnly;
    struct CudaTimerSection {
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        Timer* timer = nullptr;
    };

    struct CudaTimerQueue {
        std::vector<CudaTimerSection> sections;
        void begin(Timer& timer, void* stream);
        void end(void* stream);
        void flush();
    };

    mutable CudaTimerQueue cuda_timer_queue_;
    void flush_cuda_timers_();
#endif
    
    /* output */
    double solvation_energy_;
    double free_energy_;
    double coulombic_energy_;
    
    double pot_min_;
    double pot_max_;
    double pot_normal_min_;
    double pot_normal_max_;

    struct GmresView {
        long int n = 0;
        const double* b = nullptr;
        double* x = nullptr;
        long int restrt = 0;
        double* work = nullptr;
        long int ldw = 0;
        double* h = nullptr;
        long int ldh = 0;
        long int* iter = nullptr;
        double* residual = nullptr;
    };
    
    int gmres_(long int n, const double* b, double* x, long int restrt,
               double* work, long int ldw, double *h, long int ldh,
               long int& iter, double& residual);
    int gmres_impl_(const GmresView& view, bool enable_cuda_backend);
    int gmres_cpu_(const GmresView& view);
#ifdef USE_CUDA_CC
    int gmres_cuda_(const GmresView& view);
#endif
    
    void matrix_vector(double alpha, const double* __restrict potential_old,
                       double beta,        double* __restrict potential_new,
                       bool device_ptrs = false);
    void matrix_vector_cpu_(double alpha, const double* __restrict potential_old,
                            double beta,        double* __restrict potential_new);
#ifdef USE_CUDA_CC
    void matrix_vector_cuda(double alpha, const double* potential_old_dev,
                            double beta, double* potential_new_dev,
                            void* stream);
    void cache_device_buffers_() const;
    void reset_device_buffers_() const;
    bool validate_device_buffers_matrix_vector_(
        const double* potential_old, const double* potential_new) const;
    bool validate_device_buffers_particle_particle_() const;
    bool validate_device_buffers_particle_cluster_(bool include_pp) const;
    bool validate_device_buffers_cluster_mixed_() const;
    bool validate_device_buffers_upward_(bool use_split) const;
    bool validate_device_buffers_downward_(const double* potential) const;
    bool validate_device_buffers_clear_cluster_charges_() const;
    bool validate_device_buffers_clear_cluster_potentials_() const;
#endif
                       
    void precondition_diagonal(double* z, double* r);
    void precondition_block(double* z, double* r);
#ifdef USE_CUDA_CC
    void precondition_diagonal_cuda(double* z_dev, double* r_dev, void* stream);
#endif
    
    void particle_particle_interact(double* __restrict potential,
                              const double* __restrict potential_old,
            std::array<std::size_t, 2> target_node_particle_idxs,
            std::array<std::size_t, 2> source_node_particle_idxs);
    
    void particle_cluster_interact(double* __restrict potential,
            std::array<std::size_t, 2> target_node_particle_idxs, std::size_t source_node_idx);
                                   
    void cluster_particle_interact(double* __restrict potential,
            std::size_t target_node_idx, std::array<std::size_t, 2> source_node_particle_idxs);
            
    void cluster_cluster_interact(double* __restrict potential,
            std::size_t target_node_idx, std::size_t source_node_idx);

    void particle_particle_interact_all(double* __restrict potential,
                              const double* __restrict potential_old);
    void particle_cluster_interact_all(double* __restrict potential,
                              const double* __restrict potential_old,
                              bool include_pp);
    void cluster_particle_interact_all(double* __restrict potential);
    void cluster_cluster_interact_all(double* __restrict potential);
            
    void upward_pass();
    void downward_pass(double* __restrict potential);
    
    void clear_cluster_charges();
    void clear_cluster_potentials();
    void copyin_clusters_to_device() const;
    void delete_clusters_from_device() const;
    
    const std::array<std::size_t, 2> cluster_charges_idxs(std::size_t node_idx) const {
        return std::array<std::size_t, 2> {num_charges_per_node_ *  node_idx,
                                           num_charges_per_node_ * (node_idx + 1)};
    };

    
public:
    struct DeviceView {
        bool ready = false;
        bool owns_clusters_xyz = true;
        const double* clusters_x = nullptr;
        const double* clusters_y = nullptr;
        const double* clusters_z = nullptr;
        double* clusters_q = nullptr;
        double* clusters_q_dx = nullptr;
        double* clusters_q_dy = nullptr;
        double* clusters_q_dz = nullptr;
        double* clusters_p = nullptr;
        double* clusters_p_dx = nullptr;
        double* clusters_p_dy = nullptr;
        double* clusters_p_dz = nullptr;
        const double* elements_x = nullptr;
        const double* elements_y = nullptr;
        const double* elements_z = nullptr;
        const double* elements_nx = nullptr;
        const double* elements_ny = nullptr;
        const double* elements_nz = nullptr;
        const double* elements_area = nullptr;
        double* targets_q = nullptr;
        double* targets_q_dx = nullptr;
        double* targets_q_dy = nullptr;
        double* targets_q_dz = nullptr;
        double* sources_q = nullptr;
        double* sources_q_dx = nullptr;
        double* sources_q_dy = nullptr;
        double* sources_q_dz = nullptr;
        const double* weights = nullptr;
        double* potential_temp = nullptr;
        int* exact_idx_x = nullptr;
        int* exact_idx_y = nullptr;
        int* exact_idx_z = nullptr;
        double* denominator = nullptr;
        const std::uint32_t* node_begin = nullptr;
        const std::uint32_t* node_end = nullptr;
        const std::uint32_t* element_node_idx = nullptr;
        const std::uint32_t* pp_offsets = nullptr;
        const std::uint32_t* pp_sources = nullptr;
        const std::uint32_t* pc_offsets = nullptr;
        const std::uint32_t* pc_sources = nullptr;
        const std::uint32_t* cp_offsets = nullptr;
        const std::uint32_t* cp_sources = nullptr;
        const std::uint32_t* cc_offsets = nullptr;
        const std::uint32_t* cc_sources = nullptr;
        const std::size_t* level_nodes = nullptr;
        std::size_t level_nodes_num = 0;
        std::size_t num_nodes = 0;
        std::size_t potential_num = 0;
#ifdef USE_CUDA_CC
        CudaDeviceState state = CudaDeviceState::HostOnly;
#endif
    };

    DeviceView device_view() const {
        DeviceView view;
#ifdef USE_CUDA_CC
        view.ready = device_buffers_.ready;
        view.owns_clusters_xyz = device_buffers_.owns_clusters_xyz;
        view.clusters_x = device_buffers_.clusters_x;
        view.clusters_y = device_buffers_.clusters_y;
        view.clusters_z = device_buffers_.clusters_z;
        view.clusters_q = device_buffers_.clusters_q;
        view.clusters_q_dx = device_buffers_.clusters_q_dx;
        view.clusters_q_dy = device_buffers_.clusters_q_dy;
        view.clusters_q_dz = device_buffers_.clusters_q_dz;
        view.clusters_p = device_buffers_.clusters_p;
        view.clusters_p_dx = device_buffers_.clusters_p_dx;
        view.clusters_p_dy = device_buffers_.clusters_p_dy;
        view.clusters_p_dz = device_buffers_.clusters_p_dz;
        view.elements_x = device_buffers_.elements_x;
        view.elements_y = device_buffers_.elements_y;
        view.elements_z = device_buffers_.elements_z;
        view.elements_nx = device_buffers_.elements_nx;
        view.elements_ny = device_buffers_.elements_ny;
        view.elements_nz = device_buffers_.elements_nz;
        view.elements_area = device_buffers_.elements_area;
        view.targets_q = device_buffers_.targets_q;
        view.targets_q_dx = device_buffers_.targets_q_dx;
        view.targets_q_dy = device_buffers_.targets_q_dy;
        view.targets_q_dz = device_buffers_.targets_q_dz;
        view.sources_q = device_buffers_.sources_q;
        view.sources_q_dx = device_buffers_.sources_q_dx;
        view.sources_q_dy = device_buffers_.sources_q_dy;
        view.sources_q_dz = device_buffers_.sources_q_dz;
        view.weights = device_buffers_.weights;
        view.potential_temp = device_buffers_.potential_temp;
        view.exact_idx_x = device_buffers_.exact_idx_x;
        view.exact_idx_y = device_buffers_.exact_idx_y;
        view.exact_idx_z = device_buffers_.exact_idx_z;
        view.denominator = device_buffers_.denominator;
        view.node_begin = device_buffers_.node_begin;
        view.node_end = device_buffers_.node_end;
        view.element_node_idx = device_buffers_.element_node_idx;
        view.pp_offsets = device_buffers_.pp_offsets;
        view.pp_sources = device_buffers_.pp_sources;
        view.pc_offsets = device_buffers_.pc_offsets;
        view.pc_sources = device_buffers_.pc_sources;
        view.cp_offsets = device_buffers_.cp_offsets;
        view.cp_sources = device_buffers_.cp_sources;
        view.cc_offsets = device_buffers_.cc_offsets;
        view.cc_sources = device_buffers_.cc_sources;
        view.level_nodes = device_buffers_.level_nodes;
        view.level_nodes_num = device_buffers_.level_nodes_num;
        view.num_nodes = device_buffers_.num_nodes;
        view.potential_num = potential_.size();
        view.state = device_state_;
#endif
        return view;
    }
    using View = DeviceView;

    View host_view() {
        auto elem_view = elements_.host_view();
        View view;
        view.ready = true;
        view.owns_clusters_xyz = false;
        view.clusters_x = interp_pts_.interp_x_ptr();
        view.clusters_y = interp_pts_.interp_y_ptr();
        view.clusters_z = interp_pts_.interp_z_ptr();
        view.clusters_q = interp_charge_.data();
        view.clusters_q_dx = interp_charge_dx_.data();
        view.clusters_q_dy = interp_charge_dy_.data();
        view.clusters_q_dz = interp_charge_dz_.data();
        view.clusters_p = interp_potential_.data();
        view.clusters_p_dx = interp_potential_dx_.data();
        view.clusters_p_dy = interp_potential_dy_.data();
        view.clusters_p_dz = interp_potential_dz_.data();
        view.elements_x = elem_view.x;
        view.elements_y = elem_view.y;
        view.elements_z = elem_view.z;
        view.elements_nx = elem_view.nx;
        view.elements_ny = elem_view.ny;
        view.elements_nz = elem_view.nz;
        view.elements_area = elem_view.area;
        view.targets_q = elem_view.target_q;
        view.targets_q_dx = elem_view.target_q_dx;
        view.targets_q_dy = elem_view.target_q_dy;
        view.targets_q_dz = elem_view.target_q_dz;
        view.sources_q = elem_view.source_q;
        view.sources_q_dx = elem_view.source_q_dx;
        view.sources_q_dy = elem_view.source_q_dy;
        view.sources_q_dz = elem_view.source_q_dz;
        view.weights = weights_.data();
        view.potential_temp = potential_temp_.data();
        view.exact_idx_x = exact_idx_x_.data();
        view.exact_idx_y = exact_idx_y_.data();
        view.exact_idx_z = exact_idx_z_.data();
        view.denominator = denominator_.data();
        view.node_begin = node_particles_begin_u32_.data();
        view.node_end = node_particles_end_u32_.data();
        view.element_node_idx = element_node_idx_u32_.data();
        view.pp_offsets = pp_offsets_u32_.data();
        view.pp_sources = pp_sources_u32_.data();
        view.pc_offsets = pc_offsets_u32_.data();
        view.pc_sources = pc_sources_u32_.data();
        view.cp_offsets = cp_offsets_u32_.data();
        view.cp_sources = cp_sources_u32_.data();
        view.cc_offsets = cc_offsets_u32_.data();
        view.cc_sources = cc_sources_u32_.data();
        view.level_nodes = level_nodes_.data();
        view.level_nodes_num = level_nodes_.size();
        view.num_nodes = tree_.num_nodes();
        view.potential_num = potential_.size();
#ifdef USE_CUDA_CC
        view.state = CudaDeviceState::HostOnly;
#endif
        return view;
    }

    BoundaryElement(class Elements& elements, const class InterpolationPoints& interp_pts,
             const class Tree& tree, const class InteractionList& interaction_list,
             const class Molecule& molecule, const struct Params& params, class Output& output,
             struct Timers_BoundaryElement& timers);
    ~BoundaryElement() = default;
    
    void run_GMRES();
    //void finalize();

};


struct Timers_BoundaryElement
{
    Timer ctor;
    Timer run_GMRES;
    
    Timer clear_charges;
    Timer clear_potentials;

    Timer matrix_vector;
    Timer precondition;
    
    Timer particle_particle_interact;
    Timer particle_cluster_interact;
    Timer cluster_particle_interact;
    Timer cluster_cluster_interact;
    Timer upward_pass;
    Timer downward_pass;
    
    Timer clear_cluster_charges;
    Timer clear_cluster_potentials;
    Timer copyin_clusters_to_device;
    Timer delete_clusters_from_device;

    void print() const;
    std::string get_durations() const;
    std::string get_headers() const;

    Timers_BoundaryElement() = default;
    ~Timers_BoundaryElement() = default;
};

#endif /* H_TABIPB_TREECODE_STRUCT_H */
