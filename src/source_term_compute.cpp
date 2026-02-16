#include <cmath>
// #include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#ifdef USE_CUDA_CC
#include "source_term_cuda.h"
#endif

#include "elements.h"
#include "constants.h"
#include "source_term_compute.h"

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#endif

SourceTermCompute::SourceTermCompute(std::vector<double>& source_term,
                      class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps_solute)
    : TreeCompute(mol_tree, elem_tree, interaction_list),
      elements_(elements),  elem_interp_pts_(elem_interp_pts),
      molecule_(molecule),  mol_interp_pts_ (mol_interp_pts),
      one_over_4pi_eps_solute_(constants::ONE_OVER_4PI / phys_eps_solute),
      source_term_(source_term), source_term_offset_(elements_.num())
{
//    timers_.ctor.start();
    
    /* Target clusters */
    
    num_elem_interp_pts_per_node_        = elem_interp_pts_.num_interp_pts_per_node();
    num_elem_interp_potentials_per_node_ = std::pow(num_elem_interp_pts_per_node_, 3);
    num_elem_potentials_                 = target_tree_.num_nodes() * num_elem_interp_potentials_per_node_;
    
    elem_interp_potential_   .assign(num_elem_potentials_, 0.);
    elem_interp_potential_dx_.assign(num_elem_potentials_, 0.);
    elem_interp_potential_dy_.assign(num_elem_potentials_, 0.);
    elem_interp_potential_dz_.assign(num_elem_potentials_, 0.);
    
    
    /* Source clusters */
    
    num_mol_interp_pts_per_node_     = mol_interp_pts_.num_interp_pts_per_node();
    num_mol_interp_charges_per_node_ = std::pow(num_mol_interp_pts_per_node_, 3);
    num_mol_charges_                 = source_tree_.num_nodes() * num_mol_interp_charges_per_node_;
    
    mol_interp_charge_.assign(num_mol_charges_, 0.);

    max_mol_particles_per_node_ = 0;
    for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
        auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
        std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
        if (num_particles > max_mol_particles_per_node_) {
            max_mol_particles_per_node_ = num_particles;
        }
    }

    mol_weights_.resize(num_mol_interp_pts_per_node_);
    for (int i = 0; i < num_mol_interp_pts_per_node_; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        if (i == 0 || i == num_mol_interp_pts_per_node_ - 1) {
            w *= 0.5;
        }
        mol_weights_[i] = w;
    }

    elem_weights_.resize(num_elem_interp_pts_per_node_);
    for (int i = 0; i < num_elem_interp_pts_per_node_; ++i) {
        double w = (i % 2 == 0) ? 1.0 : -1.0;
        if (i == 0 || i == num_elem_interp_pts_per_node_ - 1) {
            w *= 0.5;
        }
        elem_weights_[i] = w;
    }

    exact_idx_x_.assign(max_mol_particles_per_node_, -1);
    exact_idx_y_.assign(max_mol_particles_per_node_, -1);
    exact_idx_z_.assign(max_mol_particles_per_node_, -1);
    denominator_.assign(max_mol_particles_per_node_, 0.0);
    
//    timers_.ctor.stop();
}




void SourceTermCompute::compute()
{
    SourceTermCompute::copyin_clusters_to_device();
#if defined(USE_CUDA_CC)
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);
    if (require_all && !validate_device_buffers_common_()) {
        std::cerr << "[CUDA_SOURCE_TERM] require_all set but device buffers not ready. "
                  << "Aborting to avoid CPU/OpenACC fallback.\n";
        std::exit(1);
    }
#endif
    SourceTermCompute::run();
    SourceTermCompute::delete_clusters_from_device();
}


void SourceTermCompute::particle_particle_interact(std::array<std::size_t, 2> target_node_idxs,
                                                   std::array<std::size_t, 2> source_node_idxs)
{
//    timers_.particle_particle_interact.start();

    /* Targets */
    
    std::size_t target_node_begin          = target_node_idxs[0];
    std::size_t target_node_end            = target_node_idxs[1];

    const double* __restrict elem_x_ptr    = elements_.x_ptr();
    const double* __restrict elem_y_ptr    = elements_.y_ptr();
    const double* __restrict elem_z_ptr    = elements_.z_ptr();
    
    const double* __restrict elem_q_dx_ptr = elements_.nx_ptr();
    const double* __restrict elem_q_dy_ptr = elements_.ny_ptr();
    const double* __restrict elem_q_dz_ptr = elements_.nz_ptr();


    /* Sources */
    
    std::size_t source_node_begin = source_node_idxs[0];
    std::size_t source_node_end   = source_node_idxs[1];
    
    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();

    const double* __restrict mol_q_ptr = molecule_.charge_ptr();
    
    
    /* Potential */
    
    double* __restrict source_term_ptr = source_term_.data();

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_PP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_particle_particle_();
        if (present_ok) {
            const auto elem_dev = elements_.device_view();
            const auto mol_dev = molecule_.device_view();
            void* stream = nullptr;
            source_term_pp_cuda(
                elem_dev.x, elem_dev.y, elem_dev.z,
                elem_dev.nx, elem_dev.ny, elem_dev.nz,
                mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                target_node_begin, target_node_end,
                source_node_begin, source_node_end,
                one_over_4pi_eps_solute_,
                elem_dev.source_term, source_term_offset_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
    }
#endif


    for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
        
        double target_x = elem_x_ptr[j];
        double target_y = elem_y_ptr[j];
        double target_z = elem_z_ptr[j];
        
        double pot_temp_1  = 0.;
        double pot_temp_dx = 0.;
        double pot_temp_dy = 0.;
        double pot_temp_dz = 0.;
        
        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {

            double dx = mol_x_ptr[k] - target_x;
            double dy = mol_y_ptr[k] - target_y;
            double dz = mol_z_ptr[k] - target_z;

            double rinv  = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            double G0    = one_over_4pi_eps_solute_ * rinv;
            double Gn    = G0 * rinv * rinv;

            pot_temp_1  += G0 * mol_q_ptr[k];
            pot_temp_dx += Gn * mol_q_ptr[k] * dx;
            pot_temp_dy += Gn * mol_q_ptr[k] * dy;
            pot_temp_dz += Gn * mol_q_ptr[k] * dz;
        }
        
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        source_term_ptr[j]                       += pot_temp_1;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        source_term_ptr[j + source_term_offset_] += elem_q_dx_ptr[j] * pot_temp_dx
                                                  + elem_q_dy_ptr[j] * pot_temp_dy
                                                  + elem_q_dz_ptr[j] * pot_temp_dz;
    }

//    timers_.particle_particle_interact.stop();
}


void SourceTermCompute::particle_cluster_interact(std::array<std::size_t, 2> target_node_idxs,
                                         std::size_t source_node_idx)
{
//    timers_.particle_cluster_interact.start();
    
    /* Targets */
    
    std::size_t target_node_begin          = target_node_idxs[0];
    std::size_t target_node_end            = target_node_idxs[1];
    
    const double* __restrict elem_x_ptr    = elements_.x_ptr();
    const double* __restrict elem_y_ptr    = elements_.y_ptr();
    const double* __restrict elem_z_ptr    = elements_.z_ptr();
    
    const double* __restrict elem_q_dx_ptr = elements_.nx_ptr();
    const double* __restrict elem_q_dy_ptr = elements_.ny_ptr();
    const double* __restrict elem_q_dz_ptr = elements_.nz_ptr();


    /* Sources */
    
    int num_mol_interp_pts_per_node                 = num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node             = num_mol_interp_charges_per_node_;

    std::size_t source_cluster_interp_pts_begin     = source_node_idx * num_mol_interp_pts_per_node_;
    std::size_t source_cluster_interp_charges_begin = source_node_idx * num_mol_interp_charges_per_node_;
    
    const double* __restrict mol_clusters_x_ptr     = mol_interp_pts_.interp_x_ptr();
    const double* __restrict mol_clusters_y_ptr     = mol_interp_pts_.interp_y_ptr();
    const double* __restrict mol_clusters_z_ptr     = mol_interp_pts_.interp_z_ptr();
    
    const double* __restrict mol_clusters_q_ptr     = mol_interp_charge_.data();
    
    
    /* Potential */
    
    double* __restrict source_term_ptr = source_term_.data();
    
    
#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_PC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_particle_cluster_();
        if (present_ok) {
            const auto elem_dev = elements_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = nullptr;
            source_term_pc_cuda(
                elem_dev.x, elem_dev.y, elem_dev.z,
                elem_dev.nx, elem_dev.ny, elem_dev.nz,
                mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                self_dev.q,
                source_node_idx,
                num_mol_interp_pts_per_node_,
                num_mol_interp_charges_per_node_,
                target_node_begin,
                target_node_end,
                one_over_4pi_eps_solute_,
                elem_dev.source_term, source_term_offset_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
    }
#endif

    for (std::size_t j = target_node_begin; j < target_node_end; ++j) {

        double target_x = elem_x_ptr[j];
        double target_y = elem_y_ptr[j];
        double target_z = elem_z_ptr[j];
        
        double pot_temp_1  = 0.;
        double pot_temp_dx = 0.;
        double pot_temp_dy = 0.;
        double pot_temp_dz = 0.;
        
        for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
        for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                
            std::size_t kk = source_cluster_interp_charges_begin
                           + k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                           + k2 * num_mol_interp_pts_per_node + k3;

            double dx = mol_clusters_x_ptr[source_cluster_interp_pts_begin + k1] - target_x;
            double dy = mol_clusters_y_ptr[source_cluster_interp_pts_begin + k2] - target_y;
            double dz = mol_clusters_z_ptr[source_cluster_interp_pts_begin + k3] - target_z;

            double rinv  = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            double G0    = one_over_4pi_eps_solute_ * rinv;
            double Gn    = G0 * rinv * rinv;

            pot_temp_1  += G0 * mol_clusters_q_ptr[kk];
            pot_temp_dx += Gn * mol_clusters_q_ptr[kk] * dx;
            pot_temp_dy += Gn * mol_clusters_q_ptr[kk] * dy;
            pot_temp_dz += Gn * mol_clusters_q_ptr[kk] * dz;
        }
        }
        }
        
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        source_term_ptr[j]                       += pot_temp_1;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        source_term_ptr[j + source_term_offset_] += elem_q_dx_ptr[j] * pot_temp_dx
                                                  + elem_q_dy_ptr[j] * pot_temp_dy
                                                  + elem_q_dz_ptr[j] * pot_temp_dz;
    }

//    timers_.particle_cluster_interact.stop();
}


void SourceTermCompute::cluster_particle_interact(std::size_t target_node_idx,
                                                  std::array<std::size_t, 2> source_node_idxs)
{
//    timers_.cluster_particle_interact.start();

    /* Targets */
    
    int num_elem_interp_pts_per_node        = num_elem_interp_pts_per_node_;
    int num_elem_interp_potentials_per_node = num_elem_interp_potentials_per_node_;

    std::size_t target_cluster_interp_pts_begin        = target_node_idx * num_elem_interp_pts_per_node_;
    std::size_t target_cluster_interp_potentials_begin = target_node_idx * num_elem_interp_potentials_per_node_;

    const double* __restrict elem_clusters_x_ptr    = elem_interp_pts_.interp_x_ptr();
    const double* __restrict elem_clusters_y_ptr    = elem_interp_pts_.interp_y_ptr();
    const double* __restrict elem_clusters_z_ptr    = elem_interp_pts_.interp_z_ptr();

    double*       __restrict elem_clusters_p_ptr    = elem_interp_potential_.data();
    double*       __restrict elem_clusters_p_dx_ptr = elem_interp_potential_dx_.data();
    double*       __restrict elem_clusters_p_dy_ptr = elem_interp_potential_dy_.data();
    double*       __restrict elem_clusters_p_dz_ptr = elem_interp_potential_dz_.data();


    /* Sources */
    
    std::size_t source_node_begin      = source_node_idxs[0];
    std::size_t source_node_end        = source_node_idxs[1];

    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();

    const double* __restrict mol_q_ptr = molecule_.charge_ptr();

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_CP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_cluster_particle_();
        if (present_ok) {
            const auto elem_interp_dev = elem_interp_pts_.device_view();
            const auto mol_dev = molecule_.device_view();
            const auto self_dev = device_view();
            void* stream = nullptr;
            source_term_cp_cuda(
                elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
                self_dev.p, self_dev.p_dx,
                self_dev.p_dy, self_dev.p_dz,
                mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                target_node_idx,
                num_elem_interp_pts_per_node_,
                num_elem_interp_potentials_per_node_,
                source_node_begin,
                source_node_end,
                one_over_4pi_eps_solute_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
    }
#endif


    for (int j1 = 0; j1 < num_elem_interp_pts_per_node; ++j1) {
    for (int j2 = 0; j2 < num_elem_interp_pts_per_node; ++j2) {
    for (int j3 = 0; j3 < num_elem_interp_pts_per_node; ++j3) {
    
        std::size_t jj = target_cluster_interp_potentials_begin
                       + j1 * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                       + j2 * num_elem_interp_pts_per_node + j3;

        double target_x = elem_clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = elem_clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = elem_clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_temp_1  = 0.;
        double pot_temp_dx = 0.;
        double pot_temp_dy = 0.;
        double pot_temp_dz = 0.;
    
        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {

            double dx = mol_x_ptr[k] - target_x;
            double dy = mol_y_ptr[k] - target_y;
            double dz = mol_z_ptr[k] - target_z;

            double rinv  = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            double G0    = one_over_4pi_eps_solute_ * rinv;
            double Gn    = G0 * rinv * rinv;

            pot_temp_1  += G0 * mol_q_ptr[k];
            pot_temp_dx += Gn * mol_q_ptr[k] * dx;
            pot_temp_dy += Gn * mol_q_ptr[k] * dy;
            pot_temp_dz += Gn * mol_q_ptr[k] * dz;
        }
    
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_ptr   [jj] += pot_temp_1;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dx_ptr[jj] += pot_temp_dx;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dy_ptr[jj] += pot_temp_dy;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dz_ptr[jj] += pot_temp_dz;
    }
    }
    }

//    timers_.cluster_particle_interact.stop();
}


void SourceTermCompute::cluster_cluster_interact(std::size_t target_node_idx,
                                                 std::size_t source_node_idx)
{
//    timers_.cluster_cluster_interact.start();
        
    /* Targets */
    
    int num_elem_interp_pts_per_node                   = num_elem_interp_pts_per_node_;
    int num_elem_interp_potentials_per_node            = num_elem_interp_potentials_per_node_;
    
    std::size_t target_cluster_interp_pts_begin        = target_node_idx * num_elem_interp_pts_per_node_;
    std::size_t target_cluster_interp_potentials_begin = target_node_idx * num_elem_interp_potentials_per_node_;
    
    const double* __restrict elem_clusters_x_ptr       = elem_interp_pts_.interp_x_ptr();
    const double* __restrict elem_clusters_y_ptr       = elem_interp_pts_.interp_y_ptr();
    const double* __restrict elem_clusters_z_ptr       = elem_interp_pts_.interp_z_ptr();

    double*       __restrict elem_clusters_p_ptr       = elem_interp_potential_.data();
    double*       __restrict elem_clusters_p_dx_ptr    = elem_interp_potential_dx_.data();
    double*       __restrict elem_clusters_p_dy_ptr    = elem_interp_potential_dy_.data();
    double*       __restrict elem_clusters_p_dz_ptr    = elem_interp_potential_dz_.data();
    
    
    /* Sources */
    
    int num_mol_interp_pts_per_node                 = num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node             = num_mol_interp_charges_per_node_;

    std::size_t source_cluster_interp_pts_begin     = source_node_idx * num_mol_interp_pts_per_node;
    std::size_t source_cluster_interp_charges_begin = source_node_idx * num_mol_interp_charges_per_node;
    
    const double* __restrict mol_clusters_x_ptr     = mol_interp_pts_.interp_x_ptr();
    const double* __restrict mol_clusters_y_ptr     = mol_interp_pts_.interp_y_ptr();
    const double* __restrict mol_clusters_z_ptr     = mol_interp_pts_.interp_z_ptr();
    
    const double* __restrict mol_clusters_q_ptr     = mol_interp_charge_.data();

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_CC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_cluster_cluster_();
        if (present_ok) {
            const auto elem_interp_dev = elem_interp_pts_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = nullptr;
            source_term_cc_cuda(
                elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
                self_dev.p, self_dev.p_dx,
                self_dev.p_dy, self_dev.p_dz,
                mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                self_dev.q,
                target_node_idx,
                source_node_idx,
                num_elem_interp_pts_per_node_,
                num_elem_interp_potentials_per_node_,
                num_mol_interp_pts_per_node_,
                num_mol_interp_charges_per_node_,
                one_over_4pi_eps_solute_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
    }
#endif


    for (int j1 = 0; j1 < num_elem_interp_pts_per_node; j1++) {
    for (int j2 = 0; j2 < num_elem_interp_pts_per_node; j2++) {
    for (int j3 = 0; j3 < num_elem_interp_pts_per_node; j3++) {
    
        std::size_t jj = target_cluster_interp_potentials_begin
                       + j1 * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                       + j2 * num_elem_interp_pts_per_node + j3;

        double target_x = elem_clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = elem_clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = elem_clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_temp_1  = 0.;
        double pot_temp_dx = 0.;
        double pot_temp_dy = 0.;
        double pot_temp_dz = 0.;
    
        for (int k1 = 0; k1 < num_mol_interp_pts_per_node; k1++) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; k2++) {
        for (int k3 = 0; k3 < num_mol_interp_pts_per_node; k3++) {
            
            std::size_t kk = source_cluster_interp_charges_begin
                           + k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                           + k2 * num_mol_interp_pts_per_node + k3;

            double dx = mol_clusters_x_ptr[source_cluster_interp_pts_begin + k1] - target_x;
            double dy = mol_clusters_y_ptr[source_cluster_interp_pts_begin + k2] - target_y;
            double dz = mol_clusters_z_ptr[source_cluster_interp_pts_begin + k3] - target_z;

            double rinv  = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz);
            double G0    = one_over_4pi_eps_solute_ * rinv;
            double Gn    = G0 * rinv * rinv;

            pot_temp_1  += G0 * mol_clusters_q_ptr[kk];
            pot_temp_dx += Gn * mol_clusters_q_ptr[kk] * dx;
            pot_temp_dy += Gn * mol_clusters_q_ptr[kk] * dy;
            pot_temp_dz += Gn * mol_clusters_q_ptr[kk] * dz;
        }
        }
        }
    
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_ptr   [jj] += pot_temp_1;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dx_ptr[jj] += pot_temp_dx;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dy_ptr[jj] += pot_temp_dy;
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        elem_clusters_p_dz_ptr[jj] += pot_temp_dz;
    }
    }
    }

//    timers_.cluster_cluster_interact.stop();
}


void SourceTermCompute::upward_pass()
{
//    timers_.upward_pass.start();

    int num_mol_interp_pts_per_node     = num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node = num_mol_interp_charges_per_node_;
    
    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();
    
    const double* __restrict mol_q_ptr = molecule_.charge_ptr();
    
    const double* __restrict mol_clusters_x_ptr = mol_interp_pts_.interp_x_ptr();
    const double* __restrict mol_clusters_y_ptr = mol_interp_pts_.interp_y_ptr();
    const double* __restrict mol_clusters_z_ptr = mol_interp_pts_.interp_z_ptr();
    
    double*       __restrict mol_clusters_q_ptr = mol_interp_charge_.data();
    
        
    double* weights_ptr = mol_weights_.data();
    int* exact_idx_x_ptr = exact_idx_x_.data();
    int* exact_idx_y_ptr = exact_idx_y_.data();
    int* exact_idx_z_ptr = exact_idx_z_.data();
    double* denominator_ptr = denominator_.data();

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_UP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_upward_pass_();
        if (present_ok) {
            const auto mol_dev = molecule_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = nullptr;
            for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
                auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
                std::size_t particle_start = particle_idxs[0];
                std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
                if (num_particles == 0) {
                    continue;
                }
                source_term_up_cuda(
                    mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                    mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                    self_dev.q,
                    self_dev.mol_weights,
                    self_dev.exact_idx_x, self_dev.exact_idx_y, self_dev.exact_idx_z,
                    self_dev.denominator,
                    node_idx,
                    num_mol_interp_pts_per_node_,
                    num_mol_interp_charges_per_node_,
                    particle_start,
                    num_particles,
                    stream);
                CUDA_CHECK_LAST_KERNEL();
            }
            CUDA_SYNC_AND_CHECK();
            return;
        }
    }
#endif
    
    for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
        
        auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
        
        std::size_t node_interp_pts_start = node_idx * num_mol_interp_pts_per_node;
        std::size_t node_charges_start    = node_idx * num_mol_interp_charges_per_node;
        
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles  = particle_idxs[1] - particle_idxs[0];
        
        {

        for (std::size_t i = 0; i < num_particles; ++i) {
            exact_idx_x_ptr[i] = -1;
            exact_idx_y_ptr[i] = -1;
            exact_idx_z_ptr[i] = -1;
        }

        for (std::size_t i = 0; i < num_particles; ++i) {
        
            double denominator_x = 0.;
            double denominator_y = 0.;
            double denominator_z = 0.;
            int ex = -1, ey = -1, ez = -1;
            
            double xx = mol_x_ptr[particle_start + i];
            double yy = mol_y_ptr[particle_start + i];
            double zz = mol_z_ptr[particle_start + i];

            // because there's a reduction over exact_idx[i], this loop carries a
            // backward dependence and won't actually parallelize
            for (int j = 0; j < num_mol_interp_pts_per_node; ++j) {
            
                double dist_x = xx - mol_clusters_x_ptr[node_interp_pts_start + j];
                double dist_y = yy - mol_clusters_y_ptr[node_interp_pts_start + j];
                double dist_z = zz - mol_clusters_z_ptr[node_interp_pts_start + j];
                
                denominator_x += weights_ptr[j] / dist_x;
                denominator_y += weights_ptr[j] / dist_y;
                denominator_z += weights_ptr[j] / dist_z;
                
                const int cx = (std::abs(dist_x) < std::numeric_limits<double>::min()) ? j : -1;
                const int cy = (std::abs(dist_y) < std::numeric_limits<double>::min()) ? j : -1;
                const int cz = (std::abs(dist_z) < std::numeric_limits<double>::min()) ? j : -1;

                ex = (ex > cx) ? ex : cx;
                ey = (ey > cy) ? ey : cy;
                ez = (ez > cz) ? ez : cz;
            }

            exact_idx_x_ptr[i] = ex;
            exact_idx_y_ptr[i] = ey;
            exact_idx_z_ptr[i] = ez;
            
            denominator_ptr[i] = 1.0;
            if (exact_idx_x_ptr[i] == -1) denominator_ptr[i] /= denominator_x;
            if (exact_idx_y_ptr[i] == -1) denominator_ptr[i] /= denominator_y;
            if (exact_idx_z_ptr[i] == -1) denominator_ptr[i] /= denominator_z;
        }

        for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
        for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
        
            std::size_t kk = node_charges_start
                   + k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                   + k2 * num_mol_interp_pts_per_node + k3;
                   
            double cx = mol_clusters_x_ptr[node_interp_pts_start + k1];
            double w1 = weights_ptr[k1];

            double cy = mol_clusters_y_ptr[node_interp_pts_start + k2];
            double w2 = weights_ptr[k2];
            
            double cz = mol_clusters_z_ptr[node_interp_pts_start + k3];
            double w3 = weights_ptr[k3];
            
            double q_temp = 0.;
            
            for (std::size_t i = 0; i < num_particles; i++) {  // loop over source points
            
                double dist_x = mol_x_ptr[particle_start + i] - cx;
                double dist_y = mol_y_ptr[particle_start + i] - cy;
                double dist_z = mol_z_ptr[particle_start + i] - cz;
                
                double numerator = 1.;

                // If exact_idx[i] == -1, then no issues.
                // If exact_idx[i] != -1, then we want to zero out terms EXCEPT when exactInd=k1.
                if (exact_idx_x_ptr[i] == -1) {
                    numerator *= w1 / dist_x;
                } else {
                    if (exact_idx_x_ptr[i] != k1) numerator *= 0.;
                }

                if (exact_idx_y_ptr[i] == -1) {
                    numerator *= w2 / dist_y;
                } else {
                    if (exact_idx_y_ptr[i] != k2) numerator *= 0.;
                }

                if (exact_idx_z_ptr[i] == -1) {
                    numerator *= w3 / dist_z;
                } else {
                    if (exact_idx_z_ptr[i] != k3) numerator *= 0.;
                }

                q_temp += mol_q_ptr[particle_start + i] * numerator * denominator_ptr[i];
            }
            
            mol_clusters_q_ptr[kk] += q_temp;
        }
        }
        }
        
        } // end parallel region
    } // end loop over nodes

//    timers_.upward_pass.stop();
}


void SourceTermCompute::downward_pass()
{
//    timers_.downward_pass.start();

    int num_elem_interp_pts_per_node        = num_elem_interp_pts_per_node_;
    int num_elem_interp_potentials_per_node = num_elem_interp_potentials_per_node_;
    
    double*       __restrict source_term_ptr = source_term_.data();
    
    const double* __restrict elem_x_ptr = elements_.x_ptr();
    const double* __restrict elem_y_ptr = elements_.y_ptr();
    const double* __restrict elem_z_ptr = elements_.z_ptr();
    
    const double* __restrict elem_q_dx_ptr = elements_.nx_ptr();
    const double* __restrict elem_q_dy_ptr = elements_.ny_ptr();
    const double* __restrict elem_q_dz_ptr = elements_.nz_ptr();
    
    const double* __restrict elem_clusters_x_ptr = elem_interp_pts_.interp_x_ptr();
    const double* __restrict elem_clusters_y_ptr = elem_interp_pts_.interp_y_ptr();
    const double* __restrict elem_clusters_z_ptr = elem_interp_pts_.interp_z_ptr();
    
    const double* __restrict elem_clusters_p_ptr    = elem_interp_potential_.data();
    const double* __restrict elem_clusters_p_dx_ptr = elem_interp_potential_dx_.data();
    const double* __restrict elem_clusters_p_dy_ptr = elem_interp_potential_dy_.data();
    const double* __restrict elem_clusters_p_dz_ptr = elem_interp_potential_dz_.data();
    
    double* weights_ptr = elem_weights_.data();

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_SOURCE_TERM_DOWN");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_downward_pass_();
        if (present_ok) {
            const auto elem_dev = elements_.device_view();
            const auto elem_interp_dev = elem_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = nullptr;
            for (std::size_t node_idx = 0; node_idx < target_tree_.num_nodes(); ++node_idx) {
                auto particle_idxs = target_tree_.node_particle_idxs(node_idx);
                std::size_t particle_start = particle_idxs[0];
                std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
                if (num_particles == 0) {
                    continue;
                }
                source_term_down_cuda(
                    elem_dev.x, elem_dev.y, elem_dev.z,
                    elem_dev.nx, elem_dev.ny, elem_dev.nz,
                    elem_interp_dev.interp_x, elem_interp_dev.interp_y, elem_interp_dev.interp_z,
                    self_dev.p, self_dev.p_dx,
                    self_dev.p_dy, self_dev.p_dz,
                    self_dev.elem_weights,
                    node_idx,
                    num_elem_interp_pts_per_node_,
                    num_elem_interp_potentials_per_node_,
                    particle_start,
                    num_particles,
                    elem_dev.source_term,
                    source_term_offset_,
                    stream);
                CUDA_CHECK_LAST_KERNEL();
            }
            CUDA_SYNC_AND_CHECK();
            return;
        }
    }
#endif
    
    for (std::size_t node_idx = 0; node_idx < target_tree_.num_nodes(); ++node_idx) {
        
        auto particle_idxs = target_tree_.node_particle_idxs(node_idx);
        std::size_t node_interp_pts_start = node_idx * num_elem_interp_pts_per_node;
        std::size_t node_potentials_start = node_idx * num_elem_interp_potentials_per_node;
        
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles  = particle_idxs[1] - particle_idxs[0];

        for (std::size_t i = 0; i < num_particles; ++i) {
        
            double denominator_x = 0.;
            double denominator_y = 0.;
            double denominator_z = 0.;
            
            int exact_idx_x = -1;
            int exact_idx_y = -1;
            int exact_idx_z = -1;
            
            double xx = elem_x_ptr[particle_start + i];
            double yy = elem_y_ptr[particle_start + i];
            double zz = elem_z_ptr[particle_start + i];
            
            for (int j = 0; j < num_elem_interp_pts_per_node; ++j) {
            
                double dist_x = xx - elem_clusters_x_ptr[node_interp_pts_start + j];
                double dist_y = yy - elem_clusters_y_ptr[node_interp_pts_start + j];
                double dist_z = zz - elem_clusters_z_ptr[node_interp_pts_start + j];
                
                denominator_x += weights_ptr[j] / dist_x;
                denominator_y += weights_ptr[j] / dist_y;
                denominator_z += weights_ptr[j] / dist_z;
                
                const int cx = (std::abs(dist_x) < std::numeric_limits<double>::min()) ? j : -1;
                const int cy = (std::abs(dist_y) < std::numeric_limits<double>::min()) ? j : -1;
                const int cz = (std::abs(dist_z) < std::numeric_limits<double>::min()) ? j : -1;

                exact_idx_x = (exact_idx_x > cx) ? exact_idx_x : cx;
                exact_idx_y = (exact_idx_y > cy) ? exact_idx_y : cy;
                exact_idx_z = (exact_idx_z > cz) ? exact_idx_z : cz;
            }
            
            double denominator = 1.;
            if (exact_idx_x == -1) denominator /= denominator_x;
            if (exact_idx_y == -1) denominator /= denominator_y;
            if (exact_idx_z == -1) denominator /= denominator_z;

            double pot_temp_1  = 0.;
            double pot_temp_dx = 0.;
            double pot_temp_dy = 0.;
            double pot_temp_dz = 0.;
            
            for (int k1 = 0; k1 < num_elem_interp_pts_per_node; ++k1) {
            for (int k2 = 0; k2 < num_elem_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_elem_interp_pts_per_node; ++k3) {
                    
                std::size_t kk = node_potentials_start
                               + k1 * num_elem_interp_pts_per_node * num_elem_interp_pts_per_node
                               + k2 * num_elem_interp_pts_per_node + k3;
                               
                double dist_x = xx - elem_clusters_x_ptr[node_interp_pts_start + k1];
                double dist_y = yy - elem_clusters_y_ptr[node_interp_pts_start + k2];
                double dist_z = zz - elem_clusters_z_ptr[node_interp_pts_start + k3];
                
                double numerator = 1.;

                // If exact_idx == -1, then no issues.
                // If exact_idx != -1, then we want to zero out terms EXCEPT when exactInd=k1.
                if (exact_idx_x == -1) {
                    numerator *= weights_ptr[k1] / dist_x;
                } else {
                    if (exact_idx_x != k1) numerator *= 0.;
                }

                if (exact_idx_y == -1) {
                    numerator *= weights_ptr[k2] / dist_y;
                } else {
                    if (exact_idx_y != k2) numerator *= 0.;
                }

                if (exact_idx_z == -1) {
                    numerator *= weights_ptr[k3] / dist_z;
                } else {
                    if (exact_idx_z != k3) numerator *= 0.;
                }

                pot_temp_1  += numerator * denominator * elem_clusters_p_ptr   [kk];
                pot_temp_dx += numerator * denominator * elem_clusters_p_dx_ptr[kk];
                pot_temp_dy += numerator * denominator * elem_clusters_p_dy_ptr[kk];
                pot_temp_dz += numerator * denominator * elem_clusters_p_dz_ptr[kk];
            }
            }
            }
            
            double pot_temp_2 = elem_q_dx_ptr[particle_start + i] * pot_temp_dx
                              + elem_q_dy_ptr[particle_start + i] * pot_temp_dy
                              + elem_q_dz_ptr[particle_start + i] * pot_temp_dz;
            source_term_ptr[particle_start + i]                       += pot_temp_1;
            source_term_ptr[particle_start + i + source_term_offset_] += pot_temp_2;
        }
    } //end loop over nodes

//    timers_.downward_pass.stop();
}

#ifdef USE_CUDA_CC
bool SourceTermCompute::validate_device_buffers_common_() const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!elements_.cuda_device_ready() || !molecule_.cuda_device_ready() ||
        !elem_interp_pts_.cuda_device_ready() || !mol_interp_pts_.cuda_device_ready()) {
        return false;
    }

    const auto& buf = device_buffers_;
    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = elem_interp_potential_.size();
    const std::size_t p_dx_num = elem_interp_potential_dx_.size();
    const std::size_t p_dy_num = elem_interp_potential_dy_.size();
    const std::size_t p_dz_num = elem_interp_potential_dz_.size();
    const std::size_t mol_weights_num = mol_weights_.size();
    const std::size_t elem_weights_num = elem_weights_.size();
    const std::size_t scratch_num = exact_idx_x_.size();

    if (buf.q_num != q_num || buf.p_num != p_num || buf.p_dx_num != p_dx_num ||
        buf.p_dy_num != p_dy_num || buf.p_dz_num != p_dz_num ||
        buf.mol_weights_num != mol_weights_num ||
        buf.elem_weights_num != elem_weights_num ||
        buf.scratch_num != scratch_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (p_dx_num > 0 && !buf.p_dx_dev) ||
        (p_dy_num > 0 && !buf.p_dy_dev) ||
        (p_dz_num > 0 && !buf.p_dz_dev) ||
        (mol_weights_num > 0 && !buf.mol_weights_dev) ||
        (elem_weights_num > 0 && !buf.elem_weights_dev) ||
        (scratch_num > 0 &&
         (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
          !buf.denominator_dev))) {
        return false;
    }
    return true;
}

bool SourceTermCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}

bool SourceTermCompute::validate_device_buffers_downward_pass_() const
{
    return validate_device_buffers_common_();
}
#endif

void SourceTermCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);

    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();
    
    const double* p_ptr    = elem_interp_potential_.data();
    const double* p_dx_ptr = elem_interp_potential_dx_.data();
    const double* p_dy_ptr = elem_interp_potential_dy_.data();
    const double* p_dz_ptr = elem_interp_potential_dz_.data();
    
    std::size_t p_num    = elem_interp_potential_.size();
    std::size_t p_dx_num = elem_interp_potential_dx_.size();
    std::size_t p_dy_num = elem_interp_potential_dy_.size();
    std::size_t p_dz_num = elem_interp_potential_dz_.size();
    
    const double* mol_weights_ptr = mol_weights_.data();
    std::size_t mol_weights_num = mol_weights_.size();
    const double* elem_weights_ptr = elem_weights_.data();
    std::size_t elem_weights_num = elem_weights_.size();

    const std::size_t scratch_num = max_mol_particles_per_node_;

    auto &buf = device_buffers_;
    if (buf.q_num != 0 && (buf.q_num != q_num || buf.p_num != p_num ||
                           buf.p_dx_num != p_dx_num || buf.p_dy_num != p_dy_num ||
                           buf.p_dz_num != p_dz_num || buf.mol_weights_num != mol_weights_num ||
                           buf.elem_weights_num != elem_weights_num ||
                           buf.scratch_num != scratch_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.p_dx_dev);
        CUDA_FREE_AND_NULL(buf.p_dy_dev);
        CUDA_FREE_AND_NULL(buf.p_dz_dev);
        CUDA_FREE_AND_NULL(buf.mol_weights_dev);
        CUDA_FREE_AND_NULL(buf.elem_weights_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        buf = DeviceBuffers{};
    }

    if (buf.q_num == 0 && (q_num > 0 || p_num > 0 || mol_weights_num > 0 ||
                           elem_weights_num > 0 || scratch_num > 0)) {
        if (q_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.q_dev, q_num * sizeof(double));
            buf.q_num = q_num;
        }
        if (p_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dev, p_num * sizeof(double));
            buf.p_num = p_num;
        }
        if (p_dx_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dx_dev, p_dx_num * sizeof(double));
            buf.p_dx_num = p_dx_num;
        }
        if (p_dy_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dy_dev, p_dy_num * sizeof(double));
            buf.p_dy_num = p_dy_num;
        }
        if (p_dz_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dz_dev, p_dz_num * sizeof(double));
            buf.p_dz_num = p_dz_num;
        }
        if (mol_weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.mol_weights_dev, mol_weights_num * sizeof(double));
            buf.mol_weights_num = mol_weights_num;
        }
        if (elem_weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.elem_weights_dev, elem_weights_num * sizeof(double));
            buf.elem_weights_num = elem_weights_num;
        }
        if (scratch_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_x_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_y_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_z_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.denominator_dev, scratch_num * sizeof(double));
            buf.scratch_num = scratch_num;
        }
    }

    cudaStream_t stream = nullptr;

    if (q_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dx_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dx_dev, p_dx_ptr, p_dx_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dy_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dy_dev, p_dy_ptr, p_dy_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_dz_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dz_dev, p_dz_ptr, p_dz_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (mol_weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.mol_weights_dev, mol_weights_ptr,
                          mol_weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (elem_weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.elem_weights_dev, elem_weights_ptr,
                          elem_weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    buf.ready = true;
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        if ((q_num > 0 && !buf.q_dev) ||
            (p_num > 0 && !buf.p_dev) ||
            (p_dx_num > 0 && !buf.p_dx_dev) ||
            (p_dy_num > 0 && !buf.p_dy_dev) ||
            (p_dz_num > 0 && !buf.p_dz_dev) ||
            (mol_weights_num > 0 && !buf.mol_weights_dev) ||
            (elem_weights_num > 0 && !buf.elem_weights_dev) ||
            (scratch_num > 0 && (!buf.exact_idx_x_dev ||
                                 !buf.exact_idx_y_dev ||
                                 !buf.exact_idx_z_dev ||
                                 !buf.denominator_dev))) {
            std::fprintf(stderr,
                         "[CUDA] SourceTermCompute missing device buffers "
                         "under TABIPB_CUDA_REQUIRE_ALL=1\n");
            std::abort();
        }
    }
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void SourceTermCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    auto &buf = device_buffers_;
    CUDA_FREE_AND_NULL(buf.q_dev);
    CUDA_FREE_AND_NULL(buf.p_dev);
    CUDA_FREE_AND_NULL(buf.p_dx_dev);
    CUDA_FREE_AND_NULL(buf.p_dy_dev);
    CUDA_FREE_AND_NULL(buf.p_dz_dev);
    CUDA_FREE_AND_NULL(buf.mol_weights_dev);
    CUDA_FREE_AND_NULL(buf.elem_weights_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
    CUDA_FREE_AND_NULL(buf.denominator_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
#endif

//    timers_.delete_clusters_from_device.stop();
}
