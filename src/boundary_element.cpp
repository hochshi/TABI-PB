#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "constants.h"
#include "boundary_element.h"
#ifdef USE_CUDA_CC
#include <openacc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cc_cuda.h"
#include "up_cuda.h"

extern "C" {
    CUcontext acc_get_cuda_context(void) __attribute__((weak));
}
#endif


BoundaryElement::BoundaryElement(class Elements& elements, const class InterpolationPoints& interp_pts,
         const class Tree& tree, const class InteractionList& interaction_list,
         const class Molecule& molecule, 
         const struct Params& params, class Output& output, struct Timers_BoundaryElement& timers)
    : elements_(elements), interp_pts_(interp_pts), tree_(tree),
      interaction_list_(interaction_list), molecule_(molecule), 
      params_(params), output_(output), timers_(timers)
{
    timers_.ctor.start();

    potential_.assign(2 * elements_.num(), 0.);
    
    num_charges_per_node_ = std::pow(interp_pts_.num_interp_pts_per_node(), 3);
    num_charges_          = tree_.num_nodes() * num_charges_per_node_;
    
    interp_charge_.resize(num_charges_);
    interp_charge_dx_.resize(num_charges_);
    interp_charge_dy_.resize(num_charges_);
    interp_charge_dz_.resize(num_charges_);
    
    interp_potential_.resize(num_charges_);
    interp_potential_dx_.resize(num_charges_);
    interp_potential_dy_.resize(num_charges_);
    interp_potential_dz_.resize(num_charges_);

    potential_temp_.resize(potential_.size());

    weights_.resize(interp_pts_.num_interp_pts_per_node());
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        double w = ((i % 2 == 0) ? 1.0 : -1.0);
        if (i == 0 || i + 1 == weights_.size()) w *= 0.5;
        weights_[i] = w;
    }

    std::size_t max_particles = elements_.num();
    exact_idx_x_.resize(max_particles);
    exact_idx_y_.resize(max_particles);
    exact_idx_z_.resize(max_particles);
    denominator_.resize(max_particles);

    std::size_t num_nodes = tree_.num_nodes();
    node_particles_begin_.resize(num_nodes);
    node_particles_end_.resize(num_nodes);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        auto idxs = tree_.node_particle_idxs(i);
        node_particles_begin_[i] = idxs[0];
        node_particles_end_[i] = idxs[1];
    }

    const auto& node_levels = tree_.node_levels();
    std::size_t max_depth = tree_.max_depth();
    level_offsets_.assign(max_depth + 1, 0);
    for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
        std::size_t lvl = node_levels[node_idx];
        if (lvl + 1 >= level_offsets_.size()) {
            level_offsets_.resize(lvl + 2, 0);
        }
        level_offsets_[lvl + 1]++;
    }
    for (std::size_t lvl = 1; lvl < level_offsets_.size(); ++lvl) {
        level_offsets_[lvl] += level_offsets_[lvl - 1];
    }
    level_nodes_.resize(num_nodes);
    std::vector<std::size_t> level_cursor = level_offsets_;
    for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
        std::size_t lvl = node_levels[node_idx];
        level_nodes_[level_cursor[lvl]++] = node_idx;
    }

    element_node_idx_.resize(elements_.num());
    for (std::size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
        std::size_t begin = node_particles_begin_[node_idx];
        std::size_t end = node_particles_end_[node_idx];
        for (std::size_t j = begin; j < end; ++j) {
            element_node_idx_[j] = node_idx;
        }
    }

    node_particles_begin_u32_.resize(num_nodes);
    node_particles_end_u32_.resize(num_nodes);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        node_particles_begin_u32_[i] = static_cast<std::uint32_t>(node_particles_begin_[i]);
        node_particles_end_u32_[i]   = static_cast<std::uint32_t>(node_particles_end_[i]);
    }

    element_node_idx_u32_.resize(element_node_idx_.size());
    for (std::size_t i = 0; i < element_node_idx_.size(); ++i)
        element_node_idx_u32_[i] = static_cast<std::uint32_t>(element_node_idx_[i]);

    const auto& pp_offsets = interaction_list_.particle_particle_offsets();
    const auto& pp_sources = interaction_list_.particle_particle_flat();
    const auto& pc_offsets = interaction_list_.particle_cluster_offsets();
    const auto& pc_sources = interaction_list_.particle_cluster_flat();
    const auto& cp_offsets = interaction_list_.cluster_particle_offsets();
    const auto& cp_sources = interaction_list_.cluster_particle_flat();
    const auto& cc_offsets = interaction_list_.cluster_cluster_offsets();
    const auto& cc_sources = interaction_list_.cluster_cluster_flat();

    pp_offsets_u32_.resize(pp_offsets.size());
    pp_sources_u32_.resize(pp_sources.size());
    pc_offsets_u32_.resize(pc_offsets.size());
    pc_sources_u32_.resize(pc_sources.size());
    cp_offsets_u32_.resize(cp_offsets.size());
    cp_sources_u32_.resize(cp_sources.size());
    cc_offsets_u32_.resize(cc_offsets.size());
    cc_sources_u32_.resize(cc_sources.size());

    for (std::size_t i = 0; i < pp_offsets.size(); ++i)
        pp_offsets_u32_[i] = static_cast<std::uint32_t>(pp_offsets[i]);
    for (std::size_t i = 0; i < pp_sources.size(); ++i)
        pp_sources_u32_[i] = static_cast<std::uint32_t>(pp_sources[i]);
    for (std::size_t i = 0; i < pc_offsets.size(); ++i)
        pc_offsets_u32_[i] = static_cast<std::uint32_t>(pc_offsets[i]);
    for (std::size_t i = 0; i < pc_sources.size(); ++i)
        pc_sources_u32_[i] = static_cast<std::uint32_t>(pc_sources[i]);
    for (std::size_t i = 0; i < cp_offsets.size(); ++i)
        cp_offsets_u32_[i] = static_cast<std::uint32_t>(cp_offsets[i]);
    for (std::size_t i = 0; i < cp_sources.size(); ++i)
        cp_sources_u32_[i] = static_cast<std::uint32_t>(cp_sources[i]);
    for (std::size_t i = 0; i < cc_offsets.size(); ++i)
        cc_offsets_u32_[i] = static_cast<std::uint32_t>(cc_offsets[i]);
    for (std::size_t i = 0; i < cc_sources.size(); ++i)
        cc_sources_u32_[i] = static_cast<std::uint32_t>(cc_sources[i]);

    timers_.ctor.stop();
}
          
void BoundaryElement::run_GMRES()
{
    timers_.run_GMRES.start();

    long int restrt = params_.gmres_restart_;
    long int length = output_.potential().size();
    long int ldw    = length;
    long int ldh    = restrt + 1;
    
    // These values are modified on return
    double residual   = params_.gmres_residual_;
    long int num_iter = params_.gmres_num_iter_;

    std::vector<double> work_vec(ldw * (restrt + 4));
    std::vector<double> h_vec   (ldh * (restrt + 2));
    
    double* work = work_vec.data();
    double* h    = h_vec.data();

    BoundaryElement::copyin_clusters_to_device();
    
    int err_code = BoundaryElement::gmres_(length, elements_.source_term_ptr(), output_.potential().data(),
                                    restrt, work, ldw, h, ldh, num_iter, residual);

    BoundaryElement::delete_clusters_from_device();
    
    output_.set_residual(residual);
    output_.set_num_iter(num_iter);

    if (err_code) {
        std::cout << "GMRES error code " << err_code << ". Exiting.";
        std::exit(1);
    }
    
    std::cout << "GMRES completed. " << num_iter << " iterations, " << residual << " residual.";

    timers_.run_GMRES.stop();
}


void BoundaryElement::matrix_vector(double alpha, const double* __restrict potential_old,
                                     double beta,       double* __restrict potential_new)
{
    timers_.matrix_vector.start();

    double potential_coeff_1 = 0.5 * (1. +      params_.phys_eps_);
    double potential_coeff_2 = 0.5 * (1. + 1. / params_.phys_eps_);
    
    std::size_t potential_num = potential_.size();
    double* potential_temp = potential_temp_.data();

#ifdef OPENACC_ENABLED
    #pragma acc enter data copyin(potential_old[0:potential_num], \
                                  potential_new[0:potential_num])
    #pragma acc parallel loop present(potential_new[0:potential_num], \
                                      potential_temp[0:potential_num])
    for (std::size_t i = 0; i < potential_num; ++i)
        potential_temp[i] = potential_new[i];

    #pragma acc parallel loop present(potential_new[0:potential_num])
    for (std::size_t i = 0; i < potential_num; ++i)
        potential_new[i] = 0.;
#else
    std::memcpy(potential_temp, potential_new, potential_num * sizeof(double));
    std::memset(potential_new, 0, potential_num * sizeof(double));
#endif

    BoundaryElement::clear_cluster_charges();
    BoundaryElement::clear_cluster_potentials();

    elements_.compute_charges(potential_old);
    BoundaryElement::upward_pass();

    BoundaryElement::particle_cluster_interact_all(potential_new, potential_old);
    BoundaryElement::cluster_cluster_interact_all(potential_new);

#ifdef OPENACC_ENABLED
    #pragma acc wait
#endif
    
    BoundaryElement::downward_pass(potential_new);

#ifdef OPENACC_ENABLED
    #pragma acc parallel loop present(potential_old[0:potential_num], \
                                      potential_new[0:potential_num], \
                                      potential_temp[0:potential_num])
    for (std::size_t i = 0; i < potential_num / 2; ++i)
        potential_new[i] = beta * potential_temp[i]
                + alpha * (potential_coeff_1 * potential_old[i] - potential_new[i]);

    #pragma acc parallel loop present(potential_old[0:potential_num], \
                                      potential_new[0:potential_num], \
                                      potential_temp[0:potential_num])
    for (std::size_t i = potential_num / 2; i < potential_num; ++i)
        potential_new[i] =  beta * potential_temp[i]
                + alpha * (potential_coeff_2 * potential_old[i] - potential_new[i]);

    #pragma acc exit data copyout(potential_new[0:potential_num])
    #pragma acc exit data delete(potential_old[0:potential_num])
#else
    for (std::size_t i = 0; i < potential_.size() / 2; ++i)
        potential_new[i] = beta * potential_temp[i]
                + alpha * (potential_coeff_1 * potential_old[i] - potential_new[i]);

    for (std::size_t i = potential_.size() / 2; i < potential_.size(); ++i)
        potential_new[i] =  beta * potential_temp[i]
                + alpha * (potential_coeff_2 * potential_old[i] - potential_new[i]);
#endif
                
    timers_.matrix_vector.stop();
}


void BoundaryElement::particle_particle_interact(double* __restrict potential,
                                          const double* __restrict potential_old,
                                          std::array<std::size_t, 2> target_node_element_idxs,
                                          std::array<std::size_t, 2> source_node_element_idxs)
{
    timers_.particle_particle_interact.start();

    std::size_t target_node_element_begin = target_node_element_idxs[0];
    std::size_t target_node_element_end   = target_node_element_idxs[1];

    std::size_t source_node_element_begin = source_node_element_idxs[0];
    std::size_t source_node_element_end   = source_node_element_idxs[1];
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;
    
    const double* __restrict elements_x_ptr    = elements_.x_ptr();
    const double* __restrict elements_y_ptr    = elements_.y_ptr();
    const double* __restrict elements_z_ptr    = elements_.z_ptr();
    
    const double* __restrict elements_nx_ptr   = elements_.nx_ptr();
    const double* __restrict elements_ny_ptr   = elements_.ny_ptr();
    const double* __restrict elements_nz_ptr   = elements_.nz_ptr();

    const double* __restrict elements_area_ptr = elements_.area_ptr();
    
    std::size_t num_elements = elements_.num();

#ifdef OPENACC_ENABLED
    #pragma acc parallel loop present(elements_x_ptr,  elements_y_ptr,  elements_z_ptr, \
                                      elements_nx_ptr, elements_ny_ptr, elements_nz_ptr, \
                                      elements_area_ptr, potential, potential_old)
#endif
    for (std::size_t j = target_node_element_begin; j < target_node_element_end; ++j) {
        
        double target_x = elements_x_ptr[j];
        double target_y = elements_y_ptr[j];
        double target_z = elements_z_ptr[j];
        
        double target_nx = elements_nx_ptr[j];
        double target_ny = elements_ny_ptr[j];
        double target_nz = elements_nz_ptr[j];
        
        double pot_temp_1 = 0.;
        double pot_temp_2 = 0.;

#ifdef OPENACC_ENABLED
        #pragma acc loop reduction(+:pot_temp_1,pot_temp_2)
#endif
        for (std::size_t k = source_node_element_begin; k < source_node_element_end; ++k) {
        
            double source_x = elements_x_ptr[k];
            double source_y = elements_y_ptr[k];
            double source_z = elements_z_ptr[k];
            
            double source_nx = elements_nx_ptr[k];
            double source_ny = elements_ny_ptr[k];
            double source_nz = elements_nz_ptr[k];
            double source_area = elements_area_ptr[k];
            
            double potential_old_0 = potential_old[k];
            double potential_old_1 = potential_old[k + num_elements];
            
            double dist_x = source_x - target_x;
            double dist_y = source_y - target_y;
            double dist_z = source_z - target_z;
            double r = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);
            
            if (r > 0) {
                double one_over_r = 1. / r;
                double G0 = constants::ONE_OVER_4PI * one_over_r;
                double kappa_r = kappa * r;
                double exp_kappa_r = std::exp(-kappa_r);
                double Gk = exp_kappa_r * G0;
                
                double source_cos  = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                double target_cos = (target_nx * dist_x + target_ny * dist_y + target_nz * dist_z) * one_over_r;
                
                double tp1 = G0 * one_over_r;
                double tp2 = (1. + kappa_r) * exp_kappa_r;

                double dot_tqsq = source_nx * target_nx + source_ny * target_ny + source_nz * target_nz;
                double G3 = (dot_tqsq - 3. * target_cos * source_cos) * one_over_r * tp1;
                double G4 = tp2 * G3 - kappa2 * target_cos * source_cos * Gk;

                double L1 = source_cos  * tp1 * (1. - tp2 * eps);
                double L2 = G0 - Gk;
                double L3 = G4 - G3;
                double L4 = target_cos * tp1 * (1. - tp2 / eps);
                
                pot_temp_1 += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                pot_temp_2 += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
            }
        }
        
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j]                += pot_temp_1;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j + num_elements] += pot_temp_2;
    }

    timers_.particle_particle_interact.stop();
}


void BoundaryElement::particle_cluster_interact(double* __restrict potential,
                                         std::array<std::size_t, 2> target_node_element_idxs,
                                         std::size_t source_node_idx)
{
    timers_.particle_cluster_interact.start();

    std::size_t num_elements   = elements_.num();
    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_charges_per_node    = num_charges_per_node_;

    std::size_t target_node_element_begin      = target_node_element_idxs[0];
    std::size_t target_node_element_end        = target_node_element_idxs[1];

    std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
    std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;
    
    const double* __restrict elements_x_ptr   = elements_.x_ptr();
    const double* __restrict elements_y_ptr   = elements_.y_ptr();
    const double* __restrict elements_z_ptr   = elements_.z_ptr();
    
    const double* __restrict targets_q_ptr     = elements_.target_charge_ptr();
    const double* __restrict targets_q_dx_ptr  = elements_.target_charge_dx_ptr();
    const double* __restrict targets_q_dy_ptr  = elements_.target_charge_dy_ptr();
    const double* __restrict targets_q_dz_ptr  = elements_.target_charge_dz_ptr();
    
    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();

    const double* __restrict clusters_q_ptr    = interp_charge_.data();
    const double* __restrict clusters_q_dx_ptr = interp_charge_dx_.data();
    const double* __restrict clusters_q_dy_ptr = interp_charge_dy_.data();
    const double* __restrict clusters_q_dz_ptr = interp_charge_dz_.data();
    
#ifdef OPENACC_ENABLED
    #pragma acc parallel loop present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                    targets_q_ptr, targets_q_dx_ptr, targets_q_dy_ptr, targets_q_dz_ptr, \
                    clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                    clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                    potential)
#endif
    for (std::size_t j = target_node_element_begin; j < target_node_element_end; ++j) {

        double target_x = elements_x_ptr[j];
        double target_y = elements_y_ptr[j];
        double target_z = elements_z_ptr[j];
        
        double pot_comp_   = 0.;
        double pot_comp_dx = 0.;
        double pot_comp_dy = 0.;
        double pot_comp_dz = 0.;
        
#ifdef OPENACC_ENABLED
        #pragma acc loop collapse(3) reduction(+:pot_comp_,   pot_comp_dx, \
                                                 pot_comp_dy, pot_comp_dz)
#endif
        for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
        for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                
            std::size_t kk = source_cluster_charges_begin
                           + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                           + k2 * num_interp_pts_per_node + k3;

            double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
            double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
            double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];

            double r2    = dx*dx + dy*dy + dz*dz;
            double r     = std::sqrt(r2);
            double rinv  = 1. / r;
            double r3inv = rinv  * rinv * rinv;
            double r5inv = r3inv * rinv * rinv;

            double kappa_r = kappa * r;
            double expkr   =  std::exp(-kappa_r);
            double d1term  =  r3inv * expkr * (1. + kappa_r);
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                   + (kappa2 * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

            pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                      + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                 + clusters_q_dy_ptr[kk] * dy
                                                 + clusters_q_dz_ptr[kk] * dz));
                                    
            pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                          - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                          +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                          +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));
                         
            pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                          - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                          +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                          +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));
                         
            pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                          - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                          +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                          +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
        }
        }
        }
        
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j]                += targets_q_ptr   [j] * pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j + num_elements] += targets_q_dx_ptr[j] * pot_comp_dx
                                     + targets_q_dy_ptr[j] * pot_comp_dy
                                     + targets_q_dz_ptr[j] * pot_comp_dz;
    }

    timers_.particle_cluster_interact.stop();
}


void BoundaryElement::cluster_particle_interact(double* __restrict potential,
                                         std::size_t target_node_idx,
                                         std::array<std::size_t, 2> source_node_element_idxs)
{
    timers_.cluster_particle_interact.start();

    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_potentials_per_node = num_charges_per_node_;
    
    std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
    std::size_t target_cluster_potentials_begin = target_node_idx * num_potentials_per_node;

    std::size_t source_node_element_begin       = source_node_element_idxs[0];
    std::size_t source_node_element_end         = source_node_element_idxs[1];
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;
    
    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();
    
    double* __restrict clusters_p_ptr          = interp_potential_.data();
    double* __restrict clusters_p_dx_ptr       = interp_potential_dx_.data();
    double* __restrict clusters_p_dy_ptr       = interp_potential_dy_.data();
    double* __restrict clusters_p_dz_ptr       = interp_potential_dz_.data();
    
    const double* __restrict elements_x_ptr    = elements_.x_ptr();
    const double* __restrict elements_y_ptr    = elements_.y_ptr();
    const double* __restrict elements_z_ptr    = elements_.z_ptr();
    
    const double* __restrict sources_q_ptr     = elements_.source_charge_ptr();
    const double* __restrict sources_q_dx_ptr  = elements_.source_charge_dx_ptr();
    const double* __restrict sources_q_dy_ptr  = elements_.source_charge_dy_ptr();
    const double* __restrict sources_q_dz_ptr  = elements_.source_charge_dz_ptr();
    
#ifdef OPENACC_ENABLED
    #pragma acc parallel loop collapse(3) present(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                    clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                    elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                    sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                    potential)
#endif
    for (int j1 = 0; j1 < num_interp_pts_per_node; ++j1) {
    for (int j2 = 0; j2 < num_interp_pts_per_node; ++j2) {
    for (int j3 = 0; j3 < num_interp_pts_per_node; ++j3) {
    
        std::size_t jj = target_cluster_potentials_begin
                       + j1 * num_interp_pts_per_node * num_interp_pts_per_node
                       + j2 * num_interp_pts_per_node + j3;

        double target_x = clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_comp_   = 0.;
        double pot_comp_dx = 0.;
        double pot_comp_dy = 0.;
        double pot_comp_dz = 0.;
    
#ifdef OPENACC_ENABLED
        #pragma acc loop reduction(+:pot_comp_,   pot_comp_dx, \
                                     pot_comp_dy, pot_comp_dz)
#endif
        for (std::size_t k = source_node_element_begin; k < source_node_element_end; ++k) {

            double dx = target_x - elements_x_ptr[k];
            double dy = target_y - elements_y_ptr[k];
            double dz = target_z - elements_z_ptr[k];

            double r2    = dx*dx + dy*dy + dz*dz;
            double r     = std::sqrt(r2);
            double rinv  = 1. / r;
            double r3inv = rinv  * rinv * rinv;
            double r5inv = r3inv * rinv * rinv;

            double kappa_r = kappa * r;
            double expkr   =  std::exp(-kappa_r);
            double d1term  =  r3inv * expkr * (1. + kappa_r);
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                   + (kappa2 * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

            pot_comp_    += (rinv * (1. - expkr) * (sources_q_ptr   [k])
                                      + d1term1 * (sources_q_dx_ptr[k] * dx
                                                 + sources_q_dy_ptr[k] * dy
                                                 + sources_q_dz_ptr[k] * dz));
                                    
            pot_comp_dx  += (sources_q_ptr   [k]  * (d1term2 * dx)
                          - (sources_q_dx_ptr[k]  * (dx * dx * d2term + d3term)
                          +  sources_q_dy_ptr[k]  * (dx * dy * d2term)
                          +  sources_q_dz_ptr[k]  * (dx * dz * d2term)));
                         
            pot_comp_dy  += (sources_q_ptr   [k]  *  d1term2 * dy
                          - (sources_q_dx_ptr[k]  * (dx * dy * d2term)
                          +  sources_q_dy_ptr[k]  * (dy * dy * d2term + d3term)
                          +  sources_q_dz_ptr[k]  * (dy * dz * d2term)));
                         
            pot_comp_dz  += (sources_q_ptr   [k]  *  d1term2 * dz
                          - (sources_q_dx_ptr[k]  * (dx * dz * d2term)
                          +  sources_q_dy_ptr[k]  * (dy * dz * d2term)
                          +  sources_q_dz_ptr[k]  * (dz * dz * d2term + d3term)));
        }
    
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_ptr   [jj] += pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dx_ptr[jj] += pot_comp_dx;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dy_ptr[jj] += pot_comp_dy;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dz_ptr[jj] += pot_comp_dz;
    }
    }
    }

    timers_.cluster_particle_interact.stop();
}


void BoundaryElement::cluster_cluster_interact(double* __restrict potential,
                                        std::size_t target_node_idx,
                                        std::size_t source_node_idx)
{
    timers_.cluster_cluster_interact.start();

    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_charges_per_node    = num_charges_per_node_;

    std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
    std::size_t target_cluster_potentials_begin = target_node_idx * num_charges_per_node;
    
    std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
    std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;
    
    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();

    double* __restrict clusters_p_ptr          = interp_potential_.data();
    double* __restrict clusters_p_dx_ptr       = interp_potential_dx_.data();
    double* __restrict clusters_p_dy_ptr       = interp_potential_dy_.data();
    double* __restrict clusters_p_dz_ptr       = interp_potential_dz_.data();
    
    const double* __restrict clusters_q_ptr    = interp_charge_.data();
    const double* __restrict clusters_q_dx_ptr = interp_charge_dx_.data();
    const double* __restrict clusters_q_dy_ptr = interp_charge_dy_.data();
    const double* __restrict clusters_q_dz_ptr = interp_charge_dz_.data();

#ifdef OPENACC_ENABLED
    #pragma acc parallel loop collapse(3) present(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                    clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                    clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                    potential)
#endif
    for (int j1 = 0; j1 < num_interp_pts_per_node; j1++) {
    for (int j2 = 0; j2 < num_interp_pts_per_node; j2++) {
    for (int j3 = 0; j3 < num_interp_pts_per_node; j3++) {
    
        std::size_t jj = target_cluster_potentials_begin
                       + j1 * num_interp_pts_per_node * num_interp_pts_per_node
                       + j2 * num_interp_pts_per_node + j3;

        double target_x = clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_comp_   = 0.;
        double pot_comp_dx = 0.;
        double pot_comp_dy = 0.;
        double pot_comp_dz = 0.;
    
#ifdef OPENACC_ENABLED
        #pragma acc loop collapse(3) reduction(+:pot_comp_,   pot_comp_dx, \
                                                 pot_comp_dy, pot_comp_dz)
#endif
        for (int k1 = 0; k1 < num_interp_pts_per_node; k1++) {
        for (int k2 = 0; k2 < num_interp_pts_per_node; k2++) {
        for (int k3 = 0; k3 < num_interp_pts_per_node; k3++) {
            
            std::size_t kk = source_cluster_charges_begin
                           + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                           + k2 * num_interp_pts_per_node + k3;

            double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
            double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
            double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];

            double r2    = dx*dx + dy*dy + dz*dz;
            double r     = std::sqrt(r2);
            double rinv  = 1.0 / r;
            double r3inv = rinv  * rinv * rinv;
            double r5inv = r3inv * rinv * rinv;

            double kappa_r = kappa * r;
            double expkr   =  std::exp(-kappa_r);
            double d1term  =  r3inv * expkr * (1. + kappa_r);
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                   + (kappa2 * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

            pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                      + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                 + clusters_q_dy_ptr[kk] * dy
                                                 + clusters_q_dz_ptr[kk] * dz));
                                    
            pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                          - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                          +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                          +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));
                         
            pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                          - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                          +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                          +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));
                         
            pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                          - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                          +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                          +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
        }
        }
        }
    
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_ptr   [jj] += pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dx_ptr[jj] += pot_comp_dx;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dy_ptr[jj] += pot_comp_dy;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        clusters_p_dz_ptr[jj] += pot_comp_dz;
    }
    }
    }

    timers_.cluster_cluster_interact.stop();
}


void BoundaryElement::particle_particle_interact_all(double* __restrict potential,
                                                     const double* __restrict potential_old)
{
    timers_.particle_particle_interact.start();

    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;

    const double* __restrict elements_x_ptr    = elements_.x_ptr();
    const double* __restrict elements_y_ptr    = elements_.y_ptr();
    const double* __restrict elements_z_ptr    = elements_.z_ptr();

    const double* __restrict elements_nx_ptr   = elements_.nx_ptr();
    const double* __restrict elements_ny_ptr   = elements_.ny_ptr();
    const double* __restrict elements_nz_ptr   = elements_.nz_ptr();

    const double* __restrict elements_area_ptr = elements_.area_ptr();

    const auto& pp_offsets = interaction_list_.particle_particle_offsets();
    const auto& pp_sources = interaction_list_.particle_particle_flat();

#ifdef OPENACC_ENABLED
    const std::uint32_t* __restrict node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* __restrict node_end_ptr   = node_particles_end_u32_.data();
    const std::uint32_t* __restrict offsets_ptr = pp_offsets_u32_.data();
    const std::uint32_t* __restrict sources_ptr = pp_sources_u32_.data();
    std::size_t num_nodes = node_particles_begin_u32_.size();
#else
    const std::size_t* __restrict node_begin_ptr = node_particles_begin_.data();
    const std::size_t* __restrict node_end_ptr   = node_particles_end_.data();
    const std::size_t* __restrict offsets_ptr = pp_offsets.data();
    const std::size_t* __restrict sources_ptr = pp_sources.data();
    std::size_t num_nodes = node_particles_begin_.size();
#endif
    std::size_t num_elements = elements_.num();

#ifdef OPENACC_ENABLED
    std::size_t offsets_num = pp_offsets_u32_.size();
    std::size_t sources_num = pp_sources_u32_.size();
    #pragma acc parallel loop gang present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                           elements_nx_ptr, elements_ny_ptr, elements_nz_ptr, \
                                           elements_area_ptr, potential, potential_old, \
                                           node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                           offsets_ptr[0:offsets_num], sources_ptr[0:sources_num])
#elif defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t target_node_idx = 0; target_node_idx < num_nodes; ++target_node_idx) {
        std::size_t target_begin = node_begin_ptr[target_node_idx];
        std::size_t target_end   = node_end_ptr[target_node_idx];

        std::size_t src_start = offsets_ptr[target_node_idx];
        std::size_t src_end   = offsets_ptr[target_node_idx + 1];

#ifdef OPENACC_ENABLED
        #pragma acc loop vector
#endif
        for (std::size_t j = target_begin; j < target_end; ++j) {
            double target_x = elements_x_ptr[j];
            double target_y = elements_y_ptr[j];
            double target_z = elements_z_ptr[j];

            double target_nx = elements_nx_ptr[j];
            double target_ny = elements_ny_ptr[j];
            double target_nz = elements_nz_ptr[j];

            double pot_temp_1 = 0.;
            double pot_temp_2 = 0.;

            for (std::size_t s = src_start; s < src_end; ++s) {
                std::size_t source_node = sources_ptr[s];
                std::size_t source_begin = node_begin_ptr[source_node];
                std::size_t source_end   = node_end_ptr[source_node];

                for (std::size_t k = source_begin; k < source_end; ++k) {
                    double source_x = elements_x_ptr[k];
                    double source_y = elements_y_ptr[k];
                    double source_z = elements_z_ptr[k];

                    double source_nx = elements_nx_ptr[k];
                    double source_ny = elements_ny_ptr[k];
                    double source_nz = elements_nz_ptr[k];
                    double source_area = elements_area_ptr[k];

                    double potential_old_0 = potential_old[k];
                    double potential_old_1 = potential_old[k + num_elements];

                    double dist_x = source_x - target_x;
                    double dist_y = source_y - target_y;
                    double dist_z = source_z - target_z;
                    double r = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

                    if (r > 0) {
                        double one_over_r = 1. / r;
                        double G0 = constants::ONE_OVER_4PI * one_over_r;
                        double kappa_r = kappa * r;
                        double exp_kappa_r = std::exp(-kappa_r);
                        double Gk = exp_kappa_r * G0;

                        double source_cos = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                        double target_cos = (target_nx * dist_x + target_ny * dist_y + target_nz * dist_z) * one_over_r;

                        double tp1 = G0 * one_over_r;
                        double tp2 = (1. + kappa_r) * exp_kappa_r;

                        double dot_tqsq = source_nx * target_nx + source_ny * target_ny + source_nz * target_nz;
                        double G3 = (dot_tqsq - 3. * target_cos * source_cos) * one_over_r * tp1;
                        double G4 = tp2 * G3 - kappa2 * target_cos * source_cos * Gk;

                        double L1 = source_cos  * tp1 * (1. - tp2 * eps);
                        double L2 = G0 - Gk;
                        double L3 = G4 - G3;
                        double L4 = target_cos * tp1 * (1. - tp2 / eps);

                        pot_temp_1 += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                        pot_temp_2 += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
                    }
                }
            }

#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            potential[j]                += pot_temp_1;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            potential[j + num_elements] += pot_temp_2;
        }
    }

    timers_.particle_particle_interact.stop();
}


void BoundaryElement::particle_cluster_interact_all(double* __restrict potential,
                                                    const double* __restrict potential_old)
{
    timers_.particle_cluster_interact.start();

    std::size_t num_elements = elements_.num();
    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_charges_per_node = num_charges_per_node_;
    constexpr int kMaxInterpPts = 16;
    constexpr int kBatchedMaxInterpPts = 8;
    constexpr int kTargetElemTile = 32;

    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;

    const double* __restrict elements_x_ptr   = elements_.x_ptr();
    const double* __restrict elements_y_ptr   = elements_.y_ptr();
    const double* __restrict elements_z_ptr   = elements_.z_ptr();
    const double* __restrict elements_nx_ptr  = elements_.nx_ptr();
    const double* __restrict elements_ny_ptr  = elements_.ny_ptr();
    const double* __restrict elements_nz_ptr  = elements_.nz_ptr();
    const double* __restrict elements_area_ptr = elements_.area_ptr();

    const double* __restrict targets_q_ptr     = elements_.target_charge_ptr();
    const double* __restrict targets_q_dx_ptr  = elements_.target_charge_dx_ptr();
    const double* __restrict targets_q_dy_ptr  = elements_.target_charge_dy_ptr();
    const double* __restrict targets_q_dz_ptr  = elements_.target_charge_dz_ptr();

    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();

    const double* __restrict clusters_q_ptr    = interp_charge_.data();
    const double* __restrict clusters_q_dx_ptr = interp_charge_dx_.data();
    const double* __restrict clusters_q_dy_ptr = interp_charge_dy_.data();
    const double* __restrict clusters_q_dz_ptr = interp_charge_dz_.data();

    const auto& pp_offsets = interaction_list_.particle_particle_offsets();
    const auto& pp_sources = interaction_list_.particle_particle_flat();
    const auto& pc_offsets = interaction_list_.particle_cluster_offsets();
    const auto& pc_sources = interaction_list_.particle_cluster_flat();

#ifdef OPENACC_ENABLED
    const std::uint32_t* __restrict node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* __restrict node_end_ptr   = node_particles_end_u32_.data();
    const std::uint32_t* __restrict pp_offsets_ptr = pp_offsets_u32_.data();
    const std::uint32_t* __restrict pp_sources_ptr = pp_sources_u32_.data();
    const std::uint32_t* __restrict pc_offsets_ptr = pc_offsets_u32_.data();
    const std::uint32_t* __restrict pc_sources_ptr = pc_sources_u32_.data();
    std::size_t num_nodes = node_particles_begin_u32_.size();
    const std::uint32_t* __restrict element_node_idx_ptr = element_node_idx_u32_.data();
#else
    const std::size_t* __restrict node_begin_ptr = node_particles_begin_.data();
    const std::size_t* __restrict node_end_ptr   = node_particles_end_.data();
    const std::size_t* __restrict pp_offsets_ptr = pp_offsets.data();
    const std::size_t* __restrict pp_sources_ptr = pp_sources.data();
    const std::size_t* __restrict pc_offsets_ptr = pc_offsets.data();
    const std::size_t* __restrict pc_sources_ptr = pc_sources.data();
    std::size_t num_nodes = node_particles_begin_.size();
    const std::size_t* __restrict element_node_idx_ptr = element_node_idx_.data();
#endif


#ifdef OPENACC_ENABLED
    std::size_t pp_offsets_num = pp_offsets_u32_.size();
    std::size_t pp_sources_num = pp_sources_u32_.size();
    std::size_t pc_offsets_num = pc_offsets_u32_.size();
    std::size_t pc_sources_num = pc_sources_u32_.size();
    if (num_interp_pts_per_node <= kBatchedMaxInterpPts) {
        int n  = num_interp_pts_per_node;
        int n2 = n * n;

        #pragma acc parallel loop gang present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                               elements_nx_ptr, elements_ny_ptr, elements_nz_ptr, elements_area_ptr, \
                                               targets_q_ptr, targets_q_dx_ptr, targets_q_dy_ptr, targets_q_dz_ptr, \
                                               clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                               clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                               potential, potential_old, node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                               pp_offsets_ptr[0:pp_offsets_num], pp_sources_ptr[0:pp_sources_num], \
                                               pc_offsets_ptr[0:pc_offsets_num], pc_sources_ptr[0:pc_sources_num])
        for (std::size_t target_node_idx = 0; target_node_idx < num_nodes; ++target_node_idx) {
            std::size_t element_begin = node_begin_ptr[target_node_idx];
            std::size_t element_end   = node_end_ptr[target_node_idx];

            std::size_t pp_start = pp_offsets_ptr[target_node_idx];
            std::size_t pp_end   = pp_offsets_ptr[target_node_idx + 1];
            std::size_t pc_start = pc_offsets_ptr[target_node_idx];
            std::size_t pc_end   = pc_offsets_ptr[target_node_idx + 1];

            for (std::size_t tile_start = element_begin; tile_start < element_end; tile_start += kTargetElemTile) {
                int tile_len = static_cast<int>(element_end - tile_start);
                if (tile_len > kTargetElemTile) tile_len = kTargetElemTile;

                double target_x_cache[kTargetElemTile];
                double target_y_cache[kTargetElemTile];
                double target_z_cache[kTargetElemTile];
                double target_nx_cache[kTargetElemTile];
                double target_ny_cache[kTargetElemTile];
                double target_nz_cache[kTargetElemTile];
                double target_q_cache[kTargetElemTile];
                double target_q_dx_cache[kTargetElemTile];
                double target_q_dy_cache[kTargetElemTile];
                double target_q_dz_cache[kTargetElemTile];

                double pot_pp_1[kTargetElemTile];
                double pot_pp_2[kTargetElemTile];
                double pot_comp_[kTargetElemTile];
                double pot_comp_dx[kTargetElemTile];
                double pot_comp_dy[kTargetElemTile];
                double pot_comp_dz[kTargetElemTile];

                #pragma acc loop vector
                for (int t = 0; t < tile_len; ++t) {
                    std::size_t j = tile_start + static_cast<std::size_t>(t);
                    target_x_cache[t] = elements_x_ptr[j];
                    target_y_cache[t] = elements_y_ptr[j];
                    target_z_cache[t] = elements_z_ptr[j];
                    target_nx_cache[t] = elements_nx_ptr[j];
                    target_ny_cache[t] = elements_ny_ptr[j];
                    target_nz_cache[t] = elements_nz_ptr[j];
                    target_q_cache[t] = targets_q_ptr[j];
                    target_q_dx_cache[t] = targets_q_dx_ptr[j];
                    target_q_dy_cache[t] = targets_q_dy_ptr[j];
                    target_q_dz_cache[t] = targets_q_dz_ptr[j];
                    pot_pp_1[t] = 0.;
                    pot_pp_2[t] = 0.;
                    pot_comp_[t] = 0.;
                    pot_comp_dx[t] = 0.;
                    pot_comp_dy[t] = 0.;
                    pot_comp_dz[t] = 0.;
                }

                for (std::size_t s = pp_start; s < pp_end; ++s) {
                    std::size_t source_node_idx = pp_sources_ptr[s];
                    std::size_t source_begin = node_begin_ptr[source_node_idx];
                    std::size_t source_end   = node_end_ptr[source_node_idx];

                    for (std::size_t k = source_begin; k < source_end; ++k) {
                        double source_x = elements_x_ptr[k];
                        double source_y = elements_y_ptr[k];
                        double source_z = elements_z_ptr[k];

                        double source_nx = elements_nx_ptr[k];
                        double source_ny = elements_ny_ptr[k];
                        double source_nz = elements_nz_ptr[k];
                        double source_area = elements_area_ptr[k];

                        double potential_old_0 = potential_old[k];
                        double potential_old_1 = potential_old[k + num_elements];

                        #pragma acc loop vector
                        for (int t = 0; t < tile_len; ++t) {
                            double dist_x = source_x - target_x_cache[t];
                            double dist_y = source_y - target_y_cache[t];
                            double dist_z = source_z - target_z_cache[t];
                            double r = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

                            if (r > 0) {
                                double one_over_r = 1. / r;
                                double G0 = constants::ONE_OVER_4PI * one_over_r;
                                double kappa_r = kappa * r;
                                double exp_kappa_r = std::exp(-kappa_r);
                                double Gk = exp_kappa_r * G0;

                                double source_cos = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                                double target_cos = (target_nx_cache[t] * dist_x + target_ny_cache[t] * dist_y + target_nz_cache[t] * dist_z) * one_over_r;

                                double tp1 = G0 * one_over_r;
                                double tp2 = (1. + kappa_r) * exp_kappa_r;

                                double dot_tqsq = source_nx * target_nx_cache[t] + source_ny * target_ny_cache[t] + source_nz * target_nz_cache[t];
                                double G3 = (dot_tqsq - 3. * target_cos * source_cos) * one_over_r * tp1;
                                double G4 = tp2 * G3 - kappa2 * target_cos * source_cos * Gk;

                                double L1 = source_cos  * tp1 * (1. - tp2 * eps);
                                double L2 = G0 - Gk;
                                double L3 = G4 - G3;
                                double L4 = target_cos * tp1 * (1. - tp2 / eps);

                                pot_pp_1[t] += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                                pot_pp_2[t] += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
                            }
                        }
                    }
                }

                for (std::size_t s = pc_start; s < pc_end; ++s) {
                    std::size_t source_node_idx = pc_sources_ptr[s];

                    std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
                    std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;

                    double source_x_cache[kBatchedMaxInterpPts];
                    double source_y_cache[kBatchedMaxInterpPts];
                    double source_z_cache[kBatchedMaxInterpPts];

                    #pragma acc loop vector
                    for (int i = 0; i < n; ++i) {
                        source_x_cache[i] = clusters_x_ptr[source_cluster_interp_pts_begin + i];
                        source_y_cache[i] = clusters_y_ptr[source_cluster_interp_pts_begin + i];
                        source_z_cache[i] = clusters_z_ptr[source_cluster_interp_pts_begin + i];
                    }

                    for (int k1 = 0; k1 < n; ++k1) {
                    for (int k2 = 0; k2 < n; ++k2) {
                    for (int k3 = 0; k3 < n; ++k3) {
                        std::size_t kk = source_cluster_charges_begin + k1 * n2 + k2 * n + k3;

                        double source_x = source_x_cache[k1];
                        double source_y = source_y_cache[k2];
                        double source_z = source_z_cache[k3];

                        double source_q    = clusters_q_ptr[kk];
                        double source_q_dx = clusters_q_dx_ptr[kk];
                        double source_q_dy = clusters_q_dy_ptr[kk];
                        double source_q_dz = clusters_q_dz_ptr[kk];

                        #pragma acc loop vector
                        for (int t = 0; t < tile_len; ++t) {
                            double dx = target_x_cache[t] - source_x;
                            double dy = target_y_cache[t] - source_y;
                            double dz = target_z_cache[t] - source_z;

                            double r2    = dx*dx + dy*dy + dz*dz;
                            double r     = std::sqrt(r2);
                            double rinv  = 1. / r;
                            double r3inv = rinv  * rinv * rinv;
                            double r5inv = r3inv * rinv * rinv;

                            double kappa_r = kappa * r;
                            double expkr   =  std::exp(-kappa_r);
                            double d1term  =  r3inv * expkr * (1. + kappa_r);
                            double d1term1 = -r3inv + d1term * eps;
                            double d1term2 = -r3inv + d1term / eps;
                            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                                   + (kappa2 * r2)));
                            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                            pot_comp_[t]    += (rinv * (1. - expkr) * (source_q)
                                                      + d1term1 * (source_q_dx * dx
                                                                 + source_q_dy * dy
                                                                 + source_q_dz * dz));

                            pot_comp_dx[t]  += (source_q     * (d1term2 * dx)
                                              - (source_q_dx * (dx * dx * d2term + d3term)
                                              +  source_q_dy * (dx * dy * d2term)
                                              +  source_q_dz * (dx * dz * d2term)));

                            pot_comp_dy[t]  += (source_q     *  d1term2 * dy
                                              - (source_q_dx * (dx * dy * d2term)
                                              +  source_q_dy * (dy * dy * d2term + d3term)
                                              +  source_q_dz * (dy * dz * d2term)));

                            pot_comp_dz[t]  += (source_q     *  d1term2 * dz
                                              - (source_q_dx * (dx * dz * d2term)
                                              +  source_q_dy * (dy * dz * d2term)
                                              +  source_q_dz * (dz * dz * d2term + d3term)));
                        }
                    }
                    }
                    }
                }

                #pragma acc loop vector
                for (int t = 0; t < tile_len; ++t) {
                    std::size_t j = tile_start + static_cast<std::size_t>(t);
                    potential[j]                += pot_pp_1[t] + target_q_cache[t] * pot_comp_[t];
                    potential[j + num_elements] += pot_pp_2[t]
                                                 + target_q_dx_cache[t] * pot_comp_dx[t]
                                                 + target_q_dy_cache[t] * pot_comp_dy[t]
                                                 + target_q_dz_cache[t] * pot_comp_dz[t];
                }
            }
        }

        timers_.particle_cluster_interact.stop();
        return;
    }
    #pragma acc parallel loop gang present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                           elements_nx_ptr, elements_ny_ptr, elements_nz_ptr, elements_area_ptr, \
                                           targets_q_ptr, targets_q_dx_ptr, targets_q_dy_ptr, targets_q_dz_ptr, \
                                           clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                           clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                           potential, potential_old, node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                           element_node_idx_ptr[0:num_elements], \
                                           pp_offsets_ptr[0:pp_offsets_num], pp_sources_ptr[0:pp_sources_num], \
                                           pc_offsets_ptr[0:pc_offsets_num], pc_sources_ptr[0:pc_sources_num])
#elif defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t j = 0; j < num_elements; ++j) {
        std::size_t target_node_idx = element_node_idx_ptr[j];

        std::size_t pp_start = pp_offsets_ptr[target_node_idx];
        std::size_t pp_end   = pp_offsets_ptr[target_node_idx + 1];
        std::size_t pc_start = pc_offsets_ptr[target_node_idx];
        std::size_t pc_end   = pc_offsets_ptr[target_node_idx + 1];

        double target_x = elements_x_ptr[j];
        double target_y = elements_y_ptr[j];
        double target_z = elements_z_ptr[j];
        double target_nx = elements_nx_ptr[j];
        double target_ny = elements_ny_ptr[j];
        double target_nz = elements_nz_ptr[j];

        double pot_pp_1 = 0.;
        double pot_pp_2 = 0.;

        for (std::size_t s = pp_start; s < pp_end; ++s) {
            std::size_t source_node_idx = pp_sources_ptr[s];
            std::size_t source_begin = node_begin_ptr[source_node_idx];
            std::size_t source_end   = node_end_ptr[source_node_idx];

#ifdef OPENACC_ENABLED
            #pragma acc loop vector reduction(+:pot_pp_1, pot_pp_2)
#endif
            for (std::size_t k = source_begin; k < source_end; ++k) {
                double source_x = elements_x_ptr[k];
                double source_y = elements_y_ptr[k];
                double source_z = elements_z_ptr[k];

                double source_nx = elements_nx_ptr[k];
                double source_ny = elements_ny_ptr[k];
                double source_nz = elements_nz_ptr[k];
                double source_area = elements_area_ptr[k];

                double potential_old_0 = potential_old[k];
                double potential_old_1 = potential_old[k + num_elements];

                double dist_x = source_x - target_x;
                double dist_y = source_y - target_y;
                double dist_z = source_z - target_z;
                double r = std::sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

                if (r > 0) {
                    double one_over_r = 1. / r;
                    double G0 = constants::ONE_OVER_4PI * one_over_r;
                    double kappa_r = kappa * r;
                    double exp_kappa_r = std::exp(-kappa_r);
                    double Gk = exp_kappa_r * G0;

                    double source_cos = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                    double target_cos = (target_nx * dist_x + target_ny * dist_y + target_nz * dist_z) * one_over_r;

                    double tp1 = G0 * one_over_r;
                    double tp2 = (1. + kappa_r) * exp_kappa_r;

                    double dot_tqsq = source_nx * target_nx + source_ny * target_ny + source_nz * target_nz;
                    double G3 = (dot_tqsq - 3. * target_cos * source_cos) * one_over_r * tp1;
                    double G4 = tp2 * G3 - kappa2 * target_cos * source_cos * Gk;

                    double L1 = source_cos  * tp1 * (1. - tp2 * eps);
                    double L2 = G0 - Gk;
                    double L3 = G4 - G3;
                    double L4 = target_cos * tp1 * (1. - tp2 / eps);

                    pot_pp_1 += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                    pot_pp_2 += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
                }
            }
        }

        double pot_comp_   = 0.;
        double pot_comp_dx = 0.;
        double pot_comp_dy = 0.;
        double pot_comp_dz = 0.;

        for (std::size_t s = pc_start; s < pc_end; ++s) {
            std::size_t source_node_idx = pc_sources_ptr[s];

            std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
            std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;

#ifdef OPENACC_ENABLED
            if (num_interp_pts_per_node <= kMaxInterpPts) {
                double dx_cache[kMaxInterpPts];
                double dy_cache[kMaxInterpPts];
                double dz_cache[kMaxInterpPts];
                double dx2_cache[kMaxInterpPts];
                double dy2_cache[kMaxInterpPts];
                double dz2_cache[kMaxInterpPts];

                #pragma acc loop seq
                for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
                    double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
                    dx_cache[k1] = dx;
                    dx2_cache[k1] = dx * dx;
                }
                #pragma acc loop seq
                for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
                    double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
                    dy_cache[k2] = dy;
                    dy2_cache[k2] = dy * dy;
                }
                #pragma acc loop seq
                for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                    double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];
                    dz_cache[k3] = dz;
                    dz2_cache[k3] = dz * dz;
                }

                #pragma acc loop collapse(3) reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
                for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
                for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
                for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                    std::size_t kk = source_cluster_charges_begin
                                   + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                                   + k2 * num_interp_pts_per_node + k3;

                    double dx = dx_cache[k1];
                    double dy = dy_cache[k2];
                    double dz = dz_cache[k3];
                    double r2 = dx2_cache[k1] + dy2_cache[k2] + dz2_cache[k3];
                    double r  = std::sqrt(r2);
                    double rinv  = 1. / r;
                    double r3inv = rinv  * rinv * rinv;
                    double r5inv = r3inv * rinv * rinv;

                    double kappa_r = kappa * r;
                    double expkr   =  std::exp(-kappa_r);
                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                    double d1term1 = -r3inv + d1term * eps;
                    double d1term2 = -r3inv + d1term / eps;
                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                           + (kappa2 * r2)));
                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                    pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                              + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                         + clusters_q_dy_ptr[kk] * dy
                                                         + clusters_q_dz_ptr[kk] * dz));

                    pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                                  - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                                  +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                                  +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));

                    pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                                  - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                                  +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                                  +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));

                    pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                                  - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                                  +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                                  +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
                }
                }
                }
                continue;
            }
#endif

#ifdef OPENACC_ENABLED
            #pragma acc loop collapse(3) reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
#endif
            for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
            for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                std::size_t kk = source_cluster_charges_begin
                               + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                               + k2 * num_interp_pts_per_node + k3;

                double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
                double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
                double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];

                double r2    = dx*dx + dy*dy + dz*dz;
                double r     = std::sqrt(r2);
                double rinv  = 1. / r;
                double r3inv = rinv  * rinv * rinv;
                double r5inv = r3inv * rinv * rinv;

                double kappa_r = kappa * r;
                double expkr   =  std::exp(-kappa_r);
                double d1term  =  r3inv * expkr * (1. + kappa_r);
                double d1term1 = -r3inv + d1term * eps;
                double d1term2 = -r3inv + d1term / eps;
                double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                       + (kappa2 * r2)));
                double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                          + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                     + clusters_q_dy_ptr[kk] * dy
                                                     + clusters_q_dz_ptr[kk] * dz));

                pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                              - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                              +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                              +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));

                pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                              - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                              +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                              +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));

                pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                              - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                              +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                              +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
            }
            }
            }
        }

#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j]                += pot_pp_1 + targets_q_ptr   [j] * pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp atomic update
#endif
        potential[j + num_elements] += pot_pp_2
                                     + targets_q_dx_ptr[j] * pot_comp_dx
                                     + targets_q_dy_ptr[j] * pot_comp_dy
                                     + targets_q_dz_ptr[j] * pot_comp_dz;
    }

    timers_.particle_cluster_interact.stop();
}


void BoundaryElement::cluster_particle_interact_all(double* __restrict potential)
{
    timers_.cluster_particle_interact.start();
    (void)potential;

    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_potentials_per_node = num_charges_per_node_;

    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;

    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();

    double* __restrict clusters_p_ptr          = interp_potential_.data();
    double* __restrict clusters_p_dx_ptr       = interp_potential_dx_.data();
    double* __restrict clusters_p_dy_ptr       = interp_potential_dy_.data();
    double* __restrict clusters_p_dz_ptr       = interp_potential_dz_.data();

    const double* __restrict elements_x_ptr    = elements_.x_ptr();
    const double* __restrict elements_y_ptr    = elements_.y_ptr();
    const double* __restrict elements_z_ptr    = elements_.z_ptr();

    const double* __restrict sources_q_ptr     = elements_.source_charge_ptr();
    const double* __restrict sources_q_dx_ptr  = elements_.source_charge_dx_ptr();
    const double* __restrict sources_q_dy_ptr  = elements_.source_charge_dy_ptr();
    const double* __restrict sources_q_dz_ptr  = elements_.source_charge_dz_ptr();

    const auto& cp_offsets = interaction_list_.cluster_particle_offsets();
    const auto& cp_sources = interaction_list_.cluster_particle_flat();
#ifdef OPENACC_ENABLED
    const std::uint32_t* __restrict node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* __restrict node_end_ptr   = node_particles_end_u32_.data();
    const std::uint32_t* __restrict offsets_ptr = cp_offsets_u32_.data();
    const std::uint32_t* __restrict sources_ptr = cp_sources_u32_.data();
    std::size_t num_nodes = node_particles_begin_u32_.size();
#else
    const std::size_t* __restrict node_begin_ptr = node_particles_begin_.data();
    const std::size_t* __restrict node_end_ptr   = node_particles_end_.data();
    const std::size_t* __restrict offsets_ptr = cp_offsets.data();
    const std::size_t* __restrict sources_ptr = cp_sources.data();
    std::size_t num_nodes = node_particles_begin_.size();
#endif

#ifdef OPENACC_ENABLED
    std::size_t offsets_num = cp_offsets_u32_.size();
    std::size_t sources_num = cp_sources_u32_.size();
    #pragma acc parallel loop gang present(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                           clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                                           elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                           sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                                           potential, node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                           offsets_ptr[0:offsets_num], sources_ptr[0:sources_num])
#elif defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t target_node_idx = 0; target_node_idx < num_nodes; ++target_node_idx) {
        std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
        std::size_t target_cluster_potentials_begin = target_node_idx * num_potentials_per_node;

        std::size_t src_start = offsets_ptr[target_node_idx];
        std::size_t src_end   = offsets_ptr[target_node_idx + 1];

#ifdef OPENACC_ENABLED
        #pragma acc loop collapse(3) vector
#endif
        for (int j1 = 0; j1 < num_interp_pts_per_node; ++j1) {
        for (int j2 = 0; j2 < num_interp_pts_per_node; ++j2) {
        for (int j3 = 0; j3 < num_interp_pts_per_node; ++j3) {
            std::size_t jj = target_cluster_potentials_begin
                           + j1 * num_interp_pts_per_node * num_interp_pts_per_node
                           + j2 * num_interp_pts_per_node + j3;

            double target_x = clusters_x_ptr[target_cluster_interp_pts_begin + j1];
            double target_y = clusters_y_ptr[target_cluster_interp_pts_begin + j2];
            double target_z = clusters_z_ptr[target_cluster_interp_pts_begin + j3];

            double pot_comp_   = 0.;
            double pot_comp_dx = 0.;
            double pot_comp_dy = 0.;
            double pot_comp_dz = 0.;

            for (std::size_t s = src_start; s < src_end; ++s) {
                std::size_t source_node_idx = sources_ptr[s];
                std::size_t source_begin = node_begin_ptr[source_node_idx];
                std::size_t source_end   = node_end_ptr[source_node_idx];

#ifdef OPENACC_ENABLED
                #pragma acc loop vector reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
#endif
                for (std::size_t k = source_begin; k < source_end; ++k) {
                    double dx = target_x - elements_x_ptr[k];
                    double dy = target_y - elements_y_ptr[k];
                    double dz = target_z - elements_z_ptr[k];

                    double r2    = dx*dx + dy*dy + dz*dz;
                    double r     = std::sqrt(r2);
                    double rinv  = 1. / r;
                    double r3inv = rinv  * rinv * rinv;
                    double r5inv = r3inv * rinv * rinv;

                    double kappa_r = kappa * r;
                    double expkr   =  std::exp(-kappa_r);
                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                    double d1term1 = -r3inv + d1term * eps;
                    double d1term2 = -r3inv + d1term / eps;
                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                           + (kappa2 * r2)));
                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                    pot_comp_    += (rinv * (1. - expkr) * (sources_q_ptr   [k])
                                              + d1term1 * (sources_q_dx_ptr[k] * dx
                                                         + sources_q_dy_ptr[k] * dy
                                                         + sources_q_dz_ptr[k] * dz));

                    pot_comp_dx  += (sources_q_ptr   [k]  * (d1term2 * dx)
                                  - (sources_q_dx_ptr[k]  * (dx * dx * d2term + d3term)
                                  +  sources_q_dy_ptr[k]  * (dx * dy * d2term)
                                  +  sources_q_dz_ptr[k]  * (dx * dz * d2term)));

                    pot_comp_dy  += (sources_q_ptr   [k]  *  d1term2 * dy
                                  - (sources_q_dx_ptr[k]  * (dx * dy * d2term)
                                  +  sources_q_dy_ptr[k]  * (dy * dy * d2term + d3term)
                                  +  sources_q_dz_ptr[k]  * (dy * dz * d2term)));

                    pot_comp_dz  += (sources_q_ptr   [k]  *  d1term2 * dz
                                  - (sources_q_dx_ptr[k]  * (dx * dz * d2term)
                                  +  sources_q_dy_ptr[k]  * (dy * dz * d2term)
                                  +  sources_q_dz_ptr[k]  * (dz * dz * d2term + d3term)));
                }
            }

#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_ptr   [jj] += pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dx_ptr[jj] += pot_comp_dx;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dy_ptr[jj] += pot_comp_dy;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dz_ptr[jj] += pot_comp_dz;
        }
        }
        }
    }

    timers_.cluster_particle_interact.stop();
}


void BoundaryElement::cluster_cluster_interact_all(double* __restrict potential)
{
    timers_.cluster_cluster_interact.start();
    (void)potential;

    constexpr int kMaxInterpPts = 16;
    constexpr int kBatchedMaxInterpPts = 8;
    constexpr int kTargetTile = 4;
    constexpr int kTargetTileCharges = kTargetTile * kTargetTile * kTargetTile;

    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    int num_charges_per_node    = num_charges_per_node_;

    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;

    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();

    double* __restrict clusters_p_ptr          = interp_potential_.data();
    double* __restrict clusters_p_dx_ptr       = interp_potential_dx_.data();
    double* __restrict clusters_p_dy_ptr       = interp_potential_dy_.data();
    double* __restrict clusters_p_dz_ptr       = interp_potential_dz_.data();

    const double* __restrict clusters_q_ptr    = interp_charge_.data();
    const double* __restrict clusters_q_dx_ptr = interp_charge_dx_.data();
    const double* __restrict clusters_q_dy_ptr = interp_charge_dy_.data();
    const double* __restrict clusters_q_dz_ptr = interp_charge_dz_.data();

    const double* __restrict elements_x_ptr    = elements_.x_ptr();
    const double* __restrict elements_y_ptr    = elements_.y_ptr();
    const double* __restrict elements_z_ptr    = elements_.z_ptr();

    const double* __restrict sources_q_ptr     = elements_.source_charge_ptr();
    const double* __restrict sources_q_dx_ptr  = elements_.source_charge_dx_ptr();
    const double* __restrict sources_q_dy_ptr  = elements_.source_charge_dy_ptr();
    const double* __restrict sources_q_dz_ptr  = elements_.source_charge_dz_ptr();

    const auto& cp_offsets = interaction_list_.cluster_particle_offsets();
    const auto& cp_sources = interaction_list_.cluster_particle_flat();
    const auto& cc_offsets = interaction_list_.cluster_cluster_offsets();
    const auto& cc_sources = interaction_list_.cluster_cluster_flat();
#ifdef OPENACC_ENABLED
    const std::uint32_t* __restrict node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* __restrict node_end_ptr   = node_particles_end_u32_.data();
    const std::uint32_t* __restrict cp_offsets_ptr = cp_offsets_u32_.data();
    const std::uint32_t* __restrict cp_sources_ptr = cp_sources_u32_.data();
    const std::uint32_t* __restrict cc_offsets_ptr = cc_offsets_u32_.data();
    const std::uint32_t* __restrict cc_sources_ptr = cc_sources_u32_.data();
    std::size_t num_nodes = node_particles_begin_u32_.size();
#else
    const std::size_t* __restrict node_begin_ptr = node_particles_begin_.data();
    const std::size_t* __restrict node_end_ptr   = node_particles_end_.data();
    const std::size_t* __restrict cp_offsets_ptr = cp_offsets.data();
    const std::size_t* __restrict cp_sources_ptr = cp_sources.data();
    const std::size_t* __restrict cc_offsets_ptr = cc_offsets.data();
    const std::size_t* __restrict cc_sources_ptr = cc_sources.data();
    std::size_t num_nodes = node_particles_begin_.size();
#endif

#ifdef OPENACC_ENABLED
    std::size_t cp_offsets_num = cp_offsets_u32_.size();
    std::size_t cp_sources_num = cp_sources_u32_.size();
    std::size_t cc_offsets_num = cc_offsets_u32_.size();
    std::size_t cc_sources_num = cc_sources_u32_.size();
    if (num_interp_pts_per_node <= kBatchedMaxInterpPts) {
        int n  = num_interp_pts_per_node;
        int n2 = n * n;
        int n3 = n2 * n;

#ifdef USE_CUDA_CC
        bool present_ok = true;
        std::size_t num_interp_pts = static_cast<std::size_t>(num_interp_pts_per_node) * num_nodes;
        std::size_t num_charges = static_cast<std::size_t>(num_charges_per_node) * num_nodes;
        std::size_t num_elements = elements_.num();
        present_ok = present_ok && acc_is_present((void*)clusters_x_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_y_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_z_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dx_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dy_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dz_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_p_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_p_dx_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_p_dy_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_p_dz_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_x_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_y_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_z_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dx_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dy_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dz_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)node_begin_ptr, num_nodes * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)node_end_ptr, num_nodes * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)cp_offsets_ptr, cp_offsets_num * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)cp_sources_ptr, cp_sources_num * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)cc_offsets_ptr, cc_offsets_num * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)cc_sources_ptr, cc_sources_num * sizeof(std::uint32_t));

        if (present_ok) {
            int acc_dev = acc_get_device_num(acc_device_nvidia);
            (void)acc_dev;
            acc_wait(acc_async_sync);
            static bool cu_inited = false;
            if (!cu_inited) {
                cuInit(0);
                cu_inited = true;
            }
            void* stream = acc_get_cuda_stream(acc_async_sync);
            #pragma acc host_data use_device(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                             clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                             clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                                             node_begin_ptr, node_end_ptr, \
                                             cp_offsets_ptr, cp_sources_ptr, cc_offsets_ptr, cc_sources_ptr, \
                                             elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                             sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr)
            {
                CUcontext acc_ctx = nullptr;
                if (acc_get_cuda_context) {
                    acc_ctx = acc_get_cuda_context();
                }
                if (acc_ctx == nullptr) {
                    cuCtxGetCurrent(&acc_ctx);
                }
                if (acc_ctx != nullptr) {
                    cuCtxSetCurrent(acc_ctx);
                }
                timers_.cluster_cluster_interact.stop();
                timers_.cluster_particle_interact.start();
                cp_interact_cuda(n, n2, n3, num_interp_pts_per_node, num_charges_per_node,
                                 eps, kappa, kappa2,
                                 clusters_x_ptr, clusters_y_ptr, clusters_z_ptr,
                                 clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr,
                                 node_begin_ptr, node_end_ptr,
                                 cp_offsets_ptr, cp_sources_ptr,
                                 elements_x_ptr, elements_y_ptr, elements_z_ptr,
                                 sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr,
                                 num_elements,
                                 num_nodes,
                                 cp_offsets_num,
                                 cp_sources_num,
                                 stream);
                timers_.cluster_particle_interact.stop();
                timers_.cluster_cluster_interact.start();
                cc_interact_cuda(n, n2, n3, num_interp_pts_per_node, num_charges_per_node,
                                 eps, kappa, kappa2,
                                 clusters_x_ptr, clusters_y_ptr, clusters_z_ptr,
                                 clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr,
                                 clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr,
                                 node_begin_ptr, node_end_ptr,
                                 cp_offsets_ptr, cp_sources_ptr,
                                 cc_offsets_ptr, cc_sources_ptr,
                                 elements_x_ptr, elements_y_ptr, elements_z_ptr,
                                 sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr,
                                 num_elements,
                                 num_nodes,
                                 cp_offsets_num,
                                 cp_sources_num,
                                 cc_offsets_num,
                                 cc_sources_num,
                                 0,
                                 stream);
            }
            timers_.cluster_cluster_interact.stop();
            return;
        }
#endif

        #pragma acc parallel loop gang vector_length(32) present(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                               clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                                               clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                               elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                               sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                                               potential, node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                               cp_offsets_ptr[0:cp_offsets_num], cp_sources_ptr[0:cp_sources_num], \
                                               cc_offsets_ptr[0:cc_offsets_num], cc_sources_ptr[0:cc_sources_num])
        for (std::size_t target_node_idx = 0; target_node_idx < num_nodes; ++target_node_idx) {
            std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
            std::size_t target_cluster_potentials_begin = target_node_idx * num_charges_per_node;

            std::size_t cp_start = cp_offsets_ptr[target_node_idx];
            std::size_t cp_end   = cp_offsets_ptr[target_node_idx + 1];
            std::size_t cc_start = cc_offsets_ptr[target_node_idx];
            std::size_t cc_end   = cc_offsets_ptr[target_node_idx + 1];

            double target_x_cache[kBatchedMaxInterpPts];
            double target_y_cache[kBatchedMaxInterpPts];
            double target_z_cache[kBatchedMaxInterpPts];

            #pragma acc loop vector
            for (int i = 0; i < n; ++i) {
                target_x_cache[i] = clusters_x_ptr[target_cluster_interp_pts_begin + i];
                target_y_cache[i] = clusters_y_ptr[target_cluster_interp_pts_begin + i];
                target_z_cache[i] = clusters_z_ptr[target_cluster_interp_pts_begin + i];
            }

            if (n <= 3) {
                #pragma acc loop worker
                for (int j1 = 0; j1 < n; ++j1) {
                #pragma acc loop vector collapse(2)
                for (int j2 = 0; j2 < n; ++j2) {
                for (int j3 = 0; j3 < n; ++j3) {
                    std::size_t out_idx = target_cluster_potentials_begin + j1 * n2 + j2 * n + j3;

                    double target_x = target_x_cache[j1];
                    double target_y = target_y_cache[j2];
                    double target_z = target_z_cache[j3];

                    double pot_comp_   = 0.;
                    double pot_comp_dx = 0.;
                    double pot_comp_dy = 0.;
                    double pot_comp_dz = 0.;

                    for (std::size_t s = cp_start; s < cp_end; ++s) {
                        std::size_t source_node_idx = cp_sources_ptr[s];
                        std::size_t source_begin = node_begin_ptr[source_node_idx];
                        std::size_t source_end   = node_end_ptr[source_node_idx];

                        #pragma acc cache(elements_x_ptr[source_begin:source_end-source_begin], \
                                          elements_y_ptr[source_begin:source_end-source_begin], \
                                          elements_z_ptr[source_begin:source_end-source_begin], \
                                          sources_q_ptr[source_begin:source_end-source_begin], \
                                          sources_q_dx_ptr[source_begin:source_end-source_begin], \
                                          sources_q_dy_ptr[source_begin:source_end-source_begin], \
                                          sources_q_dz_ptr[source_begin:source_end-source_begin])
                        #pragma acc loop seq
                        for (std::size_t k = source_begin; k < source_end; ++k) {
                            double dx = target_x - elements_x_ptr[k];
                            double dy = target_y - elements_y_ptr[k];
                            double dz = target_z - elements_z_ptr[k];

                            double r2    = dx*dx + dy*dy + dz*dz;
                            double r     = std::sqrt(r2);
                            double rinv  = 1. / r;
                            double r3inv = rinv  * rinv * rinv;
                            double r5inv = r3inv * rinv * rinv;

                            double kappa_r = kappa * r;
                            double expkr   =  std::exp(-kappa_r);
                            double d1term  =  r3inv * expkr * (1. + kappa_r);
                            double d1term1 = -r3inv + d1term * eps;
                            double d1term2 = -r3inv + d1term / eps;
                            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                                   + (kappa2 * r2)));
                            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                            pot_comp_    += (rinv * (1. - expkr) * (sources_q_ptr   [k])
                                                      + d1term1 * (sources_q_dx_ptr[k] * dx
                                                                 + sources_q_dy_ptr[k] * dy
                                                                 + sources_q_dz_ptr[k] * dz));

                            pot_comp_dx  += (sources_q_ptr   [k]  * (d1term2 * dx)
                                          - (sources_q_dx_ptr[k]  * (dx * dx * d2term + d3term)
                                          +  sources_q_dy_ptr[k]  * (dx * dy * d2term)
                                          +  sources_q_dz_ptr[k]  * (dx * dz * d2term)));

                            pot_comp_dy  += (sources_q_ptr   [k]  *  d1term2 * dy
                                          - (sources_q_dx_ptr[k]  * (dx * dy * d2term)
                                          +  sources_q_dy_ptr[k]  * (dy * dy * d2term + d3term)
                                          +  sources_q_dz_ptr[k]  * (dy * dz * d2term)));

                            pot_comp_dz  += (sources_q_ptr   [k]  *  d1term2 * dz
                                          - (sources_q_dx_ptr[k]  * (dx * dz * d2term)
                                          +  sources_q_dy_ptr[k]  * (dy * dz * d2term)
                                          +  sources_q_dz_ptr[k]  * (dz * dz * d2term + d3term)));
                        }
                    }

                    for (std::size_t s = cc_start; s < cc_end; ++s) {
                        std::size_t source_node_idx = cc_sources_ptr[s];

                        std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
                        std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;

                        #pragma acc cache(clusters_x_ptr[source_cluster_interp_pts_begin:num_interp_pts_per_node], \
                                          clusters_y_ptr[source_cluster_interp_pts_begin:num_interp_pts_per_node], \
                                          clusters_z_ptr[source_cluster_interp_pts_begin:num_interp_pts_per_node], \
                                          clusters_q_ptr[source_cluster_charges_begin:num_charges_per_node], \
                                          clusters_q_dx_ptr[source_cluster_charges_begin:num_charges_per_node], \
                                          clusters_q_dy_ptr[source_cluster_charges_begin:num_charges_per_node], \
                                          clusters_q_dz_ptr[source_cluster_charges_begin:num_charges_per_node])
                        #pragma acc loop seq collapse(3)
                        for (int k1 = 0; k1 < n; ++k1) {
                        for (int k2 = 0; k2 < n; ++k2) {
                        for (int k3 = 0; k3 < n; ++k3) {
                            std::size_t kk = source_cluster_charges_begin + k1 * n2 + k2 * n + k3;

                            double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
                            double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
                            double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];

                            double r2    = dx*dx + dy*dy + dz*dz;
                            double r     = std::sqrt(r2);
                            double rinv  = 1.0 / r;
                            double r3inv = rinv  * rinv * rinv;
                            double r5inv = r3inv * rinv * rinv;

                            double kappa_r = kappa * r;
                            double expkr   =  std::exp(-kappa_r);
                            double d1term  =  r3inv * expkr * (1. + kappa_r);
                            double d1term1 = -r3inv + d1term * eps;
                            double d1term2 = -r3inv + d1term / eps;
                            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                                   + (kappa2 * r2)));
                            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                            pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                                      + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                                 + clusters_q_dy_ptr[kk] * dy
                                                                 + clusters_q_dz_ptr[kk] * dz));

                            pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                                          - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                                          +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                                          +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));

                            pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                                          - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                                          +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                                          +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));

                            pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                                          - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                                          +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                                          +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
                        }
                        }
                        }
                    }

                    clusters_p_ptr   [out_idx] += pot_comp_;
                    clusters_p_dx_ptr[out_idx] += pot_comp_dx;
                    clusters_p_dy_ptr[out_idx] += pot_comp_dy;
                    clusters_p_dz_ptr[out_idx] += pot_comp_dz;
                }
                }
                }
                continue;
            }

            for (int t1 = 0; t1 < n; t1 += kTargetTile) {
                int t1_max = (t1 + kTargetTile < n) ? (t1 + kTargetTile) : n;
                for (int t2 = 0; t2 < n; t2 += kTargetTile) {
                    int t2_max = (t2 + kTargetTile < n) ? (t2 + kTargetTile) : n;
                    int tn2 = t2_max - t2;
                    for (int t3 = 0; t3 < n; t3 += kTargetTile) {
                        int t3_max = (t3 + kTargetTile < n) ? (t3 + kTargetTile) : n;
                        int tn3 = t3_max - t3;
                        int tile_n2 = tn2 * tn3;

                        double pot_comp_[kTargetTileCharges];
                        double pot_comp_dx[kTargetTileCharges];
                        double pot_comp_dy[kTargetTileCharges];
                        double pot_comp_dz[kTargetTileCharges];

                        #pragma acc loop vector collapse(3)
                        for (int j1 = t1; j1 < t1_max; ++j1) {
                        for (int j2 = t2; j2 < t2_max; ++j2) {
                        for (int j3 = t3; j3 < t3_max; ++j3) {
                            int jj = (j1 - t1) * tile_n2 + (j2 - t2) * tn3 + (j3 - t3);
                            pot_comp_[jj] = 0.;
                            pot_comp_dx[jj] = 0.;
                            pot_comp_dy[jj] = 0.;
                            pot_comp_dz[jj] = 0.;
                        }
                        }
                        }

                        for (std::size_t s = cp_start; s < cp_end; ++s) {
                            std::size_t source_node_idx = cp_sources_ptr[s];
                            std::size_t source_begin = node_begin_ptr[source_node_idx];
                            std::size_t source_end   = node_end_ptr[source_node_idx];

                            for (std::size_t k = source_begin; k < source_end; ++k) {
                                double source_x = elements_x_ptr[k];
                                double source_y = elements_y_ptr[k];
                                double source_z = elements_z_ptr[k];

                                double source_q    = sources_q_ptr[k];
                                double source_q_dx = sources_q_dx_ptr[k];
                                double source_q_dy = sources_q_dy_ptr[k];
                                double source_q_dz = sources_q_dz_ptr[k];

                                #pragma acc loop vector collapse(3)
                                for (int j1 = t1; j1 < t1_max; ++j1) {
                                for (int j2 = t2; j2 < t2_max; ++j2) {
                                for (int j3 = t3; j3 < t3_max; ++j3) {
                                    int jj = (j1 - t1) * tile_n2 + (j2 - t2) * tn3 + (j3 - t3);

                                    double dx = target_x_cache[j1] - source_x;
                                    double dy = target_y_cache[j2] - source_y;
                                    double dz = target_z_cache[j3] - source_z;

                                    double r2    = dx*dx + dy*dy + dz*dz;
                                    double r     = std::sqrt(r2);
                                    double rinv  = 1. / r;
                                    double r3inv = rinv  * rinv * rinv;
                                    double r5inv = r3inv * rinv * rinv;

                                    double kappa_r = kappa * r;
                                    double expkr   =  std::exp(-kappa_r);
                                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                                    double d1term1 = -r3inv + d1term * eps;
                                    double d1term2 = -r3inv + d1term / eps;
                                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                                           + (kappa2 * r2)));
                                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                                    pot_comp_[jj]    += (rinv * (1. - expkr) * (source_q)
                                                               + d1term1 * (source_q_dx * dx
                                                                          + source_q_dy * dy
                                                                          + source_q_dz * dz));

                                    pot_comp_dx[jj]  += (source_q     * (d1term2 * dx)
                                                      - (source_q_dx * (dx * dx * d2term + d3term)
                                                      +  source_q_dy * (dx * dy * d2term)
                                                      +  source_q_dz * (dx * dz * d2term)));

                                    pot_comp_dy[jj]  += (source_q     *  d1term2 * dy
                                                      - (source_q_dx * (dx * dy * d2term)
                                                      +  source_q_dy * (dy * dy * d2term + d3term)
                                                      +  source_q_dz * (dy * dz * d2term)));

                                    pot_comp_dz[jj]  += (source_q     *  d1term2 * dz
                                                      - (source_q_dx * (dx * dz * d2term)
                                                      +  source_q_dy * (dy * dz * d2term)
                                                      +  source_q_dz * (dz * dz * d2term + d3term)));
                                }
                                }
                                }
                            }
                        }

                        for (std::size_t s = cc_start; s < cc_end; ++s) {
                            std::size_t source_node_idx = cc_sources_ptr[s];

                            std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
                            std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;

                            double source_x_cache[kBatchedMaxInterpPts];
                            double source_y_cache[kBatchedMaxInterpPts];
                            double source_z_cache[kBatchedMaxInterpPts];

                            #pragma acc loop vector
                            for (int i = 0; i < n; ++i) {
                                source_x_cache[i] = clusters_x_ptr[source_cluster_interp_pts_begin + i];
                                source_y_cache[i] = clusters_y_ptr[source_cluster_interp_pts_begin + i];
                                source_z_cache[i] = clusters_z_ptr[source_cluster_interp_pts_begin + i];
                            }

                            for (int k1 = 0; k1 < n; ++k1) {
                            for (int k2 = 0; k2 < n; ++k2) {
                            for (int k3 = 0; k3 < n; ++k3) {
                                std::size_t kk = source_cluster_charges_begin + k1 * n2 + k2 * n + k3;

                                double source_x = source_x_cache[k1];
                                double source_y = source_y_cache[k2];
                                double source_z = source_z_cache[k3];

                                double source_q    = clusters_q_ptr[kk];
                                double source_q_dx = clusters_q_dx_ptr[kk];
                                double source_q_dy = clusters_q_dy_ptr[kk];
                                double source_q_dz = clusters_q_dz_ptr[kk];

                                #pragma acc loop vector collapse(3)
                                for (int j1 = t1; j1 < t1_max; ++j1) {
                                for (int j2 = t2; j2 < t2_max; ++j2) {
                                for (int j3 = t3; j3 < t3_max; ++j3) {
                                    int jj = (j1 - t1) * tile_n2 + (j2 - t2) * tn3 + (j3 - t3);

                                    double dx = target_x_cache[j1] - source_x;
                                    double dy = target_y_cache[j2] - source_y;
                                    double dz = target_z_cache[j3] - source_z;

                                    double r2    = dx*dx + dy*dy + dz*dz;
                                    double r     = std::sqrt(r2);
                                    double rinv  = 1.0 / r;
                                    double r3inv = rinv  * rinv * rinv;
                                    double r5inv = r3inv * rinv * rinv;

                                    double kappa_r = kappa * r;
                                    double expkr   =  std::exp(-kappa_r);
                                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                                    double d1term1 = -r3inv + d1term * eps;
                                    double d1term2 = -r3inv + d1term / eps;
                                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                                           + (kappa2 * r2)));
                                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                                    pot_comp_[jj]    += (rinv * (1. - expkr) * (source_q)
                                                               + d1term1 * (source_q_dx * dx
                                                                          + source_q_dy * dy
                                                                          + source_q_dz * dz));

                                    pot_comp_dx[jj]  += (source_q     * (d1term2 * dx)
                                                      - (source_q_dx * (dx * dx * d2term + d3term)
                                                      +  source_q_dy * (dx * dy * d2term)
                                                      +  source_q_dz * (dx * dz * d2term)));

                                    pot_comp_dy[jj]  += (source_q     *  d1term2 * dy
                                                      - (source_q_dx * (dx * dy * d2term)
                                                      +  source_q_dy * (dy * dy * d2term + d3term)
                                                      +  source_q_dz * (dy * dz * d2term)));

                                    pot_comp_dz[jj]  += (source_q     *  d1term2 * dz
                                                      - (source_q_dx * (dx * dz * d2term)
                                                      +  source_q_dy * (dy * dz * d2term)
                                                      +  source_q_dz * (dz * dz * d2term + d3term)));
                                }
                                }
                                }
                            }
                            }
                            }
                        }

                        #pragma acc loop vector collapse(3)
                        for (int j1 = t1; j1 < t1_max; ++j1) {
                        for (int j2 = t2; j2 < t2_max; ++j2) {
                        for (int j3 = t3; j3 < t3_max; ++j3) {
                            int jj = (j1 - t1) * tile_n2 + (j2 - t2) * tn3 + (j3 - t3);
                            std::size_t out_idx = target_cluster_potentials_begin + j1 * n2 + j2 * n + j3;
                            clusters_p_ptr   [out_idx] += pot_comp_   [jj];
                            clusters_p_dx_ptr[out_idx] += pot_comp_dx[jj];
                            clusters_p_dy_ptr[out_idx] += pot_comp_dy[jj];
                            clusters_p_dz_ptr[out_idx] += pot_comp_dz[jj];
                        }
                        }
                        }
                    }
                }
            }
        }

        timers_.cluster_cluster_interact.stop();
        return;
    }

    #pragma acc parallel loop gang present(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                           clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                                           clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                           elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                           sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                                           potential, node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                           cp_offsets_ptr[0:cp_offsets_num], cp_sources_ptr[0:cp_sources_num], \
                                           cc_offsets_ptr[0:cc_offsets_num], cc_sources_ptr[0:cc_sources_num])
#elif defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t target_node_idx = 0; target_node_idx < num_nodes; ++target_node_idx) {
        std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
        std::size_t target_cluster_potentials_begin = target_node_idx * num_charges_per_node;

        std::size_t cp_start = cp_offsets_ptr[target_node_idx];
        std::size_t cp_end   = cp_offsets_ptr[target_node_idx + 1];
        std::size_t cc_start = cc_offsets_ptr[target_node_idx];
        std::size_t cc_end   = cc_offsets_ptr[target_node_idx + 1];

#ifdef OPENACC_ENABLED
        #pragma acc loop collapse(3) vector
#endif
        for (int j1 = 0; j1 < num_interp_pts_per_node; j1++) {
        for (int j2 = 0; j2 < num_interp_pts_per_node; j2++) {
        for (int j3 = 0; j3 < num_interp_pts_per_node; j3++) {
            std::size_t jj = target_cluster_potentials_begin
                           + j1 * num_interp_pts_per_node * num_interp_pts_per_node
                           + j2 * num_interp_pts_per_node + j3;

            double target_x = clusters_x_ptr[target_cluster_interp_pts_begin + j1];
            double target_y = clusters_y_ptr[target_cluster_interp_pts_begin + j2];
            double target_z = clusters_z_ptr[target_cluster_interp_pts_begin + j3];

            double pot_comp_   = 0.;
            double pot_comp_dx = 0.;
            double pot_comp_dy = 0.;
            double pot_comp_dz = 0.;

            for (std::size_t s = cp_start; s < cp_end; ++s) {
                std::size_t source_node_idx = cp_sources_ptr[s];
                std::size_t source_begin = node_begin_ptr[source_node_idx];
                std::size_t source_end   = node_end_ptr[source_node_idx];

#ifdef OPENACC_ENABLED
                #pragma acc loop vector reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
#endif
                for (std::size_t k = source_begin; k < source_end; ++k) {
                    double dx = target_x - elements_x_ptr[k];
                    double dy = target_y - elements_y_ptr[k];
                    double dz = target_z - elements_z_ptr[k];

                    double r2    = dx*dx + dy*dy + dz*dz;
                    double r     = std::sqrt(r2);
                    double rinv  = 1. / r;
                    double r3inv = rinv  * rinv * rinv;
                    double r5inv = r3inv * rinv * rinv;

                    double kappa_r = kappa * r;
                    double expkr   =  std::exp(-kappa_r);
                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                    double d1term1 = -r3inv + d1term * eps;
                    double d1term2 = -r3inv + d1term / eps;
                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                           + (kappa2 * r2)));
                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                    pot_comp_    += (rinv * (1. - expkr) * (sources_q_ptr   [k])
                                              + d1term1 * (sources_q_dx_ptr[k] * dx
                                                         + sources_q_dy_ptr[k] * dy
                                                         + sources_q_dz_ptr[k] * dz));

                    pot_comp_dx  += (sources_q_ptr   [k]  * (d1term2 * dx)
                                  - (sources_q_dx_ptr[k]  * (dx * dx * d2term + d3term)
                                  +  sources_q_dy_ptr[k]  * (dx * dy * d2term)
                                  +  sources_q_dz_ptr[k]  * (dx * dz * d2term)));

                    pot_comp_dy  += (sources_q_ptr   [k]  *  d1term2 * dy
                                  - (sources_q_dx_ptr[k]  * (dx * dy * d2term)
                                  +  sources_q_dy_ptr[k]  * (dy * dy * d2term + d3term)
                                  +  sources_q_dz_ptr[k]  * (dy * dz * d2term)));

                    pot_comp_dz  += (sources_q_ptr   [k]  *  d1term2 * dz
                                  - (sources_q_dx_ptr[k]  * (dx * dz * d2term)
                                  +  sources_q_dy_ptr[k]  * (dy * dz * d2term)
                                  +  sources_q_dz_ptr[k]  * (dz * dz * d2term + d3term)));
                }
            }

            for (std::size_t s = cc_start; s < cc_end; ++s) {
                std::size_t source_node_idx = cc_sources_ptr[s];

                std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
                std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;

#ifdef OPENACC_ENABLED
                if (num_interp_pts_per_node <= kMaxInterpPts) {
                    double dx_cache[kMaxInterpPts];
                    double dy_cache[kMaxInterpPts];
                    double dz_cache[kMaxInterpPts];
                    double dx2_cache[kMaxInterpPts];
                    double dy2_cache[kMaxInterpPts];
                    double dz2_cache[kMaxInterpPts];

                    #pragma acc loop seq
                    for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
                        double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
                        dx_cache[k1] = dx;
                        dx2_cache[k1] = dx * dx;
                    }
                    #pragma acc loop seq
                    for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
                        double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
                        dy_cache[k2] = dy;
                        dy2_cache[k2] = dy * dy;
                    }
                    #pragma acc loop seq
                    for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                        double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];
                        dz_cache[k3] = dz;
                        dz2_cache[k3] = dz * dz;
                    }

                    #pragma acc loop collapse(3) reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
                    for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
                    for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
                    for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                        std::size_t kk = source_cluster_charges_begin
                                       + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                                       + k2 * num_interp_pts_per_node + k3;

                        double dx = dx_cache[k1];
                        double dy = dy_cache[k2];
                        double dz = dz_cache[k3];
                        double r2 = dx2_cache[k1] + dy2_cache[k2] + dz2_cache[k3];
                        double r  = std::sqrt(r2);
                        double rinv  = 1.0 / r;
                        double r3inv = rinv  * rinv * rinv;
                        double r5inv = r3inv * rinv * rinv;

                        double kappa_r = kappa * r;
                        double expkr   =  std::exp(-kappa_r);
                        double d1term  =  r3inv * expkr * (1. + kappa_r);
                        double d1term1 = -r3inv + d1term * eps;
                        double d1term2 = -r3inv + d1term / eps;
                        double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                               + (kappa2 * r2)));
                        double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                        pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                                  + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                             + clusters_q_dy_ptr[kk] * dy
                                                             + clusters_q_dz_ptr[kk] * dz));

                        pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                                      - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                                      +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                                      +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));

                        pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                                      - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                                      +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                                      +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));

                        pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                                      - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                                      +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                                      +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
                    }
                    }
                    }
                    continue;
                }
#endif

#ifdef OPENACC_ENABLED
                #pragma acc loop collapse(3) reduction(+:pot_comp_, pot_comp_dx, pot_comp_dy, pot_comp_dz)
#endif
                for (int k1 = 0; k1 < num_interp_pts_per_node; k1++) {
                for (int k2 = 0; k2 < num_interp_pts_per_node; k2++) {
                for (int k3 = 0; k3 < num_interp_pts_per_node; k3++) {
                    std::size_t kk = source_cluster_charges_begin
                                   + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                                   + k2 * num_interp_pts_per_node + k3;

                    double dx = target_x - clusters_x_ptr[source_cluster_interp_pts_begin + k1];
                    double dy = target_y - clusters_y_ptr[source_cluster_interp_pts_begin + k2];
                    double dz = target_z - clusters_z_ptr[source_cluster_interp_pts_begin + k3];

                    double r2    = dx*dx + dy*dy + dz*dz;
                    double r     = std::sqrt(r2);
                    double rinv  = 1.0 / r;
                    double r3inv = rinv  * rinv * rinv;
                    double r5inv = r3inv * rinv * rinv;

                    double kappa_r = kappa * r;
                    double expkr   =  std::exp(-kappa_r);
                    double d1term  =  r3inv * expkr * (1. + kappa_r);
                    double d1term1 = -r3inv + d1term * eps;
                    double d1term2 = -r3inv + d1term / eps;
                    double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa_r)
                                                           + (kappa2 * r2)));
                    double d3term  =  r3inv * ( 1. - expkr * (1. + kappa_r));

                    pot_comp_    += (rinv * (1. - expkr) * (clusters_q_ptr   [kk])
                                              + d1term1 * (clusters_q_dx_ptr[kk] * dx
                                                         + clusters_q_dy_ptr[kk] * dy
                                                         + clusters_q_dz_ptr[kk] * dz));

                    pot_comp_dx  += (clusters_q_ptr   [kk]  * (d1term2 * dx)
                                  - (clusters_q_dx_ptr[kk]  * (dx * dx * d2term + d3term)
                                  +  clusters_q_dy_ptr[kk]  * (dx * dy * d2term)
                                  +  clusters_q_dz_ptr[kk]  * (dx * dz * d2term)));

                    pot_comp_dy  += (clusters_q_ptr   [kk]  *  d1term2 * dy
                                  - (clusters_q_dx_ptr[kk]  * (dx * dy * d2term)
                                  +  clusters_q_dy_ptr[kk]  * (dy * dy * d2term + d3term)
                                  +  clusters_q_dz_ptr[kk]  * (dy * dz * d2term)));

                    pot_comp_dz  += (clusters_q_ptr   [kk]  *  d1term2 * dz
                                  - (clusters_q_dx_ptr[kk]  * (dx * dz * d2term)
                                  +  clusters_q_dy_ptr[kk]  * (dy * dz * d2term)
                                  +  clusters_q_dz_ptr[kk]  * (dz * dz * d2term + d3term)));
                }
                }
                }
            }

#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_ptr   [jj] += pot_comp_;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dx_ptr[jj] += pot_comp_dx;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dy_ptr[jj] += pot_comp_dy;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            clusters_p_dz_ptr[jj] += pot_comp_dz;
        }
        }
        }
    }

    timers_.cluster_cluster_interact.stop();
}


void BoundaryElement::upward_pass()
{
    timers_.upward_pass.start();

    constexpr int kMaxInterpPts = 16;

    const double* __restrict clusters_x_ptr   = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr   = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr   = interp_pts_.interp_z_ptr();
    
    double* __restrict clusters_q_ptr         = interp_charge_.data();
    double* __restrict clusters_q_dx_ptr      = interp_charge_dx_.data();
    double* __restrict clusters_q_dy_ptr      = interp_charge_dy_.data();
    double* __restrict clusters_q_dz_ptr      = interp_charge_dz_.data();
    
    const double* __restrict elements_x_ptr  = elements_.x_ptr();
    const double* __restrict elements_y_ptr  = elements_.y_ptr();
    const double* __restrict elements_z_ptr  = elements_.z_ptr();
    
    const double* __restrict sources_q_ptr    = elements_.source_charge_ptr();
    const double* __restrict sources_q_dx_ptr = elements_.source_charge_dx_ptr();
    const double* __restrict sources_q_dy_ptr = elements_.source_charge_dy_ptr();
    const double* __restrict sources_q_dz_ptr = elements_.source_charge_dz_ptr();
        
    double* weights_ptr = weights_.data();
    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();

    std::size_t max_particles = exact_idx_x_.size();
    int* exact_idx_x_ptr = exact_idx_x_.data();
    int* exact_idx_y_ptr = exact_idx_y_.data();
    int* exact_idx_z_ptr = exact_idx_z_.data();
    double* denominator_ptr = denominator_.data();

    const std::uint32_t* __restrict node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* __restrict node_end_ptr   = node_particles_end_u32_.data();
    std::size_t num_nodes = node_particles_begin_u32_.size();

    const std::size_t* __restrict level_offsets_ptr = level_offsets_.data();
    const std::size_t* __restrict level_nodes_ptr = level_nodes_.data();
    std::size_t level_count = level_offsets_.empty() ? 0 : (level_offsets_.size() - 1);
    std::size_t level_nodes_num = level_nodes_.size();
#ifndef OPENACC_ENABLED
    (void)num_nodes;
    (void)level_nodes_num;
#endif

#if defined(USE_CUDA_CC) && defined(OPENACC_ENABLED)
    if (num_interp_pts_per_node <= kMaxInterpPts) {
        bool present_ok = true;
        std::size_t num_interp_pts = static_cast<std::size_t>(num_interp_pts_per_node) * num_nodes;
        std::size_t num_charges = static_cast<std::size_t>(num_charges_per_node_) * num_nodes;
        std::size_t num_elements = elements_.num();

        present_ok = present_ok && acc_is_present((void*)clusters_x_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_y_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_z_ptr, num_interp_pts * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dx_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dy_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)clusters_q_dz_ptr, num_charges * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)weights_ptr, static_cast<std::size_t>(num_interp_pts_per_node) * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_x_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_y_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)elements_z_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dx_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dy_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)sources_q_dz_ptr, num_elements * sizeof(double));
        present_ok = present_ok && acc_is_present((void*)node_begin_ptr, num_nodes * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)node_end_ptr, num_nodes * sizeof(std::uint32_t));
        present_ok = present_ok && acc_is_present((void*)level_nodes_ptr, level_nodes_num * sizeof(std::size_t));
        present_ok = present_ok && acc_is_present((void*)exact_idx_x_ptr, max_particles * sizeof(int));
        present_ok = present_ok && acc_is_present((void*)exact_idx_y_ptr, max_particles * sizeof(int));
        present_ok = present_ok && acc_is_present((void*)exact_idx_z_ptr, max_particles * sizeof(int));
        present_ok = present_ok && acc_is_present((void*)denominator_ptr, max_particles * sizeof(double));

        if (present_ok) {
            int acc_dev = acc_get_device_num(acc_device_nvidia);
            (void)acc_dev;
            acc_wait(acc_async_sync);
            static bool cu_inited = false;
            if (!cu_inited) {
                cuInit(0);
                cu_inited = true;
            }
            void* stream = acc_get_cuda_stream(acc_async_sync);
            #pragma acc host_data use_device(clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                             clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                             weights_ptr, elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                             sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                                             node_begin_ptr, node_end_ptr, level_nodes_ptr)
            {
                CUcontext acc_ctx = nullptr;
                if (acc_get_cuda_context) {
                    acc_ctx = acc_get_cuda_context();
                }
                if (acc_ctx == nullptr) {
                    cuCtxGetCurrent(&acc_ctx);
                }
                if (acc_ctx != nullptr) {
                    cuCtxSetCurrent(acc_ctx);
                }

                if (level_nodes_num > 0) {
                    upward_fused_cuda(num_interp_pts_per_node, num_charges_per_node_,
                                      clusters_x_ptr, clusters_y_ptr, clusters_z_ptr,
                                      weights_ptr,
                                      elements_x_ptr, elements_y_ptr, elements_z_ptr,
                                      sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr,
                                      node_begin_ptr, node_end_ptr,
                                      level_nodes_ptr,
                                      level_nodes_num,
                                      clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr,
                                      stream);
                }
            }
            timers_.upward_pass.stop();
            return;
        }
    }
#endif

    for (std::size_t level = 0; level < level_count; ++level) {
        std::size_t level_begin = level_offsets_ptr[level];
        std::size_t level_end   = level_offsets_ptr[level + 1];
#ifdef OPENACC_ENABLED
        #pragma acc parallel loop gang present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                               sources_q_ptr, sources_q_dx_ptr, sources_q_dy_ptr, sources_q_dz_ptr, \
                                               clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                               clusters_q_ptr, clusters_q_dx_ptr, clusters_q_dy_ptr, clusters_q_dz_ptr, \
                                               weights_ptr, exact_idx_x_ptr[0:max_particles], exact_idx_y_ptr[0:max_particles], \
                                               exact_idx_z_ptr[0:max_particles], denominator_ptr[0:max_particles], \
                                               node_begin_ptr[0:num_nodes], node_end_ptr[0:num_nodes], \
                                               level_offsets_ptr[0:level_count+1], level_nodes_ptr[0:level_nodes_num])
#endif
        for (std::size_t level_idx = level_begin; level_idx < level_end; ++level_idx) {
            std::size_t node_idx = level_nodes_ptr[level_idx];

            std::size_t node_interp_pts_start =
                node_idx * static_cast<std::size_t>(num_interp_pts_per_node);
            std::size_t node_charges_start =
                node_idx * static_cast<std::size_t>(num_charges_per_node_);

            std::size_t particle_start = static_cast<std::size_t>(node_begin_ptr[node_idx]);
            std::size_t particle_end   = static_cast<std::size_t>(node_end_ptr[node_idx]);
            std::size_t num_particles  = particle_end - particle_start;

            const double* __restrict node_x_ptr = clusters_x_ptr + node_interp_pts_start;
            const double* __restrict node_y_ptr = clusters_y_ptr + node_interp_pts_start;
            const double* __restrict node_z_ptr = clusters_z_ptr + node_interp_pts_start;
            const double* __restrict weights = weights_ptr;

            double node_x_cache[kMaxInterpPts];
            double node_y_cache[kMaxInterpPts];
            double node_z_cache[kMaxInterpPts];
            double w_cache[kMaxInterpPts];
            if (num_interp_pts_per_node <= kMaxInterpPts) {
#ifdef OPENACC_ENABLED
                #pragma acc loop vector
#endif
                for (int j = 0; j < num_interp_pts_per_node; ++j) {
                    node_x_cache[j] = node_x_ptr[j];
                    node_y_cache[j] = node_y_ptr[j];
                    node_z_cache[j] = node_z_ptr[j];
                    w_cache[j] = weights_ptr[j];
                }
                node_x_ptr = node_x_cache;
                node_y_ptr = node_y_cache;
                node_z_ptr = node_z_cache;
                weights = w_cache;
            }

            const double* __restrict elements_x = elements_x_ptr + particle_start;
            const double* __restrict elements_y = elements_y_ptr + particle_start;
            const double* __restrict elements_z = elements_z_ptr + particle_start;

            const double* __restrict sources_q    = sources_q_ptr    + particle_start;
            const double* __restrict sources_q_dx = sources_q_dx_ptr + particle_start;
            const double* __restrict sources_q_dy = sources_q_dy_ptr + particle_start;
            const double* __restrict sources_q_dz = sources_q_dz_ptr + particle_start;

            int* __restrict exact_x = exact_idx_x_ptr + particle_start;
            int* __restrict exact_y = exact_idx_y_ptr + particle_start;
            int* __restrict exact_z = exact_idx_z_ptr + particle_start;
            double* __restrict denom_ptr = denominator_ptr + particle_start;

#ifdef OPENACC_ENABLED
            #pragma acc loop vector
#endif
            for (std::size_t i = 0; i < num_particles; ++i) {
                double denominator_x = 0.;
                double denominator_y = 0.;
                double denominator_z = 0.;
                int ex = -1, ey = -1, ez = -1;

                double xx = elements_x[i];
                double yy = elements_y[i];
                double zz = elements_z[i];

                // because there's a reduction over exact_idx[i], this loop carries a
                // backward dependence and won't actually parallelize
#ifdef OPENACC_ENABLED
                #pragma acc loop reduction(+:denominator_x,denominator_y,denominator_z) reduction(max:ex,ey,ez)
#endif
                for (int j = 0; j < num_interp_pts_per_node; ++j) {
                    double dist_x = xx - node_x_ptr[j];
                    double dist_y = yy - node_y_ptr[j];
                    double dist_z = zz - node_z_ptr[j];

                    denominator_x += weights[j] / dist_x;
                    denominator_y += weights[j] / dist_y;
                    denominator_z += weights[j] / dist_z;

                    const int cx = (std::abs(dist_x) < std::numeric_limits<double>::min()) ? j : -1;
                    const int cy = (std::abs(dist_y) < std::numeric_limits<double>::min()) ? j : -1;
                    const int cz = (std::abs(dist_z) < std::numeric_limits<double>::min()) ? j : -1;

                    ex = (ex > cx) ? ex : cx;
                    ey = (ey > cy) ? ey : cy;
                    ez = (ez > cz) ? ez : cz;
                }

                exact_x[i] = ex;
                exact_y[i] = ey;
                exact_z[i] = ez;

                double denom = 1.0;
                if (ex == -1) denom /= denominator_x;
                if (ey == -1) denom /= denominator_y;
                if (ez == -1) denom /= denominator_z;
                denom_ptr[i] = denom;
            }

            constexpr int kParticleTile = 128;
            double tile_x[kParticleTile];
            double tile_y[kParticleTile];
            double tile_z[kParticleTile];
            double tile_q[kParticleTile];
            double tile_q_dx[kParticleTile];
            double tile_q_dy[kParticleTile];
            double tile_q_dz[kParticleTile];
            double tile_denom[kParticleTile];
            int tile_ex[kParticleTile];
            int tile_ey[kParticleTile];
            int tile_ez[kParticleTile];

            for (std::size_t tile_start = 0; tile_start < num_particles; tile_start += kParticleTile) {
                int tile_count = static_cast<int>(num_particles - tile_start);
                if (tile_count > kParticleTile) tile_count = kParticleTile;

#ifdef OPENACC_ENABLED
                #pragma acc loop vector
#endif
                for (int ii = 0; ii < tile_count; ++ii) {
                    std::size_t pidx = tile_start + static_cast<std::size_t>(ii);
                    tile_x[ii] = elements_x[pidx];
                    tile_y[ii] = elements_y[pidx];
                    tile_z[ii] = elements_z[pidx];
                    tile_q[ii] = sources_q[pidx];
                    tile_q_dx[ii] = sources_q_dx[pidx];
                    tile_q_dy[ii] = sources_q_dy[pidx];
                    tile_q_dz[ii] = sources_q_dz[pidx];
                    tile_denom[ii] = denom_ptr[pidx];
                    tile_ex[ii] = exact_x[pidx];
                    tile_ey[ii] = exact_y[pidx];
                    tile_ez[ii] = exact_z[pidx];
                }

#ifdef OPENACC_ENABLED
                #pragma acc loop collapse(3) vector
#endif
                for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
                for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
                for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                    std::size_t kk = node_charges_start
                           + static_cast<std::size_t>(k1 * num_interp_pts_per_node * num_interp_pts_per_node)
                           + static_cast<std::size_t>(k2 * num_interp_pts_per_node + k3);

                    double cx = node_x_ptr[k1];
                    double w1 = weights[k1];

                    double cy = node_y_ptr[k2];
                    double w2 = weights[k2];

                    double cz = node_z_ptr[k3];
                    double w3 = weights[k3];

                    double q_temp    = 0.;
                    double q_dx_temp = 0.;
                    double q_dy_temp = 0.;
                    double q_dz_temp = 0.;

#ifdef OPENACC_ENABLED
                    #pragma acc loop seq
#endif
                    for (int ii = 0; ii < tile_count; ++ii) {
                        double dist_x = tile_x[ii] - cx;
                        double dist_y = tile_y[ii] - cy;
                        double dist_z = tile_z[ii] - cz;

                        double numerator = 1.;

                        // If exact_idx[i] == -1, then no issues.
                        // If exact_idx[i] != -1, then we want to zero out terms EXCEPT when exactInd=k1.
                        if (tile_ex[ii] == -1) {
                            numerator *= w1 / dist_x;
                        } else {
                            if (tile_ex[ii] != k1) numerator *= 0.;
                        }

                        if (tile_ey[ii] == -1) {
                            numerator *= w2 / dist_y;
                        } else {
                            if (tile_ey[ii] != k2) numerator *= 0.;
                        }

                        if (tile_ez[ii] == -1) {
                            numerator *= w3 / dist_z;
                        } else {
                            if (tile_ez[ii] != k3) numerator *= 0.;
                        }

                        double denom = tile_denom[ii];
                        q_temp    += tile_q[ii]    * numerator * denom;
                        q_dx_temp += tile_q_dx[ii] * numerator * denom;
                        q_dy_temp += tile_q_dy[ii] * numerator * denom;
                        q_dz_temp += tile_q_dz[ii] * numerator * denom;
                    }

                    clusters_q_ptr   [kk] += q_temp;
                    clusters_q_dx_ptr[kk] += q_dx_temp;
                    clusters_q_dy_ptr[kk] += q_dy_temp;
                    clusters_q_dz_ptr[kk] += q_dz_temp;
                }
                }
                }
            }
        }
    }
    timers_.upward_pass.stop();
}


void BoundaryElement::downward_pass(double* __restrict potential)
{
    timers_.downward_pass.start();

    const double* __restrict clusters_x_ptr    = interp_pts_.interp_x_ptr();
    const double* __restrict clusters_y_ptr    = interp_pts_.interp_y_ptr();
    const double* __restrict clusters_z_ptr    = interp_pts_.interp_z_ptr();
    
    const double* __restrict clusters_p_ptr    = interp_potential_.data();
    const double* __restrict clusters_p_dx_ptr = interp_potential_dx_.data();
    const double* __restrict clusters_p_dy_ptr = interp_potential_dy_.data();
    const double* __restrict clusters_p_dz_ptr = interp_potential_dz_.data();
    
    const double* __restrict elements_x_ptr   = elements_.x_ptr();
    const double* __restrict elements_y_ptr   = elements_.y_ptr();
    const double* __restrict elements_z_ptr   = elements_.z_ptr();
    
    const double* __restrict targets_q_ptr     = elements_.target_charge_ptr();
    const double* __restrict targets_q_dx_ptr  = elements_.target_charge_dx_ptr();
    const double* __restrict targets_q_dy_ptr  = elements_.target_charge_dy_ptr();
    const double* __restrict targets_q_dz_ptr  = elements_.target_charge_dz_ptr();
    
    double* weights_ptr = weights_.data();

    std::size_t potential_offset = elements_.num();
    int num_interp_pts_per_node = interp_pts_.num_interp_pts_per_node();
    
    for (std::size_t node_idx = 0; node_idx < tree_.num_nodes(); ++node_idx) {
        
        auto particle_idxs = tree_.node_particle_idxs(node_idx);
        std::size_t node_interp_pts_start = node_idx * num_interp_pts_per_node;
        std::size_t node_potentials_start = node_idx * num_charges_per_node_;
        
        std::size_t particle_start = particle_idxs[0];
        std::size_t num_particles  = particle_idxs[1] - particle_idxs[0];

#ifdef OPENACC_ENABLED
#pragma acc parallel loop present(elements_x_ptr, elements_y_ptr, elements_z_ptr, \
                                  targets_q_ptr, targets_q_dx_ptr, targets_q_dy_ptr, targets_q_dz_ptr, \
                                  clusters_x_ptr, clusters_y_ptr, clusters_z_ptr, \
                                  clusters_p_ptr, clusters_p_dx_ptr, clusters_p_dy_ptr, clusters_p_dz_ptr, \
                                  potential, weights_ptr)
#endif
        for (std::size_t i = 0; i < num_particles; ++i) {
        
            double denominator_x = 0.;
            double denominator_y = 0.;
            double denominator_z = 0.;
            
            int exact_idx_x = -1;
            int exact_idx_y = -1;
            int exact_idx_z = -1;
            
            double xx = elements_x_ptr[particle_start + i];
            double yy = elements_y_ptr[particle_start + i];
            double zz = elements_z_ptr[particle_start + i];
            
#ifdef OPENACC_ENABLED
            #pragma acc loop reduction(+:denominator_x,denominator_y,denominator_z) \
                             reduction(max:exact_idx_x,exact_idx_y,exact_idx_z)
#endif
            for (int j = 0; j < num_interp_pts_per_node; ++j) {
            
                double dist_x = xx - clusters_x_ptr[node_interp_pts_start + j];
                double dist_y = yy - clusters_y_ptr[node_interp_pts_start + j];
                double dist_z = zz - clusters_z_ptr[node_interp_pts_start + j];
                
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

            double pot_comp_   = 0.;
            double pot_comp_dx = 0.;
            double pot_comp_dy = 0.;
            double pot_comp_dz = 0.;
            
#ifdef OPENACC_ENABLED
            #pragma acc loop collapse(3) reduction(+:pot_comp_,  pot_comp_dx, \
                                                     pot_comp_dy,pot_comp_dz)
#endif
            for (int k1 = 0; k1 < num_interp_pts_per_node; ++k1) {
            for (int k2 = 0; k2 < num_interp_pts_per_node; ++k2) {
            for (int k3 = 0; k3 < num_interp_pts_per_node; ++k3) {
                    
                std::size_t kk = node_potentials_start
                               + k1 * num_interp_pts_per_node * num_interp_pts_per_node
                               + k2 * num_interp_pts_per_node + k3;
                               
                double dist_x = xx - clusters_x_ptr[node_interp_pts_start + k1];
                double dist_y = yy - clusters_y_ptr[node_interp_pts_start + k2];
                double dist_z = zz - clusters_z_ptr[node_interp_pts_start + k3];
                
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

                pot_comp_   += numerator * denominator * clusters_p_ptr   [kk];
                pot_comp_dx += numerator * denominator * clusters_p_dx_ptr[kk];
                pot_comp_dy += numerator * denominator * clusters_p_dy_ptr[kk];
                pot_comp_dz += numerator * denominator * clusters_p_dz_ptr[kk];
            }
            }
            }
            
            double pot_temp_1 = targets_q_ptr   [particle_start + i] * pot_comp_;
            double pot_temp_2 = targets_q_dx_ptr[particle_start + i] * pot_comp_dx
                              + targets_q_dy_ptr[particle_start + i] * pot_comp_dy
                              + targets_q_dz_ptr[particle_start + i] * pot_comp_dz;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            potential[particle_start + i]                    += pot_temp_1;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
            #pragma omp atomic update
#endif
            potential[particle_start + i + potential_offset] += pot_temp_2;
        }
    } //end loop over nodes
    timers_.downward_pass.stop();
}


void BoundaryElement::clear_cluster_charges()
{
    timers_.clear_cluster_charges.start();

#ifdef OPENACC_ENABLED
    std::size_t num_charges = num_charges_;
    double* __restrict clusters_q_ptr    = interp_charge_.data();
    double* __restrict clusters_q_dx_ptr = interp_charge_dx_.data();
    double* __restrict clusters_q_dy_ptr = interp_charge_dy_.data();
    double* __restrict clusters_q_dz_ptr = interp_charge_dz_.data();
    
    #pragma acc parallel loop present(clusters_q_ptr, clusters_q_dx_ptr, \
                                      clusters_q_dy_ptr, clusters_q_dz_ptr)
    for (std::size_t i = 0; i < num_charges; ++i) {
        clusters_q_ptr[i] = 0.;
        clusters_q_dx_ptr[i] = 0.;
        clusters_q_dy_ptr[i] = 0.;
        clusters_q_dz_ptr[i] = 0.;
    }
#else
    std::fill(interp_charge_.begin(),    interp_charge_.end(),    0);
    std::fill(interp_charge_dx_.begin(), interp_charge_dx_.end(), 0);
    std::fill(interp_charge_dy_.begin(), interp_charge_dy_.end(), 0);
    std::fill(interp_charge_dz_.begin(), interp_charge_dz_.end(), 0);
#endif

    timers_.clear_cluster_charges.stop();
}


void BoundaryElement::clear_cluster_potentials()
{
    timers_.clear_cluster_potentials.start();

#ifdef OPENACC_ENABLED
    std::size_t num_potentials = num_charges_;
    double* __restrict clusters_p_ptr    = interp_potential_.data();
    double* __restrict clusters_p_dx_ptr = interp_potential_dx_.data();
    double* __restrict clusters_p_dy_ptr = interp_potential_dy_.data();
    double* __restrict clusters_p_dz_ptr = interp_potential_dz_.data();
    
    #pragma acc parallel loop present(clusters_p_ptr, clusters_p_dx_ptr, \
                                      clusters_p_dy_ptr, clusters_p_dz_ptr)
    for (std::size_t i = 0; i < num_potentials; ++i) {
        clusters_p_ptr[i] = 0.;
        clusters_p_dx_ptr[i] = 0.;
        clusters_p_dy_ptr[i] = 0.;
        clusters_p_dz_ptr[i] = 0.;
    }
#else
    std::fill(interp_potential_.begin(),    interp_potential_.end(),    0);
    std::fill(interp_potential_dx_.begin(), interp_potential_dx_.end(), 0);
    std::fill(interp_potential_dy_.begin(), interp_potential_dy_.end(), 0);
    std::fill(interp_potential_dz_.begin(), interp_potential_dz_.end(), 0);
#endif

    timers_.clear_cluster_potentials.stop();
}


void BoundaryElement::copyin_clusters_to_device() const
{
    timers_.copyin_clusters_to_device.start();

#ifdef OPENACC_ENABLED
    const double* q_ptr    = interp_charge_.data();
    const double* q_dx_ptr = interp_charge_dx_.data();
    const double* q_dy_ptr = interp_charge_dy_.data();
    const double* q_dz_ptr = interp_charge_dz_.data();
    
    std::size_t q_num    = interp_charge_.size();
    std::size_t q_dx_num = interp_charge_dx_.size();
    std::size_t q_dy_num = interp_charge_dy_.size();
    std::size_t q_dz_num = interp_charge_dz_.size();
    
    const double* p_ptr    = interp_potential_.data();
    const double* p_dx_ptr = interp_potential_dx_.data();
    const double* p_dy_ptr = interp_potential_dy_.data();
    const double* p_dz_ptr = interp_potential_dz_.data();
    
    std::size_t p_num    = interp_potential_.size();
    std::size_t p_dx_num = interp_potential_dx_.size();
    std::size_t p_dy_num = interp_potential_dy_.size();
    std::size_t p_dz_num = interp_potential_dz_.size();

    const double* weights_ptr = weights_.data();
    std::size_t weights_num = weights_.size();

    const double* potential_temp_ptr = potential_temp_.data();
    std::size_t potential_temp_num = potential_temp_.size();

    const int* exact_idx_x_ptr = exact_idx_x_.data();
    const int* exact_idx_y_ptr = exact_idx_y_.data();
    const int* exact_idx_z_ptr = exact_idx_z_.data();
    const double* denominator_ptr = denominator_.data();
    std::size_t max_particles = exact_idx_x_.size();

    const std::uint32_t* node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* node_end_ptr = node_particles_end_u32_.data();
    std::size_t node_count = node_particles_begin_u32_.size();

    const std::uint32_t* element_node_idx_ptr = element_node_idx_u32_.data();
    std::size_t element_node_count = element_node_idx_u32_.size();

    const std::size_t* level_offsets_ptr = level_offsets_.data();
    const std::size_t* level_nodes_ptr = level_nodes_.data();
    std::size_t level_offsets_num = level_offsets_.size();
    std::size_t level_nodes_num = level_nodes_.size();

    const std::uint32_t* pp_offsets_ptr = pp_offsets_u32_.data();
    const std::uint32_t* pc_offsets_ptr = pc_offsets_u32_.data();
    const std::uint32_t* cp_offsets_ptr = cp_offsets_u32_.data();
    const std::uint32_t* cc_offsets_ptr = cc_offsets_u32_.data();
    const std::uint32_t* pp_sources_ptr = pp_sources_u32_.data();
    const std::uint32_t* pc_sources_ptr = pc_sources_u32_.data();
    const std::uint32_t* cp_sources_ptr = cp_sources_u32_.data();
    const std::uint32_t* cc_sources_ptr = cc_sources_u32_.data();

    std::size_t pp_offsets_num = pp_offsets_u32_.size();
    std::size_t pc_offsets_num = pc_offsets_u32_.size();
    std::size_t cp_offsets_num = cp_offsets_u32_.size();
    std::size_t cc_offsets_num = cc_offsets_u32_.size();
    std::size_t pp_sources_num = pp_sources_u32_.size();
    std::size_t pc_sources_num = pc_sources_u32_.size();
    std::size_t cp_sources_num = cp_sources_u32_.size();
    std::size_t cc_sources_num = cc_sources_u32_.size();
    
    #pragma acc enter data create( \
                q_ptr[0:q_num], q_dx_ptr[0:q_dx_num], q_dy_ptr[0:q_dy_num], q_dz_ptr[0:q_dz_num], \
                p_ptr[0:p_num], p_dx_ptr[0:p_dx_num], p_dy_ptr[0:p_dy_num], p_dz_ptr[0:p_dz_num], \
                potential_temp_ptr[0:potential_temp_num])
    #pragma acc enter data copyin(weights_ptr[0:weights_num])
    #pragma acc enter data create(exact_idx_x_ptr[0:max_particles], exact_idx_y_ptr[0:max_particles], \
                                  exact_idx_z_ptr[0:max_particles], denominator_ptr[0:max_particles])
    #pragma acc enter data copyin(node_begin_ptr[0:node_count], node_end_ptr[0:node_count])
    #pragma acc enter data copyin(element_node_idx_ptr[0:element_node_count])
    #pragma acc enter data copyin(level_offsets_ptr[0:level_offsets_num], \
                                  level_nodes_ptr[0:level_nodes_num])
    #pragma acc enter data copyin( \
                pp_offsets_ptr[0:pp_offsets_num], pc_offsets_ptr[0:pc_offsets_num], \
                cp_offsets_ptr[0:cp_offsets_num], cc_offsets_ptr[0:cc_offsets_num], \
                pp_sources_ptr[0:pp_sources_num], pc_sources_ptr[0:pc_sources_num], \
                cp_sources_ptr[0:cp_sources_num], cc_sources_ptr[0:cc_sources_num])
#endif

    timers_.copyin_clusters_to_device.stop();
}


void BoundaryElement::delete_clusters_from_device() const
{
    timers_.delete_clusters_from_device.start();

#ifdef OPENACC_ENABLED
    const double* q_ptr    = interp_charge_.data();
    const double* q_dx_ptr = interp_charge_dx_.data();
    const double* q_dy_ptr = interp_charge_dy_.data();
    const double* q_dz_ptr = interp_charge_dz_.data();
    
    std::size_t q_num    = interp_charge_.size();
    std::size_t q_dx_num = interp_charge_dx_.size();
    std::size_t q_dy_num = interp_charge_dy_.size();
    std::size_t q_dz_num = interp_charge_dz_.size();
    
    const double* p_ptr    = interp_potential_.data();
    const double* p_dx_ptr = interp_potential_dx_.data();
    const double* p_dy_ptr = interp_potential_dy_.data();
    const double* p_dz_ptr = interp_potential_dz_.data();
    
    std::size_t p_num    = interp_potential_.size();
    std::size_t p_dx_num = interp_potential_dx_.size();
    std::size_t p_dy_num = interp_potential_dy_.size();
    std::size_t p_dz_num = interp_potential_dz_.size();

    const double* weights_ptr = weights_.data();
    std::size_t weights_num = weights_.size();

    const double* potential_temp_ptr = potential_temp_.data();
    std::size_t potential_temp_num = potential_temp_.size();

    const int* exact_idx_x_ptr = exact_idx_x_.data();
    const int* exact_idx_y_ptr = exact_idx_y_.data();
    const int* exact_idx_z_ptr = exact_idx_z_.data();
    const double* denominator_ptr = denominator_.data();
    std::size_t max_particles = exact_idx_x_.size();

    const std::uint32_t* node_begin_ptr = node_particles_begin_u32_.data();
    const std::uint32_t* node_end_ptr = node_particles_end_u32_.data();
    std::size_t node_count = node_particles_begin_u32_.size();

    const std::uint32_t* element_node_idx_ptr = element_node_idx_u32_.data();
    std::size_t element_node_count = element_node_idx_u32_.size();

    const std::size_t* level_offsets_ptr = level_offsets_.data();
    const std::size_t* level_nodes_ptr = level_nodes_.data();
    std::size_t level_offsets_num = level_offsets_.size();
    std::size_t level_nodes_num = level_nodes_.size();

    const std::uint32_t* pp_offsets_ptr = pp_offsets_u32_.data();
    const std::uint32_t* pc_offsets_ptr = pc_offsets_u32_.data();
    const std::uint32_t* cp_offsets_ptr = cp_offsets_u32_.data();
    const std::uint32_t* cc_offsets_ptr = cc_offsets_u32_.data();
    const std::uint32_t* pp_sources_ptr = pp_sources_u32_.data();
    const std::uint32_t* pc_sources_ptr = pc_sources_u32_.data();
    const std::uint32_t* cp_sources_ptr = cp_sources_u32_.data();
    const std::uint32_t* cc_sources_ptr = cc_sources_u32_.data();

    std::size_t pp_offsets_num = pp_offsets_u32_.size();
    std::size_t pc_offsets_num = pc_offsets_u32_.size();
    std::size_t cp_offsets_num = cp_offsets_u32_.size();
    std::size_t cc_offsets_num = cc_offsets_u32_.size();
    std::size_t pp_sources_num = pp_sources_u32_.size();
    std::size_t pc_sources_num = pc_sources_u32_.size();
    std::size_t cp_sources_num = cp_sources_u32_.size();
    std::size_t cc_sources_num = cc_sources_u32_.size();
    
    #pragma acc exit data delete( \
                q_ptr[0:q_num], q_dx_ptr[0:q_dx_num], q_dy_ptr[0:q_dy_num], q_dz_ptr[0:q_dz_num], \
                p_ptr[0:p_num], p_dx_ptr[0:p_dx_num], p_dy_ptr[0:p_dy_num], p_dz_ptr[0:p_dz_num], \
                potential_temp_ptr[0:potential_temp_num])
    #pragma acc exit data delete(weights_ptr[0:weights_num])
    #pragma acc exit data delete(exact_idx_x_ptr[0:max_particles], exact_idx_y_ptr[0:max_particles], \
                                 exact_idx_z_ptr[0:max_particles], denominator_ptr[0:max_particles])
    #pragma acc exit data delete(node_begin_ptr[0:node_count], node_end_ptr[0:node_count])
    #pragma acc exit data delete(element_node_idx_ptr[0:element_node_count])
    #pragma acc exit data delete(level_offsets_ptr[0:level_offsets_num], \
                                 level_nodes_ptr[0:level_nodes_num])
    #pragma acc exit data delete( \
                pp_offsets_ptr[0:pp_offsets_num], pc_offsets_ptr[0:pc_offsets_num], \
                cp_offsets_ptr[0:cp_offsets_num], cc_offsets_ptr[0:cc_offsets_num], \
                pp_sources_ptr[0:pp_sources_num], pc_sources_ptr[0:pc_sources_num], \
                cp_sources_ptr[0:cp_sources_num], cc_sources_ptr[0:cc_sources_num])
#endif

    timers_.delete_clusters_from_device.stop();
}


//void BoundaryElement::finalize()
//{
//    timers_.finalize.start();
//
//    solvation_energy_ = constants::UNITS_PARA  * elements_.compute_solvation_energy(potential_);
//    coulombic_energy_ = constants::UNITS_COEFF * molecule_.coulombic_energy();
//    free_energy_      = solvation_energy_ + coulombic_energy_;
//
//    elements_.unorder(potential_);
//
//    constexpr double pot_scaling = constants::UNITS_COEFF * constants::PI * 4.;
//    std::transform(std::begin(potential_), std::end(potential_),
//                   std::begin(potential_), [=](double x){ return x * pot_scaling; });
//
//    auto pot_min_max = std::minmax_element(
//        potential_.begin(), potential_.begin() + potential_.size() / 2);
//
//    auto pot_normal_min_max = std::minmax_element(
//        potential_.begin() + potential_.size() / 2, potential_.end());
//
//    pot_min_ = *pot_min_max.first;
//    pot_max_ = *pot_min_max.second;
//
//    pot_normal_min_ = *pot_normal_min_max.first;
//    pot_normal_max_ = *pot_normal_min_max.second;
//
//    timers_.finalize.stop();
//}


void Timers_BoundaryElement::print() const
{
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(5);
    std::cout << "|...BoundaryElement function times (s)...." << std::endl;
    std::cout << "|   |...ctor.......................: ";
    std::cout << std::setw(12) << std::right << ctor                       .elapsed_time() << std::endl;
    std::cout << "|   |...run_GMRES..................: ";
    std::cout << std::setw(12) << std::right << run_GMRES                  .elapsed_time() << std::endl;
    std::cout << "|       |...matrix_vector..........: ";
    std::cout << std::setw(12) << std::right << matrix_vector              .elapsed_time() << std::endl;
    std::cout << "|           |...upward pass........: ";
    std::cout << std::setw(12) << std::right << upward_pass                .elapsed_time() << std::endl;
    std::cout << "|           |...PP interact........: ";
    std::cout << std::setw(12) << std::right << particle_particle_interact .elapsed_time() << std::endl;
    std::cout << "|           |...PC interact........: ";
    std::cout << std::setw(12) << std::right << particle_cluster_interact  .elapsed_time() << std::endl;
    std::cout << "|           |...CP interact........: ";
    std::cout << std::setw(12) << std::right << cluster_particle_interact  .elapsed_time() << std::endl;
    std::cout << "|           |...CC interact........: ";
    std::cout << std::setw(12) << std::right << cluster_cluster_interact   .elapsed_time() << std::endl;
    std::cout << "|           |...downward pass......: ";
    std::cout << std::setw(12) << std::right << downward_pass              .elapsed_time() << std::endl;
    std::cout << "|       |...precondition...........: ";
    std::cout << std::setw(12) << std::right << precondition               .elapsed_time() << std::endl;
    std::cout << "|" << std::endl;
}


std::string Timers_BoundaryElement::get_durations() const
{
    std::string durations;
    durations.append(std::to_string(ctor                       .elapsed_time())).append(", ");
    durations.append(std::to_string(run_GMRES                  .elapsed_time())).append(", ");
    durations.append(std::to_string(matrix_vector              .elapsed_time())).append(", ");
    durations.append(std::to_string(upward_pass                .elapsed_time())).append(", ");
    durations.append(std::to_string(particle_particle_interact .elapsed_time())).append(", ");
    durations.append(std::to_string(particle_cluster_interact  .elapsed_time())).append(", ");
    durations.append(std::to_string(cluster_particle_interact  .elapsed_time())).append(", ");
    durations.append(std::to_string(cluster_cluster_interact   .elapsed_time())).append(", ");
    durations.append(std::to_string(downward_pass              .elapsed_time())).append(", ");
    durations.append(std::to_string(precondition               .elapsed_time())).append(", ");
    
    return durations;
}


std::string Timers_BoundaryElement::get_headers() const
{
    std::string headers;
    headers.append("BoundaryElement ctor, ");
    headers.append("BoundaryElement run_GMRES, ");
    headers.append("BoundaryElement matrix_vector, ");
//    headers.append("BoundaryElement clear_cluster_charges, ");
//    headers.append("BoundaryElement clear_cluster_potentials, ");
//    headers.append("BoundaryElement copyin_clusters_to_device, ");
//    headers.append("BoundaryElement delete_clusters_from_device, ");
    headers.append("BoundaryElement upward_pass, ");
    headers.append("BoundaryElement particle_particle_interact, ");
    headers.append("BoundaryElement particle_cluster_interact, ");
    headers.append("BoundaryElement cluster_particle_interact, ");
    headers.append("BoundaryElement cluster_cluster_interact, ");
    headers.append("BoundaryElement downward_pass, ");
    headers.append("BoundaryElement precondition, ");
    
    return headers;
}
