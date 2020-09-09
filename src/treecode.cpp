#include <cstring>
#include <iostream>

#include "constants.h"
#include "treecode.h"

#include <cstdio>
#include <cstdlib>

long int Treecode::run_GMRES()
{
    static long int info;
    long int restrt = 10;
    long int length = 2 * particles_.num();
    long int ldw    = length;
    long int ldh    = restrt + 1;
    double resid    = 1e-4;
    
    long int iter = 100;
    
    std::vector<double> work(ldw * (restrt + 4));
    std::vector<double> h   (ldw * (restrt + 2));
    
    std::fill(potential_.begin(), potential_.end(), 0);
    
    Treecode::gmres_(length, particles_.source_term_ptr(), potential_.data(),
                     &restrt, work.data(), ldw, h.data(), ldh, &iter, &resid, &info);

    return iter;
}


void Treecode::matrix_vector(double alpha, double* potential_old,
                             double beta,  double* potential_new)
{
    double potential_coeff_1 = 0.5 * (1. +      params_.phys_eps_);
    double potential_coeff_2 = 0.5 * (1. + 1. / params_.phys_eps_);
    
    auto* potential_temp = (double *)std::malloc(potential_.size() * sizeof(double));
    std::memcpy(potential_temp, potential_new, potential_.size() * sizeof(double));
    std::memset(potential_new, 0, potential_.size() * sizeof(double));
    
    particles_.compute_charges(potential_old);
    clusters_.upward_pass();
    clusters_.clear_potentials();

    for (std::size_t target_node_idx = 0; target_node_idx < tree_.num_nodes(); ++target_node_idx) {
        
        for (auto source_node_idx : interaction_list_.particle_particle(target_node_idx))
            Treecode::particle_particle_interact(potential_new, potential_old,
                    tree_.node_particle_idxs(target_node_idx), tree_.node_particle_idxs(source_node_idx));
    
        for (auto source_node_idx : interaction_list_.particle_cluster(target_node_idx))
            Treecode::particle_cluster_interact(potential_new, 
                    tree_.node_particle_idxs(target_node_idx), source_node_idx);
        
        for (auto source_node_idx : interaction_list_.cluster_particle(target_node_idx))
            Treecode::cluster_particle_interact(potential_new, 
                    target_node_idx, tree_.node_particle_idxs(source_node_idx));
        
        for (auto source_node_idx : interaction_list_.cluster_cluster(target_node_idx))
            Treecode::cluster_cluster_interact(potential_new, target_node_idx, source_node_idx);
    }
    
    clusters_.downward_pass(potential_new);
    
    for (std::size_t i = 0; i < potential_.size() / 2; ++i)
        potential_new[i] = beta * potential_temp[i]
                + alpha * (potential_coeff_1 * potential_old[i] - potential_new[i]);
                                             
    for (std::size_t i = potential_.size() / 2; i < potential_.size(); ++i)
        potential_new[i] =  beta * potential_temp[i]
                + alpha * (potential_coeff_2 * potential_old[i] - potential_new[i]);
                
    std::free(potential_temp);
}


void Treecode::precondition(double *z, double *r)
{
    double potential_coeff_1 = 0.5 * (1. +      params_.phys_eps_);
    double potential_coeff_2 = 0.5 * (1. + 1. / params_.phys_eps_);
    
    for (std::size_t i = 0;                i <     particles_.num(); ++i) z[i] = r[i] / potential_coeff_1;
    for (std::size_t i = particles_.num(); i < 2 * particles_.num(); ++i) z[i] = r[i] / potential_coeff_2;
}


void Treecode::particle_particle_interact(double* potential, double* potential_old,
                                          std::array<std::size_t, 2> target_node_particle_idxs,
                                          std::array<std::size_t, 2> source_node_particle_idxs)
{
    std::size_t target_node_particle_begin = target_node_particle_idxs[0];
    std::size_t target_node_particle_end   = target_node_particle_idxs[1];

    std::size_t source_node_particle_begin = source_node_particle_idxs[0];
    std::size_t source_node_particle_end   = source_node_particle_idxs[1];
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    double kappa2 = params_.phys_kappa2_;
    
    const double* __restrict__ particles_x_ptr    = particles_.x_ptr();
    const double* __restrict__ particles_y_ptr    = particles_.y_ptr();
    const double* __restrict__ particles_z_ptr    = particles_.z_ptr();
    
    const double* __restrict__ particles_nx_ptr   = particles_.nx_ptr();
    const double* __restrict__ particles_ny_ptr   = particles_.ny_ptr();
    const double* __restrict__ particles_nz_ptr   = particles_.nz_ptr();

    const double* __restrict__ particles_area_ptr = particles_.area_ptr();
    
    std::size_t num_particles = particles_.num();

    for (std::size_t j = target_node_particle_begin; j < target_node_particle_end; ++j) {
        
        double target_x = particles_x_ptr[j];
        double target_y = particles_y_ptr[j];
        double target_z = particles_z_ptr[j];
        
        double target_nx = particles_nx_ptr[j];
        double target_ny = particles_ny_ptr[j];
        double target_nz = particles_nz_ptr[j];
    
        for (std::size_t k = source_node_particle_begin; k < source_node_particle_end; ++k) {
        
            double source_x = particles_x_ptr[k];
            double source_y = particles_y_ptr[k];
            double source_z = particles_z_ptr[k];
            
            double source_nx = particles_nx_ptr[k];
            double source_ny = particles_ny_ptr[k];
            double source_nz = particles_nz_ptr[k];
            double source_area = particles_area_ptr[k];
            
            double potential_old_0 = potential_old[k];
            double potential_old_1 = potential_old[k + num_particles];
            
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
                
                double cos_theta  = (source_nx * dist_x + source_ny * dist_y + source_nz * dist_z) * one_over_r;
                double cos_theta0 = (target_nx * dist_x + target_ny * dist_y + target_nz * dist_z) * one_over_r;
                
                double tp1 = G0 * one_over_r;
                double tp2 = (1. + kappa_r) * exp_kappa_r;

                double dot_tqsq = source_nx * target_nx + source_ny * target_ny + source_nz * target_nz;
                double G3 = (dot_tqsq - 3. * cos_theta0 * cos_theta) * one_over_r * tp1;
                double G4 = tp2 * G3 - kappa2 * cos_theta0 * cos_theta * Gk;

                double L1 = cos_theta  * tp1 * (1. - tp2 * eps);
                double L2 = G0 - Gk;
                double L3 = G4 - G3;
                double L4 = cos_theta0 * tp1 * (1. - tp2 / eps);
                
                potential[j]                 += (L1 * potential_old_0 + L2 * potential_old_1) * source_area;
                potential[j + num_particles] += (L3 * potential_old_0 + L4 * potential_old_1) * source_area;
            }
        }
    }
}


void Treecode::particle_cluster_interact(double* potential, 
                                         std::array<std::size_t, 2> target_node_particle_idxs,
                                         std::size_t source_node_idx)
{
    std::size_t num_particles   = particles_.num();
    int num_interp_pts_per_node = clusters_.num_interp_pts_per_node();
    int num_charges_per_node    = clusters_.num_charges_per_node();

    std::size_t target_node_particle_begin      = target_node_particle_idxs[0];
    std::size_t target_node_particle_end        = target_node_particle_idxs[1];

    std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
    std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    
    const double* __restrict__ particles_x_ptr   = particles_.x_ptr();
    const double* __restrict__ particles_y_ptr   = particles_.y_ptr();
    const double* __restrict__ particles_z_ptr   = particles_.z_ptr();
    
    const double* __restrict__ targets_q_ptr     = particles_.target_charge_ptr();
    const double* __restrict__ targets_q_dx_ptr  = particles_.target_charge_dx_ptr();
    const double* __restrict__ targets_q_dy_ptr  = particles_.target_charge_dy_ptr();
    const double* __restrict__ targets_q_dz_ptr  = particles_.target_charge_dz_ptr();
    
    const double* __restrict__ clusters_x_ptr    = clusters_.interp_x_ptr();
    const double* __restrict__ clusters_y_ptr    = clusters_.interp_y_ptr();
    const double* __restrict__ clusters_z_ptr    = clusters_.interp_z_ptr();

    const double* __restrict__ clusters_q_ptr    = clusters_.interp_charge_ptr();
    const double* __restrict__ clusters_q_dx_ptr = clusters_.interp_charge_dx_ptr();
    const double* __restrict__ clusters_q_dy_ptr = clusters_.interp_charge_dy_ptr();
    const double* __restrict__ clusters_q_dz_ptr = clusters_.interp_charge_dz_ptr();
    
    
    for (std::size_t j = target_node_particle_begin; j < target_node_particle_end; ++j) {

        double target_x = particles_x_ptr[j];
        double target_y = particles_y_ptr[j];
        double target_z = particles_z_ptr[j];
        
        double pot_comp_   = 0.;
        double pot_comp_dx = 0.;
        double pot_comp_dy = 0.;
        double pot_comp_dz = 0.;
        
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

            double expkr   =  std::exp(-kappa * r);
            double d1term  =  r3inv * expkr * (1. + (kappa * r));
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa * r)
                                                   + (kappa * kappa * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa * r));

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
        
        potential[j]                 += targets_q_ptr   [j] * pot_comp_;
        potential[j + num_particles] += targets_q_dx_ptr[j] * pot_comp_dx
                                      + targets_q_dy_ptr[j] * pot_comp_dy
                                      + targets_q_dz_ptr[j] * pot_comp_dz;
    }
}


void Treecode::cluster_particle_interact(double* potential, std::size_t target_node_idx,
                                         std::array<std::size_t, 2> source_node_particle_idxs)
{
    int num_interp_pts_per_node = clusters_.num_interp_pts_per_node();
    int num_potentials_per_node = clusters_.num_charges_per_node();
    
    std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
    std::size_t target_cluster_potentials_begin = target_node_idx * num_potentials_per_node;

    std::size_t source_node_particle_begin      = source_node_particle_idxs[0];
    std::size_t source_node_particle_end        = source_node_particle_idxs[1];
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    
    const double* __restrict__ clusters_x_ptr    = clusters_.interp_x_ptr();
    const double* __restrict__ clusters_y_ptr    = clusters_.interp_y_ptr();
    const double* __restrict__ clusters_z_ptr    = clusters_.interp_z_ptr();
    
    double* __restrict__ clusters_p_ptr          = clusters_.interp_potential_ptr();
    double* __restrict__ clusters_p_dx_ptr       = clusters_.interp_potential_dx_ptr();
    double* __restrict__ clusters_p_dy_ptr       = clusters_.interp_potential_dy_ptr();
    double* __restrict__ clusters_p_dz_ptr       = clusters_.interp_potential_dz_ptr();
    
    const double* __restrict__ particles_x_ptr   = particles_.x_ptr();
    const double* __restrict__ particles_y_ptr   = particles_.y_ptr();
    const double* __restrict__ particles_z_ptr   = particles_.z_ptr();
    
    const double* __restrict__ sources_q_ptr     = particles_.source_charge_ptr();
    const double* __restrict__ sources_q_dx_ptr  = particles_.source_charge_dx_ptr();
    const double* __restrict__ sources_q_dy_ptr  = particles_.source_charge_dy_ptr();
    const double* __restrict__ sources_q_dz_ptr  = particles_.source_charge_dz_ptr();
    
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
    
        for (std::size_t k = source_node_particle_begin; k < source_node_particle_end; ++k) {

            double dx = target_x - particles_x_ptr[k];
            double dy = target_y - particles_y_ptr[k];
            double dz = target_z - particles_z_ptr[k];

            double r2    = dx*dx + dy*dy + dz*dz;
            double r     = std::sqrt(r2);
            double rinv  = 1. / r;
            double r3inv = rinv  * rinv * rinv;
            double r5inv = r3inv * rinv * rinv;

            double expkr   =  std::exp(-kappa * r);
            double d1term  =  r3inv * expkr * (1. + (kappa * r));
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa * r)
                                                   + (kappa * kappa * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa * r));

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
    
        clusters_p_ptr   [jj] += pot_comp_;
        clusters_p_dx_ptr[jj] += pot_comp_dx;
        clusters_p_dy_ptr[jj] += pot_comp_dy;
        clusters_p_dz_ptr[jj] += pot_comp_dz;
    }
    }
    }
}


void Treecode::cluster_cluster_interact(double* potential, std::size_t target_node_idx, 
                                        std::size_t source_node_idx)
{
    int num_interp_pts_per_node = clusters_.num_interp_pts_per_node();
    int num_charges_per_node    = clusters_.num_charges_per_node();

    std::size_t target_cluster_interp_pts_begin = target_node_idx * num_interp_pts_per_node;
    std::size_t target_cluster_potentials_begin = target_node_idx * num_charges_per_node;
    
    std::size_t source_cluster_interp_pts_begin = source_node_idx * num_interp_pts_per_node;
    std::size_t source_cluster_charges_begin    = source_node_idx * num_charges_per_node;
    
    double eps    = params_.phys_eps_;
    double kappa  = params_.phys_kappa_;
    
    const double* __restrict__ clusters_x_ptr    = clusters_.interp_x_ptr();
    const double* __restrict__ clusters_y_ptr    = clusters_.interp_y_ptr();
    const double* __restrict__ clusters_z_ptr    = clusters_.interp_z_ptr();

    double* __restrict__ clusters_p_ptr          = clusters_.interp_potential_ptr();
    double* __restrict__ clusters_p_dx_ptr       = clusters_.interp_potential_dx_ptr();
    double* __restrict__ clusters_p_dy_ptr       = clusters_.interp_potential_dy_ptr();
    double* __restrict__ clusters_p_dz_ptr       = clusters_.interp_potential_dz_ptr();
    
    const double* __restrict__ clusters_q_ptr    = clusters_.interp_charge_ptr();
    const double* __restrict__ clusters_q_dx_ptr = clusters_.interp_charge_dx_ptr();
    const double* __restrict__ clusters_q_dy_ptr = clusters_.interp_charge_dy_ptr();
    const double* __restrict__ clusters_q_dz_ptr = clusters_.interp_charge_dz_ptr();
    
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

            double expkr   =  std::exp(-kappa * r);
            double d1term  =  r3inv * expkr * (1. + (kappa * r));
            double d1term1 = -r3inv + d1term * eps;
            double d1term2 = -r3inv + d1term / eps;
            double d2term  =  r5inv * (-3. + expkr * (3. + (3. * kappa * r)
                                                   + (kappa * kappa * r2)));
            double d3term  =  r3inv * ( 1. - expkr * (1. + kappa * r));

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
    
        clusters_p_ptr   [jj] += pot_comp_;
        clusters_p_dx_ptr[jj] += pot_comp_dx;
        clusters_p_dy_ptr[jj] += pot_comp_dy;
        clusters_p_dz_ptr[jj] += pot_comp_dz;
    }
    }
    }
}