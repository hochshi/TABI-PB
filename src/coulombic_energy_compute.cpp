#include <cmath>
// #include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// #include "constants.h"
#include "coulombic_energy_compute.h"

CoulombicEnergyCompute::CoulombicEnergyCompute(const class Molecule& molecule,
                      const class InterpolationPoints& mol_interp_pts, const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps_solute)
    : TreeCompute(mol_tree, interaction_list),
      molecule_(molecule), mol_interp_pts_(mol_interp_pts), eps_solute_(phys_eps_solute)
{
//    timers_.ctor.start();
    
    /* Target and Source clusters */
    
    num_mol_interp_pts_per_node_        = mol_interp_pts_.num_interp_pts_per_node();
    num_mol_interp_charges_per_node_    = std::pow(num_mol_interp_pts_per_node_, 3);
    num_mol_charges_                    = source_tree_.num_nodes() * num_mol_interp_charges_per_node_;
    
    num_mol_interp_potentials_per_node_ = num_mol_interp_charges_per_node_;
    num_mol_potentials_                 = num_mol_charges_;
    
    mol_interp_charge_.assign(num_mol_charges_, 0.);
    mol_interp_potential_.assign(num_mol_potentials_, 0.);

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

    exact_idx_x_.assign(max_mol_particles_per_node_, -1);
    exact_idx_y_.assign(max_mol_particles_per_node_, -1);
    exact_idx_z_.assign(max_mol_particles_per_node_, -1);
    denominator_.assign(max_mol_particles_per_node_, 0.0);

    /* Coulombic energy */

    coul_eng_vec_.resize(1);
    coul_eng_vec_[0] = 0.;

    coulombic_energy_ = 0.;

//    timers_.ctor.stop();
}


double CoulombicEnergyCompute::compute()
{
    CoulombicEnergyCompute::copyin_clusters_to_device();
    CoulombicEnergyCompute::run();
    CoulombicEnergyCompute::delete_clusters_from_device();
    
    coulombic_energy_ = coul_eng_vec_[0] / 2;
    
    return coulombic_energy_;
}


void CoulombicEnergyCompute::particle_particle_interact(std::array<std::size_t, 2> target_node_idxs,
                                                        std::array<std::size_t, 2> source_node_idxs)
{
//    timers_.particle_particle_interact.start();

    /* Targets */
    
    std::size_t target_node_begin      = target_node_idxs[0];
    std::size_t target_node_end        = target_node_idxs[1];

    /* Sources */
    
    std::size_t source_node_begin      = source_node_idxs[0];
    std::size_t source_node_end        = source_node_idxs[1];
    
    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();

    const double* __restrict mol_q_ptr = molecule_.charge_ptr();

    /* Potential */

    double* __restrict coul_eng_ptr = coul_eng_vec_.data();

#ifdef USE_CUDA_CC
    if (try_particle_particle_interact_cuda_(target_node_begin, target_node_end,
                                             source_node_begin, source_node_end)) {
        return;
    }
#endif

    for (std::size_t j = target_node_begin; j < target_node_end; ++j) {
        
        double target_x = mol_x_ptr[j];
        double target_y = mol_y_ptr[j];
        double target_z = mol_z_ptr[j];
        double target_q = mol_q_ptr[j];
        
        double pot_temp = 0.;
        
        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {

            double dx = target_x - mol_x_ptr[k];
            double dy = target_y - mol_y_ptr[k];
            double dz = target_z - mol_z_ptr[k];
            double r  = dx*dx + dy*dy + dz*dz;

            if (r > 0) pot_temp += target_q * mol_q_ptr[k] / eps_solute_ / std::sqrt(r);
        }

#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        coul_eng_ptr[0] += pot_temp;
    }

//    timers_.particle_particle_interact.stop();
}


void CoulombicEnergyCompute::particle_cluster_interact(std::array<std::size_t, 2> target_node_idxs,
                                                       std::size_t source_node_idx)
{
//    timers_.particle_cluster_interact.start();
    
    /* Targets */
    
    std::size_t target_node_begin          = target_node_idxs[0];
    std::size_t target_node_end            = target_node_idxs[1];
    
    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();

    const double* __restrict mol_q_ptr = molecule_.charge_ptr();

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

    double* __restrict coul_eng_ptr = coul_eng_vec_.data();
    
    
#ifdef USE_CUDA_CC
    if (try_particle_cluster_interact_cuda_(target_node_begin, target_node_end,
                                            source_node_idx)) {
        return;
    }
#endif

    for (std::size_t j = target_node_begin; j < target_node_end; ++j) {

        double target_x = mol_x_ptr[j];
        double target_y = mol_y_ptr[j];
        double target_z = mol_z_ptr[j];
        double target_q = mol_q_ptr[j];
        
        double pot_temp = 0.;
        
        for (int k1 = 0; k1 < num_mol_interp_pts_per_node; ++k1) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; ++k2) {
        for (int k3 = 0; k3 < num_mol_interp_pts_per_node; ++k3) {
                
            std::size_t kk = source_cluster_interp_charges_begin
                           + k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                           + k2 * num_mol_interp_pts_per_node + k3;

            double dx = target_x - mol_clusters_x_ptr[source_cluster_interp_pts_begin + k1];
            double dy = target_y - mol_clusters_y_ptr[source_cluster_interp_pts_begin + k2];
            double dz = target_z - mol_clusters_z_ptr[source_cluster_interp_pts_begin + k3];

            pot_temp += mol_clusters_q_ptr[kk] / eps_solute_ / std::sqrt(dx*dx + dy*dy + dz*dz);
        }
        }
        }
        
        pot_temp *= target_q;

#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        coul_eng_ptr[0] += pot_temp;
    }

//    timers_.particle_cluster_interact.stop();
}


void CoulombicEnergyCompute::cluster_particle_interact(std::size_t target_node_idx,
                                                       std::array<std::size_t, 2> source_node_idxs)
{
//    timers_.cluster_particle_interact.start();

    /* Targets */
    
    int num_mol_interp_pts_per_node                    = num_mol_interp_pts_per_node_;
    int num_mol_interp_potentials_per_node             = num_mol_interp_potentials_per_node_;

    std::size_t target_cluster_interp_pts_begin        = target_node_idx * num_mol_interp_pts_per_node_;
    std::size_t target_cluster_interp_potentials_begin = target_node_idx * num_mol_interp_potentials_per_node_;

    const double* __restrict mol_clusters_x_ptr = mol_interp_pts_.interp_x_ptr();
    const double* __restrict mol_clusters_y_ptr = mol_interp_pts_.interp_y_ptr();
    const double* __restrict mol_clusters_z_ptr = mol_interp_pts_.interp_z_ptr();

    double*       __restrict mol_clusters_p_ptr = mol_interp_potential_.data();


    /* Sources */
    
    std::size_t source_node_begin      = source_node_idxs[0];
    std::size_t source_node_end        = source_node_idxs[1];

    const double* __restrict mol_x_ptr = molecule_.x_ptr();
    const double* __restrict mol_y_ptr = molecule_.y_ptr();
    const double* __restrict mol_z_ptr = molecule_.z_ptr();

    const double* __restrict mol_q_ptr = molecule_.charge_ptr();


#ifdef USE_CUDA_CC
    if (try_cluster_particle_interact_cuda_(target_node_idx, source_node_begin,
                                            source_node_end)) {
        return;
    }
#endif


    for (int j1 = 0; j1 < num_mol_interp_pts_per_node; ++j1) {
    for (int j2 = 0; j2 < num_mol_interp_pts_per_node; ++j2) {
    for (int j3 = 0; j3 < num_mol_interp_pts_per_node; ++j3) {
    
        std::size_t jj = target_cluster_interp_potentials_begin
                       + j1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                       + j2 * num_mol_interp_pts_per_node + j3;

        double target_x = mol_clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = mol_clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = mol_clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_temp = 0.;
    
        for (std::size_t k = source_node_begin; k < source_node_end; ++k) {

            double dx = target_x - mol_x_ptr[k];
            double dy = target_y - mol_y_ptr[k];
            double dz = target_z - mol_z_ptr[k];

            pot_temp += mol_q_ptr[k] / eps_solute_ / std::sqrt(dx*dx + dy*dy + dz*dz);;
        }
    
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        mol_clusters_p_ptr[jj] += pot_temp;

    }
    }
    }

//    timers_.cluster_particle_interact.stop();
}


void CoulombicEnergyCompute::cluster_cluster_interact(std::size_t target_node_idx,
                                                      std::size_t source_node_idx)
{
//    timers_.cluster_cluster_interact.start();
        
    /* Targets */
    
    int num_mol_interp_potentials_per_node            = num_mol_interp_potentials_per_node_;
    
    std::size_t target_cluster_interp_pts_begin        = target_node_idx * num_mol_interp_pts_per_node_;
    std::size_t target_cluster_interp_potentials_begin = target_node_idx * num_mol_interp_potentials_per_node_;

    double*       __restrict mol_clusters_p_ptr       = mol_interp_potential_.data();
    
    
    /* Sources */
    
    int num_mol_interp_pts_per_node                 = num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node             = num_mol_interp_charges_per_node_;

    std::size_t source_cluster_interp_pts_begin     = source_node_idx * num_mol_interp_pts_per_node;
    std::size_t source_cluster_interp_charges_begin = source_node_idx * num_mol_interp_charges_per_node;
    
    const double* __restrict mol_clusters_x_ptr     = mol_interp_pts_.interp_x_ptr();
    const double* __restrict mol_clusters_y_ptr     = mol_interp_pts_.interp_y_ptr();
    const double* __restrict mol_clusters_z_ptr     = mol_interp_pts_.interp_z_ptr();
    
    const double* __restrict mol_clusters_q_ptr     = mol_interp_charge_.data();

#ifdef USE_CUDA_CC
    if (try_cluster_cluster_interact_cuda_(target_node_idx, source_node_idx)) {
        return;
    }
#endif


    for (int j1 = 0; j1 < num_mol_interp_pts_per_node; j1++) {
    for (int j2 = 0; j2 < num_mol_interp_pts_per_node; j2++) {
    for (int j3 = 0; j3 < num_mol_interp_pts_per_node; j3++) {
    
        std::size_t jj = target_cluster_interp_potentials_begin
                       + j1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                       + j2 * num_mol_interp_pts_per_node + j3;

        double target_x = mol_clusters_x_ptr[target_cluster_interp_pts_begin + j1];
        double target_y = mol_clusters_y_ptr[target_cluster_interp_pts_begin + j2];
        double target_z = mol_clusters_z_ptr[target_cluster_interp_pts_begin + j3];
        
        double pot_temp = 0.;
    
        for (int k1 = 0; k1 < num_mol_interp_pts_per_node; k1++) {
        for (int k2 = 0; k2 < num_mol_interp_pts_per_node; k2++) {
        for (int k3 = 0; k3 < num_mol_interp_pts_per_node; k3++) {
            
            std::size_t kk = source_cluster_interp_charges_begin
                           + k1 * num_mol_interp_pts_per_node * num_mol_interp_pts_per_node
                           + k2 * num_mol_interp_pts_per_node + k3;

            double dx = target_x - mol_clusters_x_ptr[source_cluster_interp_pts_begin + k1];
            double dy = target_y - mol_clusters_y_ptr[source_cluster_interp_pts_begin + k2];
            double dz = target_z - mol_clusters_z_ptr[source_cluster_interp_pts_begin + k3];

            pot_temp += mol_clusters_q_ptr[kk] / eps_solute_ / std::sqrt(dx*dx + dy*dy + dz*dz);;
        }
        }
        }
    
#ifdef OPENMP_ENABLED
        #pragma omp atomic update
#endif
        mol_clusters_p_ptr[jj] += pot_temp;
    }
    }
    }

//    timers_.cluster_cluster_interact.stop();
}


void CoulombicEnergyCompute::upward_pass()
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
    int weights_num = mol_weights_.size();
    int* exact_idx_x_ptr = exact_idx_x_.data();
    int* exact_idx_y_ptr = exact_idx_y_.data();
    int* exact_idx_z_ptr = exact_idx_z_.data();
    double* denominator_ptr = denominator_.data();

#ifdef USE_CUDA_CC
    if (try_upward_pass_cuda_()) {
        return;
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


void CoulombicEnergyCompute::downward_pass()
{
}

void CoulombicEnergyCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    copyin_clusters_to_device_cuda_();
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void CoulombicEnergyCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    delete_clusters_from_device_cuda_();
#endif

//    timers_.delete_clusters_from_device.stop();
}
