#include <cmath>
// #include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// #include "constants.h"
#include "coulombic_energy_compute.h"

#ifdef OPENACC_ENABLED
#include <openacc.h>
#endif
#ifdef USE_CUDA_CC
#include "coulombic_energy_cuda.h"
#endif
#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#endif

namespace {
constexpr int kCoulombicAsync = 9;
}


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

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_COULOMBIC_PP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_particle_particle_();
        if (present_ok) {
            const auto mol_dev = molecule_.device_view();
            const auto self_dev = device_view();
            void* stream = acc_get_cuda_stream(kCoulombicAsync);
            coulombic_pp_cuda(
                mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                target_node_begin,
                target_node_end,
                source_node_begin,
                source_node_end,
                eps_solute_,
                self_dev.coul_eng,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
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
    
    
#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_COULOMBIC_PC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_particle_cluster_();
        if (present_ok) {
            const auto mol_dev = molecule_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = acc_get_cuda_stream(kCoulombicAsync);
            coulombic_pc_cuda(
                mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                self_dev.q,
                source_node_idx,
                num_mol_interp_pts_per_node_,
                num_mol_interp_charges_per_node_,
                target_node_begin,
                target_node_end,
                eps_solute_,
                self_dev.coul_eng,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
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


#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_COULOMBIC_CP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_cluster_particle_();
        if (present_ok) {
            const auto mol_dev = molecule_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = acc_get_cuda_stream(kCoulombicAsync);
            coulombic_cp_cuda(
                mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                self_dev.p,
                mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                target_node_idx,
                num_mol_interp_pts_per_node_,
                num_mol_interp_potentials_per_node_,
                source_node_begin,
                source_node_end,
                eps_solute_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
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

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_COULOMBIC_CC");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_cluster_cluster_();
        if (present_ok) {
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = acc_get_cuda_stream(kCoulombicAsync);
            coulombic_cc_cuda(
                mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                self_dev.q, self_dev.p,
                target_node_idx,
                source_node_idx,
                num_mol_interp_pts_per_node_,
                num_mol_interp_charges_per_node_,
                num_mol_interp_potentials_per_node_,
                eps_solute_,
                stream);
            CUDA_CHECK_LAST_KERNEL();
            return;
        }
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

#if defined(OPENACC_ENABLED) && defined(USE_CUDA_CC)
    const char* env_disable = std::getenv("TABIPB_CUDA_COULOMBIC_UP");
    const bool use_cuda = !(env_disable && std::strcmp(env_disable, "0") == 0);
    if (use_cuda) {
        const bool present_ok = validate_device_buffers_upward_pass_();
        if (present_ok) {
            const auto mol_dev = molecule_.device_view();
            const auto mol_interp_dev = mol_interp_pts_.device_view();
            const auto self_dev = device_view();
            void* stream = acc_get_cuda_stream(kCoulombicAsync);
            for (std::size_t node_idx = 0; node_idx < source_tree_.num_nodes(); ++node_idx) {
                auto particle_idxs = source_tree_.node_particle_idxs(node_idx);
                std::size_t particle_start = particle_idxs[0];
                std::size_t num_particles = particle_idxs[1] - particle_idxs[0];
                if (num_particles == 0) {
                    continue;
                }
                coulombic_up_cuda(
                    mol_dev.particles_x, mol_dev.particles_y, mol_dev.particles_z, mol_dev.charge,
                    mol_interp_dev.interp_x, mol_interp_dev.interp_y, mol_interp_dev.interp_z,
                    self_dev.q,
                    self_dev.weights,
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
            #pragma acc wait(kCoulombicAsync)
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


void CoulombicEnergyCompute::downward_pass()
{
}

#ifdef USE_CUDA_CC
bool CoulombicEnergyCompute::validate_device_buffers_common_() const
{
    if (device_state_ != CudaDeviceState::DeviceMapped || !device_buffers_.ready) {
        return false;
    }
    if (!molecule_.cuda_device_ready() || !mol_interp_pts_.cuda_device_ready()) {
        return false;
    }

    const auto& buf = device_buffers_;
    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = mol_interp_potential_.size();
    const std::size_t coul_eng_num = coul_eng_vec_.size();
    const std::size_t weights_num = mol_weights_.size();
    const std::size_t scratch_num = exact_idx_x_.size();

    if (buf.q_num != q_num || buf.p_num != p_num ||
        buf.coul_eng_num != coul_eng_num || buf.weights_num != weights_num ||
        buf.scratch_num != scratch_num) {
        return false;
    }
    if ((q_num > 0 && !buf.q_dev) ||
        (p_num > 0 && !buf.p_dev) ||
        (coul_eng_num > 0 && !buf.coul_eng_dev) ||
        (weights_num > 0 && !buf.weights_dev) ||
        (scratch_num > 0 &&
         (!buf.exact_idx_x_dev || !buf.exact_idx_y_dev || !buf.exact_idx_z_dev ||
          !buf.denominator_dev))) {
        return false;
    }
    return true;
}

bool CoulombicEnergyCompute::validate_device_buffers_particle_particle_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_particle_cluster_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_cluster_particle_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_cluster_cluster_() const
{
    return validate_device_buffers_common_();
}

bool CoulombicEnergyCompute::validate_device_buffers_upward_pass_() const
{
    return validate_device_buffers_common_();
}
#endif


void CoulombicEnergyCompute::copyin_clusters_to_device() const
{
//    timers_.copyin_clusters_to_device.start();

#ifdef USE_CUDA_CC
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all =
        (require_all_env && std::strcmp(require_all_env, "0") != 0);

    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();
    
    const double* p_ptr = mol_interp_potential_.data();
    std::size_t p_num   = mol_interp_potential_.size();

    double* coul_eng_ptr = coul_eng_vec_.data();
    std::size_t coul_eng_num   = coul_eng_vec_.size();

    const double* weights_ptr = mol_weights_.data();
    std::size_t weights_num   = mol_weights_.size();
    std::size_t scratch_num   = max_mol_particles_per_node_;

    auto& buf = device_buffers_;
    if (buf.ready &&
        (buf.q_num != q_num || buf.p_num != p_num ||
         buf.coul_eng_num != coul_eng_num || buf.weights_num != weights_num ||
         buf.scratch_num != scratch_num)) {
        CUDA_FREE_AND_NULL(buf.q_dev);
        CUDA_FREE_AND_NULL(buf.p_dev);
        CUDA_FREE_AND_NULL(buf.coul_eng_dev);
        CUDA_FREE_AND_NULL(buf.weights_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
        CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
        CUDA_FREE_AND_NULL(buf.denominator_dev);
        buf = DeviceBuffers{};
    }

    if (!buf.ready &&
        (q_num > 0 || p_num > 0 || coul_eng_num > 0 || weights_num > 0 || scratch_num > 0)) {
        if (q_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.q_dev, q_num * sizeof(double));
            buf.q_num = q_num;
        }
        if (p_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.p_dev, p_num * sizeof(double));
            buf.p_num = p_num;
        }
        if (coul_eng_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.coul_eng_dev, coul_eng_num * sizeof(double));
            buf.coul_eng_num = coul_eng_num;
        }
        if (weights_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.weights_dev, weights_num * sizeof(double));
            buf.weights_num = weights_num;
        }
        if (scratch_num > 0) {
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_x_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_y_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.exact_idx_z_dev, scratch_num * sizeof(int));
            CUDA_MALLOC_OR_DIE(&buf.denominator_dev, scratch_num * sizeof(double));
            buf.scratch_num = scratch_num;
        }
        buf.ready = true;
    }

    cudaStream_t stream = nullptr;
#ifdef OPENACC_ENABLED
    stream = static_cast<cudaStream_t>(acc_get_cuda_stream(kCoulombicAsync));
#endif

    if (q_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.q_dev, q_ptr, q_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (p_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.p_dev, p_ptr, p_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (coul_eng_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.coul_eng_dev, coul_eng_ptr, coul_eng_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }
    if (weights_num > 0) {
        CUDA_MEMCPY_ASYNC(buf.weights_dev, weights_ptr, weights_num * sizeof(double),
                          cudaMemcpyHostToDevice, stream);
    }

    CUDA_SYNC_AND_CHECK();
    device_state_ = CudaDeviceState::DeviceMapped;

    if (require_all) {
        const bool ok =
            (q_num == 0 || (buf.q_dev && buf.q_num == q_num)) &&
            (p_num == 0 || (buf.p_dev && buf.p_num == p_num)) &&
            (coul_eng_num == 0 || (buf.coul_eng_dev && buf.coul_eng_num == coul_eng_num)) &&
            (weights_num == 0 || (buf.weights_dev && buf.weights_num == weights_num)) &&
            (scratch_num == 0 || (buf.exact_idx_x_dev && buf.exact_idx_y_dev &&
                                  buf.exact_idx_z_dev && buf.denominator_dev &&
                                  buf.scratch_num == scratch_num));
        if (!ok) {
            std::fprintf(stderr,
                         "[CUDA] CoulombicEnergyCompute missing or mismatched device buffers "
                         "under TABIPB_CUDA_REQUIRE_ALL=1\n");
            std::abort();
        }
    }
#elif defined(OPENACC_ENABLED)
    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();
    
    const double* p_ptr = mol_interp_potential_.data();
    std::size_t p_num   = mol_interp_potential_.size();

    const double* coul_eng_ptr = coul_eng_vec_.data();
    std::size_t coul_eng_num   = coul_eng_vec_.size();

    const double* weights_ptr = mol_weights_.data();
    std::size_t weights_num   = mol_weights_.size();
    int* exact_idx_x_ptr      = exact_idx_x_.data();
    int* exact_idx_y_ptr      = exact_idx_y_.data();
    int* exact_idx_z_ptr      = exact_idx_z_.data();
    double* denominator_ptr   = denominator_.data();
    std::size_t scratch_num   = max_mol_particles_per_node_;
    
    #pragma acc enter data copyin(q_ptr[0:q_num], p_ptr[0:p_num])
    #pragma acc enter data copyin(coul_eng_ptr[0:coul_eng_num])
    #pragma acc enter data copyin(weights_ptr[0:weights_num])
    #pragma acc enter data create(exact_idx_x_ptr[0:scratch_num], exact_idx_y_ptr[0:scratch_num], \
                                  exact_idx_z_ptr[0:scratch_num], denominator_ptr[0:scratch_num])
#endif

//    timers_.copyin_clusters_to_device.stop();
}


void CoulombicEnergyCompute::delete_clusters_from_device() const
{
//    timers_.delete_clusters_from_device.start();

#ifdef USE_CUDA_CC
    auto& buf = device_buffers_;

    const std::size_t q_num = mol_interp_charge_.size();
    const std::size_t p_num = mol_interp_potential_.size();
    const std::size_t coul_eng_num = coul_eng_vec_.size();
    const std::size_t weights_num = mol_weights_.size();
    const std::size_t scratch_num = max_mol_particles_per_node_;

    double* coul_eng_ptr = coul_eng_vec_.data();

    // Pull the scalar/vector result back before releasing device ownership.
    if (buf.coul_eng_dev && coul_eng_num > 0) {
        cudaStream_t stream = nullptr;
#ifdef OPENACC_ENABLED
        stream = static_cast<cudaStream_t>(acc_get_cuda_stream(kCoulombicAsync));
#endif
        CUDA_MEMCPY_ASYNC(coul_eng_ptr, buf.coul_eng_dev, coul_eng_num * sizeof(double),
                          cudaMemcpyDeviceToHost, stream);
        CUDA_SYNC_AND_CHECK();
    }

    CUDA_FREE_AND_NULL(buf.q_dev);
    CUDA_FREE_AND_NULL(buf.p_dev);
    CUDA_FREE_AND_NULL(buf.coul_eng_dev);
    CUDA_FREE_AND_NULL(buf.weights_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_x_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_y_dev);
    CUDA_FREE_AND_NULL(buf.exact_idx_z_dev);
    CUDA_FREE_AND_NULL(buf.denominator_dev);
    buf = DeviceBuffers{};
    device_state_ = CudaDeviceState::HostOnly;
#elif defined(OPENACC_ENABLED)
    const double* q_ptr = mol_interp_charge_.data();
    std::size_t q_num   = mol_interp_charge_.size();
    
    const double* p_ptr = mol_interp_potential_.data();
    std::size_t p_num   = mol_interp_potential_.size();
    
    const double* coul_eng_ptr = coul_eng_vec_.data();
    std::size_t coul_eng_num   = coul_eng_vec_.size();

    const double* weights_ptr = mol_weights_.data();
    std::size_t weights_num   = mol_weights_.size();
    int* exact_idx_x_ptr      = exact_idx_x_.data();
    int* exact_idx_y_ptr      = exact_idx_y_.data();
    int* exact_idx_z_ptr      = exact_idx_z_.data();
    double* denominator_ptr   = denominator_.data();
    std::size_t scratch_num   = max_mol_particles_per_node_;

    #pragma acc exit data delete(q_ptr[0:q_num], p_ptr[0:p_num])
    #pragma acc exit data copyout(coul_eng_ptr[0:coul_eng_num])
    #pragma acc exit data delete(weights_ptr[0:weights_num], \
                                 exact_idx_x_ptr[0:scratch_num], exact_idx_y_ptr[0:scratch_num], \
                                 exact_idx_z_ptr[0:scratch_num], denominator_ptr[0:scratch_num])
#endif

//    timers_.delete_clusters_from_device.stop();
}
