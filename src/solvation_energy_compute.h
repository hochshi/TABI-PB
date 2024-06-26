#ifndef H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H
#define H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H

// #include "timer.h"
#include "elements.h"
#include "molecule.h"
#include "interp_pts.h"
#include "tree_compute.h"

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
    
    
    /* Source clusters */
    
    int num_mol_interp_pts_per_node_;
    int num_mol_interp_charges_per_node_;
    std::size_t num_mol_charges_;
    
    std::vector<double> mol_interp_charge_;
    
    
    /* Solvation energy */
    
    std::vector<double> solv_eng_vec_;
    double solvation_energy_;
    
    
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

    SolvationEnergyCompute(std::vector<double>& potential,
                      class Elements& elements, const class InterpolationPoints& elem_interp_pts,
                      const class Tree& elem_tree,
                      const class Molecule& molecule, const class InterpolationPoints& mol_interp_pts,
                      const class Tree& mol_tree,
                      const class InteractionList& interaction_list, double phys_eps, double phys_kappa);

    ~SolvationEnergyCompute() = default;
    
    double compute();
    
};

#endif /* H_TABIPB_SOLVATION_ENERGY_COMPUTE_STRUCT_H */
