#ifndef H_TABIPB_TREE_COMPUTE_STRUCT_H
#define H_TABIPB_TREE_COMPUTE_STRUCT_H

//#include "particles.h"
//#include "interp_pts.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifdef USE_CUDA_CC
#include "cuda_helpers.h"
#endif

#include "tree.h"
#include "interaction_list.h"


class TreeCompute
{
protected:
    const class Tree& source_tree_;
    const class Tree& target_tree_;
    const class InteractionList& interaction_list_;
    
    virtual void particle_particle_interact(std::array<std::size_t, 2> target_node_particle_idxs,
                                            std::array<std::size_t, 2> source_node_particle_idxs) = 0;
    
    virtual void particle_cluster_interact(std::array<std::size_t, 2> target_node_particle_idxs,
                                           std::size_t source_node_idx) = 0;
                                   
    virtual void cluster_particle_interact(std::size_t target_node_idx,
                                           std::array<std::size_t, 2> source_node_particle_idxs) = 0;
            
    virtual void cluster_cluster_interact(std::size_t target_node_idx, std::size_t source_node_idx) = 0;
            
    virtual void upward_pass() = 0;
    virtual void downward_pass() = 0;
    
    virtual void copyin_clusters_to_device() const = 0;
    virtual void delete_clusters_from_device() const = 0;

    
public:
    TreeCompute(const class Tree& source_tree, const class Tree& target_tree,
                const class InteractionList& interaction_list)
        : source_tree_(source_tree), target_tree_(target_tree), interaction_list_(interaction_list) {};
        
    TreeCompute(const class Tree& tree,
                const class InteractionList& interaction_list)
        : source_tree_(tree), target_tree_(tree), interaction_list_(interaction_list) {};
        
    virtual ~TreeCompute() = default;
    void run();
};


#endif /* H_TABIPB_TREE_COMPUTE_STRUCT_H */
