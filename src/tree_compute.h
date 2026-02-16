#ifndef H_TABIPB_TREE_COMPUTE_STRUCT_H
#define H_TABIPB_TREE_COMPUTE_STRUCT_H

//#include "particles.h"
//#include "interp_pts.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

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
    
    void run() {
        const char* debug_env = std::getenv("TABIPB_DEBUG_PROGRESS");
        const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
        const bool require_all =
            (require_all_env && std::strcmp(require_all_env, "0") != 0);
        const bool debug_progress =
            require_all || (debug_env && std::strcmp(debug_env, "0") != 0);
        const char* verbose_env = std::getenv("TABIPB_DEBUG_TREE_RUN_VERBOSE");
        const bool debug_verbose =
            (verbose_env && std::strcmp(verbose_env, "0") != 0);

        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: upward_pass begin\n";
        }
        upward_pass();
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: upward_pass end\n";
            std::cerr << "[DEBUG] TreeCompute::run: interaction loops begin\n";
        }

        std::size_t pp_calls = 0;
        std::size_t pc_calls = 0;
        std::size_t cp_calls = 0;
        std::size_t cc_calls = 0;
#if defined(OPENMP_ENABLED) && !defined(OPENACC_ENABLED)
        #pragma omp parallel for
#endif
        for (std::size_t target_node_idx = 0; target_node_idx < target_tree_.num_nodes(); ++target_node_idx) {
        
            for (auto source_node_idx : interaction_list_.particle_particle(target_node_idx)) {
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: particle_particle_interact begin "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
                particle_particle_interact(target_tree_.node_particle_idxs(target_node_idx),
                                           source_tree_.node_particle_idxs(source_node_idx));
                ++pp_calls;
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: particle_particle_interact end "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
            }

            for (auto source_node_idx : interaction_list_.particle_cluster(target_node_idx)) {
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: particle_cluster_interact begin "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
                particle_cluster_interact(target_tree_.node_particle_idxs(target_node_idx), source_node_idx);
                ++pc_calls;
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: particle_cluster_interact end "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
            }
            
            for (auto source_node_idx : interaction_list_.cluster_particle(target_node_idx)) {
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: cluster_particle_interact begin "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
                cluster_particle_interact(target_node_idx, source_tree_.node_particle_idxs(source_node_idx));
                ++cp_calls;
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: cluster_particle_interact end "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
            }
            
            for (auto source_node_idx : interaction_list_.cluster_cluster(target_node_idx)) {
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: cluster_cluster_interact begin "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
                cluster_cluster_interact(target_node_idx, source_node_idx);
                ++cc_calls;
                if (debug_verbose) {
                    std::cerr << "[DEBUG] TreeCompute::run: cluster_cluster_interact end "
                              << "target_node=" << target_node_idx
                              << " source_node=" << source_node_idx << "\n";
                }
            }

            if (debug_progress && !debug_verbose) {
                const std::size_t done = target_node_idx + 1;
                const std::size_t total = target_tree_.num_nodes();
                if (done == total || done % 10000 == 0) {
                    std::cerr << "[DEBUG] TreeCompute::run: interaction progress "
                              << done << "/" << total
                              << " pp=" << pp_calls
                              << " pc=" << pc_calls
                              << " cp=" << cp_calls
                              << " cc=" << cc_calls << "\n";
                }
            }
        }
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: interaction loops end "
                      << "pp=" << pp_calls
                      << " pc=" << pc_calls
                      << " cp=" << cp_calls
                      << " cc=" << cc_calls << "\n";
        }
#ifdef OPENACC_ENABLED
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: acc wait begin\n";
        }
        #pragma acc wait
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: acc wait end\n";
        }
#endif
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: downward_pass begin\n";
        }
        downward_pass();
        if (debug_progress) {
            std::cerr << "[DEBUG] TreeCompute::run: downward_pass end\n";
        }
    }
};


#endif /* H_TABIPB_TREE_COMPUTE_STRUCT_H */
