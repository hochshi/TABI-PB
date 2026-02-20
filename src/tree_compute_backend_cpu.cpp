#include "tree_compute_backend_cpu.h"

void tree_compute_run_cpu(TreeCompute& self, const TreeComputeView& view,
                          const TreeComputeBackendParams& params)
{
    std::size_t pp_calls = 0;
    std::size_t pc_calls = 0;
    std::size_t cp_calls = 0;
    std::size_t cc_calls = 0;
#if defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t target_node_idx = 0; target_node_idx < view.target_tree->num_nodes(); ++target_node_idx) {
        for (auto source_node_idx : view.interaction_list->particle_particle(target_node_idx)) {
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: particle_particle_interact begin "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
            self.particle_particle_interact(view.target_tree->node_particle_idxs(target_node_idx),
                                            view.source_tree->node_particle_idxs(source_node_idx));
            ++pp_calls;
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: particle_particle_interact end "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
        }

        for (auto source_node_idx : view.interaction_list->particle_cluster(target_node_idx)) {
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: particle_cluster_interact begin "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
            self.particle_cluster_interact(view.target_tree->node_particle_idxs(target_node_idx),
                                           source_node_idx);
            ++pc_calls;
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: particle_cluster_interact end "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
        }

        for (auto source_node_idx : view.interaction_list->cluster_particle(target_node_idx)) {
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: cluster_particle_interact begin "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
            self.cluster_particle_interact(target_node_idx,
                                           view.source_tree->node_particle_idxs(source_node_idx));
            ++cp_calls;
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: cluster_particle_interact end "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
        }

        for (auto source_node_idx : view.interaction_list->cluster_cluster(target_node_idx)) {
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: cluster_cluster_interact begin "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
            self.cluster_cluster_interact(target_node_idx, source_node_idx);
            ++cc_calls;
            if (params.debug_verbose) {
                std::cerr << "[DEBUG] TreeCompute::run: cluster_cluster_interact end "
                          << "target_node=" << target_node_idx
                          << " source_node=" << source_node_idx << "\n";
            }
        }

        if (params.debug_progress && !params.debug_verbose) {
            const std::size_t done = target_node_idx + 1;
            const std::size_t total = view.target_tree->num_nodes();
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
    if (params.debug_progress) {
        std::cerr << "[DEBUG] TreeCompute::run: interaction loops end "
                  << "pp=" << pp_calls
                  << " pc=" << pc_calls
                  << " cp=" << cp_calls
                  << " cc=" << cc_calls << "\n";
    }
}
