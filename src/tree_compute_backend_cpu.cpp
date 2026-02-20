#include "tree_compute_backend_cpu.h"

void tree_compute_run_cpu(TreeCompute& self, const TreeComputeView& view,
                          const TreeComputeBackendParams& params)
{
    (void)params;
#if defined(OPENMP_ENABLED)
    #pragma omp parallel for
#endif
    for (std::size_t target_node_idx = 0; target_node_idx < view.target_tree->num_nodes(); ++target_node_idx) {
        for (auto source_node_idx : view.interaction_list->particle_particle(target_node_idx)) {
            self.particle_particle_interact(view.target_tree->node_particle_idxs(target_node_idx),
                                            view.source_tree->node_particle_idxs(source_node_idx));
        }

        for (auto source_node_idx : view.interaction_list->particle_cluster(target_node_idx)) {
            self.particle_cluster_interact(view.target_tree->node_particle_idxs(target_node_idx),
                                           source_node_idx);
        }

        for (auto source_node_idx : view.interaction_list->cluster_particle(target_node_idx)) {
            self.cluster_particle_interact(target_node_idx,
                                           view.source_tree->node_particle_idxs(source_node_idx));
        }

        for (auto source_node_idx : view.interaction_list->cluster_cluster(target_node_idx)) {
            self.cluster_cluster_interact(target_node_idx, source_node_idx);
        }
    }
}
