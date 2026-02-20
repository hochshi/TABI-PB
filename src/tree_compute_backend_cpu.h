#ifndef H_TABIPB_TREE_COMPUTE_BACKEND_CPU_H
#define H_TABIPB_TREE_COMPUTE_BACKEND_CPU_H

#include "tree_compute.h"

void tree_compute_run_cpu(TreeCompute& self, const TreeComputeView& view,
                          const TreeComputeBackendParams& params);

#endif
