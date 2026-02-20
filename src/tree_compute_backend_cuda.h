#ifndef H_TABIPB_TREE_COMPUTE_BACKEND_CUDA_H
#define H_TABIPB_TREE_COMPUTE_BACKEND_CUDA_H

#include "tree_compute.h"

#ifdef USE_CUDA_CC
void tree_compute_run_cuda(TreeCompute& self, const TreeComputeView& view,
                           const TreeComputeBackendParams& params);
#endif

#endif
